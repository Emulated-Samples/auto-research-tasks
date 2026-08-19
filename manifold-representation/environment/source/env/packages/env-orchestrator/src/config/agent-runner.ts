/**
 * Agent Runner
 *
 * Spawns the agent in a child process with an explicit environment allowlist.
 * The orchestrator is the gatekeeper: it decides which credentials the agent
 * can see based on agent type and config (awsAccess, etc.).
 *
 * Two modes:
 *
 * 1. "linux-user" (default): Child process runs as hyperfocal-agent via sudo.
 *    Filesystem lockdown via workspace-isolation.ts — the KERNEL enforces that
 *    the agent cannot read grader state, the gold solution, or env internals.
 *    Requires ensureAgentUser() and setupWorkspacePermissions() before spawn.
 *
 * 2. "claude-permissions" (opt-in): Child process runs as root.
 *    Tool-level sandboxing handled only by the Claude CLI's --allowedTools/
 *    --disallowedTools (string patterns, bypassable by a root agent). Use when
 *    an env genuinely needs a root agent; otherwise prefer linux-user.
 *
 * Both modes use the same allowlist env construction, so credential gating
 * (ANTHROPIC_API_KEY, AWS_*, etc.) is consistent regardless of mode.
 *
 * Why a child process instead of in-process?
 * Node's spawn() with the `env` option fully replaces the child's environment.
 * This prevents credentials from the orchestrator's process.env (injected by
 * main.ts for the environment module's AWS SDK, inherited from cloud-init, etc.)
 * from leaking to the agent. See main.ts lines 84-106 for the contract this upholds.
 */

import { spawn, execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { HyperfocalConfig } from "./yaml-config.js";
import {
  getProblemMetadata,
  type PermissionsMode,
} from "@hyperfocal/env-base";
import {
  ensureAgentUser,
  getAgentUser,
  lockdownSensitiveDirectories,
  setupWorkspacePermissions,
  setupClaudeCredentials,
} from "./workspace-isolation.js";
import {
  getCodexAuthPath,
  loadCredentials,
  loadLiteLlmProviderEnv,
  loadProviderCredentials,
} from "./credentials.js";
import {
  agentFailureFromSession,
  agentSessionKeys,
  latestNewAgentSession,
  type AgentFailureDetails,
} from "../internal/agent-failure.js";

export type AgentType = "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent";

export interface AgentRunOptions {
  prompt: string;
  workspacePath: string;
  model: string;
  config: HyperfocalConfig;
  apiKey: string;
  problemId: string;
  schemaPath?: string;
  agentType?: AgentType;
  permissionsMode?: PermissionsMode;
}

export interface AgentRunResult {
  exitCode: number;
  pid: number;
  timedOut: boolean;
  /** Detailed failure persisted by the child agent telemetry session. */
  failure?: AgentFailureDetails;
  /**
   * "exit"             — child process exited (any non-timeout reason).
   * "timeout"          — orchestrator killed the agent after the hard cap.
   * "aborted_upstream" — child's stderr contained an Anthropic API 5xx
   *                      pattern (e.g. "API Error: 529 Overloaded"). Set
   *                      regardless of exit code; the rollout should be
   *                      filtered from score distributions, not counted
   *                      as a hard failure. We do NOT auto-retry here
   *                      because the agent may have already mutated env
   *                      state (created publications, transitioned ticket,
   *                      pushed to gitea) and re-running mid-rollout would
   *                      corrupt the test signal.
   */
  exitReason: "exit" | "timeout" | "aborted_upstream";
}

/**
 * Matches Anthropic API 5xx error patterns that appear in claude-code /
 * codex / opencode stderr when an upstream API outage interrupts the agent.
 * Examples we've seen in traces:
 *   API Error: 529 Overloaded
 *   API Error: Overloaded
 *   API Error: 503 Service Unavailable
 * The regex is intentionally permissive (any 5xx) and matches both bare
 * "Overloaded" and explicit status codes.
 */
const UPSTREAM_5XX_REGEX = /API Error:\s*(5\d\d|Overloaded)/i;

/** Bytes of child stderr we keep in a rolling buffer for post-exit scanning. */
const STDERR_TAIL_BYTES = 8192;

/**
 * Build the environment allowlist for the agent child process.
 *
 * Starts with a minimal base (PATH, HOME, LANG) and conditionally adds
 * credentials based on agent type and config:
 * - ANTHROPIC_API_KEY: only for anthropic-coding (Claude Code uses OAuth)
 * - LiteLLM provider API keys: only for mini-swe-agent
 * - AWS_*: only when config.agent.awsAccess is true
 */
function buildAgentEnv(options: {
  agentType: AgentType;
  apiKey: string;
  config: HyperfocalConfig;
  codexHome?: string;
  isLegacyMode?: boolean;
}): Record<string, string> {
  const { agentType, apiKey, config, codexHome, isLegacyMode = false } = options;

  // CLI agents need HOME pointed at the auth/config directory we prepared:
  // - Claude Code: ~/.claude/.credentials.json
  // - OpenCode: ~/.local/share/opencode/auth.json + provider registry
  // - Codex: CODEX_HOME defaults under ~/.codex when not overridden
  // - mini-swe-agent: ~/.local/share/mini-swe-agent runtime/config files
  // In linux-user mode the child runs as hyperfocal-agent, so HOME must
  // follow the prepared /home/<agent> auth tree instead of /root.
  // AnthropicCodingAgent has no $HOME dependency, uses workspace for isolation.
  const homeDir = (
    agentType === "claude-code" ||
    agentType === "opencode" ||
    agentType === "codex" ||
    agentType === "mini-swe-agent"
  )
    ? (isLegacyMode ? `/home/${getAgentUser()}` : "/root")
    : "/hyperfocal/env/workspace";

  const agentEnv: Record<string, string> = {
    PATH: process.env.PATH!,
    HOME: homeDir,
    SHELL: "/bin/bash",
    LANG: process.env.LANG || "en_US.UTF-8",
    CLAUDE_CODE_MAX_OUTPUT_TOKENS: process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS || "512000",
    // Git defense-in-depth: hide /hyperfocal/env/.git from upward walks.
    // Git's ceiling means "don't chdir up INTO this dir" — so we list
    // /hyperfocal/env (the dir containing the .git we want to hide), NOT
    // /hyperfocal/env/workspace (which is the agent's cwd; ceiling is
    // exclusive of cwd, so listing the cwd is a no-op). With this, an
    // agent running `git log` from anywhere under workspace gets
    // "not a git repository" instead of leaking gold-state history.
    // Doesn't close `cd /hyperfocal/env && git log` (cwd-direct attack);
    // see DEFAULT_DISALLOWED_TOOLS in env-base for subcommand-level blocks.
    GIT_CEILING_DIRECTORIES: "/hyperfocal/env",
  };

  // Env-declared agent knobs (hyperfocal.yaml `agent.env`). Namespace-gated:
  // yaml may tune HYPERFOCAL_CLAUDE_* agent behavior (idle cutoffs, the
  // background-task reaper) but can never inject credentials or arbitrary
  // process config into the child. Applied BEFORE the process-env loop below
  // so a run-level override (run-request envVars) always wins over the repo.
  for (const [key, value] of Object.entries(config.agent?.env ?? {})) {
    if (key.startsWith("HYPERFOCAL_CLAUDE_") && typeof value === "string") {
      agentEnv[key] = value;
    } else {
      console.warn(
        `Ignoring hyperfocal.yaml agent.env["${key}"]: only HYPERFOCAL_CLAUDE_* string values may cross into the agent process`
      );
    }
  }
  for (const key of [
    "HYPERFOCAL_CLAUDE_RESULT_IDLE_CUTOFF",
    "HYPERFOCAL_CLAUDE_RESULT_IDLE_NO_PENDING_MS",
    "HYPERFOCAL_CLAUDE_RESULT_IDLE_PENDING_MS",
    "HYPERFOCAL_CLAUDE_BG_TASK_MAX_AGE_MS",
    "HYPERFOCAL_CLAUDE_BG_TASK_REAP_SYNTH_GRACE_MS",
  ]) {
    if (process.env[key] !== undefined) {
      agentEnv[key] = process.env[key]!;
    }
  }

  // Claude Code uses OAuth from ~/.claude/.credentials.json -- never needs ANTHROPIC_API_KEY.
  // AnthropicCodingAgent calls the API directly and requires it.
  // OpenCode reads auth.json from $HOME/.local/share/opencode/ -- no env var keys needed.
  // Codex prefers CODEX_HOME/auth.json, but CODEX_API_KEY is supported for exec automation.
  // mini-swe-agent uses LiteLLM, so it receives only explicit provider API keys.
  if (agentType === "anthropic-coding") {
    agentEnv.ANTHROPIC_API_KEY = apiKey;
  }

  const creds = loadCredentials(config);
  if (agentType === "codex") {
    if (codexHome) {
      agentEnv.CODEX_HOME = codexHome;
    }
    if (creds.codexApiKey) {
      agentEnv.CODEX_API_KEY = creds.codexApiKey;
    }
  }

  if (agentType === "mini-swe-agent") {
    Object.assign(agentEnv, loadLiteLlmProviderEnv(config));
    agentEnv.MSWEA_CONFIGURED = "true";
    agentEnv.MSWEA_SILENT_STARTUP = "1";
    agentEnv.MSWEA_MODEL_NAME = config.agent.defaultModel;
    if (process.env.HYPERFOCAL_MINI_SWE_AGENT_BIN) {
      agentEnv.HYPERFOCAL_MINI_SWE_AGENT_BIN = process.env.HYPERFOCAL_MINI_SWE_AGENT_BIN;
    }
  }

  // Gate AWS credentials on config. The orchestrator's own process.env has AWS_*
  // injected by main.ts for the environment module's SDK clients. Without this
  // allowlist, those would leak to the agent even when awsAccess is false.
  if (config.agent.awsAccess) {
    const awsCreds = creds.aws;
    if (awsCreds) {
      console.log("  Agent will have AWS access (from .env file)");
      agentEnv.AWS_ACCESS_KEY_ID = awsCreds.accessKeyId;
      agentEnv.AWS_SECRET_ACCESS_KEY = awsCreds.secretAccessKey;
      if (awsCreds.sessionToken) {
        agentEnv.AWS_SESSION_TOKEN = awsCreds.sessionToken;
      }
      agentEnv.AWS_REGION = awsCreds.region;
      if (awsCreds.accountId) {
        agentEnv.AWS_ACCOUNT_ID = awsCreds.accountId;
      }
    } else {
      console.log("  Agent will NOT have AWS access (no credentials in .env or process.env)");
    }
  } else {
    console.log("  Agent will NOT have AWS access (awsAccess: false in config)");
  }

  return agentEnv;
}

/**
 * Log the agent environment for auditability.
 * Redacts values for keys containing SECRET, KEY, or TOKEN.
 */
function logAgentEnv(agentEnv: Record<string, string>): void {
  console.log("\nAgent environment (what the agent will see):");
  for (const [key, value] of Object.entries(agentEnv)) {
    if (key.includes("SECRET") || key.includes("KEY") || key.includes("TOKEN")) {
      console.log(`  ${key}=${value.slice(0, 12)}...`);
    } else {
      console.log(`  ${key}=${value}`);
    }
  }
  console.log();
}

/**
 * Run the agent in a child process with a controlled environment.
 *
 * Always spawns a separate process via _internal-run-agent, passing only
 * the allowlisted environment variables. The permissions mode determines
 * whether the child runs as root (claude-permissions) or as the
 * hyperfocal-agent user via sudo (linux-user).
 */
export async function runAgent(
  options: AgentRunOptions
): Promise<AgentRunResult> {
  const {
    prompt,
    workspacePath,
    model,
    config,
    apiKey,
    problemId,
    schemaPath,
    agentType = "claude-code",
    permissionsMode = "linux-user",
  } = options;

  const isLegacyMode = permissionsMode === "linux-user";
  // Snapshot before spawning so an old failed session cannot be mistaken for
  // this execution if the child fails before creating telemetry.
  const previousAgentSessionKeys = agentSessionKeys(getProblemMetadata(problemId));

  // --- Legacy mode prerequisites ---
  // These must run before the child process spawns because they create the
  // user account, set workspace ownership, and share Claude CLI credentials.
  if (isLegacyMode) {
    ensureAgentUser();
    setupWorkspacePermissions(workspacePath);
    if (agentType === "claude-code") {
      setupClaudeCredentials();
    }
    // Reapply lock-down in rollout path (env setup command may not have been run).
    lockdownSensitiveDirectories(config.paths.root);
  }

  // --- OpenCode: write auth.json before spawning ---
  if (agentType === "opencode") {
    const providerCreds = loadProviderCredentials(config);
    const authHomeDir = isLegacyMode
      ? `/home/${getAgentUser()}`
      : (process.env.HOME || "/root");
    setupOpenCodeAuth(providerCreds, authHomeDir);

    if (isLegacyMode) {
      const agentUser = getAgentUser();
      execSync(`chown -R ${agentUser}:${agentUser} ${authHomeDir}`);
    }
  }

  // --- Codex: create isolated CODEX_HOME before spawning ---
  const codexHome = agentType === "codex"
    ? setupCodexRuntimeHome(config, isLegacyMode)
    : undefined;

  // --- Write prompt file for child process ---
  const promptFile = path.join(workspacePath, ".agent-prompt.txt");
  fs.writeFileSync(promptFile, prompt);

  if (isLegacyMode) {
    const agentUser = getAgentUser();
    execSync(`chown ${agentUser}:${agentUser} ${promptFile}`);
  }

  // --- Stage the output schema where the agent can read it ---
  // Some agent CLIs read the schema FILE by path (e.g. Codex's --output-schema).
  // The schema normally lives under environment/, which lockdownSensitiveDirectories
  // locks root-only — so under linux-user the unprivileged agent gets EACCES and the
  // CLI aborts before doing anything (Codex: "Failed to read output schema file ...
  // Permission denied"). The schema is NOT secret — its content is interpolated into
  // the prompt for agents like claude-code — so stage a world-readable copy outside
  // the locked tree and point the CLI at that. (claude-code is unaffected either way;
  // it never reads the file.)
  let agentSchemaPath = schemaPath;
  if (schemaPath) {
    try {
      const stagedDir = path.join(workspacePath, ".hyperfocal");
      fs.mkdirSync(stagedDir, { recursive: true });
      const stagedSchema = path.join(stagedDir, "agent-output-schema.json");
      fs.copyFileSync(schemaPath, stagedSchema);
      fs.chmodSync(stagedSchema, 0o644);
      agentSchemaPath = stagedSchema;
    } catch (error) {
      console.warn(
        `  Warning: failed to stage output schema for the agent: ${error}. ` +
        `Passing the original path (may be unreadable under linux-user).`
      );
    }
  }

  // --- Build allowlist environment ---
  const agentEnv = buildAgentEnv({ agentType, apiKey, config, codexHome, isLegacyMode });
  logAgentEnv(agentEnv);

  // --- Resolve paths ---
  const orchestratorPath = path.join(
    config.paths.root,
    "packages/env-orchestrator/bin/env-orchestrator.js"
  );

  let nodePath = "node";
  try {
    nodePath = execSync("which node", { encoding: "utf-8" }).trim();
  } catch {
    console.warn("  Warning: Could not find node path, using 'node'");
  }

  // --- Build CLI args for _internal-run-agent ---
  const agentCliArgs = [
    orchestratorPath,
    "_internal-run-agent",
    "--prompt-file", promptFile,
    "--workspace", workspacePath,
    "--model", model,
    "--problem-id", problemId,
    "--agent-type", agentType,
    "--permissions-mode", permissionsMode,
  ];

  if (agentSchemaPath) {
    agentCliArgs.push("--schema-file", agentSchemaPath);
  }
  // Diagnostic flag for OpenCode silent-termination investigations.
  // Read here (orchestrator boundary) once and forwarded as a CLI flag
  // so the child process doesn't need to inherit OPENCODE_DEBUG via the
  // env allowlist, and OpenCodeAgent stays free of process.env reads.
  if (process.env.OPENCODE_DEBUG === "1" && agentType === "opencode") {
    agentCliArgs.push("--opencode-debug");
  }
  // Raw coding-agent transcript capture is enabled by default for now so
  // fresh rollouts preserve provider protocol evidence while Codex support
  // is stabilizing. Set HYPERFOCAL_CODING_AGENT_DEBUG_LOGS=0 to opt out.
  //
  // TODO(coding-agent-debug-logs): split the default by runtime once the
  // control plane gives the orchestrator an explicit dev-container vs rollout
  // signal. Dev containers should default on; production rollouts may default
  // off once Codex telemetry is stable.
  if (process.env.HYPERFOCAL_CODING_AGENT_DEBUG_LOGS !== "0" &&
      process.env.HYPERFOCAL_CODING_AGENT_DEBUG_LOGS !== "false") {
    agentCliArgs.push("--coding-agent-debug-logs");
  }
  // Per-env additive tool overrides (hyperfocal.yaml agent.disallowedTools /
  // allowedTools). The parent (root) reads them from config and threads them to
  // the child as base64'd JSON. This MUST be passed in: under linux-user the
  // child runs as the unprivileged agent and cannot read hyperfocal.yaml (it is
  // locked root-only by lockdownSensitiveDirectories), so the child can no
  // longer loadConfig() it itself. base64 avoids arg-splitting on patterns that
  // contain commas/spaces (e.g. "Bash(git *)").
  const encodeList = (xs: string[]) =>
    Buffer.from(JSON.stringify(xs)).toString("base64");
  agentCliArgs.push(
    "--yaml-disallowed-tools", encodeList(config.agent?.disallowedTools ?? []),
    "--yaml-allowed-tools", encodeList(config.agent?.allowedTools ?? []),
  );
  // --- Spawn child process ---
  // `detached: true` puts the child in its own process group so we can SIGKILL
  // the whole tree (claude → bash → ssh → remote pgrep loop, etc.) on timeout
  // or after exit. Orphaned bash/SSH from the agent's Bash tool would otherwise
  // outlive the agent process and inherit stdio fds; we settle on the agent
  // wrapper's 'exit' event (process-end) rather than 'close' (stdio-drained)
  // so those orphans don't block completion, then SIGKILL the group to reap
  // them before the test phase runs.
  let child;

  if (isLegacyMode) {
    // Legacy: run as hyperfocal-agent via sudo.
    // Uses `sudo -u <user> env -i KEY=VALUE ... node <script>` to fully
    // replace the environment with only the allowlisted variables. The -i
    // flag is critical: without it, sudo injects SHELL from /etc/passwd
    // (set to /bin/false for the system account) and SUDO_* metadata,
    // which leak into the child process.
    const agentUser = getAgentUser();
    const envArgs = Object.entries(agentEnv).map(([key, value]) => `${key}=${value}`);

    console.log(`Starting agent as ${agentUser} (permissions: ${permissionsMode})...`);
    console.log(`  Using node: ${nodePath}`);

    child = spawn("sudo", ["-u", agentUser, "env", "-i", ...envArgs, nodePath, ...agentCliArgs], {
      cwd: workspacePath,
      // stdio piped (not inherited) so we can tee a rolling stderr tail for
      // post-exit 5xx detection. Output is still forwarded to the
      // orchestrator's stdout/stderr below via .pipe(), so console behavior
      // is unchanged for human observers.
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    });
  } else {
    console.log(`Starting agent (permissions: ${permissionsMode})...`);
    console.log(`  Using node: ${nodePath}`);

    child = spawn(nodePath, agentCliArgs, {
      cwd: workspacePath,
      env: agentEnv,
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    });
  }

  // Tee child stdout/stderr through to the orchestrator console (preserving
  // the prior "inherit" UX) while capturing a rolling tail of stderr so we
  // can scan for Anthropic API 5xx patterns after the child exits. The tail
  // is bounded so a long-running rollout doesn't accumulate memory.
  let stderrTail = "";
  if (child.stdout) child.stdout.pipe(process.stdout);
  if (child.stderr) {
    child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(chunk);
      stderrTail += chunk.toString("utf-8");
      if (stderrTail.length > STDERR_TAIL_BYTES) {
        stderrTail = stderrTail.slice(stderrTail.length - STDERR_TAIL_BYTES);
      }
    });
  }

  const pid = child.pid!;
  console.log(`Agent started with PID: ${pid} (process group: ${pid})`);

  // Hard cap on the agent phase. Without this, a hung child (claude leaving
  // orphan SSH/docker exec processes) blocks the orchestrator forever and the
  // test phase never runs. Override via HYPERFOCAL_AGENT_TIMEOUT_MS.
  const agentTimeoutMs = Number(process.env.HYPERFOCAL_AGENT_TIMEOUT_MS) || 180 * 60 * 1000;

  function killGroup(signal: NodeJS.Signals): void {
    try {
      // Negative pid → entire process group.
      process.kill(-pid, signal);
    } catch {
      // Group already gone, or we're not allowed; fall back to direct kill.
      try { process.kill(pid, signal); } catch { /* ignore */ }
    }
  }

  return new Promise((resolve) => {
    let settled = false;
    let timedOut = false;
    let spawnErrorMessage: string | undefined;

    const timeout = setTimeout(() => {
      timedOut = true;
      console.error(
        `Agent exceeded ${Math.round(agentTimeoutMs / 1000)}s timeout — sending SIGTERM to process group ${pid}`
      );
      killGroup("SIGTERM");
      // Hard kill 10s later if SIGTERM didn't take.
      setTimeout(() => killGroup("SIGKILL"), 10_000);
    }, agentTimeoutMs);

    function settle(exitCode: number): void {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      try { fs.unlinkSync(promptFile); } catch { /* ignore */ }
      // Remove the staged schema copy too (only when we actually staged one).
      if (agentSchemaPath && agentSchemaPath !== schemaPath) {
        try { fs.unlinkSync(agentSchemaPath); } catch { /* ignore */ }
      }
      // Always clean up the process group on exit so leftover bash/ssh/docker
      // exec children from the agent's Bash tool don't survive into the test
      // phase (where they could block grader queries through pgbouncer, etc.).
      killGroup("SIGKILL");

      // Detect upstream API outages so the batch aggregator can filter these
      // rollouts out of score distributions. We do this AFTER timeout takes
      // precedence — a hard cap is a real env outcome, an API 5xx is not.
      let exitReason: AgentRunResult["exitReason"] = timedOut ? "timeout" : "exit";
      if (!timedOut && UPSTREAM_5XX_REGEX.test(stderrTail)) {
        exitReason = "aborted_upstream";
        const match = stderrTail.match(UPSTREAM_5XX_REGEX);
        console.warn(
          `Agent failure attributed to upstream API outage (matched: "${match?.[0]}"). ` +
          `Surfacing exitReason="aborted_upstream" so batch analysis can filter this rollout. ` +
          `Not auto-retrying — agent may have already mutated env state mid-rollout.`
        );
      }
      const latestSession = latestNewAgentSession(
        getProblemMetadata(problemId),
        previousAgentSessionKeys
      );
      const failure = exitCode === 0
        ? undefined
        : agentFailureFromSession(
            latestSession,
            spawnErrorMessage || `Agent exited with code ${exitCode}`
          );
      resolve({ exitCode, pid, timedOut, exitReason, failure });
    }

    // Listen on 'exit', not 'close'. 'exit' fires when the agent wrapper
    // process ends — which is exactly the signal we want. 'close' fires
    // later, after all stdio streams of the child AND its descendants drain;
    // when the agent's Bash tool leaves backgrounded ssh/bash/nohup-spawned
    // commands behind (a common pattern for cutover scripts, sleep loops,
    // pgbouncer KILL handlers), those orphan grandchildren inherit stdio
    // fds and prevent 'close' from ever firing — the orchestrator then sits
    // through the full HYPERFOCAL_AGENT_TIMEOUT_MS waiting for stdio drain
    // that will never happen. settle() runs killGroup("SIGKILL") which
    // reaps the orphans, so we're not leaking processes by switching events.
    child.on("exit", (code) => {
      const finalCode = timedOut ? 124 : (code ?? 1);
      const reason = timedOut ? " (timeout)" : "";
      console.log(`Agent exited with code: ${finalCode}${reason}`);
      settle(finalCode);
    });

    child.on("error", (error) => {
      spawnErrorMessage = `Failed to start agent: ${error.message}`;
      console.error(`Failed to start agent: ${error.message}`);
      settle(1);
    });
  });
}

/**
 * Write OpenCode's auth.json to the target user's data directory.
 *
 * auth.json format (from OpenCode's Zod schema in auth/index.ts):
 *   { "provider": { type: "api", key: "..." } | { type: "oauth", access, refresh, expires } }
 *
 * Called by runAgent() before spawning the child process so credentials
 * are on disk when OpenCode starts.
 */
function setupOpenCodeAuth(
  credentials: Record<string, unknown>,
  homeDir: string
): void {
  const authDir = path.join(homeDir, ".local/share/opencode");
  const authFile = path.join(authDir, "auth.json");

  fs.mkdirSync(authDir, { recursive: true });
  fs.writeFileSync(authFile, JSON.stringify(credentials, null, 2), { mode: 0o600 });

  const providers = Object.keys(credentials);
  console.log(`  Wrote OpenCode auth.json with providers: ${providers.join(", ")}`);
}

/**
 * Create a per-run CODEX_HOME so generated config.toml and copied auth do not
 * mutate the global ~/.codex directory. CodexAgent writes config.toml there.
 */
function setupCodexRuntimeHome(
  config: HyperfocalConfig,
  isLegacyMode: boolean
): string {
  const homeRoot = isLegacyMode
    ? `/home/${getAgentUser()}`
    : (process.env.HOME || "/root");
  const runtimeDir = path.join(
    homeRoot,
    ".local/share/codex/runtime/hyperfocal",
    `${Date.now()}-${process.pid}`
  );
  fs.mkdirSync(runtimeDir, { recursive: true });

  const sourceAuth = getCodexAuthPath(process.env.HOME || "/root");
  const targetAuth = path.join(runtimeDir, "auth.json");
  const creds = loadCredentials(config);

  if (fs.existsSync(sourceAuth)) {
    fs.copyFileSync(sourceAuth, targetAuth);
    fs.chmodSync(targetAuth, 0o600);
    console.log(`  Copied Codex auth.json into isolated CODEX_HOME: ${runtimeDir}`);
  } else if (creds.codexApiKey) {
    console.log("  Codex will authenticate with CODEX_API_KEY from .env/process.env");
  } else {
    throw new Error(
      "Codex credentials not found. Run `codex login` to create ~/.codex/auth.json " +
      "or set CODEX_API_KEY in /hyperfocal/env/environment/.env."
    );
  }

  if (isLegacyMode) {
    const agentUser = getAgentUser();
    execSync(`chown -R ${agentUser}:${agentUser} ${runtimeDir}`);
  }

  return runtimeDir;
}
