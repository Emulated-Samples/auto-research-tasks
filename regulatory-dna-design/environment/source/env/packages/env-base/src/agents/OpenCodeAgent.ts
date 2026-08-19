/**
 * OpenCode Agent
 *
 * Wraps the OpenCode CLI (anomalyco/opencode) for multi-provider coding tasks.
 * Uses OpenCode's built-in tools with nd-JSON streaming output.
 *
 * Key differences from ClaudeCodeAgent:
 * - Multi-provider: supports anthropic, openai, google, xai, openrouter, etc.
 * - Auth via auth.json (not OAuth credentials file)
 * - nd-JSON event format: step_start, tool_use, text, step_finish, error
 * - Model format: "provider/model" (e.g. "anthropic/claude-opus-4-6")
 * - No --max-turns flag — long-horizon tasks with process timeout as safety net
 * - All tool permissions auto-approved in `run` mode (no --yolo needed)
 *
 * Prerequisites:
 * - OpenCode CLI installed (auto-detected via `which opencode`)
 * - auth.json written to ~/.local/share/opencode/auth.json (by orchestrator)
 */

import { execSync, ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { OpenCodeConfiguration, AgentRunOptions, PermissionsMode } from "../types.js";
import type { TelemetrySession } from "../telemetry/index.js";
import type { LogEventType } from "../telemetry/types.js";
import { readMcpServersFromPath } from "../mcp/index.js";
import type { McpServerSpec, McpServersFile } from "../mcp/index.js";
import { shellResultEvent, toolCallEvent, type ToolFamily } from "./events.js";
import { runStreamingCli } from "./streaming-cli.js";

function classifyOpenCodeToolFamily(tool: string): ToolFamily {
  const lower = tool.toLowerCase();
  if (lower === "bash" || lower === "shell") return "shell";
  if (
    lower === "read" ||
    lower === "edit" ||
    lower === "write" ||
    lower === "glob" ||
    lower === "grep"
  ) {
    return "file";
  }
  if (lower === "todowrite" || lower === "todo_list") return "todo";
  if (lower.startsWith("mcp:")) return "mcp";
  return "generic";
}

/**
 * Per-build iteration ceiling injected into the synthesized opencode.json.
 *
 * OpenCode's `run` mode otherwise caps a session at its own internal
 * default (~100 steps in v1.15.x). The cap is a budget, not a hard fence:
 * when the model reaches the ceiling, OpenCode forcibly switches it into
 * "summarize and stop" mode — a successful but truncated session. We
 * raise it well above typical long-horizon cutover task usage and rely on
 * the close-handler check (lastStepFinishReason !== "stop") to surface a
 * truncation as a failed rollout instead of a falsely "completed" one.
 */
const OPENCODE_BUILD_STEPS = 600;

/**
 * Retry policy for transient provider errors (HTTP 429 / rate-limit).
 *
 * Sized for long-horizon environments (e.g. pg-engine-feature-cutover)
 * where losing 10+ minutes of progress to a noisy-neighbor TPM 429 is
 * the dominant failure mode. The exponential backoff schedule is
 * 15s → 30s → 60s → 120s (plus up-to-1s jitter), so 5 attempts spend
 * at most ~4 min of wall clock on sleeps. `maxTotalMs` is the hard
 * fence — beyond it we surface the rate-limit error and let the
 * orchestrator decide whether to re-rollout.
 */
const OPENCODE_RETRY_CONFIG = {
  maxAttempts: 5,
  baseDelayMs: 15_000,
  maxTotalMs: 15 * 60 * 1_000,
} as const;

/**
 * Continuation prompt sent on resumed attempts via `opencode run
 * --session <id>`. Deliberately minimal: tells the model the prior
 * tool results, todos, and plan are still valid so it doesn't reset
 * its work after a transient provider hiccup.
 */
const OPENCODE_RESUME_PROMPT =
  "Resume the previous task. Your earlier plan, todos, and tool " +
  "results are still valid — continue from where you left off " +
  "instead of restarting.";

/**
 * Outcome of a single `opencode run` invocation. Distinguishes
 * "transient provider rate-limit, safe to retry" from "fatal, give up
 * now" so the retry loop in `run()` can route on `kind` instead of
 * regex-matching error strings.
 */
type RunOutcome =
  | { kind: "completed"; stepCount: number; sessionID?: string }
  | {
      kind: "rate_limit";
      message: string;
      stepCount: number;
      sessionID?: string;
    }
  | {
      kind: "provider_error";
      message: string;
      stepCount: number;
      sessionID?: string;
    }
  | {
      kind: "permission_rejected";
      message: string;
      stepCount: number;
      sessionID?: string;
    }
  | {
      kind: "incomplete";
      message: string;
      stepCount: number;
      sessionID?: string;
    }
  | {
      kind: "cli_error";
      message: string;
      stepCount: number;
      sessionID?: string;
    };

/**
 * Inspect an OpenCode `error` event payload and decide whether it's a
 * transient provider rate-limit (HTTP 429) we can retry or a fatal
 * error that should stop the rollout.
 *
 * The 429 envelope OpenAI returns is stringified JSON nested inside
 * `error.data.message`, e.g.:
 *   "{\"code\":429,\"message\":\"Request too large...\",\"metadata\":{\"error_type\":\"rate_limit_exceeded\"}}"
 * OpenCode does NOT populate `error.data.statusCode` for provider errors,
 * so we parse the inner JSON. Anthropic/other providers may emit
 * different shapes; the substring fallback catches the common cases.
 */
function classifyOpenCodeError(
  rawMessage: string,
  statusCode?: number,
): { kind: "rate_limit" | "provider_error"; message: string } {
  let code: number | undefined = statusCode;
  let errorType: string | undefined;
  try {
    const parsed = JSON.parse(rawMessage) as {
      code?: unknown;
      metadata?: { error_type?: unknown };
    };
    if (typeof parsed?.code === "number") code = parsed.code;
    if (typeof parsed?.metadata?.error_type === "string") {
      errorType = parsed.metadata.error_type;
    }
  } catch {
    // Not JSON — fall back to substring match below.
  }

  const isRateLimit =
    code === 429 ||
    errorType === "rate_limit_exceeded" ||
    /\brate[_ ]?limit/i.test(rawMessage);

  if (isRateLimit) {
    const preview =
      rawMessage.length > 240 ? rawMessage.slice(0, 240) + "…" : rawMessage;
    return {
      kind: "rate_limit",
      message: `Provider rate limit (HTTP 429): ${preview}`,
    };
  }
  return {
    kind: "provider_error",
    message: `OpenCode error: ${rawMessage}${code ? ` (${code})` : ""}`,
  };
}

function extractPermissionRejection(stderr: string): string | undefined {
  const lines = stderr
    .split(/\r?\n/)
    .filter((line) => /permission requested: .*auto-rejecting/i.test(line));
  return lines.at(-1)?.trim();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Exponential backoff with up-to-1s of jitter, clamped to whatever
 * wall-clock budget is left in the overall retry window. `attempt` is
 * 1-indexed from the perspective of "how many failures so far"; the
 * first sleep (after attempt 1's failure) uses `baseDelayMs`.
 */
function computeBackoffMs(
  attempt: number,
  baseDelayMs: number,
  remainingMs: number,
): number {
  const raw = baseDelayMs * Math.pow(2, attempt - 1);
  const jitter = Math.floor(Math.random() * 1000);
  return Math.min(remainingMs, raw + jitter);
}

/**
 * Translate the neutral `McpServersFile` into OpenCode's `opencode.json`
 * shape. OpenCode wants `{ mcp: { name: { type: "remote"|"local",
 * enabled: true, ... } } }` — different field names than Claude's CLI:
 *
 *   neutral transport "http"  → opencode type "remote" (url + enabled)
 *   neutral transport "stdio" → opencode type "local"  (command + environment + enabled)
 */
function toOpencodeMcp(file: McpServersFile): Record<string, Record<string, unknown>> {
  const mcp: Record<string, Record<string, unknown>> = {};
  for (const [name, spec] of Object.entries(file.servers)) {
    mcp[name] = opencodeServerEntry(spec);
  }
  return mcp;
}

function opencodeServerEntry(spec: McpServerSpec): Record<string, unknown> {
  if (spec.transport === "http") {
    return { type: "remote", url: spec.url, enabled: true };
  }
  // stdio
  const out: Record<string, unknown> = {
    type: "local",
    command: spec.command,
    enabled: true,
  };
  if (spec.env) out.environment = spec.env;
  return out;
}

/**
 * nd-JSON event types from OpenCode's --format json output.
 * Verified experimentally against OpenCode v1.2.10.
 */
interface OpenCodeStepStart {
  type: "step_start";
  timestamp: number;
  sessionID: string;
  part: {
    id: string;
    sessionID: string;
    messageID: string;
    type: "step-start";
  };
}

interface OpenCodeToolUse {
  type: "tool_use";
  timestamp: number;
  sessionID: string;
  part: {
    id: string;
    sessionID: string;
    messageID: string;
    type: "tool";
    callID: string;
    tool: string;
    state: {
      status: string;
      input: Record<string, unknown>;
      output?: string;
      title?: string;
      metadata?: Record<string, unknown>;
      time?: { start: number; end: number };
    };
  };
}

interface OpenCodeText {
  type: "text";
  timestamp: number;
  sessionID: string;
  part: {
    id: string;
    sessionID: string;
    messageID: string;
    type: "text";
    text: string;
    time?: { start: number; end: number };
  };
}

interface OpenCodeStepFinish {
  type: "step_finish";
  timestamp: number;
  sessionID: string;
  part: {
    id: string;
    sessionID: string;
    messageID: string;
    type: "step-finish";
    reason: string;
    cost: number;
    tokens: {
      total: number;
      input: number;
      output: number;
      reasoning: number;
      cache?: { read: number; write: number };
    };
  };
}

interface OpenCodeError {
  type: "error";
  timestamp: number;
  sessionID: string;
  error: {
    name: string;
    data?: {
      message?: string;
      statusCode?: number;
    };
  };
}

type OpenCodeEvent =
  | OpenCodeStepStart
  | OpenCodeToolUse
  | OpenCodeText
  | OpenCodeStepFinish
  | OpenCodeError;

/**
 * Resolve the path to the OpenCode CLI binary.
 *
 * Tries `which opencode` first, then falls back to known locations.
 * Throws if the binary cannot be found.
 */
function resolveOpenCodeBinary(): string {
  try {
    const result = execSync("which opencode", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    if (result) return result;
  } catch {
    // `which` failed — try known fallback paths
  }

  const knownPaths = [
    "/root/.opencode/bin/opencode",
    "/usr/local/bin/opencode",
  ];

  for (const p of knownPaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  throw new Error(
    "OpenCode CLI not found. Checked: `which opencode`, /root/.opencode/bin/opencode, /usr/local/bin/opencode.\n" +
    "Install with: curl -fsSL https://opencode.ai/install | bash"
  );
}

/**
 * OpenCode Agent - executes coding tasks using OpenCode CLI
 */
export class OpenCodeAgent {
  private config: OpenCodeConfiguration;
  private session: TelemetrySession | null = null;
  private cliProcess: ChildProcess | null = null;
  private stepCount: number = 0;
  private opencodeBinary: string;
  private fatalErrorMessage: string = "";
  /**
   * Classification of the most recent provider error captured in
   * `fatalErrorMessage`. `"rate_limit"` is recoverable by the retry
   * loop in `run()`; provider errors (auth, model-not-found, content
   * filter, malformed requests, ...) are treated as terminal. Reset at
   * the start of each `runOnce` attempt so a prior attempt's
   * classification can't leak into the next attempt's outcome.
   */
  private lastFatalErrorKind: "rate_limit" | "provider_error" | undefined;
  /**
   * Most recently observed OpenCode session ID. Captured from any
   * event that carries `sessionID` (step_start, tool_use, text,
   * step_finish, error). Sticky across attempts so the retry loop can
   * pass `--session <id>` to resume work rather than restart from
   * scratch on a transient provider rate-limit.
   */
  private sessionID: string | undefined;
  /**
   * Reason from the most recent `step_finish` event. OpenCode emits
   * `"stop"` only when the model voluntarily ends the conversation; any
   * other terminal value (`"tool-calls"`, etc.) combined with a clean
   * `exitCode === 0` indicates a non-graceful termination — typically
   * the per-build step ceiling (`OPENCODE_BUILD_STEPS`) was hit, the
   * model returned an empty/filtered response, or the upstream provider
   * silently dropped the connection. The close handler treats those as
   * failed runs to keep the rollout's `metadata.json` honest.
   */
  private lastStepFinishReason: string | undefined;

  constructor(config: OpenCodeConfiguration) {
    this.config = config;
    this.opencodeBinary = resolveOpenCodeBinary();
    this.log("info", `OpenCodeAgent initialized (model: ${config.model}, binary: ${this.opencodeBinary})`);
  }

  /**
   * Set telemetry session for logging
   */
  setTelemetrySession(session: TelemetrySession): void {
    this.session = session;
  }

  /**
   * Log helper - uses session if available, else console
   */
  private log(
    type: LogEventType,
    message: string,
    data?: Record<string, unknown>
  ): void {
    if (this.session) {
      this.session.log(type, message, data);
    } else {
      console.log(message);
    }
  }

  /**
   * Log error helper
   */
  private logError(message: string, data?: Record<string, unknown>): void {
    if (this.session) {
      this.session.log("error", message, data);
    } else {
      console.error(message);
    }
  }

  /**
   * Write auth.json to the OpenCode data directory.
   * Called before spawning the CLI process so credentials are available.
   */
  private setupAuth(): void {
    const homeDir = process.env.HOME || "/root";
    const authDir = path.join(homeDir, ".local/share/opencode");
    const authFile = path.join(authDir, "auth.json");

    fs.mkdirSync(authDir, { recursive: true });
    fs.writeFileSync(
      authFile,
      JSON.stringify(this.config.credentials, null, 2),
      { mode: 0o600 }
    );

    const providers = Object.keys(this.config.credentials);
    this.log("info", `Wrote auth.json with providers: ${providers.join(", ")}`, {
      authFile,
      providers,
    });
  }

  /**
   * Synthesize a per-pid `opencode.json` carrying:
   *   1. `mcp` block translated from the canonical neutral
   *      `mcp-servers.json` (when one exists for this rollout).
   *   2. `agent.build.steps` set to `OPENCODE_BUILD_STEPS` so long-horizon
   *      tasks aren't cut off at the upstream default.
   *
   * Returns the absolute path to the synthesized config so the caller
   * can set `OPENCODE_CONFIG=<path>` on spawn. OpenCode has no
   * `--mcp-config` flag — env-var pickup is the canonical injection
   * point.
   *
   * Always writes a config (even with no MCP) because the step-ceiling
   * bump is independent of MCP and benefits every OpenCode rollout.
   * A malformed neutral file silently disables MCP for this run rather
   * than failing the agent.
   */
  private setupOpencodeConfig(): string {
    const config: Record<string, unknown> = {
      $schema: "https://opencode.ai/config.json",
      agent: {
        build: { steps: OPENCODE_BUILD_STEPS },
      },
      // Use a fully permissive opencode.json for Hyperfocal rollouts and rely
      // on linux-user / filesystem isolation as the security boundary. This
      // also prevents non-interactive `opencode run` from auto-rejecting
      // external_directory or *.env prompts and then exiting idle without a
      // true completion signal.
      permission: {
        "*": "allow",
      },
    };

    const neutralPath = this.config.mcpConfigPath;
    if (neutralPath) {
      const file = readMcpServersFromPath(neutralPath);
      if (!file && fs.existsSync(neutralPath)) {
        this.logError(
          `Failed to parse neutral mcp-servers file at ${neutralPath} — running without MCP`,
        );
      }
      if (file && Object.keys(file.servers).length > 0) {
        config.mcp = toOpencodeMcp(file);
        this.log("info", `MCP active: ${Object.keys(file.servers).join(", ")}`, {
          servers: Object.keys(file.servers),
          source: neutralPath,
        });
      }
    }

    const homeDir = process.env.HOME || "/root";
    const runtimeDir = path.join(homeDir, ".local/share/opencode/runtime", String(process.pid));
    fs.mkdirSync(runtimeDir, { recursive: true });
    const configPath = path.join(runtimeDir, "opencode.json");
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return configPath;
  }

  /**
   * Build CLI arguments for `opencode run`.
   *
   * OpenCode's `run` mode auto-approves all tool permissions (no --yolo needed).
   * No --max-turns equivalent — long-horizon tasks run to completion.
   *
   * When `resumeSessionID` is provided, the CLI re-enters the existing
   * session (`--session <id>`); the caller is expected to pass a small
   * continuation prompt instead of the original task prompt so the
   * model picks up where it left off.
   */
  private buildCLIArgs(
    prompt: string,
    workingDir: string,
    resumeSessionID?: string,
  ): string[] {
    const args = [
      "run",
      "--format", "json",
      "--model", this.config.model,
      "--dir", workingDir,
    ];
    if (resumeSessionID) {
      args.push("--session", resumeSessionID);
    }
    if (this.config.debug) {
      args.push("--print-logs", "--log-level", "DEBUG");
    }
    args.push(prompt);
    return args;
  }

  /**
   * When `config.debug` is on, create an empty file alongside the
   * synthesized opencode.json to receive the full OpenCode stderr
   * stream (the `--print-logs --log-level DEBUG` payload is too large
   * for the in-memory tail-500 buffer). Returns the path so the
   * stderr `'data'` listener can append to it; returns undefined when
   * debug is off or file creation fails.
   */
  private maybeInitDebugStderrFile(opencodeConfigPath: string): string | undefined {
    if (!this.config.debug) return undefined;
    const stderrFilePath = path.join(path.dirname(opencodeConfigPath), "stderr.log");
    try {
      fs.writeFileSync(stderrFilePath, "");
      this.log("info", `OpenCode DEBUG stderr → ${stderrFilePath}`);
      return stderrFilePath;
    } catch (e) {
      this.logError(`Failed to init stderr debug file: ${(e as Error).message}`);
      return undefined;
    }
  }

  /**
   * Compose the error message for the incomplete-termination case. This
   * is evidence reporting, not root-cause logic: OpenCode's non-interactive
   * `run` exits when the session goes idle, and idle can mean "done",
   * "permission rejected", "provider went quiet", or "budget exhausted".
   * The close-handler only treats it as successful when the final
   * `step_finish` reason was explicitly `"stop"`.
   */
  private formatIncompleteOutcome(evidence: string[]): string {
    const possibleCauses: string[] = [];
    if (this.stepCount >= OPENCODE_BUILD_STEPS) {
      possibleCauses.push(`step-budget ceiling (${OPENCODE_BUILD_STEPS}) reached`);
    }
    possibleCauses.push(
      "empty / content-filtered model response",
      "silent provider or network error",
    );
    const evidenceText =
      evidence.length > 0 ? ` Evidence: ${evidence.join(" | ")}.` : "";
    return (
      `OpenCode CLI exited cleanly but the model never declared completion ` +
      `(last step_finish reason: ${this.lastStepFinishReason ?? "none"}, steps: ${this.stepCount}). ` +
      `Possible causes: ${possibleCauses.join("; ")}.` +
      evidenceText +
      " " +
      `Re-run with OPENCODE_DEBUG=1 and inspect ` +
      `~/.local/share/opencode/runtime/<pid>/stderr.log for the proximate reason.`
    );
  }

  /**
   * Run the agent with the given prompt.
   *
   * Wraps one or more `opencode run` invocations in a retry loop so
   * transient provider rate-limits (HTTP 429) don't kill a long-horizon
   * rollout. The original 7-step pg-cutover failure (provider TPM 429
   * on step 7, ~10 minutes of setup wasted) is the motivating case.
   *
   * Retries resume the OpenCode session via `--session <id>` so the
   * model picks up its existing plan, todos, and tool history rather
   * than restarting. Non-rate-limit errors (auth, model-not-found,
   * content filter, silent truncation) are treated as terminal —
   * retrying those would just burn time and tokens.
   */
  async run(
    prompt: string,
    workingDir: string,
    _options?: AgentRunOptions
  ): Promise<void> {
    if (!prompt?.trim()) {
      throw new Error("Prompt must be a non-empty string");
    }

    this.log("info", `Using OpenCode CLI (${this.config.model})`);
    // OpenCode runs the explicit provider/model string verbatim (no alias
    // expansion), so the resolved id is the requested one.
    this.session?.setResolvedModel(this.config.model);
    this.log(
      "info",
      `Prompt: ${prompt.substring(0, 100)}${prompt.length > 100 ? "..." : ""}`,
      { promptLength: prompt.length }
    );
    this.log("info", `Working directory: ${workingDir}`);

    // Write auth.json if credentials were provided (may already be written by orchestrator)
    if (Object.keys(this.config.credentials).length > 0) {
      this.setupAuth();
    }

    // Synthesize and point OpenCode at our runtime config (raises step
    // ceiling, wires MCP servers when the canonical file is present).
    // Done once; reused across retry attempts.
    const opencodeConfigPath = this.setupOpencodeConfig();
    this.log("info", `Wrote opencode.json: ${opencodeConfigPath}`, {
      configPath: opencodeConfigPath,
      buildSteps: OPENCODE_BUILD_STEPS,
    });

    const { maxAttempts, baseDelayMs, maxTotalMs } = OPENCODE_RETRY_CONFIG;
    const overallStart = Date.now();

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      // Resume the prior session iff we observed a session ID on a
      // previous attempt. If attempt 1 dies before any event arrives,
      // there's no session to resume — fall back to a fresh run.
      const resumeSessionID = attempt > 1 ? this.sessionID : undefined;
      const effectivePrompt =
        resumeSessionID != null ? OPENCODE_RESUME_PROMPT : prompt;

      if (attempt > 1) {
        this.log(
          "info",
          `Retry attempt ${attempt}/${maxAttempts}` +
            (resumeSessionID ? ` (resuming session ${resumeSessionID})` : ""),
          {
            attempt,
            maxAttempts,
            resumeSessionID,
          },
        );
      }

      const outcome = await this.runOnce(
        effectivePrompt,
        workingDir,
        opencodeConfigPath,
        resumeSessionID,
      );

      if (outcome.kind === "completed") {
        return;
      }

      if (outcome.kind !== "rate_limit") {
        // Don't waste retry budget on auth/model/content-filter/permission
        // errors or incomplete idle exits. Only provider rate limits have a
        // proven safe resume path.
        throw new Error(outcome.message);
      }

      // rate_limit — decide whether we still have budget for another try
      if (attempt >= maxAttempts) {
        throw new Error(
          `${outcome.message} (exhausted ${maxAttempts} retry attempts)`,
        );
      }
      const elapsed = Date.now() - overallStart;
      const remaining = maxTotalMs - elapsed;
      if (remaining <= 0) {
        throw new Error(
          `${outcome.message} (retry wall-clock budget ${maxTotalMs}ms exhausted)`,
        );
      }
      const delay = computeBackoffMs(attempt, baseDelayMs, remaining);
      this.log(
        "info",
        `Rate-limited — backing off ${delay}ms before retry ${attempt + 1}/${maxAttempts}`,
        {
          reason: "rate_limit",
          delayMs: delay,
          attempt: attempt + 1,
          maxAttempts,
          sessionID: this.sessionID,
        },
      );
      await sleep(delay);
    }

    // Unreachable: the loop body either returns, throws, or sleeps and
    // continues. Defensive guard so future edits don't accidentally
    // fall through to a silent success.
    throw new Error("OpenCode retry loop terminated without an outcome");
  }

  /**
   * Spawn `opencode run` once and return a `RunOutcome` describing why
   * the process ended. Never rejects — even fatal errors come back as
   * a typed terminal outcome so the retry loop in `run()` is the single
   * place that decides whether to surface, retry, or give up.
   */
  private async runOnce(
    prompt: string,
    workingDir: string,
    opencodeConfigPath: string,
    resumeSessionID?: string,
  ): Promise<RunOutcome> {
    // Reset per-attempt state. `this.sessionID` deliberately survives
    // across attempts (it's the resume handle); everything else starts
    // fresh so a prior attempt's error/step counts can't contaminate
    // this attempt's outcome.
    this.stepCount = 0;
    this.lastStepFinishReason = undefined;
    this.fatalErrorMessage = "";
    this.lastFatalErrorKind = undefined;

    const cliArgs = this.buildCLIArgs(prompt, workingDir, resumeSessionID);

    this.log("info", `Starting OpenCode CLI`, {
      args: cliArgs,
      binary: this.opencodeBinary,
      resumeSessionID,
    });

    const stderrFilePath = this.maybeInitDebugStderrFile(opencodeConfigPath);

    let result: Awaited<ReturnType<typeof runStreamingCli>>;
    try {
      result = await runStreamingCli({
        command: this.opencodeBinary,
        args: cliArgs,
        cwd: workingDir,
        stdin: "ignore",
        env: {
          ...process.env,
          HOME: process.env.HOME || "/root",
          // OpenCode has no --mcp-config flag; OPENCODE_CONFIG is the
          // documented runtime-override path. Picked up at CLI startup.
          OPENCODE_CONFIG: opencodeConfigPath,
        },
        debugLogStream: this.createDebugLogStream(),
        onProcess: (child) => {
          this.cliProcess = child;
        },
        onStdoutLine: (line) => {
          if (!line.trim()) return;
          try {
            const event = JSON.parse(line) as OpenCodeEvent;
            this.handleEvent(event);
          } catch {
            // Not JSON — might be raw output, log it through the
            // session so OPENCODE_DEBUG investigations can correlate.
            this.log("info", line);
          }
        },
        onStderrText: (text) => {
          if (stderrFilePath) {
            try { fs.appendFileSync(stderrFilePath, text); } catch { /* best effort */ }
          }
          if (text.includes("Error") || text.includes("error")) {
            this.logError(`CLI stderr: ${text.trim()}`);
          }
        },
      });
    } catch (error) {
      // Spawn-level failures are always terminal — `opencode` binary
      // missing / unexecutable / etc. Don't retry.
      const message = `Failed to start OpenCode CLI: ${error instanceof Error ? error.message : String(error)}`;
      this.logError(`Error: ${message}`);
      return {
        kind: "cli_error",
        message,
        stepCount: this.stepCount,
        sessionID: this.sessionID,
      };
    }

    if (result.completionSource === "exit") {
      this.log("info", `OpenCode CLI exited before stdio closed; settled after drain grace`, {
        exitCode: result.code,
        signal: result.signal,
      });
    }

    const outcome = this.classifyCloseOutcome(result.code, result.stderrChunks, result.signal);
    if (outcome.kind === "completed") {
      this.log("info", `OpenCode CLI completed`, {
        steps: this.stepCount,
        exitCode: result.code,
        completionSource: result.completionSource,
      });
    } else {
      this.logError(`Error: ${outcome.message}`, {
        errorKind: outcome.kind,
        stepCount: this.stepCount,
        completionSource: result.completionSource,
      });
    }
    return outcome;
  }

  private createDebugLogStream(): fs.WriteStream | undefined {
    if (!this.config.codingAgentDebugLogs || !this.session) return undefined;
    return fs.createWriteStream(this.session.getDebugLogPath(), { flags: "a" });
  }

  /**
   * Decide the `RunOutcome` for the just-completed CLI process based
   * on its exit code, any error event captured mid-stream, and the
   * step-finish state.
   *
   * Routing order matters:
   *   1. Non-zero exit code → fatal (covers CLI-level crashes).
   *   2. A captured error event → its classification (rate_limit | other).
   *      This branch fires regardless of stepCount because a mid-run
   *      provider 429 (post step 1) was previously masked by the
   *      stepCount==0 gate and reported as a silent truncation.
   *   3. Step ran but never said `stop` → completed with a warning.
   *   4. Otherwise → completed.
   */
  private classifyCloseOutcome(
    code: number | null,
    stderrChunks: string[],
    signal: NodeJS.Signals | null = null,
  ): RunOutcome {
    if (code !== 0 || signal) {
      let message = signal
        ? `OpenCode CLI terminated with signal ${signal}`
        : `OpenCode CLI exited with code ${code}`;
      if (stderrChunks.length > 0) {
        message += `\nStderr: ${stderrChunks.join("").slice(-500)}`;
      }
      return {
        kind: "cli_error",
        message,
        stepCount: this.stepCount,
        sessionID: this.sessionID,
      };
    }

    if (this.fatalErrorMessage) {
      const kind = this.lastFatalErrorKind === "rate_limit" ? "rate_limit" : "provider_error";
      return {
        kind,
        message: this.fatalErrorMessage,
        stepCount: this.stepCount,
        sessionID: this.sessionID,
      };
    }

    if (this.stepCount > 0 && this.lastStepFinishReason !== "stop") {
      const stderr = stderrChunks.join("");
      const permissionRejection = extractPermissionRejection(stderr);
      if (permissionRejection) {
        return {
          kind: "permission_rejected",
          message:
            `OpenCode permission was auto-rejected before the model declared completion ` +
            `(last step_finish reason: ${this.lastStepFinishReason ?? "none"}, steps: ${this.stepCount}). ` +
            `Evidence: ${permissionRejection}`,
          stepCount: this.stepCount,
          sessionID: this.sessionID,
        };
      }
      const evidence = [
        `lastStepFinishReason=${this.lastStepFinishReason ?? "none"}`,
        `steps=${this.stepCount}`,
      ];
      this.log("warn", this.formatIncompleteOutcome(evidence), {
        reason: "missing_terminal_stop",
        lastStepFinishReason: this.lastStepFinishReason,
        stepCount: this.stepCount,
        sessionID: this.sessionID,
      });
    }

    return {
      kind: "completed",
      stepCount: this.stepCount,
      sessionID: this.sessionID,
    };
  }

  /**
   * Handle an nd-JSON event from the CLI stream
   */
  private handleEvent(event: OpenCodeEvent): void {
    // Sticky-capture the session ID off any event that carries one so
    // the retry loop can resume via `--session <id>` even when the
    // failure arrives before a step_finish.
    if (
      "sessionID" in event &&
      typeof event.sessionID === "string" &&
      event.sessionID
    ) {
      this.sessionID = event.sessionID;
    }
    switch (event.type) {
      case "step_start":
        this.handleStepStart(event);
        break;
      case "tool_use":
        this.handleToolUse(event);
        break;
      case "text":
        this.handleText(event);
        break;
      case "step_finish":
        this.handleStepFinish(event);
        break;
      case "error":
        this.handleError(event);
        break;
      default:
        this.log("info", `Unknown OpenCode event: ${JSON.stringify(event).slice(0, 200)}`);
    }
  }

  private handleStepStart(event: OpenCodeStepStart): void {
    this.stepCount++;
    this.log("info", `Step ${this.stepCount} started`, {
      sessionID: event.sessionID,
      messageID: event.part.messageID,
    });
  }

  private handleToolUse(event: OpenCodeToolUse): void {
    const { tool, state, callID } = event.part;
    const inputPreview = JSON.stringify(state.input).slice(0, 150);
    const isShellTool = tool.toLowerCase() === "bash" || tool.toLowerCase() === "shell";

    if (isShellTool) {
      const call = toolCallEvent({
        tool,
        toolUseId: callID,
        family: "shell",
        input: state.input,
        message: `${tool}: ${inputPreview}${inputPreview.length >= 150 ? "..." : ""}`,
      });
      this.log(call.type, call.message, call.data);

      if (state.output !== undefined) {
        const output = String(state.output);
        const result = shellResultEvent({
          tool,
          toolUseId: callID,
          stdout: output,
          stderr: "",
          status: state.status,
          message: `Result: ${output.slice(0, 200)}${output.length > 200 ? "..." : ""}`,
        });
        this.log(result.type, result.message, result.data);
      }
      return;
    }

    const call = toolCallEvent({
      tool,
      toolUseId: callID,
      family: classifyOpenCodeToolFamily(tool),
      input: state.input,
      message: `${tool}: ${inputPreview}${inputPreview.length >= 150 ? "..." : ""}`,
    });
    this.log(call.type, call.message, call.data);

    // OpenCode embeds tool results in the same event (state.output)
    if (state.output !== undefined) {
      const outputPreview = String(state.output).slice(0, 200);
      this.log("tool_result", `Result: ${outputPreview}${String(state.output).length > 200 ? "..." : ""}`, {
        tool,
        tool_use_id: callID,
        output: state.output,
        status: state.status,
      });
    }
  }

  private handleText(event: OpenCodeText): void {
    this.log("agent_text", event.part.text, { text: event.part.text });
  }

  private handleStepFinish(event: OpenCodeStepFinish): void {
    const { reason, cost, tokens } = event.part;
    const status = reason === "stop" ? "completed" : "continuing";

    // Track for the close-handler's "exited 0 but never said stop" check.
    this.lastStepFinishReason = reason;

    // TODO(metadata-tokens): define a provider-neutral metadata shape before
    // projecting usage here. OpenCode emits per-step token deltas, while
    // Claude Code's result.usage is already a session aggregate, so the two
    // sources need different accumulation rules behind one public contract.
    this.log("info", `Step finished (${status}, ${tokens.total} tokens, $${cost.toFixed(4)})`, {
      reason,
      cost,
      tokens,
      stepCount: this.stepCount,
    });
  }

  private handleError(event: OpenCodeError): void {
    const errMsg = event.error.data?.message || event.error.name;
    const statusCode = event.error.data?.statusCode;
    // Classify before storing so the close handler can route on
    // `lastFatalErrorKind` (rate-limit → retryable, other → terminal).
    const classified = classifyOpenCodeError(errMsg, statusCode);
    this.fatalErrorMessage = classified.message;
    this.lastFatalErrorKind = classified.kind;
    this.logError(this.fatalErrorMessage, {
      errorName: event.error.name,
      errorData: event.error.data,
      errorKind: classified.kind,
    });
  }

  /**
   * Get the model name
   */
  getModel(): string {
    return this.config.model;
  }

  /**
   * Get the configuration
   */
  getConfiguration(): OpenCodeConfiguration {
    return { ...this.config };
  }
}
