/**
 * Agent Entry Point
 *
 * Runs inside the agent child process, invoked by runAgent() in agent-runner.ts
 * via the _internal-run-agent CLI command.
 *
 * The parent process constructs an allowlist environment before spawning this
 * entry point, so process.env here contains only what the orchestrator explicitly
 * granted (PATH, HOME, LANG, and conditionally ANTHROPIC_API_KEY or AWS_* based
 * on agent type and config).
 */

import * as fs from "fs";
import {
  ClaudeCodeAgent,
  AnthropicCodingAgent,
  OpenCodeAgent,
  CodexAgent,
  MiniSweAgent,
  createSession,
  DEFAULT_ALLOWED_TOOLS,
  DEFAULT_DISALLOWED_TOOLS,
  mcpServersFilePath,
} from "@hyperfocal/env-base";
import type { PermissionsMode } from "@hyperfocal/env-base";
import { rejectedClaudeRateLimitFromError } from "./agent-failure.js";

type AgentType = "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent";

function isAgentType(value: string): value is AgentType {
  return value === "claude-code" ||
    value === "anthropic-coding" ||
    value === "opencode" ||
    value === "codex" ||
    value === "mini-swe-agent";
}

interface AgentArgs {
  promptFile: string;
  workspace: string;
  model: string;
  problemId: string;
  agentType: AgentType;
  schemaFile?: string;
  permissionsMode: PermissionsMode;
  /** Diagnostic flag: forwards to OpenCodeAgent.config.debug. */
  openCodeDebug: boolean;
  /** Mirror raw provider CLI stdout/stderr to the session debug log. */
  codingAgentDebugLogs: boolean;
  /**
   * Per-env additive tool overrides from hyperfocal.yaml, threaded from the
   * parent (which reads the now root-only config) as base64'd JSON arrays.
   * The child cannot read hyperfocal.yaml under linux-user, so these arrive
   * via CLI instead of loadConfig().
   */
  yamlDisallowedTools: string[];
  yamlAllowedTools: string[];
}

function parseArgs(): AgentArgs {
  const args = process.argv.slice(3);
  const result: Partial<AgentArgs> = {
    agentType: "claude-code",
    permissionsMode: "linux-user",
    openCodeDebug: false,
    codingAgentDebugLogs: false,
    yamlDisallowedTools: [],
    yamlAllowedTools: [],
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--prompt-file":
        result.promptFile = args[++i];
        break;
      case "--workspace":
        result.workspace = args[++i];
        break;
      case "--model":
        result.model = args[++i];
        break;
      case "--problem-id":
        result.problemId = args[++i];
        break;
      case "--schema-file":
        result.schemaFile = args[++i];
        break;
      case "--agent-type":
        {
          const value = args[++i];
          if (!isAgentType(value)) {
            console.error(
              `Invalid --agent-type "${value}". Must be "claude-code", "anthropic-coding", "opencode", "codex", or "mini-swe-agent".`
            );
            process.exit(1);
          }
          result.agentType = value;
        }
        break;
      case "--permissions-mode":
        result.permissionsMode = args[++i] as PermissionsMode;
        break;
      case "--opencode-debug":
        result.openCodeDebug = true;
        break;
      case "--coding-agent-debug-logs":
        result.codingAgentDebugLogs = true;
        break;
      case "--yaml-disallowed-tools":
        result.yamlDisallowedTools = JSON.parse(
          Buffer.from(args[++i], "base64").toString("utf-8")
        );
        break;
      case "--yaml-allowed-tools":
        result.yamlAllowedTools = JSON.parse(
          Buffer.from(args[++i], "base64").toString("utf-8")
        );
        break;
    }
  }

  if (!result.promptFile || !result.workspace || !result.model || !result.problemId) {
    console.error(
      "Usage: _internal-run-agent --prompt-file <path> --workspace <path> --model <model> --problem-id <id> [--agent-type <type>] [--permissions-mode <mode>] [--opencode-debug] [--coding-agent-debug-logs]"
    );
    process.exit(1);
  }

  return result as AgentArgs;
}

export async function runAgentEntry(): Promise<void> {
  const args = parseArgs();

  // Both `model` and `agentType` are persisted into the per-problem
  // metadata.json by TelemetrySession.updateProblemMetadata. The control
  // plane (rolloutStatusSyncService) reads them and populates runs.model /
  // rollouts.model / runs.agent_type, which removes the "—" placeholder
  // from the runs list when no explicit override is passed in the request.
  const session = createSession(
    args.problemId,
    "agent",
    "solve",
    args.model,
    undefined,
    args.agentType,
  );
  const prompt = fs.readFileSync(args.promptFile, "utf-8");

  try {
    // Orchestrator owns the single agent-session lifecycle event. The
    // concrete agent classes only emit provider-specific `info` rows so the
    // trace has one `session_start` / `session_end` pair from TelemetrySession.
    // Frontend pairing is now id-based; env-base agents are responsible for
    // emitting `tool_use_id` on both tool_call and tool_result events.
    session.log("session_start", "AGENT started (model: " + args.model + ")");
    session.log("info", `   Type: ${args.agentType}`);
    session.log("info", `   Model: ${args.model}`);
    session.log("info", `   Permissions: ${args.permissionsMode}`);
    session.log("info", `   Workspace: ${args.workspace}`);
    if (args.schemaFile) {
      session.log("info", `   Schema: ${args.schemaFile}`);
    }
    session.log("info", "=".repeat(60));

    // Canonical MCP config discovery: if setupProblem wrote the neutral
    // mcp-servers.json at the canonical workspace path, pass it to
    // whichever agent class is constructed below. Each agent translates
    // the neutral schema to its CLI's native shape; an unset/missing path
    // is treated as "no MCP for this run".
    const neutralMcpPath = mcpServersFilePath(args.workspace);
    const mcpConfigPath = fs.existsSync(neutralMcpPath) ? neutralMcpPath : undefined;
    if (mcpConfigPath) {
      session.log("info", `MCP config detected: ${mcpConfigPath}`);
    }

    if (args.agentType === "claude-code") {
      // Per-environment tool overrides come from hyperfocal.yaml. Both
      // disallowedTools and allowedTools are additive — they append to
      // the env-base defaults, never replace. They are threaded in from the
      // parent as CLI args (not read here) because under linux-user the child
      // runs unprivileged and cannot read the now root-only hyperfocal.yaml.
      const yamlDisallow = args.yamlDisallowedTools;
      const yamlAllow = args.yamlAllowedTools;

      const disallowedTools = [
        ...DEFAULT_DISALLOWED_TOOLS,
        ...yamlDisallow,
      ];

      // Slow-query problems block docker + sandbox access to force the
      // agent through SSH + observability tools. Per-PROBLEM, not
      // per-environment, so it stays imperative here.
      if (args.problemId.startsWith("slow-query")) {
        disallowedTools.push(
          // Docker access (forces SSH + observability instead of god-mode)
          "Bash(docker exec *)",
          "Bash(docker ps*)",
          "Bash(docker restart *)",
          "Bash(docker inspect *)",
          "Bash(docker logs *)",
          "Bash(docker cp *)",
          "Bash(docker stop *)",
          "Bash(docker kill *)",
          "Bash(docker rm *)",
          "Bash(docker-compose *)",
          // Sandbox source (prevents reading API code to shortcut investigation)
          "Read(//hyperfocal/env/sandbox/**)",
          "Edit(//hyperfocal/env/sandbox/**)",
          "Write(//hyperfocal/env/sandbox/**)",
          "Bash(* /hyperfocal/env/sandbox*)",
        );
      }

      if (yamlDisallow.length > 0) {
        session.log("info", `   Extra disallowedTools from hyperfocal.yaml: ${yamlDisallow.length}`);
      }
      if (yamlAllow.length > 0) {
        session.log("info", `   Extra allowedTools from hyperfocal.yaml: ${yamlAllow.length}`);
      }

      const allowedTools = [
        ...DEFAULT_ALLOWED_TOOLS,
        ...yamlAllow,
      ];

      const agent = new ClaudeCodeAgent({
        type: "claude-code",
        model: args.model,
        permissionsMode: args.permissionsMode,
        allowedTools,
        disallowedTools,
        mcpConfigPath,
        codingAgentDebugLogs: args.codingAgentDebugLogs,
      });

      agent.setTelemetrySession(session);
      await agent.run(prompt, args.workspace, { schemaPath: args.schemaFile });
    } else if (args.agentType === "opencode") {
      // OpenCode reads auth.json from ~/.local/share/opencode/auth.json
      // which was already written by agent-runner.ts before spawning this process.
      // We pass empty credentials here since auth is file-based, not config-based.
      //
      // TODO(opencode-disallowed-tools): hyperfocal.yaml's
      // `agent.disallowedTools` is currently honored only by ClaudeCodeAgent
      // (see the `claude-code` branch above and DEFAULT_DISALLOWED_TOOLS in
      // env-base). When an environment is run via OpenCode for cross-model
      // benchmarking — e.g., pg-engine-feature-cutover with `gold-state` /
      // sandbox read blocks — those guards are silently skipped, which lets
      // the agent read solution hints. Fix: translate the Claude-style
      // patterns ("Read(//path/**)", "Bash(git log*)") into OpenCode's
      // tool-permission config (its run-mode supports a denylist via
      // ~/.opencode/config.json or per-invocation flags), and apply both
      // the env-base defaults and the per-env yaml additions before
      // invoking OpenCodeAgent.run(). Until that lands, treat OpenCode
      // benchmark scores on isolation-sensitive envs with skepticism.
      const agent = new OpenCodeAgent({
        type: "opencode",
        model: args.model,
        credentials: {},
        mcpConfigPath,
        // Opt-in DEBUG flag propagated from the orchestrator via the
        // `--opencode-debug` CLI flag — threaded as a config field
        // rather than read from process.env so OpenCodeAgent stays
        // free of global-state coupling.
        debug: args.openCodeDebug,
        codingAgentDebugLogs: args.codingAgentDebugLogs,
      });

      agent.setTelemetrySession(session);
      await agent.run(prompt, args.workspace, { schemaPath: args.schemaFile });
    } else if (args.agentType === "codex") {
      const agent = new CodexAgent({
        type: "codex",
        model: args.model,
        sandbox: "danger-full-access",
        approvalPolicy: "never",
        mcpConfigPath,
        codingAgentDebugLogs: args.codingAgentDebugLogs,
      });

      agent.setTelemetrySession(session);
      await agent.run(prompt, args.workspace, { schemaPath: args.schemaFile });
    } else if (args.agentType === "mini-swe-agent") {
      const agent = new MiniSweAgent({
        type: "mini-swe-agent",
        model: args.model,
        mcpConfigPath,
        codingAgentDebugLogs: args.codingAgentDebugLogs,
      });

      agent.setTelemetrySession(session);
      await agent.run(prompt, args.workspace, { schemaPath: args.schemaFile });
    } else {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        console.error("ANTHROPIC_API_KEY not set in agent environment.");
        console.error(
          "This usually means the orchestrator failed to pass credentials to the child process."
        );
        process.exit(1);
      }

      const agent = new AnthropicCodingAgent({
        type: "anthropic-coding",
        model: args.model,
        credentials: { anthropic: apiKey },
      });

      agent.setTelemetrySession(session);
      await agent.run(prompt, args.workspace, { schemaPath: args.schemaFile });
    }

    session.end("completed");
    console.log("Agent completed");
    process.exit(0);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    const rateLimit = rejectedClaudeRateLimitFromError(error);
    const status = rateLimit ? "errored" : "failed";
    session.log("error", `Agent failed: ${msg}`, rateLimit ? {
      failureKind: "provider_rate_limit",
      observation: rateLimit,
    } : undefined);
    session.end(status, msg);
    console.error("Agent failed:", msg);
    process.exit(1);
  }
}
