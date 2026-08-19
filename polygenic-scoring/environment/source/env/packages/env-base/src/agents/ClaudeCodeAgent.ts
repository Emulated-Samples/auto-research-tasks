/**
 * Claude Code Agent
 *
 * Wraps the official Claude Code CLI for coding tasks.
 * Uses Claude Code's built-in tools (Bash, Read, Write, Edit, WebFetch, etc.)
 * with bidirectional streaming via stream-json format.
 *
 * Permission modes:
 *
 * 1. "claude-permissions" (default, recommended):
 *    Four-layer defense for tool isolation:
 *      - --tools: hard-restricts which tools exist in the model's context
 *      - --permission-mode dontAsk: auto-denies tools not pre-approved
 *      - --allowedTools: pre-approves specific tools for dontAsk mode
 *      - --disallowedTools: blocks specific patterns (deny > allow)
 *    Runs as the current user (root) — no sudo needed.
 *    CWD set to workspace (/hyperfocal/env/workspace).
 *
 *    Key pattern syntax for --disallowedTools:
 *      Read(//path/**)  → absolute path (// prefix required!)
 *      Read(/path/**)   → relative to settings file (NOT absolute)
 *      Bash(git *)      → Bash command starting with "git "
 *      Bash(* /path*)   → any command containing "/path"
 *
 * 2. "linux-user" (legacy):
 *    - Uses --permission-mode bypassPermissions
 *    - Requires non-root user (hyperfocal-agent)
 *    - Requires setupClaudeCredentials() for credential sharing
 *    - Filesystem lockdown via workspace-isolation.ts
 *
 * Prerequisites:
 * - Claude Code CLI installed (auto-detected via `which claude`)
 * - OAuth credentials at /root/.claude/.credentials.json
 */

import { execSync, ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { ClaudeCodeConfiguration, AgentRunOptions, PermissionsMode } from "../types.js";
import type { TelemetrySession } from "../telemetry/index.js";
import type { LogEventType } from "../telemetry/types.js";
import * as os from "os";
import { readMcpServersFromPath } from "../mcp/index.js";
import type { McpServerSpec, McpServersFile } from "../mcp/index.js";
import {
  agentReasoningEvent,
  shellResultEvent,
  structuredResultEvent,
  toolCallEvent,
  type ToolFamily,
} from "./events.js";
import { runStreamingCli } from "./streaming-cli.js";
import {
  ClaudeCodeRateLimitError,
  normalizeClaudeApiRateLimitResult,
  normalizeClaudeRateLimitEvent,
} from "./claude-rate-limit.js";
import type { ProviderRateLimitObservation } from "../telemetry/types.js";

const DEFAULT_IDLE_CUTOFF_NO_PENDING_TASKS_MS = 5 * 60 * 1000;
const DEFAULT_IDLE_CUTOFF_WITH_PENDING_TASKS_MS = 15 * 60 * 1000;
const IDLE_CUTOFF_SIGKILL_GRACE_MS = 10_000;

/**
 * Per-background-task max age ("reaper", HYPERFOCAL_CLAUDE_BG_TASK_MAX_AGE_MS,
 * 0 disables). Kill the TASK, never the rollout: agents launch builds/fits
 * with run_in_background and some of those genuinely never return (a
 * non-converging fit has no exit condition). Without a per-task cap the agent
 * burns its remaining budget polling and dies at the agent-phase hard timeout
 * with work unfinished — MEASURED (sc-silentbench run_5, 2026-07-17): 5/9
 * rollouts hit the 180-min cap with 2-11 tasks still pending, each pending
 * 2.0-2.7h. The 45-min default is set from the same batch's 255 COMPLETED
 * tasks: p50=3min, p90=14min, p99=24min, max=51.5min — ~2x p99 of legitimate
 * work, while reaping every observed hang with >=2h of agent budget left.
 *
 * Reaping kills the task's processes (found by their open fd on the task's
 * output file) and lets the CLI observe the death and report the task
 * terminal through the normal resume path. Only if the CLI never reports
 * within a grace window do we synthesize the terminal state ourselves.
 */
const DEFAULT_BACKGROUND_TASK_MAX_AGE_MS = 45 * 60 * 1000;
const REAPER_SIGKILL_GRACE_MS = 10_000;
const DEFAULT_REAPER_SYNTHESIZE_TERMINAL_GRACE_MS = 30_000;

function reapExplanation(maxAgeMs: number): string {
  const minutes = Math.round(maxAgeMs / 60_000);
  return (
    `[harness] This background task was killed after exceeding the ` +
    `${minutes}-minute background-task limit. Treat it as hung (e.g. a ` +
    `non-converging run or a build that will never finish): do not restart ` +
    `it unchanged — use a cheaper check or a different approach, and finish ` +
    `your remaining work in the foreground.`
  );
}

/**
 * Headless background-task resume (HYPERFOCAL_CLAUDE_RESUME_ON_TASK_COMPLETION,
 * default on; set =0 to restore the legacy close-stdin-immediately behavior).
 *
 * In stream-json print mode the CLI exits as soon as the agent yields IF
 * stdin is closed — abandoning any still-running background task (observed
 * rollout 2026-06-11A: agent backgrounded "wait for the low-traffic window
 * then cut over", yielded, and the CLI exited 7 minutes before the window;
 * tests graded a pre-cutover sandbox). Interactive Claude Code re-invokes
 * the model when a background task finishes; headless mode has no one to
 * send the next user message — so we do: keep stdin open while tasks are
 * pending, write a follow-up user message when a task reaches a terminal
 * state, and close stdin once the agent yields with nothing pending. The
 * pending-tasks idle cutoff (15 min) remains the backstop for tasks that
 * hang forever.
 */
function taskResumeMessage(task: { taskId: string; description?: string }, status: string, summary?: string): string {
  return (
    `Background task ${task.taskId}${task.description ? ` (${task.description})` : ""} ` +
    `finished with status: ${status}.${summary ? ` Summary: ${summary}` : ""} ` +
    `Review its output and continue your work. If everything you intended is already complete, wrap up.`
  );
}

interface ClaudeIdleCutoffPolicy {
  enabled: boolean;
  noPendingTasksMs: number;
  withPendingTasksMs: number;
}

interface ClaudeBackgroundTask {
  taskId: string;
  toolUseId?: string;
  description?: string;
  taskType?: string;
  startedAt: number;
  lastKnownStatus?: string;
}

function envFlagEnabled(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return defaultValue;
  return raw !== "0" && raw.toLowerCase() !== "false";
}

function envDurationMs(name: string, defaultValue: number): number {
  const raw = process.env[name];
  if (raw === undefined) return defaultValue;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : defaultValue;
}

function classifyClaudeToolFamily(tool: string): ToolFamily {
  const lower = tool.toLowerCase();
  if (lower === "bash") return "shell";
  if (lower === "read" || lower === "edit" || lower === "write") return "file";
  if (lower === "todowrite") return "todo";
  if (lower.startsWith("mcp__")) return "mcp";
  return "generic";
}

/**
 * Translate the neutral `McpServersFile` into the schema Claude Code's
 * CLI accepts via `--mcp-config`. Claude wants `{ mcpServers: { name:
 * { type: "http"|"stdio", ... } } }`. The mapping is mechanical:
 *
 *   neutral transport "http"  → claude type "http"  (url)
 *   neutral transport "stdio" → claude type "stdio" (command + env)
 */
function toClaudeMcpConfig(file: McpServersFile): {
  mcpServers: Record<string, Record<string, unknown>>;
} {
  const mcpServers: Record<string, Record<string, unknown>> = {};
  for (const [name, spec] of Object.entries(file.servers)) {
    mcpServers[name] = claudeServerEntry(spec);
  }
  return { mcpServers };
}

function claudeServerEntry(spec: McpServerSpec): Record<string, unknown> {
  if (spec.transport === "http") {
    return { type: "http", url: spec.url };
  }
  // stdio
  const out: Record<string, unknown> = { type: "stdio", command: spec.command };
  if (spec.env) out.env = spec.env;
  return out;
}

/**
 * Default tools the agent is allowed to use in claude-permissions mode.
 * These give the agent shell, file I/O, and web access — everything needed
 * for coding tasks. Used by both --tools (availability) and --allowedTools
 * (permission pre-approval).
 */
export const DEFAULT_ALLOWED_TOOLS = [
  "Bash",
  "Read",
  "Edit",
  "Write",
  "WebSearch",
  "WebFetch",
];

/**
 * Git subcommands that read repo history (the gold-state leak vector).
 * Allow add/commit/push/pull/remote/status/init/clone/fetch so envs that
 * deploy via `git push` (Gitea, GitHub Actions, etc.) work without
 * per-env overrides. WebFetch/curl already grant remote read anyway, so
 * blocking clone/fetch wouldn't add real defense.
 */
export const BLOCKED_GIT_SUBCOMMANDS = [
  "log",
  "show",
  "diff",
  "blame",
  "checkout",
  "switch",
  "stash",
  "reflog",
  "fsck",
  "cat-file",
  "rev-list",
  "ls-tree",
  "ls-files",
  "archive",
  "bundle",
  "filter-branch",
  "format-patch",
];

/**
 * Default tool patterns blocked in claude-permissions mode.
 * Prevents the agent from reading the gold-state / environment directory,
 * running orchestrator commands that leak test info, and inspecting repo history.
 *
 * PATTERN SYNTAX (from Claude Code docs):
 *
 *   Read/Edit patterns follow gitignore spec:
 *     //path  → absolute filesystem path (double slash!)
 *     /path   → relative to settings file (NOT absolute — common mistake)
 *     **      → matches recursively across directories
 *     *       → matches within a single directory only
 *
 *   Bash patterns match against the command string:
 *     Bash(git *)         → any command starting with "git " (word boundary)
 *     Bash(* /some/path*) → any command containing "/some/path"
 *
 * IMPORTANT: Read patterns MUST use // prefix for absolute paths.
 * Without //, a path like /hyperfocal/... is interpreted as relative
 * to the settings file and silently matches nothing.
 *
 * NOTE: Bash restrictions are inherently best-effort. A determined agent
 * can read files via creative shell tricks (env vars, python one-liners,
 * etc.). The layered approach (--tools + dontAsk + disallowedTools)
 * raises the bar significantly but does not guarantee perfect isolation.
 * For that, use linux-user mode with filesystem permissions.
 */
export const DEFAULT_DISALLOWED_TOOLS = [
  // --- Git ---
  // Block subcommands that read repo history (see BLOCKED_GIT_SUBCOMMANDS).
  // Each subcommand is blocked both bare (`git log ...`) and with git global
  // options between `git` and the subcommand (`git --no-pager log`,
  // `git -c core.pager=cat log`, `git -C <path> log`, `git -P log`,
  // `git --git-dir=<path> log`, ...). Every git global option starts with
  // "-", so the `git -*` variant covers any option prefix — the bare
  // patterns alone were bypassed by `git --no-pager log` (lgb-004
  // hardening note). Slight over-blocking of benign commands that happen
  // to contain " <subcommand>" after a global option is accepted: deny-side
  // false positives are the safe direction.
  //
  // Defense-in-depth: agent-runner.ts also sets
  // GIT_CEILING_DIRECTORIES=/hyperfocal/env/workspace so the agent can't
  // walk up from the workspace cwd to find the env repo's .git. The
  // residual `cd /hyperfocal/env && git log` hole is documented and
  // accepted under claude-permissions; use linux-user mode if your env
  // needs adversarial-grade isolation.
  ...BLOCKED_GIT_SUBCOMMANDS.flatMap((sub) => [
    `Bash(git ${sub}*)`,
    `Bash(git -* ${sub}*)`,
  ]),
  // Defeat explicit pointing at the env repo via -C / --git-dir / GIT_DIR.
  "Bash(git -C /hyperfocal/env*)",
  "Bash(*--git-dir*hyperfocal/env*)",
  "Bash(*GIT_DIR=*hyperfocal/env*)",

  // --- Environment directory (tests, gold-state, setup code) ---
  // Block Read/Edit/Write to the entire environment directory tree
  "Read(//hyperfocal/env/environment/**)",
  "Edit(//hyperfocal/env/environment/**)",
  "Write(//hyperfocal/env/environment/**)",
  // Block ANY Bash command that mentions the environment path.
  // Catches: cat, head, tail, less, python, cp, find, etc.
  "Bash(* /hyperfocal/env/environment*)",
  // Block commands mentioning gold-state by name (catches creative paths)
  "Bash(*gold-state*)",

  // --- Additional sensitive paths ---
  // Temp dirs can expose historical eval artifacts.
  "Read(//hyperfocal/tmp/**)",
  "Edit(//hyperfocal/tmp/**)",
  "Write(//hyperfocal/tmp/**)",
  "Bash(* /hyperfocal/tmp*)",

  // Migration internals (both canonical and legacy locations)
  "Read(//hyperfocal/migration-env-builder/**)",
  "Edit(//hyperfocal/migration-env-builder/**)",
  "Write(//hyperfocal/migration-env-builder/**)",
  "Bash(* /hyperfocal/migration-env-builder*)",

  "Read(//hyperfocal/env/packages/migration-env-builder/**)",
  "Edit(//hyperfocal/env/packages/migration-env-builder/**)",
  "Write(//hyperfocal/env/packages/migration-env-builder/**)",
  "Bash(* /hyperfocal/env/packages/migration-env-builder*)",

  // Block package tree reads in eval mode to avoid orchestrator internals leakage.
  "Read(//hyperfocal/env/packages/**)",
  "Edit(//hyperfocal/env/packages/**)",
  "Write(//hyperfocal/env/packages/**)",
  "Bash(* /hyperfocal/env/packages*)",

  // Block Claude per-project memory/logs under /root.
  "Read(//root/.claude/projects/**)",
  "Edit(//root/.claude/projects/**)",
  "Write(//root/.claude/projects/**)",
  "Bash(* /root/.claude/projects*)",

  // --- Orchestrator commands (leak test names, failure messages, problem structure) ---
  "Bash(*env-orchestrator test*)",
  "Bash(*env-orchestrator problems*)",
  "Bash(*env-orchestrator prompt*)",

  // --- Config files (leak architecture, paths, problem definitions) ---
  "Read(//hyperfocal/env/hyperfocal.yaml)",
  "Bash(*hyperfocal.yaml*)",
  "Bash(*problems.yaml*)",
];

/**
 * CLI stream message types
 */
interface CLISystemMessage {
  type: "system";
  subtype?: string;
  cwd?: string;
  session_id?: string;
  tools?: string[];
  model?: string | null;
  permissionMode?: string;
  claude_code_version?: string;
  task_id?: string;
  tool_use_id?: string;
  description?: string;
  task_type?: string;
  status?: string;
  output_file?: string;
  summary?: string;
  patch?: {
    status?: string;
    is_backgrounded?: boolean;
    end_time?: number;
    [key: string]: unknown;
  };
}

interface CLIToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface CLITextBlock {
  type: "text";
  text: string;
}

interface CLIThinkingBlock {
  type: "thinking";
  thinking: string;
  signature?: string;
}

interface CLIAssistantMessage {
  type: "assistant";
  message: {
    model: string;
    content: (CLIToolUseBlock | CLITextBlock | CLIThinkingBlock)[];
  };
  session_id: string;
}

/**
 * Bash-style tool result (from Shell, Bash commands)
 */
interface BashToolResult {
  stdout: string;
  stderr: string;
  interrupted: boolean;
  backgroundTaskId?: string;
}

/**
 * User message from CLI - contains tool results in various formats
 */
interface CLIUserMessage {
  type: "user";
  message: {
    role: "user";
    content: Array<{
      type: "tool_result";
      tool_use_id: string;
      content: string;
      is_error?: boolean;
    }>;
  };
  session_id: string;
  uuid?: string;
  parent_tool_use_id?: string | null;
  // Tool results vary by tool type:
  // - Bash/Shell: { stdout, stderr, interrupted }
  // - Write/Read/TodoWrite/etc: structured object specific to the tool
  tool_use_result?: BashToolResult | Record<string, unknown>;
}

interface CLIResultMessage {
  type: "result";
  subtype: "success" | "error_during_execution";
  is_error: boolean;
  stop_reason?: string;
  terminal_reason?: string;
  duration_ms: number;
  num_turns: number;
  result: string;
  session_id: string;
  total_cost_usd: number;
  usage?: ClaudeCodeUsage;
  api_error_status?: number | null;
  origin?: {
    kind?: string;
  };
}

interface ClaudeCodeUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  [key: string]: unknown;
}

interface CLIRateLimitEvent {
  type: "rate_limit_event";
  rate_limit_info?: unknown;
  rateLimitInfo?: unknown;
}

type CLIMessage =
  | CLISystemMessage
  | CLIAssistantMessage
  | CLIUserMessage
  | CLIResultMessage
  | CLIRateLimitEvent;

/**
 * Resolve the path to the Claude Code CLI binary.
 * 
 * Tries `which claude` first, then falls back to known locations.
 * Throws if the binary cannot be found.
 */
function resolveClaudeBinary(): string {
  // Try `which claude` for auto-detection
  try {
    const result = execSync("which claude", { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] }).trim();
    if (result) return result;
  } catch {
    // `which` failed — try known fallback paths
  }

  // Fallback: check known locations
  const knownPaths = [
    "/root/.local/bin/claude",
    "/usr/local/bin/claude",
  ];

  const fs = require("fs");
  for (const p of knownPaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  throw new Error(
    "Claude Code CLI not found. Checked: `which claude`, /root/.local/bin/claude, /usr/local/bin/claude.\n" +
    "Install with: curl -fsSL https://claude.ai/install.sh | bash"
  );
}

/**
 * Claude Code Agent - executes coding tasks using Claude Code CLI
 */
export class ClaudeCodeAgent {
  private config: ClaudeCodeConfiguration;
  private session: TelemetrySession | null = null;
  private cliProcess: ChildProcess | null = null;
  private turnCount: number = 0;
  private claudeBinary: string;
  private systemInitLogged: boolean = false;
  private pendingBackgroundTasks = new Map<string, ClaudeBackgroundTask>();
  private idleCutoffPolicy: ClaudeIdleCutoffPolicy;
  private idleCutoffTimer: NodeJS.Timeout | undefined;
  private idleCutoffForceKillTimer: NodeJS.Timeout | undefined;
  private idleCutoffTriggered = false;
  private idleCutoffTelemetry: Record<string, unknown> | undefined;
  private lastProviderResult: CLIResultMessage | undefined;
  private providerRateLimits = new Map<string, ProviderRateLimitObservation>();
  private resumeOnTaskCompletion = envFlagEnabled("HYPERFOCAL_CLAUDE_RESUME_ON_TASK_COMPLETION", true);
  /** True between a provider result and the next assistant activity. */
  private agentIdle = false;
  private stdinClosed = false;
  private readonly streamSessionId = "session-" + Date.now();
  private readonly backgroundTaskMaxAgeMs = envDurationMs(
    "HYPERFOCAL_CLAUDE_BG_TASK_MAX_AGE_MS",
    DEFAULT_BACKGROUND_TASK_MAX_AGE_MS
  );
  private readonly reaperSynthesizeGraceMs = envDurationMs(
    "HYPERFOCAL_CLAUDE_BG_TASK_REAP_SYNTH_GRACE_MS",
    DEFAULT_REAPER_SYNTHESIZE_TERMINAL_GRACE_MS
  );
  private backgroundTaskReapTimers = new Map<string, NodeJS.Timeout>();
  private reapedBackgroundTaskIds = new Set<string>();

  constructor(config: ClaudeCodeConfiguration) {
    this.config = config;
    this.claudeBinary = resolveClaudeBinary();
    this.idleCutoffPolicy = {
      enabled: envFlagEnabled("HYPERFOCAL_CLAUDE_RESULT_IDLE_CUTOFF", true),
      noPendingTasksMs: envDurationMs(
        "HYPERFOCAL_CLAUDE_RESULT_IDLE_NO_PENDING_MS",
        DEFAULT_IDLE_CUTOFF_NO_PENDING_TASKS_MS
      ),
      withPendingTasksMs: envDurationMs(
        "HYPERFOCAL_CLAUDE_RESULT_IDLE_PENDING_MS",
        DEFAULT_IDLE_CUTOFF_WITH_PENDING_TASKS_MS
      ),
    };
    this.log("info", `ClaudeCodeAgent initialized (model: ${config.model}, binary: ${this.claudeBinary})`);
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

  private emit(event: ReturnType<typeof shellResultEvent>): void {
    this.log(event.type, event.message, event.data);
  }

  /**
   * Get the effective permissions mode, defaulting to "claude-permissions"
   */
  private getPermissionsMode(): PermissionsMode {
    return this.config.permissionsMode || "claude-permissions";
  }

  /**
   * Build CLI arguments based on the permissions mode.
   *
   * claude-permissions (default, recommended):
   *   Four-layer defense for tool isolation:
   *     Layer 1: --tools          → hard-restrict available tool set
   *     Layer 2: --permission-mode dontAsk → auto-deny unapproved tools
   *     Layer 3: --allowedTools   → pre-approve tools for dontAsk
   *     Layer 4: --disallowedTools → deny specific patterns (deny > allow)
   *
   *   Why dontAsk over bypassPermissions:
   *     - bypassPermissions has a known bug where --allowedTools is ignored
   *       (GitHub: anthropics/claude-code#12232)
   *     - dontAsk correctly respects both --allowedTools and --disallowedTools
   *
   * linux-user (legacy):
   *   Uses --permission-mode bypassPermissions with OS-level user isolation.
   */
  private buildCLIArgs(): string[] {
    const maxTurns = this.config.options?.maxTurns || 500;
    const mode = this.getPermissionsMode();

    // MCP wiring: opt-in via mcpConfigPath. The agent reads the canonical
    // neutral file (`mcp-servers.json`), translates it to Claude's native
    // `--mcp-config` schema, and writes the translated config to a
    // per-pid temp file. Tool prefixes ("mcp__<server>") are derived from
    // the same file so adding a second server (e.g. Sentry) Just Works
    // without code changes. A missing/malformed file silently disables MCP.
    const mcpSetup = this.setupClaudeMcpConfig();

    // Common args shared by both modes
    const args = [
      "--print",
      "--verbose",
      "--output-format", "stream-json",
      "--input-format", "stream-json",
      "--model", this.config.model,
      "--max-turns", String(maxTurns),
      "--thinking-display", "summarized",
    ];

    if (mcpSetup) {
      args.push("--mcp-config", mcpSetup.translatedConfigPath);
      args.push("--strict-mcp-config");
    }

    if (mode === "claude-permissions") {
      const baseAllowed = this.config.allowedTools || DEFAULT_ALLOWED_TOOLS;
      // mcp__<server> prefixes grant every mcp__<server>__* tool the server
      // registers. Only appended when MCP is active so non-MCP problems
      // never see these tools in their context.
      const allowedTools = mcpSetup ? [...baseAllowed, ...mcpSetup.toolPrefixes] : baseAllowed;
      const disallowedTools = this.config.disallowedTools || DEFAULT_DISALLOWED_TOOLS;

      // Layer 1: Hard-restrict available tool set.
      // Tools not listed here are removed from the model's context entirely —
      // the agent won't even know they exist (e.g. Task, Grep, Glob).
      args.push("--tools", allowedTools.join(","));

      // Layer 2: Permission mode.
      // dontAsk auto-denies any tool not pre-approved via --allowedTools.
      // This prevents the agent from ever pausing for human approval.
      args.push("--permission-mode", "dontAsk");

      // Layer 3: Pre-approve listed tools so they work under dontAsk mode.
      // Without this, dontAsk would auto-deny everything.
      args.push("--allowedTools", allowedTools.join(","));

      // Layer 4: Deny-list for specific patterns.
      // Deny rules take precedence over allow rules (deny > ask > allow).
      // Uses comma separator to safely handle patterns containing spaces
      // (e.g. "Bash(git *)"). Space-separated would mis-parse these.
      if (disallowedTools.length > 0) {
        args.push("--disallowedTools", disallowedTools.join(","));
      }
    } else {
      // Legacy: bypass permission prompts (requires non-root user).
      // Still pass disallowedTools for defense-in-depth.
      const disallowedTools = this.config.disallowedTools || DEFAULT_DISALLOWED_TOOLS;
      args.push("--permission-mode", "bypassPermissions");
      if (disallowedTools.length > 0) {
        args.push("--disallowedTools", disallowedTools.join(","));
      }
    }

    return args;
  }

  /**
   * Read the neutral `mcp-servers.json`, translate it to Claude's
   * `--mcp-config` schema, and write the translated config to a per-pid
   * temp file. Returns the temp path and the derived `mcp__<server>`
   * tool prefixes — or `null` when MCP isn't active for this run
   * (path unset, file missing, file malformed, or no servers declared).
   *
   * A `null` return cleanly disables every MCP-related flag in
   * `buildCLIArgs`; we never half-enable MCP.
   */
  private setupClaudeMcpConfig(): { translatedConfigPath: string; toolPrefixes: string[] } | null {
    const neutralPath = this.config.mcpConfigPath;
    if (!neutralPath) return null;

    const file = readMcpServersFromPath(neutralPath);
    if (!file) {
      // Either missing (silent no-MCP) or malformed (we want to know).
      // readMcpServersFromPath already returns null for both; differentiate
      // with a cheap existsSync so missing stays quiet.
      if (fs.existsSync(neutralPath)) {
        this.logError(
          `Failed to parse neutral mcp-servers file at ${neutralPath} — running without MCP`,
        );
      }
      return null;
    }

    const serverNames = Object.keys(file.servers);
    if (serverNames.length === 0) return null;

    const translated = toClaudeMcpConfig(file);
    const runtimeDir = path.join(os.homedir(), ".local/share/claude-code/runtime", String(process.pid));
    fs.mkdirSync(runtimeDir, { recursive: true });
    const translatedConfigPath = path.join(runtimeDir, "mcp-config.json");
    fs.writeFileSync(translatedConfigPath, JSON.stringify(translated, null, 2));

    return {
      translatedConfigPath,
      toolPrefixes: serverNames.map((name) => `mcp__${name}`),
    };
  }

  /**
   * Run the agent with the given prompt
   */
  async run(
    prompt: string,
    workingDir: string,
    _options?: AgentRunOptions
  ): Promise<void> {
    if (!prompt?.trim()) {
      throw new Error("Prompt must be a non-empty string");
    }
    this.providerRateLimits.clear();

    const mode = this.getPermissionsMode();
    this.log("info", `Using Claude Code CLI (${this.config.model}, permissions: ${mode})`);
    this.log(
      "info",
      `Prompt: ${prompt.substring(0, 100)}${prompt.length > 100 ? "..." : ""}`,
      { promptLength: prompt.length }
    );
    this.log("info", `Working directory: ${workingDir}`);

    const cliArgs = this.buildCLIArgs();

    this.log("info", `Starting Claude Code CLI`, {
      args: cliArgs,
      binary: this.claudeBinary,
      permissionsMode: mode,
    });

    const inputMessage = {
      type: "user",
      message: {
        role: "user",
        content: prompt,
      },
      session_id: this.streamSessionId,
    };

    let result: Awaited<ReturnType<typeof runStreamingCli>>;
    try {
      result = await runStreamingCli({
        command: this.claudeBinary,
        args: cliArgs,
        cwd: workingDir,
        stdin: "pipe",
        env: {
          ...process.env,
          // HOME should be set by agent-runner, but ensure it's set.
          HOME: process.env.HOME || "/root",
          // Disable Claude Code auto memory for agent rollouts.
          CLAUDE_CODE_DISABLE_AUTO_MEMORY: "1",
        },
        debugLogStream: this.createDebugLogStream(),
        onProcess: (child) => {
          this.cliProcess = child;
        },
        onStdoutLine: (line) => {
          if (!line.trim()) return;
          this.cancelIdleCutoffTimer("stdout_event");

          try {
            const msg = JSON.parse(line) as CLIMessage;
            this.handleCLIMessage(msg);
          } catch (e) {
            // Not JSON - might be raw output, log it
            this.log("info", line);
          }
        },
        onStderrText: (text) => {
          this.cancelIdleCutoffTimer("stderr_event");
          // Log errors but filter out noise
          if (text.includes("Error") || text.includes("error")) {
            this.logError(`CLI stderr: ${text.trim()}`);
          }
        },
        writeStdin: (child) => {
          child.stdin?.write(JSON.stringify(inputMessage) + "\n");
          if (this.resumeOnTaskCompletion) {
            // Keep stdin open: closing it makes the CLI exit at the first
            // yield, abandoning running background tasks. closeStdin() ends
            // the session once the agent yields with nothing pending.
            return;
          }
          child.stdin?.end();
          this.stdinClosed = true;
        },
      });
    } catch (error) {
      const message = `Failed to start Claude Code CLI: ${error instanceof Error ? error.message : String(error)}`;
      this.logError(`Error: ${message}`);
      throw new Error(message);
    } finally {
      this.cancelIdleCutoffTimer("cli_finished");
      this.clearAllBackgroundTaskReapers();
    }

    if (result.completionSource === "exit") {
      this.log("info", `Claude Code CLI exited before stdio closed; settled after drain grace`, {
        exitCode: result.code,
        signal: result.signal,
      });
    }

    if (this.idleCutoffTriggered) {
      this.log("warn", "Claude Code CLI stopped after post-result idle cutoff", {
        ...this.idleCutoffTelemetry,
        exitCode: result.code,
        signal: result.signal,
        completionSource: result.completionSource,
      });
    }

    const rejectedRateLimit = this.latestRejectedRateLimit();
    if (rejectedRateLimit) {
      const error = new ClaudeCodeRateLimitError(rejectedRateLimit);
      this.logError(`Error: ${error.message}`, {
        errorCode: error.code,
        provider: error.provider,
        limitType: error.limitType,
        observedAt: error.observedAt,
        resetsAt: error.resetsAt,
      });
      throw error;
    }

    if (result.code !== 0 || result.signal) {
      if (this.idleCutoffTriggered) {
        this.log("info", "Treating Claude Code post-result idle cutoff as agent completion for grading", {
          exitCode: result.code,
          signal: result.signal,
          completionSource: result.completionSource,
        });
        return;
      }

      let errorMessage = result.signal
        ? `Claude Code CLI terminated with signal ${result.signal}`
        : `Claude Code CLI exited with code ${result.code}`;
      if (result.stderrChunks.length > 0) {
        errorMessage += `\nStderr: ${result.stderrChunks.join("").slice(-500)}`;
      }
      this.logError(`Error: ${errorMessage}`);
      throw new Error(errorMessage);
    }

    this.log("info", `Claude Code CLI completed`, {
      turns: this.turnCount,
      exitCode: result.code,
      completionSource: result.completionSource,
    });
  }

  private createDebugLogStream(): fs.WriteStream | undefined {
    if (!this.config.codingAgentDebugLogs || !this.session) return undefined;
    return fs.createWriteStream(this.session.getDebugLogPath(), { flags: "a" });
  }

  /**
   * Handle a message from the CLI stream
   */
  private handleCLIMessage(msg: CLIMessage): void {
    switch (msg.type) {
      case "system":
        this.handleSystemMessage(msg);
        this.handleTaskSystemMessage(msg);
        break;
      case "assistant":
        this.handleAssistantMessage(msg);
        break;
      case "user":
        this.handleUserMessage(msg);
        break;
      case "result":
        this.handleResultMessage(msg);
        this.scheduleIdleCutoffTimer(msg);
        this.maybeCloseStdinAfterResult(msg);
        break;
      case "rate_limit_event":
        this.handleRateLimitEvent(msg);
        break;
      default:
        // Unknown message type, log raw
        this.log("info", `Unknown CLI message type: ${JSON.stringify(msg).slice(0, 200)}`);
    }
  }

  private handleRateLimitEvent(msg: CLIRateLimitEvent): void {
    const observation = normalizeClaudeRateLimitEvent(msg, new Date().toISOString());
    if (!observation) {
      this.log("warn", "Ignoring malformed Claude Code rate-limit event", {
        providerEventType: "provider_rate_limit_invalid",
        provider: "claude-code",
      });
      return;
    }

    this.recordRateLimitObservation(observation);
  }

  private recordRateLimitObservation(observation: ProviderRateLimitObservation): void {
    const current = this.providerRateLimits.get(observation.limitType);
    if (!current || Date.parse(observation.observedAt) >= Date.parse(current.observedAt)) {
      this.providerRateLimits.set(observation.limitType, observation);
    }

    if (this.session) {
      this.session.recordProviderRateLimitObservation(observation);
      return;
    }

    this.log("info", `Provider rate limit ${observation.status} (${observation.limitType})`, {
      providerEventType: "provider_rate_limit",
      observation,
    });
  }

  private latestRejectedRateLimit(): ProviderRateLimitObservation | undefined {
    return [...this.providerRateLimits.values()]
      .filter((observation) => observation.status === "rejected")
      .sort((a, b) => Date.parse(b.observedAt) - Date.parse(a.observedAt))[0];
  }

  private handleSystemMessage(msg: CLISystemMessage): void {
    // The CLI expands the requested string (e.g. the alias "opus") into a
    // concrete id here. Subsequent init events report model: null, so the
    // truthy guard keeps the first real value.
    if (msg.model) {
      this.session?.setResolvedModel(msg.model);
    }

    if (this.systemInitLogged || !this.isMeaningfulSystemInit(msg)) {
      return;
    }
    this.systemInitLogged = true;

    this.log("info", `Claude Code ${msg.claude_code_version || ""} initialized`, {
      subtype: msg.subtype,
      session_id: msg.session_id,
      model: msg.model,
      permissionMode: msg.permissionMode,
      tools: msg.tools,
      cwd: msg.cwd,
    });
  }

  private handleTaskSystemMessage(msg: CLISystemMessage): void {
    if (!msg.task_id) return;

    if (msg.subtype === "task_started") {
      this.pendingBackgroundTasks.set(msg.task_id, {
        taskId: msg.task_id,
        toolUseId: msg.tool_use_id,
        description: msg.description,
        taskType: msg.task_type,
        startedAt: Date.now(),
      });
      this.armBackgroundTaskReaper(msg.task_id);
      this.log("info", "Claude background task started", {
        providerEventType: "claude_background_task_started",
        taskId: msg.task_id,
        toolUseId: msg.tool_use_id,
        description: msg.description,
        taskType: msg.task_type,
        pendingBackgroundTasks: this.pendingBackgroundTasks.size,
      });
      return;
    }

    if (msg.subtype === "task_updated") {
      const task = this.pendingBackgroundTasks.get(msg.task_id);
      if (task && msg.patch?.is_backgrounded) {
        task.lastKnownStatus = "backgrounded";
      }

      if (msg.patch?.status === "completed" || msg.patch?.status === "failed") {
        const wasPending = this.pendingBackgroundTasks.delete(msg.task_id);
        this.clearBackgroundTaskReaper(msg.task_id);
        this.log("info", "Claude background task updated", {
          providerEventType: "claude_background_task_updated",
          taskId: msg.task_id,
          status: msg.patch.status,
          pendingBackgroundTasks: this.pendingBackgroundTasks.size,
        });
        if (wasPending) {
          this.onBackgroundTaskTerminal(task ?? { taskId: msg.task_id, startedAt: 0 }, msg.patch.status);
        }
      }
      return;
    }

    if (msg.subtype === "task_notification") {
      let terminalTask: ClaudeBackgroundTask | undefined;
      if (msg.status === "completed" || msg.status === "failed") {
        terminalTask = this.pendingBackgroundTasks.get(msg.task_id);
        this.pendingBackgroundTasks.delete(msg.task_id);
        this.clearBackgroundTaskReaper(msg.task_id);
      }
      this.log("info", "Claude background task notification", {
        providerEventType: "claude_background_task_notification",
        taskId: msg.task_id,
        toolUseId: msg.tool_use_id,
        status: msg.status,
        summary: msg.summary,
        outputFile: msg.output_file,
        pendingBackgroundTasks: this.pendingBackgroundTasks.size,
      });
      if (terminalTask) {
        this.onBackgroundTaskTerminal(terminalTask, msg.status ?? "completed", msg.summary);
      } else if (this.agentIdle) {
        // Non-terminal notification while the agent is yielded: the stdout
        // line just cancelled the idle cutoff without waking the agent, so
        // re-arm it — otherwise the session hangs until the agent-runner
        // hard timeout.
        this.rearmIdleCutoffAfterNotification();
      }
    }
  }

  /**
   * A background task reached a terminal state. If the agent has yielded
   * (headless mode has no user to send the next message), write a follow-up
   * user message so the session resumes with the task's outcome — the same
   * thing interactive Claude Code does. Falls back to re-arming the idle
   * cutoff when the resume can't be delivered.
   */
  private onBackgroundTaskTerminal(task: ClaudeBackgroundTask, status: string, summary?: string): void {
    if (!this.resumeOnTaskCompletion || !this.agentIdle) return;

    // A reaped task's CLI-reported summary (if any) says only that the
    // process died — tell the agent WHY, or it will relaunch the same hung
    // command and wait out the cap a second time.
    if (this.reapedBackgroundTaskIds.has(task.taskId)) {
      const explanation = reapExplanation(this.backgroundTaskMaxAgeMs);
      summary = summary ? `${summary} ${explanation}` : explanation;
    }

    const resumed = this.writeFollowupUserMessage(taskResumeMessage(task, status, summary));
    this.log(resumed ? "info" : "warn",
      resumed
        ? "Resuming yielded agent after background task completion"
        : "Could not resume yielded agent (stdin unavailable); re-arming idle cutoff",
      {
        providerEventType: "claude_background_task_resume",
        taskId: task.taskId,
        status,
        resumed,
        pendingBackgroundTasks: this.pendingBackgroundTasks.size,
      });
    if (resumed) {
      this.agentIdle = false;
    }
    // Either way the cutoff was cancelled by this stdout line; re-arm so a
    // failed resume (or a model that never answers) still terminates.
    this.rearmIdleCutoffAfterNotification();
  }

  private rearmIdleCutoffAfterNotification(): void {
    if (this.idleCutoffTimer || this.idleCutoffTriggered) return;
    if (this.lastProviderResult) {
      this.scheduleIdleCutoffTimer(this.lastProviderResult);
    }
  }

  private writeFollowupUserMessage(text: string): boolean {
    const stdin = this.cliProcess?.stdin;
    if (!stdin || stdin.destroyed || this.stdinClosed) return false;
    try {
      stdin.write(JSON.stringify({
        type: "user",
        message: { role: "user", content: text },
        session_id: this.streamSessionId,
      }) + "\n");
      return true;
    } catch (e) {
      this.log("warn", `Failed to write follow-up user message: ${e instanceof Error ? e.message : e}`);
      return false;
    }
  }

  private closeStdin(reason: string): void {
    if (this.stdinClosed) return;
    this.stdinClosed = true;
    try {
      this.cliProcess?.stdin?.end();
    } catch { /* already gone */ }
    this.log("info", "Closed CLI stdin; session will end", {
      providerEventType: "claude_stdin_closed",
      reason,
    });
  }

  /**
   * Resume-mode session termination: after a yield with no pending
   * background tasks (or any error result), close stdin so the CLI exits.
   * With pending tasks the session stays open — onBackgroundTaskTerminal
   * resumes it, and the pending-tasks idle cutoff backstops a hang.
   */
  private maybeCloseStdinAfterResult(msg: CLIResultMessage): void {
    if (!this.resumeOnTaskCompletion || this.stdinClosed) return;
    const pendingCount = this.pendingBackgroundTasks.size;
    if (msg.is_error || msg.subtype !== "success" || pendingCount === 0) {
      this.closeStdin(pendingCount === 0 ? "result_no_pending_tasks" : "error_result");
      return;
    }
    this.log("info", "Agent yielded with pending background tasks; keeping session open for resume", {
      providerEventType: "claude_session_held_open",
      pendingBackgroundTasks: this.summarizePendingBackgroundTasks(),
    });
  }

  private handleAssistantMessage(msg: CLIAssistantMessage): void {
    this.turnCount++;
    this.agentIdle = false;
    
    for (const block of msg.message.content) {
      if (block.type === "tool_use") {
        const inputPreview = JSON.stringify(block.input).slice(0, 150);
        this.emit(toolCallEvent({
          tool: block.name,
          toolUseId: block.id,
          family: classifyClaudeToolFamily(block.name),
          input: block.input,
          message: `${block.name}: ${inputPreview}${inputPreview.length >= 150 ? "..." : ""}`,
        }));
      } else if (block.type === "text") {
        this.log("agent_text", block.text, { text: block.text });
      } else if (block.type === "thinking" && block.thinking) {
        this.emit(agentReasoningEvent(block.thinking));
      }
    }
  }

  private handleUserMessage(msg: CLIUserMessage): void {
    // Tool results from CLI - format varies by tool type
    if (msg.tool_use_result) {
      const result = msg.tool_use_result;
      const toolUseId = this.getToolResultId(msg);
      
      // Bash-style results have stdout/stderr/interrupted
      if (this.isBashToolResult(result)) {
        const outputPreview = result.stdout.slice(0, 200);
        this.emit(shellResultEvent({
          toolUseId,
          stdout: result.stdout,
          stderr: result.stderr,
          interrupted: result.interrupted,
          message: `Result: ${outputPreview}${result.stdout.length > 200 ? "..." : ""}`,
        }));
      } else {
        // Structured results from other tools (Write, Read, TodoWrite, Edit, etc.)
        const resultJson = JSON.stringify(result);
        const preview = resultJson.slice(0, 200);
        this.emit(structuredResultEvent({
          toolUseId,
          result,
          message: `Result: ${preview}${resultJson.length > 200 ? "..." : ""}`,
        }));
      }
    } else if (msg.message?.content) {
      // Some tools only send results via message.content (fallback path)
      for (const block of msg.message.content) {
        if (block.type === "tool_result") {
          const preview = block.content.slice(0, 200);
          const toolIdSuffix = block.tool_use_id.slice(-8);
          this.log("tool_result", `${toolIdSuffix}: ${preview}${block.content.length > 200 ? "..." : ""}`, {
            tool_use_id: block.tool_use_id,
            content: block.content,
            is_error: block.is_error,
          });
        }
      }
    }
  }

  private isMeaningfulSystemInit(msg: CLISystemMessage): boolean {
    return Boolean(
      msg.claude_code_version ||
      msg.model ||
      msg.permissionMode ||
      msg.cwd ||
      (Array.isArray(msg.tools) && msg.tools.length > 0)
    );
  }

  private getToolResultId(msg: CLIUserMessage): string | undefined {
    const contentId = msg.message?.content?.find(
      (block) => block.type === "tool_result" && block.tool_use_id
    )?.tool_use_id;
    return contentId || msg.parent_tool_use_id || undefined;
  }

  /**
   * Type guard to check if a tool result is Bash-style (has stdout/stderr/interrupted)
   */
  private isBashToolResult(result: unknown): result is BashToolResult {
    return (
      typeof result === "object" &&
      result !== null &&
      "stdout" in result &&
      typeof (result as BashToolResult).stdout === "string"
    );
  }

  private handleResultMessage(msg: CLIResultMessage): void {
    this.lastProviderResult = msg;
    this.agentIdle = true;
    this.log("info", `Claude Code provider result: ${msg.subtype} (${msg.num_turns} turns, $${msg.total_cost_usd.toFixed(4)})`, {
      providerEventType: "claude_result",
      subtype: msg.subtype,
      is_error: msg.is_error,
      stop_reason: msg.stop_reason,
      terminal_reason: msg.terminal_reason,
      origin: msg.origin,
      duration_ms: msg.duration_ms,
      num_turns: msg.num_turns,
      total_cost_usd: msg.total_cost_usd,
      usage: msg.usage,
      api_error_status: msg.api_error_status,
      result: msg.result,
      pendingBackgroundTasks: this.summarizePendingBackgroundTasks(),
    });

    // Some Claude versions report only api_error_status=429 and omit the
    // structured rate_limit_event. Preserve the richer observation when one
    // was already emitted; otherwise record a confirmed rejection with an
    // explicitly unknown window.
    if (!this.latestRejectedRateLimit()) {
      const observation = normalizeClaudeApiRateLimitResult(msg, new Date());
      if (observation) this.recordRateLimitObservation(observation);
    }
  }

  private scheduleIdleCutoffTimer(msg: CLIResultMessage): void {
    if (!this.idleCutoffPolicy.enabled || msg.is_error || msg.subtype !== "success") return;
    const pendingTasks = this.summarizePendingBackgroundTasks();
    const pendingCount = pendingTasks.length;
    const idleMs = pendingCount > 0
      ? this.idleCutoffPolicy.withPendingTasksMs
      : this.idleCutoffPolicy.noPendingTasksMs;

    // This is not a Claude Code completion contract. It is a Hyperfocal
    // grading policy: once Claude has yielded and stayed quiet long enough,
    // stop waiting for provider-managed background work and let tests grade
    // the current environment state.
    this.idleCutoffTimer = setTimeout(() => {
      this.idleCutoffTriggered = true;
      this.idleCutoffTelemetry = {
        providerEventType: "claude_result_idle_cutoff",
        idleMs,
        pendingBackgroundTasks: pendingTasks,
        pendingBackgroundTaskCount: pendingCount,
        resultPreview: msg.result.slice(0, 500),
        stop_reason: msg.stop_reason,
        terminal_reason: msg.terminal_reason,
        origin: msg.origin,
      };

      this.log("warn", "Claude Code post-result idle cutoff fired; terminating CLI so tests can grade current state", {
        ...this.idleCutoffTelemetry,
      });

      this.cliProcess?.kill("SIGTERM");
      this.idleCutoffForceKillTimer = setTimeout(() => {
        this.cliProcess?.kill("SIGKILL");
      }, IDLE_CUTOFF_SIGKILL_GRACE_MS);
    }, idleMs);

    this.log("info", "Claude Code post-result idle cutoff armed", {
      providerEventType: "claude_result_idle_cutoff_armed",
      idleMs,
      pendingBackgroundTasks: pendingTasks,
      pendingBackgroundTaskCount: pendingCount,
      resultPreview: msg.result.slice(0, 500),
      stop_reason: msg.stop_reason,
      terminal_reason: msg.terminal_reason,
      origin: msg.origin,
    });
  }

  private cancelIdleCutoffTimer(reason: string): void {
    if (this.idleCutoffTriggered && reason !== "cli_finished") {
      return;
    }

    const hadTimer = Boolean(this.idleCutoffTimer || this.idleCutoffForceKillTimer);
    if (this.idleCutoffTimer) {
      clearTimeout(this.idleCutoffTimer);
      this.idleCutoffTimer = undefined;
    }
    if (this.idleCutoffForceKillTimer) {
      clearTimeout(this.idleCutoffForceKillTimer);
      this.idleCutoffForceKillTimer = undefined;
    }
    if (hadTimer && reason !== "cli_finished") {
      this.log("info", "Claude Code post-result idle cutoff cancelled", {
        providerEventType: "claude_result_idle_cutoff_cancelled",
        reason,
      });
    }
  }

  private armBackgroundTaskReaper(taskId: string): void {
    if (this.backgroundTaskMaxAgeMs <= 0) return;
    this.clearBackgroundTaskReaper(taskId);
    const timer = setTimeout(() => this.reapBackgroundTask(taskId), this.backgroundTaskMaxAgeMs);
    timer.unref?.();
    this.backgroundTaskReapTimers.set(taskId, timer);
  }

  private clearBackgroundTaskReaper(taskId: string): void {
    const timer = this.backgroundTaskReapTimers.get(taskId);
    if (timer) {
      clearTimeout(timer);
      this.backgroundTaskReapTimers.delete(taskId);
    }
  }

  private clearAllBackgroundTaskReapers(): void {
    for (const timer of this.backgroundTaskReapTimers.values()) clearTimeout(timer);
    this.backgroundTaskReapTimers.clear();
  }

  /**
   * The per-task max age fired. Kill the task's process tree and let the CLI
   * observe the death: it reports the task terminal through the normal
   * task_updated/notification path, which already resumes a yielded agent.
   * Synthesizing the terminal state immediately would race that report and
   * double-deliver the resume message, so we only synthesize after a grace
   * window in which the CLI stayed silent (e.g. the fd scan found nothing
   * because the processes were already zombies).
   */
  private reapBackgroundTask(taskId: string): void {
    this.backgroundTaskReapTimers.delete(taskId);
    const task = this.pendingBackgroundTasks.get(taskId);
    if (!task) return;

    this.reapedBackgroundTaskIds.add(taskId);
    const terminated = this.killBackgroundTaskProcesses(taskId, "SIGTERM");
    this.log("warn", "Claude background task exceeded max age; killing its processes", {
      providerEventType: "claude_background_task_reaped",
      taskId,
      description: task.description,
      ageMs: Date.now() - task.startedAt,
      maxAgeMs: this.backgroundTaskMaxAgeMs,
      signaledPids: terminated,
      pendingBackgroundTasks: this.pendingBackgroundTasks.size,
    });

    const sigkill = setTimeout(() => {
      if (this.pendingBackgroundTasks.has(taskId)) {
        this.killBackgroundTaskProcesses(taskId, "SIGKILL");
      }
    }, REAPER_SIGKILL_GRACE_MS);
    sigkill.unref?.();

    const synthesize = setTimeout(() => {
      this.backgroundTaskReapTimers.delete(taskId);
      const still = this.pendingBackgroundTasks.get(taskId);
      if (!still) return;
      this.pendingBackgroundTasks.delete(taskId);
      this.log("warn", "Reaped background task never reported terminal; synthesizing failure", {
        providerEventType: "claude_background_task_reap_synthesized",
        taskId,
        pendingBackgroundTasks: this.pendingBackgroundTasks.size,
      });
      this.onBackgroundTaskTerminal(still, "failed");
    }, this.reaperSynthesizeGraceMs);
    synthesize.unref?.();
    this.backgroundTaskReapTimers.set(taskId, synthesize);
  }

  /**
   * Find the task's processes by their open fd on the task output file
   * (…/tasks/<taskId>.output — the shell and all its children inherit it) and
   * signal them. Never signals this process or the CLI itself: the CLI also
   * holds an fd on the file to tail it, and killing the CLI would end the
   * session this reaper exists to protect.
   */
  private killBackgroundTaskProcesses(taskId: string, signal: NodeJS.Signals): number[] {
    const suffix = `/tasks/${taskId}.output`;
    const signaled: number[] = [];
    let entries: string[];
    try {
      entries = fs.readdirSync("/proc");
    } catch {
      return signaled;
    }
    for (const entry of entries) {
      if (!/^\d+$/.test(entry)) continue;
      const pid = Number(entry);
      if (pid === process.pid || pid === this.cliProcess?.pid) continue;
      let fds: string[];
      try {
        fds = fs.readdirSync(`/proc/${entry}/fd`);
      } catch {
        continue;
      }
      for (const fd of fds) {
        let target: string;
        try {
          target = fs.readlinkSync(`/proc/${entry}/fd/${fd}`);
        } catch {
          continue;
        }
        if (target.endsWith(suffix)) {
          try {
            process.kill(pid, signal);
            signaled.push(pid);
          } catch {
            // Already gone, or not ours to signal.
          }
          break;
        }
      }
    }
    return signaled;
  }

  private summarizePendingBackgroundTasks(): Array<Record<string, unknown>> {
    const now = Date.now();
    return Array.from(this.pendingBackgroundTasks.values()).map((task) => ({
      taskId: task.taskId,
      toolUseId: task.toolUseId,
      description: task.description,
      taskType: task.taskType,
      ageMs: now - task.startedAt,
      lastKnownStatus: task.lastKnownStatus,
    }));
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
  getConfiguration(): ClaudeCodeConfiguration {
    return { ...this.config };
  }
}
