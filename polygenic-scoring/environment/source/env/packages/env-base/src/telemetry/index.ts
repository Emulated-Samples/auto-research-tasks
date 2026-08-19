/**
 * Telemetry module for persistent logging
 *
 * Provides dual output: human-readable .log and structured .jsonl
 *
 * Directory structure:
 * /hyperfocal/logs/
 * └── {problem-id}/
 *     ├── metadata.json           # Session metadata (updated each run)
 *     ├── combined.log            # Chronological log of ALL activity (appends)
 *     ├── agent/
 *     │   ├── {timestamp}.log       # Human-readable agent session
 *     │   ├── {timestamp}.jsonl     # Structured agent events
 *     │   └── {timestamp}.debug.log # Raw coding-agent CLI stream (when enabled)
 *     ├── tests/
 *     │   └── ...
 *     └── setup/
 *         └── ...
 *
 * Note: We use a fixed path /hyperfocal/logs/ instead of $HOME/.hyperfocal/logs/
 * because both the orchestrator (root) and agent (hyperfocal-agent) need to write
 * to the same location. The directory is created with mode 777 to allow both users.
 */

import * as fs from "fs";
import * as path from "path";
import type {
  LogEvent,
  LogCategory,
  LogEventType,
  SessionMetadata,
  ProblemMetadata,
  SessionSummary,
  ProviderRateLimitObservation,
  RolloutPhase,
  RolloutStatus,
} from "./types.js";

/**
 * Logs directory - shared between orchestrator and agent
 *
 * On EC2: /hyperfocal/logs/ (created by userdata with mode 777)
 * For local dev: Falls back to $HOME/.hyperfocal/logs/
 */
const LOGS_DIR = process.env.HYPERFOCAL_LOGS_DIR ||
  (fs.existsSync("/hyperfocal/logs")
    ? "/hyperfocal/logs"
    : path.join(process.env.HOME || "/root", ".hyperfocal", "logs"));
const MAX_JSONL_OUTPUT_LENGTH = 10000; // Truncate output in JSONL to this length

/**
 * TelemetrySession - manages logging for a single command invocation
 *
 * Creates and writes to:
 * - Category-specific .log file (human-readable, full output)
 * - Category-specific .jsonl file (structured, truncated output)
 * - Combined log file (appends to problem-level chronological log)
 */
export class TelemetrySession {
  private problemId: string;
  private category: LogCategory;
  private timestamp: string;
  private logFile: string;
  private jsonlFile: string;
  private combinedLogFile: string;
  private metadata: SessionMetadata;
  private ended: boolean = false;
  /** The interpolated problem prompt to store in metadata (optional) */
  private problemPrompt?: string;

  constructor(
    problemId: string,
    category: LogCategory,
    command: string,
    model?: string,
    problemPrompt?: string,
    agentType?: string
  ) {
    this.problemId = problemId;
    this.category = category;
    this.problemPrompt = problemPrompt;
    // Use filesystem-safe timestamp format
    this.timestamp = new Date().toISOString().replace(/[:.]/g, "-");

    // Ensure directories exist with world-writable permissions
    // This is necessary because both root (orchestrator) and hyperfocal-agent need to write
    const problemDir = path.join(LOGS_DIR, problemId);
    const categoryDir = path.join(problemDir, category);
    fs.mkdirSync(categoryDir, { recursive: true, mode: 0o777 });
    
    // Ensure parent directories have correct permissions (recursive: true may not set mode on existing dirs)
    try {
      fs.chmodSync(problemDir, 0o777);
      fs.chmodSync(categoryDir, 0o777);
    } catch {
      // Ignore permission errors (may not be owner)
    }

    // File paths
    this.logFile = path.join(categoryDir, `${this.timestamp}.log`);
    this.jsonlFile = path.join(categoryDir, `${this.timestamp}.jsonl`);
    this.combinedLogFile = path.join(problemDir, "combined.log");

    // Initialize metadata
    this.metadata = {
      problemId,
      category,
      command,
      model,
      agentType,
      startTime: new Date().toISOString(),
      status: "running",
    };

    // Write session start marker to combined log
    this.writeCombinedHeader();

    // Update problem metadata
    this.updateProblemMetadata();
  }

  /**
   * Log an event - writes to console, .log, and .jsonl
   */
  log(
    type: LogEventType,
    message: string,
    data?: Record<string, unknown>
  ): void {
    if (this.ended) return;

    const event: LogEvent = {
      timestamp: new Date().toISOString(),
      type,
      category: this.category,
      message,
      data: this.truncateData(data),
    };

    // Console output
    console.log(message);

    // Human-readable log (full output)
    const logLine = `[${event.timestamp}] ${message}\n`;
    this.safeAppend(this.logFile, logLine);
    this.safeAppend(this.combinedLogFile, logLine);

    // Structured JSONL (truncated output)
    this.safeAppend(this.jsonlFile, JSON.stringify(event) + "\n");
  }

  /**
   * Log without console output (for raw data passthrough like command output)
   */
  logRaw(text: string): void {
    if (this.ended) return;
    this.safeAppend(this.logFile, text);
    this.safeAppend(this.combinedLogFile, text);
  }

  /**
   * Log only to console and files, no JSONL event (for streaming output)
   */
  logOutput(text: string): void {
    if (this.ended) return;
    process.stdout.write(text);
    this.safeAppend(this.logFile, text);
    this.safeAppend(this.combinedLogFile, text);
  }

  /**
   * End the session - marks completion status and writes footer
   * 
   * @param status - Session outcome:
   *   - "completed": Session finished successfully
   *   - "failed": Session failed (agent's work was incorrect for tests)
   *   - "errored": Session couldn't complete due to environment/infrastructure issue
   */
  end(status: "completed" | "failed" | "errored", error?: string): void {
    if (this.ended) return;
    this.ended = true;

    this.metadata.endTime = new Date().toISOString();
    this.metadata.status = status;
    if (error) {
      this.metadata.error = error;
    }

    // Write session end marker
    const duration =
      new Date(this.metadata.endTime).getTime() -
      new Date(this.metadata.startTime).getTime();
    const marker =
      `\n${"=".repeat(80)}\n` +
      `[${this.metadata.endTime}] ${this.category.toUpperCase()} ${status} ` +
      `(${Math.round(duration / 1000)}s)\n` +
      `${"=".repeat(80)}\n\n`;

    this.safeAppend(this.logFile, marker);
    this.safeAppend(this.combinedLogFile, marker);

    // Log end event to JSONL
    const endEvent: LogEvent = {
      timestamp: this.metadata.endTime,
      type: "session_end",
      category: this.category,
      message: `Session ${status}`,
      data: {
        duration,
        status,
        error,
      },
    };
    this.safeAppend(this.jsonlFile, JSON.stringify(endEvent) + "\n");

    // Update problem metadata with final status
    this.updateProblemMetadata();
  }

  /**
   * Record the concrete model the agent resolved to (e.g. the CLI expands
   * the alias "opus" into "claude-opus-4-8"). Agents call this once the
   * provider reports the real id. Ignores empty values and no-op repeats so
   * a stream that re-reports the same id doesn't rewrite metadata each turn.
   */
  setResolvedModel(model: string): void {
    if (!model || this.metadata.resolvedModel === model) return;
    this.metadata.resolvedModel = model;
    this.updateProblemMetadata();
  }

  /**
   * Emit the canonical provider-rate-limit event and project the newest
   * observation for this limit window into metadata.json.
   */
  recordProviderRateLimitObservation(observation: ProviderRateLimitObservation): void {
    if (this.ended) return;

    this.log(
      "info",
      `Provider rate limit ${observation.status} (${observation.limitType})`,
      {
        providerEventType: "provider_rate_limit",
        observation,
      }
    );

    const current = this.metadata.providerRateLimits?.[observation.limitType];
    if (current && Date.parse(current.observedAt) > Date.parse(observation.observedAt)) {
      return;
    }

    this.metadata.providerRateLimits = {
      ...(this.metadata.providerRateLimits ?? {}),
      [observation.limitType]: observation,
    };
    this.updateProblemMetadata();
  }

  /**
   * Get paths to log files
   */
  getPaths(): { log: string; jsonl: string; combined: string; debug: string } {
    return {
      log: this.logFile,
      jsonl: this.jsonlFile,
      combined: this.combinedLogFile,
      debug: this.getDebugLogPath(),
    };
  }

  /**
   * Path for raw coding-agent protocol capture.
   *
   * This is a sidecar to product JSONL: callers append the CLI stdout/stderr
   * bytes directly, without parsing or reshaping, so protocol regressions can
   * be investigated without polluting the normalized trace events.
   */
  getDebugLogPath(): string {
    return path.join(
      path.dirname(this.jsonlFile),
      `${this.timestamp}.debug.log`
    );
  }

  /**
   * Get the problem ID for this session
   */
  getProblemId(): string {
    return this.problemId;
  }

  /**
   * Check if session has ended
   */
  hasEnded(): boolean {
    return this.ended;
  }

  private writeCombinedHeader(): void {
    const header =
      `\n${"=".repeat(80)}\n` +
      `[${this.metadata.startTime}] ${this.category.toUpperCase()} started` +
      `${this.metadata.model ? ` (model: ${this.metadata.model})` : ""}\n` +
      `Command: ${this.metadata.command}\n` +
      `${"=".repeat(80)}\n`;

    this.safeAppend(this.combinedLogFile, header);
  }

  private updateProblemMetadata(): void {
    const metadataPath = path.join(LOGS_DIR, this.problemId, "metadata.json");

    let problemMeta: ProblemMetadata;
    try {
      if (fs.existsSync(metadataPath)) {
        problemMeta = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
      } else {
        problemMeta = {
          problemId: this.problemId,
          lastActivity: this.metadata.startTime,
          sessions: [],
        };
      }
    } catch {
      // If metadata is corrupted, start fresh
      problemMeta = {
        problemId: this.problemId,
        lastActivity: this.metadata.startTime,
        sessions: [],
      };
    }

    // Store problem prompt if provided (only set once, first session wins)
    // This captures exactly what prompt the agent received
    if (this.problemPrompt && !problemMeta.problemPrompt) {
      problemMeta.problemPrompt = this.problemPrompt;
    }

    // Update or add session summary
    const existingIdx = problemMeta.sessions.findIndex(
      (s) => s.timestamp === this.timestamp && s.category === this.category
    );

    const summary: SessionSummary = {
      timestamp: this.timestamp,
      category: this.category,
      command: this.metadata.command,
      status: this.metadata.status,
      duration: this.metadata.endTime
        ? new Date(this.metadata.endTime).getTime() -
          new Date(this.metadata.startTime).getTime()
        : undefined,
      error: this.metadata.error,
      // Forward agent-session identity so the control plane can populate
      // runs / rollouts model + agent_type from telemetry. These are only
      // set on agent sessions (producer convention); leaving them undefined
      // on setup/tests sessions is intentional. `model` is the requested
      // string; `resolvedModel` is the concrete id the agent reported.
      model: this.metadata.model,
      resolvedModel: this.metadata.resolvedModel,
      agentType: this.metadata.agentType,
      providerRateLimits: this.metadata.providerRateLimits,
    };

    if (existingIdx >= 0) {
      problemMeta.sessions[existingIdx] = summary;
    } else {
      problemMeta.sessions.push(summary);
    }

    problemMeta.lastActivity = new Date().toISOString();

    try {
      fs.writeFileSync(metadataPath, JSON.stringify(problemMeta, null, 2), { mode: 0o666 });
      // Ensure file is world-writable so both root (orchestrator) and hyperfocal-agent can update it
      try {
        fs.chmodSync(metadataPath, 0o666);
      } catch {
        // Ignore - may not be owner
      }
    } catch (err) {
      console.error("Failed to write problem metadata:", err);
    }
  }

  private truncateData(
    data?: Record<string, unknown>
  ): Record<string, unknown> | undefined {
    if (!data) return undefined;

    const truncated: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === "string" && value.length > MAX_JSONL_OUTPUT_LENGTH) {
        truncated[key] =
          value.substring(0, MAX_JSONL_OUTPUT_LENGTH) +
          `\n... [truncated ${value.length - MAX_JSONL_OUTPUT_LENGTH} chars]`;
      } else {
        truncated[key] = value;
      }
    }
    return truncated;
  }

  private safeAppend(filePath: string, content: string): void {
    try {
      const isNewFile = !fs.existsSync(filePath);
      fs.appendFileSync(filePath, content);
      // Make new files world-writable so both root (orchestrator) and hyperfocal-agent can write
      if (isNewFile) {
        try {
          fs.chmodSync(filePath, 0o666);
        } catch {
          // Ignore - may not be owner of parent directory
        }
      }
    } catch (err) {
      // Don't let logging failures break execution
      console.error(`Failed to write to ${filePath}:`, err);
    }
  }
}

/**
 * Create a new telemetry session
 *
 * @param problemId - The problem identifier (used as directory name)
 * @param category - Log category: 'agent', 'tests', or 'setup'
 * @param command - The command being run: 'solve', 'test', 'setup', 'rollout'
 * @param model - Optional model name (agent sessions only)
 * @param problemPrompt - Optional interpolated problem prompt to store in metadata
 * @param agentType - Optional agent implementation id (agent sessions only):
 *                    "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent"
 */
export function createSession(
  problemId: string,
  category: LogCategory,
  command: string,
  model?: string,
  problemPrompt?: string,
  agentType?: string
): TelemetrySession {
  return new TelemetrySession(problemId, category, command, model, problemPrompt, agentType);
}

/**
 * Get the logs directory path
 */
export function getLogsDir(): string {
  return LOGS_DIR;
}

/**
 * List all problems that have logs
 */
export function listProblemsWithLogs(): string[] {
  if (!fs.existsSync(LOGS_DIR)) {
    return [];
  }
  return fs.readdirSync(LOGS_DIR).filter((name) => {
    try {
      const stat = fs.statSync(path.join(LOGS_DIR, name));
      return stat.isDirectory();
    } catch {
      return false;
    }
  });
}

/**
 * Get metadata for a specific problem
 */
export function getProblemMetadata(problemId: string): ProblemMetadata | null {
  const metadataPath = path.join(LOGS_DIR, problemId, "metadata.json");
  if (!fs.existsSync(metadataPath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
  } catch {
    return null;
  }
}

/**
 * Get the combined log file path for a problem
 */
export function getCombinedLogPath(problemId: string): string {
  return path.join(LOGS_DIR, problemId, "combined.log");
}

/**
 * Read the latest agent trace for a problem.
 *
 * Looks up metadata.json to find the most recent agent session,
 * then reads the corresponding .jsonl (preferred) or .log file.
 *
 * This is the read counterpart to TelemetrySession's write path,
 * used by environment rubrics to provide agent trace context
 * to the LLM judge.
 *
 * @param problemId - The problem identifier
 * @returns The trace file contents, or null if no agent trace exists
 */
export function readAgentTrace(problemId: string): string | null {
  const meta = getProblemMetadata(problemId);
  if (!meta) return null;

  const agentSession = [...(meta.sessions || [])]
    .reverse()
    .find((s) => s.category === "agent" && s.timestamp);

  if (!agentSession?.timestamp) return null;

  const agentDir = path.join(LOGS_DIR, problemId, "agent");

  for (const ext of [".jsonl", ".log"]) {
    const filePath = path.join(agentDir, `${agentSession.timestamp}${ext}`);
    if (fs.existsSync(filePath)) {
      return fs.readFileSync(filePath, "utf-8");
    }
  }

  return null;
}

/**
 * Update the current execution phase in problem metadata.
 * 
 * This helps the control plane and UI show accurate state during rollout:
 * - "setup": Running environment setup
 * - "agent": AI agent is solving the problem  
 * - "tests": Running test suite
 * - "complete": Rollout has finished
 * 
 * @param problemId - The problem identifier
 * @param phase - Current execution phase
 */
export function updateProblemPhase(
  problemId: string,
  phase: RolloutPhase
): void {
  const metadataPath = path.join(LOGS_DIR, problemId, "metadata.json");
  
  if (!fs.existsSync(metadataPath)) {
    console.warn(`[telemetry] Cannot update phase: metadata.json not found for ${problemId}`);
    return;
  }

  try {
    const meta: ProblemMetadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
    meta.currentPhase = phase;
    meta.lastActivity = new Date().toISOString();
    fs.writeFileSync(metadataPath, JSON.stringify(meta, null, 2), { mode: 0o666 });
    
    try {
      fs.chmodSync(metadataPath, 0o666);
    } catch {
      // Ignore - may not be owner
    }
  } catch (err) {
    console.error(`[telemetry] Failed to update phase for ${problemId}:`, err);
  }
}

function timestampToIso(timestamp: string): string | undefined {
  const match = timestamp.match(/^(.+T)(\d{2})-(\d{2})-(\d{2})-(\d{3})Z$/);
  if (!match) return undefined;
  return `${match[1]}${match[2]}:${match[3]}:${match[4]}.${match[5]}Z`;
}

/**
 * Record a forced end for the latest agent session from outside the owning
 * process. This is used when the orchestrator has to kill the agent child
 * before the child can call TelemetrySession.end() itself.
 */
export function recordForcedAgentSessionEnd(
  problemId: string,
  status: "completed" | "failed" | "errored",
  error?: string,
  data?: Record<string, unknown>
): void {
  const category: LogCategory = "agent";
  const metadataPath = path.join(LOGS_DIR, problemId, "metadata.json");

  if (!fs.existsSync(metadataPath)) {
    console.warn(`[telemetry] Cannot mark latest ${category} session ended: metadata.json not found for ${problemId}`);
    return;
  }

  try {
    const meta: ProblemMetadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
    const session = [...(meta.sessions || [])]
      .reverse()
      .find((s) => s.category === category);

    if (!session) {
      console.warn(`[telemetry] Cannot mark latest ${category} session ended: no session found for ${problemId}`);
      return;
    }

    const endedAt = new Date().toISOString();
    const startedAt = timestampToIso(session.timestamp);
    session.status = status;
    session.error = error;
    if (startedAt) {
      session.duration = new Date(endedAt).getTime() - new Date(startedAt).getTime();
    }
    meta.lastActivity = endedAt;

    fs.writeFileSync(metadataPath, JSON.stringify(meta, null, 2), { mode: 0o666 });
    try {
      fs.chmodSync(metadataPath, 0o666);
    } catch {
      // Ignore - may not be owner
    }

    const categoryDir = path.join(LOGS_DIR, problemId, category);
    const logFile = path.join(categoryDir, `${session.timestamp}.log`);
    const jsonlFile = path.join(categoryDir, `${session.timestamp}.jsonl`);
    const combinedLogFile = path.join(LOGS_DIR, problemId, "combined.log");
    const durationText = session.duration !== undefined
      ? ` (${Math.round(session.duration / 1000)}s)`
      : "";
    const marker =
      `\n${"=".repeat(80)}\n` +
      `[${endedAt}] ${category.toUpperCase()} ${status}${durationText}\n` +
      `${error ? `Reason: ${error}\n` : ""}` +
      `${"=".repeat(80)}\n\n`;
    const event: LogEvent = {
      timestamp: endedAt,
      type: "session_end",
      category,
      message: `Session ${status}`,
      data: {
        duration: session.duration,
        status,
        error,
        synthetic: true,
        ...data,
      },
    };

    fs.appendFileSync(logFile, marker);
    fs.appendFileSync(combinedLogFile, marker);
    fs.appendFileSync(jsonlFile, JSON.stringify(event) + "\n");
  } catch (err) {
    console.error(`[telemetry] Failed to mark latest ${category} session ended for ${problemId}:`, err);
  }
}

/**
 * Finalize the rollout with explicit status.
 *
 * This sets the rolloutStatus field which takes precedence over session-derived
 * status in the control plane. Call this at the end of every rollout execution
 * path (success, failure, or error).
 *
 * @param problemId - The problem identifier
 * @param status - Final rollout status:
 *   - "completed": All tests passed
 *   - "failed": Tests failed (agent didn't solve the problem)
 *   - "errored": Infrastructure/environment issue
 * @param error - Optional error message for failed/errored status
 * @param rolloutScore - Optional aggregate rollout score. Signed weighted
 *   mean produced by aggregateTestResults; not clamped to [0, 1].
 *
 * The execution phase is marked complete only when tests were reached, or
 * when the rollout itself completed successfully. Earlier failures retain
 * their last phase so artifact consumers can distinguish phase_not_reached
 * from artifacts that are merely unavailable after a finished phase.
 */
export function finalizeRollout(
  problemId: string,
  status: RolloutStatus,
  error?: string,
  rolloutScore?: number
): void {
  const metadataPath = path.join(LOGS_DIR, problemId, "metadata.json");
  
  if (!fs.existsSync(metadataPath)) {
    console.warn(`[telemetry] Cannot finalize rollout: metadata.json not found for ${problemId}`);
    return;
  }

  try {
    const meta: ProblemMetadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
    meta.rolloutStatus = status;
    meta.rolloutFinalizedAt = new Date().toISOString();
    if (status === "completed" || meta.currentPhase === "tests") {
      meta.currentPhase = "complete";
    }
    if (error) {
      meta.rolloutError = error;
    }
    if (rolloutScore !== undefined) {
      meta.rolloutScore = rolloutScore;
    }
    meta.lastActivity = new Date().toISOString();
    
    fs.writeFileSync(metadataPath, JSON.stringify(meta, null, 2), { mode: 0o666 });
    
    try {
      fs.chmodSync(metadataPath, 0o666);
    } catch {
      // Ignore - may not be owner
    }
    
    console.log(`[telemetry] Rollout finalized: ${status}${error ? ` - ${error}` : ""}`);
  } catch (err) {
    console.error(`[telemetry] Failed to finalize rollout for ${problemId}:`, err);
  }
}

// Re-export types for convenience
export type {
  LogEvent,
  LogCategory,
  LogEventType,
  SessionMetadata,
  ProblemMetadata,
  SessionSummary,
  ProviderRateLimitObservation,
  RolloutPhase,
  RolloutStatus,
} from "./types.js";

// Re-export S3 sync utilities
export { S3TelemetrySync, createS3SyncFromEnv } from "./s3-sync.js";
export type { S3TelemetrySyncConfig } from "./s3-sync.js";
