/**
 * Telemetry types for persistent logging
 */

/**
 * Log categories - separate log files per category
 */
export type LogCategory = "agent" | "tests" | "setup" | "cleanup";

/**
 * Event types for structured logging
 */
export type LogEventType =
  // Session lifecycle
  | "session_start"
  | "session_end"
  // Agent events
  | "turn_start"
  | "turn_end"
  | "agent_reasoning"
  | "agent_text"
  | "tool_call"
  | "tool_result"
  // Command execution
  | "command_start"
  | "command_end"
  // Test events
  | "test_start"
  | "test_result"
  // Setup events
  | "setup_start"
  | "setup_end"
  // Cleanup events
  | "cleanup_start"
  | "cleanup_end"
  // Generic
  | "info"
  | "warn"
  | "error";

/**
 * Structured log event for JSONL files
 */
export interface LogEvent {
  timestamp: string; // ISO 8601
  type: LogEventType;
  category: LogCategory;
  message: string; // Human-readable message
  data?: Record<string, unknown>; // Structured data (truncated for large outputs)
}

/**
 * Provider-neutral rate-limit state observed during an agent session.
 *
 * `resetsAt` is provider-reported and diagnostic. Scheduling policy belongs
 * to the control plane rather than this telemetry package.
 */
export interface ProviderRateLimitObservation {
  provider: "claude-code";
  status: "allowed" | "allowed_warning" | "rejected";
  limitType: string;
  observedAt: string;
  resetsAt?: string;
  utilization?: number;
  threshold?: number;
  isUsingOverage?: boolean;
  overageStatus?: string;
}

/**
 * Session metadata stored in metadata.json
 * 
 * Status meanings:
 * - "running": Session is in progress
 * - "completed": Session finished successfully
 * - "failed": Session failed (agent's work was incorrect for tests)
 * - "errored": Session couldn't complete due to environment/infrastructure issue
 */
export interface SessionMetadata {
  problemId: string;
  category: LogCategory;
  command: string; // e.g., 'solve', 'test', 'setup', 'rollout'
  // `model` is the requested string (may be an alias like "opus" or a
  // concrete id); `resolvedModel` is the concrete id the agent actually ran,
  // reported by the agent process. They diverge for aliases.
  model?: string;
  resolvedModel?: string;
  /**
   * For agent sessions: which agent implementation was used.
   * Kept as a loose string in env-base so this package doesn't take on a
   * dependency on the env-orchestrator's AgentType union. Producers in
   * env-orchestrator pass one of:
   * "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent".
   */
  agentType?: string;
  startTime: string; // ISO 8601
  endTime?: string;
  status: "running" | "completed" | "failed" | "errored";
  error?: string;
  /** Latest provider rate-limit observation for each provider limit window. */
  providerRateLimits?: Record<string, ProviderRateLimitObservation>;
}

/**
 * Rollout execution phase
 */
export type RolloutPhase = "setup" | "agent" | "tests" | "complete";

/**
 * Explicit rollout status (set when rollout finishes)
 * 
 * Status meanings:
 * - "completed": All tests passed, rollout successful
 * - "failed": Tests failed (agent didn't solve the problem)
 * - "errored": Infrastructure/environment issue prevented completion
 */
export type RolloutStatus = "completed" | "failed" | "errored";

/**
 * Problem-level metadata (aggregates all sessions)
 * 
 * The control plane reads this file to determine rollout status.
 * When rolloutStatus is present, it takes precedence over session-derived status.
 */
export interface ProblemMetadata {
  problemId: string;
  lastActivity: string; // ISO 8601
  sessions: SessionSummary[];

  /**
   * The interpolated problem prompt given to the agent.
   * Set once when the first session starts (typically setup session).
   * This captures exactly what the agent saw, after variable substitution.
   */
  problemPrompt?: string;

  /**
   * Explicit rollout status - set when rollout finishes.
   * Takes precedence over session-derived status when present.
   */
  rolloutStatus?: RolloutStatus;

  /**
   * Timestamp when rollout was finalized (ISO 8601)
   */
  rolloutFinalizedAt?: string;

  /**
   * Error message if rollout failed or errored
   */
  rolloutError?: string;

  /**
   * Current execution phase - helps UI show accurate state during rollout
   */
  currentPhase?: RolloutPhase;

  /**
   * Aggregate rollout score. Weighted mean of individual test scores:
   *   sum(weight * score) / sum(weight)
   * over non-skipped, non-errored tests. Tests default to weight=1 when
   * unspecified, so suites that don't declare weights reproduce the old
   * flat-mean math exactly.
   *
   * Signed and not clamped to [0, 1] — when penalty tests fire, this value
   * can be negative. Researchers see the full signal; the categorical
   * rolloutStatus remains the success/failure source of truth for any
   * consumer that needs a boolean outcome.
   */
  rolloutScore?: number;
}

/**
 * Summary of a single session for metadata.json
 *
 * `model` and `agentType` are only populated for `agent` category sessions.
 * They were added so the control plane can surface the resolved model/agent
 * in run + rollout list views without parsing the agent .jsonl trace. See
 * rolloutStatusSyncService.ts in the control-plane backend for the consumer.
 */
export interface SessionSummary {
  timestamp: string;
  category: LogCategory;
  command: string;
  status: "running" | "completed" | "failed" | "errored";
  duration?: number; // milliseconds
  error?: string;
  /** Requested model string for agent sessions (may be an alias like "opus"). */
  model?: string;
  /** Concrete id the agent actually ran (e.g., "claude-opus-4-8"); differs from `model` for aliases. */
  resolvedModel?: string;
  /** Agent implementation used: "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent" */
  agentType?: string;
  /** Latest provider rate-limit observation for each provider limit window. */
  providerRateLimits?: Record<string, ProviderRateLimitObservation>;
}
