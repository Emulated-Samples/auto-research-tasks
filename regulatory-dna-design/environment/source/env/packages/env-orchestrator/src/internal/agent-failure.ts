/**
 * Cross-process agent failure handling.
 *
 * Agent implementations live in env-base and run in a child process. Their
 * Error instances cannot cross that process boundary, so the parent consumes
 * the child session summary written to metadata.json instead of importing
 * provider-specific error classes. Keep these guards structural so an
 * env-orchestrator release remains compatible with environments pinned to an
 * older or newer env-base package.
 */

export type AgentFailureStatus = "failed" | "errored";

export interface ClaudeRateLimitObservation {
  provider: "claude-code";
  status: "rejected";
  limitType?: string;
  observedAt?: string;
  resetsAt?: string;
}

export interface AgentFailureDetails {
  status: AgentFailureStatus;
  message: string;
  kind?: "provider_rate_limit";
  rateLimit?: ClaudeRateLimitObservation;
}

interface AgentExecutionResultLike {
  exitCode: number;
  failure?: AgentFailureDetails;
}

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null
    ? value as UnknownRecord
    : undefined;
}

function stringField(record: UnknownRecord | undefined, key: string): string | undefined {
  const value = record?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function normalizeRejectedObservation(value: unknown): ClaudeRateLimitObservation | undefined {
  const record = asRecord(value);
  if (
    stringField(record, "provider") !== "claude-code" ||
    stringField(record, "status") !== "rejected"
  ) {
    return undefined;
  }

  return {
    provider: "claude-code",
    status: "rejected",
    limitType: stringField(record, "limitType"),
    observedAt: stringField(record, "observedAt"),
    resetsAt: stringField(record, "resetsAt"),
  };
}

/**
 * Recognize env-base's typed Claude rejection without importing its class.
 * The observation shape is also accepted so compatible env-base versions can
 * add fields without requiring an orchestrator release.
 */
export function rejectedClaudeRateLimitFromError(
  error: unknown
): ClaudeRateLimitObservation | undefined {
  const record = asRecord(error);
  if (!record) return undefined;

  const observation = normalizeRejectedObservation(record.observation) ||
    normalizeRejectedObservation(record);
  if (!observation) return undefined;

  const hasStableIdentity =
    stringField(record, "code") === "CLAUDE_CODE_RATE_LIMITED" ||
    stringField(record, "name") === "ClaudeCodeRateLimitError" ||
    asRecord(record.observation) !== undefined;

  return hasStableIdentity ? observation : undefined;
}

export function agentSessionKeys(metadata: unknown): Set<string> {
  const sessions = asRecord(metadata)?.sessions;
  if (!Array.isArray(sessions)) return new Set();

  return new Set(
    sessions.flatMap((session) => {
      const record = asRecord(session);
      const category = stringField(record, "category");
      const timestamp = stringField(record, "timestamp");
      return category === "agent" && timestamp
        ? [`${category}:${timestamp}`]
        : [];
    })
  );
}

/** Find the newest agent session created after the parent took its snapshot. */
export function latestNewAgentSession(
  metadata: unknown,
  previousSessionKeys: ReadonlySet<string>
): UnknownRecord | undefined {
  const sessions = asRecord(metadata)?.sessions;
  if (!Array.isArray(sessions)) return undefined;

  for (let index = sessions.length - 1; index >= 0; index--) {
    const session = asRecord(sessions[index]);
    const category = stringField(session, "category");
    const timestamp = stringField(session, "timestamp");
    if (
      category === "agent" &&
      timestamp &&
      !previousSessionKeys.has(`${category}:${timestamp}`)
    ) {
      return session;
    }
  }

  return undefined;
}

function rejectedRateLimitFromSession(
  session: UnknownRecord | undefined
): ClaudeRateLimitObservation | undefined {
  const windows = asRecord(session?.providerRateLimits);
  if (!windows) return undefined;

  for (const observation of Object.values(windows)) {
    const rejected = normalizeRejectedObservation(observation);
    if (rejected) return rejected;
  }

  return undefined;
}

function formatRateLimitMessage(observation: ClaudeRateLimitObservation): string {
  const window = observation.limitType ? ` (${observation.limitType})` : "";
  const reset = observation.resetsAt ? `; provider reset at ${observation.resetsAt}` : "";
  return `Claude Code rate limit rejected${window}${reset}`;
}

/**
 * Convert a child session summary into the failure contract returned by
 * runAgent. A structured rejected observation wins over the legacy "failed"
 * session status; allowed and warning observations do not affect status.
 */
export function agentFailureFromSession(
  session: unknown,
  fallbackMessage: string
): AgentFailureDetails {
  const record = asRecord(session);
  const detailedMessage = stringField(record, "error");
  const rejectedRateLimit = rejectedRateLimitFromSession(record);

  if (rejectedRateLimit) {
    return {
      status: "errored",
      message: detailedMessage || formatRateLimitMessage(rejectedRateLimit),
      kind: "provider_rate_limit",
      rateLimit: rejectedRateLimit,
    };
  }

  return {
    status: stringField(record, "status") === "errored" ? "errored" : "failed",
    message: detailedMessage || fallbackMessage,
  };
}

/** Error thrown in the orchestrator parent while retaining child semantics. */
export class AgentExecutionError extends Error {
  readonly rolloutStatus: AgentFailureStatus;
  readonly exitCode: number;
  readonly failureKind?: AgentFailureDetails["kind"];
  readonly rateLimit?: ClaudeRateLimitObservation;

  constructor(result: AgentExecutionResultLike) {
    const failure = result.failure;
    super(failure?.message || `Agent exited with code ${result.exitCode}`);
    this.name = "AgentExecutionError";
    this.rolloutStatus = failure?.status || "failed";
    this.exitCode = result.exitCode;
    this.failureKind = failure?.kind;
    this.rateLimit = failure?.rateLimit;
  }
}

export function agentExecutionStatus(error: unknown): AgentFailureStatus | undefined {
  const record = asRecord(error);
  if (stringField(record, "name") !== "AgentExecutionError") return undefined;

  const status = stringField(record, "rolloutStatus");
  return status === "failed" || status === "errored" ? status : undefined;
}
