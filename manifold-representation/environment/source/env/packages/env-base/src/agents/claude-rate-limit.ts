import type { ProviderRateLimitObservation } from "../telemetry/types.js";

const RATE_LIMIT_STATUSES = new Set<ProviderRateLimitObservation["status"]>([
  "allowed",
  "allowed_warning",
  "rejected",
]);

type UnknownRecord = Record<string, unknown>;

/**
 * Claude does not identify the quota window on result messages that only
 * report an HTTP 429. Keep a stable synthetic window so those confirmed
 * rejections still reach metadata and scheduler classification.
 */
export const UNKNOWN_CLAUDE_RATE_LIMIT_TYPE = "unknown";

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function firstDefined(record: UnknownRecord, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined) return record[key];
  }
  return undefined;
}

function optionalFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function optionalString(value: unknown, maxLength = 100): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.replace(/\u0000/g, "").trim();
  return normalized ? normalized.slice(0, maxLength) : undefined;
}

function normalizeTimestamp(value: unknown): string | undefined {
  let milliseconds: number;
  if (value instanceof Date) {
    milliseconds = value.getTime();
  } else if (typeof value === "number" && Number.isFinite(value)) {
    // Claude currently reports resetsAt in Unix seconds. Accept milliseconds
    // as well so a protocol update cannot silently create a date in 1970.
    milliseconds = Math.abs(value) < 1e12 ? value * 1000 : value;
  } else if (typeof value === "string" && value.trim()) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      milliseconds = Math.abs(numeric) < 1e12 ? numeric * 1000 : numeric;
    } else {
      milliseconds = Date.parse(value);
    }
  } else {
    return undefined;
  }

  if (!Number.isFinite(milliseconds)) return undefined;
  try {
    return new Date(milliseconds).toISOString();
  } catch {
    return undefined;
  }
}

/**
 * Convert a Claude Code stream-json `rate_limit_event` into the provider-
 * neutral telemetry contract. The caller supplies the observation time so
 * this function remains deterministic and straightforward to test.
 *
 * Only explicitly allowed fields are copied. In particular, raw provider
 * payloads, credential material, and scheduler slot identifiers never enter
 * product telemetry through this path.
 */
export function normalizeClaudeRateLimitEvent(
  event: unknown,
  observedAt: string | number | Date
): ProviderRateLimitObservation | null {
  if (!isRecord(event) || event.type !== "rate_limit_event") return null;

  const info = firstDefined(event, "rate_limit_info", "rateLimitInfo");
  if (!isRecord(info)) return null;

  const status = firstDefined(info, "status");
  const rawLimitType = firstDefined(info, "rateLimitType", "rate_limit_type");
  const limitType = optionalString(rawLimitType);
  const normalizedObservedAt = normalizeTimestamp(observedAt);
  if (
    typeof status !== "string" ||
    !RATE_LIMIT_STATUSES.has(status as ProviderRateLimitObservation["status"]) ||
    typeof rawLimitType !== "string" ||
    rawLimitType.replace(/\u0000/g, "").trim().length > 100 ||
    !limitType ||
    !normalizedObservedAt
  ) {
    return null;
  }

  const observation: ProviderRateLimitObservation = {
    provider: "claude-code",
    status: status as ProviderRateLimitObservation["status"],
    limitType,
    observedAt: normalizedObservedAt,
  };

  const resetsAt = normalizeTimestamp(firstDefined(info, "resetsAt", "resets_at"));
  if (resetsAt) observation.resetsAt = resetsAt;

  const utilization = optionalFiniteNumber(firstDefined(info, "utilization"));
  if (utilization !== undefined) observation.utilization = utilization;

  const threshold = optionalFiniteNumber(firstDefined(info, "threshold"));
  if (threshold !== undefined) observation.threshold = threshold;

  const isUsingOverage = firstDefined(info, "isUsingOverage", "is_using_overage");
  if (typeof isUsingOverage === "boolean") {
    observation.isUsingOverage = isUsingOverage;
  }

  const overageStatus = optionalString(firstDefined(info, "overageStatus", "overage_status"));
  if (overageStatus) observation.overageStatus = overageStatus;

  return observation;
}

/**
 * Convert a Claude Code result carrying a confirmed API 429 into the same
 * provider-neutral contract as a structured `rate_limit_event`.
 *
 * Result messages do not include a limit window or provider reset. Those
 * fields therefore remain unknown instead of being inferred from error text.
 */
export function normalizeClaudeApiRateLimitResult(
  result: unknown,
  observedAt: string | number | Date
): ProviderRateLimitObservation | null {
  if (
    !isRecord(result) ||
    result.type !== "result" ||
    result.api_error_status !== 429
  ) {
    return null;
  }

  const normalizedObservedAt = normalizeTimestamp(observedAt);
  if (!normalizedObservedAt) return null;

  return {
    provider: "claude-code",
    status: "rejected",
    limitType: UNKNOWN_CLAUDE_RATE_LIMIT_TYPE,
    observedAt: normalizedObservedAt,
  };
}

/**
 * Stable error surfaced when Claude reports a rejected provider limit and
 * then exits. The structural fields let callers classify it without parsing
 * the human-readable message.
 */
export class ClaudeCodeRateLimitError extends Error {
  readonly code = "CLAUDE_CODE_RATE_LIMITED" as const;
  readonly provider = "claude-code" as const;
  readonly status = "rejected" as const;
  readonly limitType: string;
  readonly observedAt: string;
  readonly resetsAt?: string;
  readonly observation: ProviderRateLimitObservation;

  constructor(observation: ProviderRateLimitObservation) {
    const resetSuffix = observation.resetsAt
      ? `; provider reset at ${observation.resetsAt}`
      : "";
    super(`Claude Code rate limit rejected (${observation.limitType})${resetSuffix}`);
    this.name = "ClaudeCodeRateLimitError";
    this.limitType = observation.limitType;
    this.observedAt = observation.observedAt;
    this.resetsAt = observation.resetsAt;
    this.observation = observation;
  }
}
