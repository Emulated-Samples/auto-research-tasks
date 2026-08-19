import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const logsDir = await mkdtemp(path.join(os.tmpdir(), "env-base-rate-limit-"));
process.env.HYPERFOCAL_LOGS_DIR = logsDir;

const {
  ClaudeCodeRateLimitError,
  UNKNOWN_CLAUDE_RATE_LIMIT_TYPE,
  normalizeClaudeApiRateLimitResult,
  normalizeClaudeRateLimitEvent,
} = await import("../dist/agents/claude-rate-limit.js");
const {
  createSession,
  getProblemMetadata,
} = await import("../dist/telemetry/index.js");

test.after(async () => {
  await rm(logsDir, { recursive: true, force: true });
});

test("normalizes a rejected Claude event without copying unknown fields", () => {
  const observation = normalizeClaudeRateLimitEvent(
    {
      type: "rate_limit_event",
      credential: "must-not-escape",
      rate_limit_info: {
        status: "rejected",
        rateLimitType: "five_hour",
        resetsAt: 1_783_833_600,
        isUsingOverage: false,
        overageStatus: "rejected",
        overageDisabledReason: "org_level_disabled",
        token: "must-not-escape",
      },
    },
    "2026-07-11T20:58:03.000Z"
  );

  assert.deepEqual(observation, {
    provider: "claude-code",
    status: "rejected",
    limitType: "five_hour",
    observedAt: "2026-07-11T20:58:03.000Z",
    resetsAt: "2026-07-12T05:20:00.000Z",
    isUsingOverage: false,
    overageStatus: "rejected",
  });
  assert.equal(JSON.stringify(observation).includes("must-not-escape"), false);
  assert.equal(JSON.stringify(observation).includes("overageDisabledReason"), false);
});

test("normalizes warning telemetry and ISO reset timestamps", () => {
  assert.deepEqual(
    normalizeClaudeRateLimitEvent(
      {
        type: "rate_limit_event",
        rate_limit_info: {
          status: "allowed_warning",
          rateLimitType: "seven_day",
          resetsAt: "2026-07-13T08:20:00Z",
          utilization: 0.94,
          threshold: 0.9,
          isUsingOverage: true,
        },
      },
      1_783_855_200
    ),
    {
      provider: "claude-code",
      status: "allowed_warning",
      limitType: "seven_day",
      observedAt: "2026-07-12T11:20:00.000Z",
      resetsAt: "2026-07-13T08:20:00.000Z",
      utilization: 0.94,
      threshold: 0.9,
      isUsingOverage: true,
    }
  );
});

test("rejects malformed, unknown, and non-rate-limit events", () => {
  const observedAt = "2026-07-11T20:58:03.000Z";
  const invalidEvents = [
    null,
    { type: "assistant" },
    { type: "rate_limit_event" },
    { type: "rate_limit_event", rate_limit_info: { status: "new_status", rateLimitType: "five_hour" } },
    { type: "rate_limit_event", rate_limit_info: { status: "rejected" } },
  ];

  for (const event of invalidEvents) {
    assert.equal(normalizeClaudeRateLimitEvent(event, observedAt), null);
  }
  assert.equal(
    normalizeClaudeRateLimitEvent(
      { type: "rate_limit_event", rate_limit_info: { status: "rejected", rateLimitType: "five_hour" } },
      "not-a-date"
    ),
    null
  );
});

test("normalizes a confirmed API 429 without copying result payload fields", () => {
  const observation = normalizeClaudeApiRateLimitResult(
    {
      type: "result",
      api_error_status: 429,
      result: "secret-bearing provider error",
      credential: "must-not-escape",
    },
    "2026-07-11T20:58:03Z"
  );

  assert.deepEqual(observation, {
    provider: "claude-code",
    status: "rejected",
    limitType: UNKNOWN_CLAUDE_RATE_LIMIT_TYPE,
    observedAt: "2026-07-11T20:58:03.000Z",
  });
  assert.equal(JSON.stringify(observation).includes("secret-bearing"), false);
  assert.equal(JSON.stringify(observation).includes("must-not-escape"), false);
});

test("does not synthesize rate-limit telemetry for non-429 or malformed results", () => {
  const observedAt = "2026-07-11T20:58:03Z";
  assert.equal(
    normalizeClaudeApiRateLimitResult({ type: "result", api_error_status: 500 }, observedAt),
    null
  );
  assert.equal(
    normalizeClaudeApiRateLimitResult({ type: "assistant", api_error_status: 429 }, observedAt),
    null
  );
  assert.equal(
    normalizeClaudeApiRateLimitResult({ type: "result", api_error_status: 429 }, "not-a-date"),
    null
  );
});

test("persists the latest observation per window and emits canonical JSONL", async () => {
  const session = createSession("rate-limit-metadata", "agent", "solve", "opus", undefined, "claude-code");
  const older = {
    provider: "claude-code",
    status: "allowed",
    limitType: "five_hour",
    observedAt: "2026-07-11T20:00:00.000Z",
    resetsAt: "2026-07-12T01:00:00.000Z",
  };
  const newest = {
    ...older,
    status: "rejected",
    observedAt: "2026-07-11T20:05:00.000Z",
  };
  const stale = {
    ...older,
    status: "allowed_warning",
    observedAt: "2026-07-11T20:01:00.000Z",
  };
  const weekly = {
    provider: "claude-code",
    status: "allowed_warning",
    limitType: "seven_day",
    observedAt: "2026-07-11T20:06:00.000Z",
    utilization: 0.91,
  };

  session.recordProviderRateLimitObservation(older);
  session.recordProviderRateLimitObservation(newest);
  session.recordProviderRateLimitObservation(stale);
  session.recordProviderRateLimitObservation(weekly);

  const metadata = getProblemMetadata("rate-limit-metadata");
  assert.deepEqual(metadata.sessions[0].providerRateLimits, {
    five_hour: newest,
    seven_day: weekly,
  });

  const events = (await readFile(session.getPaths().jsonl, "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line))
    .filter((event) => event.data?.providerEventType === "provider_rate_limit");
  assert.equal(events.length, 4);
  assert.equal(events.every((event) => event.type === "info"), true);
  assert.deepEqual(events.at(-1).data.observation, weekly);
});

test("rate-limit errors expose stable structural fields", () => {
  const observation = {
    provider: "claude-code",
    status: "rejected",
    limitType: "five_hour",
    observedAt: "2026-07-11T20:58:03.000Z",
    resetsAt: "2026-07-12T05:20:00.000Z",
  };
  const error = new ClaudeCodeRateLimitError(observation);

  assert.equal(error.name, "ClaudeCodeRateLimitError");
  assert.equal(error.code, "CLAUDE_CODE_RATE_LIMITED");
  assert.equal(error.provider, "claude-code");
  assert.equal(error.status, "rejected");
  assert.equal(error.limitType, "five_hour");
  assert.equal(error.resetsAt, observation.resetsAt);
  assert.equal(error.observation, observation);
});

test("metadata projection is independent of raw debug logging", () => {
  const session = createSession("debug-disabled", "agent", "solve", "opus", undefined, "claude-code");
  session.recordProviderRateLimitObservation({
    provider: "claude-code",
    status: "allowed",
    limitType: "five_hour",
    observedAt: "2026-07-11T21:00:00.000Z",
  });

  assert.equal(getProblemMetadata("debug-disabled").sessions[0].providerRateLimits.five_hour.status, "allowed");
});
