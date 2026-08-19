import assert from "node:assert/strict";
import test from "node:test";

import {
  AgentExecutionError,
  agentExecutionStatus,
  agentFailureFromSession,
  agentSessionKeys,
  latestNewAgentSession,
  rejectedClaudeRateLimitFromError,
} from "../dist/internal/agent-failure.js";

const rejectedObservation = {
  provider: "claude-code",
  status: "rejected",
  limitType: "five_hour",
  observedAt: "2026-07-12T03:44:00.000Z",
  resetsAt: "2026-07-12T05:20:00.000Z",
};

test("recognizes the structural env-base rate-limit error contract", () => {
  const observation = rejectedClaudeRateLimitFromError({
    name: "ClaudeCodeRateLimitError",
    code: "CLAUDE_CODE_RATE_LIMITED",
    provider: "claude-code",
    status: "rejected",
    observation: rejectedObservation,
  });

  assert.deepEqual(observation, rejectedObservation);
});

test("does not classify allowed or warning observations as rejected", () => {
  for (const status of ["allowed", "allowed_warning"]) {
    assert.equal(rejectedClaudeRateLimitFromError({
      name: "ClaudeCodeRateLimitError",
      code: "CLAUDE_CODE_RATE_LIMITED",
      observation: { ...rejectedObservation, status },
    }), undefined);
  }
});

test("preserves a detailed child error and promotes rejected limits to errored", () => {
  const failure = agentFailureFromSession({
    category: "agent",
    status: "failed",
    error: "Claude Code rejected the five_hour limit; reset at 05:20Z",
    providerRateLimits: { five_hour: rejectedObservation },
  }, "Agent exited with code 1");

  assert.deepEqual(failure, {
    status: "errored",
    message: "Claude Code rejected the five_hour limit; reset at 05:20Z",
    kind: "provider_rate_limit",
    rateLimit: rejectedObservation,
  });
});

test("allowed and warning metadata leave ordinary failures unchanged", () => {
  for (const status of ["allowed", "allowed_warning"]) {
    const failure = agentFailureFromSession({
      category: "agent",
      status: "failed",
      error: "Agent implementation failed",
      providerRateLimits: {
        five_hour: { ...rejectedObservation, status },
      },
    }, "Agent exited with code 1");

    assert.deepEqual(failure, {
      status: "failed",
      message: "Agent implementation failed",
    });
  }
});

test("selects only an agent session created by the current child", () => {
  const oldMetadata = {
    sessions: [{
      category: "agent",
      timestamp: "2026-07-12T03-00-00-000Z",
      status: "failed",
      providerRateLimits: { five_hour: rejectedObservation },
    }],
  };
  const previousKeys = agentSessionKeys(oldMetadata);
  const currentSession = {
    category: "agent",
    timestamp: "2026-07-12T04-00-00-000Z",
    status: "failed",
    error: "Current detailed error",
  };

  assert.equal(latestNewAgentSession(oldMetadata, previousKeys), undefined);
  assert.deepEqual(latestNewAgentSession({
    sessions: [...oldMetadata.sessions, currentSession],
  }, previousKeys), currentSession);
});

test("AgentExecutionError carries the child message and rollout status", () => {
  const error = new AgentExecutionError({
    exitCode: 1,
    failure: {
      status: "errored",
      message: "Rate limit rejected",
      kind: "provider_rate_limit",
      rateLimit: rejectedObservation,
    },
  });

  assert.equal(error.message, "Rate limit rejected");
  assert.equal(error.exitCode, 1);
  assert.equal(error.failureKind, "provider_rate_limit");
  assert.equal(agentExecutionStatus(error), "errored");
});
