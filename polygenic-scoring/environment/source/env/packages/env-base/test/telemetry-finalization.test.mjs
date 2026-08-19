import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const logsDir = await mkdtemp(path.join(os.tmpdir(), "env-base-finalization-"));
process.env.HYPERFOCAL_LOGS_DIR = logsDir;

const {
  createSession,
  finalizeRollout,
  getProblemMetadata,
  updateProblemPhase,
} = await import("../dist/telemetry/index.js");

test.after(async () => {
  await rm(logsDir, { recursive: true, force: true });
});

function initializeProblem(problemId, phase) {
  createSession(problemId, "setup", "rollout");
  updateProblemPhase(problemId, phase);
}

test("setup errors retain the setup phase", () => {
  const problemId = "setup-error";
  initializeProblem(problemId, "setup");

  finalizeRollout(problemId, "errored", "dependency installation failed");

  const metadata = getProblemMetadata(problemId);
  assert.equal(metadata.rolloutStatus, "errored");
  assert.equal(metadata.currentPhase, "setup");
});

test("agent failures retain the agent phase", () => {
  const problemId = "agent-failure";
  initializeProblem(problemId, "agent");

  finalizeRollout(problemId, "failed", "agent exited before tests");

  const metadata = getProblemMetadata(problemId);
  assert.equal(metadata.rolloutStatus, "failed");
  assert.equal(metadata.currentPhase, "agent");
});

for (const status of ["failed", "errored"]) {
  test(`test ${status} finalization marks the phase complete`, () => {
    const problemId = `tests-${status}`;
    initializeProblem(problemId, "tests");

    finalizeRollout(problemId, status, `tests ${status}`);

    const metadata = getProblemMetadata(problemId);
    assert.equal(metadata.rolloutStatus, status);
    assert.equal(metadata.currentPhase, "complete");
  });
}

test("successful finalization marks the phase complete", () => {
  const problemId = "successful-rollout";
  initializeProblem(problemId, "agent");

  finalizeRollout(problemId, "completed");

  const metadata = getProblemMetadata(problemId);
  assert.equal(metadata.rolloutStatus, "completed");
  assert.equal(metadata.currentPhase, "complete");
});
