import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const fixture = new URL("./fixtures/rollout-telemetry-scenario.js", import.meta.url);
const problemId = "phase-telemetry";

function runScenario(scenario, agentScript) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "env-orchestrator-phase-"));
  const logsRoot = path.join(root, "logs");
  const fakeBin = path.join(root, "bin");
  const fakeNode = path.join(fakeBin, "node");
  fs.mkdirSync(fakeBin, { recursive: true });
  fs.writeFileSync(fakeNode, `#!/bin/sh\n${agentScript}\n`, { mode: 0o755 });

  const result = spawnSync(process.execPath, [fixture.pathname, scenario, root], {
    encoding: "utf8",
    env: {
      ...process.env,
      ANTHROPIC_API_KEY: "test-key",
      HYPERFOCAL_AGENT_TIMEOUT_MS: scenario === "agent-timeout" ? "50" : "10000",
      HYPERFOCAL_LOGS_DIR: logsRoot,
      PATH: `${fakeBin}:${process.env.PATH}`,
    },
  });

  const problemLogs = path.join(logsRoot, problemId);
  const metadata = JSON.parse(
    fs.readFileSync(path.join(problemLogs, "metadata.json"), "utf8")
  );

  return { result, root, problemLogs, metadata };
}

function sessionCategories(metadata) {
  return metadata.sessions.map((session) => session.category);
}

test("setup failure records setup phase only and creates no tests artifact", (t) => {
  const { result, root, problemLogs, metadata } = runScenario("setup-failure", "exit 0");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  assert.equal(result.status, 1, result.stderr || result.stdout);
  assert.equal(metadata.currentPhase, "setup");
  assert.equal(metadata.rolloutStatus, "errored");
  assert.equal(metadata.rolloutError, "No module named flashinfer.jit.aot_config");
  assert.deepEqual(sessionCategories(metadata), ["setup"]);
  assert.equal(fs.existsSync(path.join(problemLogs, "tests")), false);
  assert.equal(metadata.sessions[0].status, "errored");
});

test("agent failure preserves agent phase and creates no tests session", (t) => {
  const { result, root, problemLogs, metadata } = runScenario("agent-failure", "exit 17");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  assert.equal(result.status, 1, result.stderr || result.stdout);
  assert.equal(metadata.currentPhase, "agent");
  assert.equal(metadata.rolloutStatus, "failed");
  assert.deepEqual(sessionCategories(metadata), ["setup"]);
  assert.equal(fs.existsSync(path.join(problemLogs, "tests")), false);
});

test("test failure creates and finalizes tests telemetry", (t) => {
  const { result, root, problemLogs, metadata } = runScenario("test-failure", "exit 0");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  assert.equal(result.status, 1, result.stderr || result.stdout);
  assert.equal(metadata.currentPhase, "complete");
  assert.deepEqual(sessionCategories(metadata), ["setup", "tests"]);
  const testsSession = metadata.sessions.find((session) => session.category === "tests");
  assert.equal(testsSession.status, "failed");

  const testsDir = path.join(problemLogs, "tests");
  assert.equal(fs.existsSync(testsDir), true);
  assert.equal(
    fs.readdirSync(testsDir).some((name) => name.endsWith(".jsonl")),
    true
  );
});

test("tests-phase exception preserves existing failure classification", (t) => {
  const { result, root, problemLogs, metadata } = runScenario("tests-exception", "exit 0");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  assert.equal(result.status, 1, result.stderr || result.stdout);
  assert.equal(metadata.currentPhase, "complete");
  assert.equal(metadata.rolloutStatus, "failed");
  assert.deepEqual(sessionCategories(metadata), ["setup", "tests"]);
  assert.equal(
    metadata.sessions.find((session) => session.category === "tests").status,
    "failed"
  );
  assert.equal(fs.existsSync(path.join(problemLogs, "tests")), true);
});

test("agent timeout creates tests telemetry only when grading continues", (t) => {
  const { result, root, problemLogs, metadata } = runScenario("agent-timeout", "sleep 5");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  assert.equal(result.status, 1, result.stderr || result.stdout);
  assert.equal(metadata.currentPhase, "complete");
  assert.deepEqual(sessionCategories(metadata), ["setup", "tests"]);

  const testsDir = path.join(problemLogs, "tests");
  const jsonlName = fs.readdirSync(testsDir).find((name) => name.endsWith(".jsonl"));
  assert.ok(jsonlName);
  const events = fs.readFileSync(path.join(testsDir, jsonlName), "utf8");
  assert.match(events, /"testsContinuedAfterAgentTimeout":true/);
});
