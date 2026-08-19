import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chmodSync, mkdirSync, openSync, writeFileSync, closeSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { setTimeout as sleep } from "node:timers/promises";

// The reaper caps how long a run_in_background task may live. Sized down for
// tests; the synth grace is what fires when the CLI never observes the kill.
process.env.HYPERFOCAL_CLAUDE_BG_TASK_MAX_AGE_MS = "120";
process.env.HYPERFOCAL_CLAUDE_BG_TASK_REAP_SYNTH_GRACE_MS = "250";

const logsDir = await mkdtemp(path.join(os.tmpdir(), "env-base-reaper-"));
process.env.HYPERFOCAL_LOGS_DIR = logsDir;

// resolveClaudeBinary() runs at construction; give it a stub on PATH so the
// test never depends on a real CLI install.
const binDir = await mkdtemp(path.join(os.tmpdir(), "env-base-reaper-bin-"));
const stub = path.join(binDir, "claude");
writeFileSync(stub, "#!/bin/sh\nexit 0\n");
chmodSync(stub, 0o755);
process.env.PATH = `${binDir}:${process.env.PATH}`;

const { ClaudeCodeAgent } = await import("../dist/agents/ClaudeCodeAgent.js");

const workDir = await mkdtemp(path.join(os.tmpdir(), "env-base-reaper-work-"));
const tasksDir = path.join(workDir, "tasks");
mkdirSync(tasksDir, { recursive: true });

test.after(async () => {
  await rm(logsDir, { recursive: true, force: true });
  await rm(binDir, { recursive: true, force: true });
  await rm(workDir, { recursive: true, force: true });
});

function makeAgent() {
  const agent = new ClaudeCodeAgent({ model: "claude-test" });
  agent.resumeOnTaskCompletion = true;
  agent.agentIdle = true;
  agent.resumeMessages = [];
  agent.writeFollowupUserMessage = (message) => {
    agent.resumeMessages.push(message);
    return true;
  };
  return agent;
}

/** Spawn a long sleeper whose stdout fd is the task's output file — the same
 * signature (open fd on …/tasks/<id>.output) the reaper's /proc scan keys on. */
function spawnTaskProcess(taskId) {
  const outFd = openSync(path.join(tasksDir, `${taskId}.output`), "w");
  const child = spawn("sleep", ["300"], { stdio: ["ignore", outFd, outFd] });
  closeSync(outFd);
  return child;
}

function startTask(agent, taskId) {
  agent.handleTaskSystemMessage({
    type: "system",
    subtype: "task_started",
    task_id: taskId,
    description: `test task ${taskId}`,
    task_type: "local_bash",
  });
}

test("reaper kills an over-age task's processes; CLI report resumes with the explanation", async () => {
  const agent = makeAgent();
  const child = spawnTaskProcess("reap1");
  const exited = new Promise((resolve) => child.once("exit", (code, signal) => resolve(signal)));

  startTask(agent, "reap1");
  assert.equal(agent.pendingBackgroundTasks.size, 1);

  const signal = await Promise.race([exited, sleep(2_000, "timeout")]);
  assert.equal(signal, "SIGTERM", "task process must be killed at the cap");
  assert.ok(agent.reapedBackgroundTaskIds.has("reap1"));
  // Task stays pending until the CLI (or the synth fallback) reports terminal.
  assert.equal(agent.pendingBackgroundTasks.size, 1);

  // CLI observes the death and reports terminal — the normal path.
  agent.handleTaskSystemMessage({
    type: "system",
    subtype: "task_updated",
    task_id: "reap1",
    patch: { status: "failed" },
  });
  assert.equal(agent.pendingBackgroundTasks.size, 0);
  assert.equal(agent.resumeMessages.length, 1);
  assert.match(agent.resumeMessages[0], /background-task limit/);
  assert.match(agent.resumeMessages[0], /do not restart it unchanged/);

  // Synth grace elapsing later must NOT double-deliver.
  await sleep(400);
  assert.equal(agent.resumeMessages.length, 1);
});

test("synth fallback reports failure when the CLI never observes the kill", async () => {
  const agent = makeAgent();
  // No real process — the fd scan finds nothing, so only the fallback can end the wait.
  startTask(agent, "reap2");

  await sleep(600); // > cap (120ms) + synth grace (250ms)
  assert.equal(agent.pendingBackgroundTasks.size, 0);
  assert.equal(agent.resumeMessages.length, 1);
  assert.match(agent.resumeMessages[0], /status: failed/);
  assert.match(agent.resumeMessages[0], /background-task limit/);
});

test("a task finishing under the cap is never reaped", async () => {
  const agent = makeAgent();
  const child = spawnTaskProcess("fast1");
  startTask(agent, "fast1");

  agent.handleTaskSystemMessage({
    type: "system",
    subtype: "task_notification",
    task_id: "fast1",
    status: "completed",
    summary: "done",
  });
  assert.equal(agent.pendingBackgroundTasks.size, 0);
  assert.equal(agent.resumeMessages.length, 1);
  assert.doesNotMatch(agent.resumeMessages[0], /background-task limit/);

  await sleep(300); // past the cap
  assert.equal(child.exitCode, null, "completed task's process must not be signaled");
  assert.equal(agent.resumeMessages.length, 1);
  child.kill("SIGKILL");
});
