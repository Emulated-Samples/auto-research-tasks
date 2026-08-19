import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const { loadProblems } = await import("../dist/problems.js");

const dir = await mkdtemp(path.join(os.tmpdir(), "env-base-min-replay-score-"));

test.after(async () => {
  await rm(dir, { recursive: true, force: true });
});

let counter = 0;
async function loadYaml(problemFields) {
  const file = path.join(dir, `problems-${counter++}.yaml`);
  const lines = ["- id: p1", "  prompt: do the thing"];
  for (const [key, value] of Object.entries(problemFields)) {
    lines.push(`  ${key}: ${value}`);
  }
  await writeFile(file, lines.join("\n") + "\n");
  return loadProblems(file);
}

test("minReplayScore is optional and absent by default", async () => {
  const [p] = await loadYaml({});
  assert.equal(p.minReplayScore, undefined);
});

test("minReplayScore accepts values in (0, 1]", async () => {
  for (const value of [0.1, 0.8, 1]) {
    const [p] = await loadYaml({ minReplayScore: value });
    assert.equal(p.minReplayScore, value);
  }
});

test("minReplayScore rejects out-of-range and non-numeric values", async () => {
  for (const value of [0, -0.5, 1.5, '"high"', "true"]) {
    await assert.rejects(
      async () => loadYaml({ minReplayScore: value }),
      /invalid 'minReplayScore'/,
      `expected rejection for minReplayScore: ${value}`
    );
  }
});

test("minReplayScore rejects NaN and infinities", async () => {
  for (const value of [".nan", ".inf"]) {
    await assert.rejects(
      async () => loadYaml({ minReplayScore: value }),
      /invalid 'minReplayScore'/,
      `expected rejection for minReplayScore: ${value}`
    );
  }
});
