import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const { loadProblems } = await import("../dist/problems.js");

const dir = await mkdtemp(path.join(os.tmpdir(), "env-base-contract-v2-"));

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

test("problemStateRef is rejected with a migration error", async () => {
  await assert.rejects(
    async () => loadYaml({ problemStateRef: "problem/p1" }),
    /'problemStateRef', which was removed \(contract v2\).*setupProblem\(problemId\)/s
  );
});

test("solution mode field is rejected with a migration error", async () => {
  for (const value of ["none", "hook"]) {
    await assert.rejects(
      async () => loadYaml({ solution: value }),
      /'solution', which was removed \(contract v2\).*solveProblem\(\)/s,
      `expected rejection for solution: ${value}`
    );
  }
});

test("oracleMinScore is rejected with a rename error", async () => {
  await assert.rejects(
    async () => loadYaml({ oracleMinScore: 0.8 }),
    /'oracleMinScore', which was renamed to minReplayScore/
  );
});

test("solutionRef and solutionPatch are mutually exclusive", async () => {
  await assert.rejects(
    async () => loadYaml({ solutionRef: "main", solutionPatch: "solutions/p1.patch" }),
    /both 'solutionRef' and 'solutionPatch'/
  );
});

test("solutionPatch accepts a repo-relative path", async () => {
  const [p] = await loadYaml({ solutionPatch: "solutions/p1.patch" });
  assert.equal(p.solutionPatch, "solutions/p1.patch");
  assert.equal(p.solutionRef, undefined);
});

test("solutionPatch rejects empty and non-string values", async () => {
  for (const value of ['""', '"   "', "true"]) {
    await assert.rejects(
      async () => loadYaml({ solutionPatch: value }),
      /invalid 'solutionPatch'/,
      `expected rejection for solutionPatch: ${value}`
    );
  }
});
