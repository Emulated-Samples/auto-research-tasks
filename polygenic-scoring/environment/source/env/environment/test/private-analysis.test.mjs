import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const adapter = fs.readFileSync(path.join(root, "environment/src/index.ts"), "utf8");

test("private state dirs are owner-only and never materialized for the solver", () => {
  // Source-trim (owner-ruled): gold/ and rollout_analysis/ were removed from the
  // tree (answer-bearing oracle + retained analysis, no live grade-path reader).
  // reference/ MUST STAY sealed: grader/grade.py does a live `from
  // reference.protocol import ...` at grade time (grade.py:66, used at
  // grade.py:425-435 for provenance validation), so it ships and stays
  // owner-only (sealed from the solver, readable by the root grader). Assert
  // reference remains in the list and the two trimmed dirs are gone.
  assert.match(
    adapter,
    /const PRIVATE_STATE_DIRS = \[[\s\S]*?"corpus",[\s\S]*?"datagen",[\s\S]*?"grader",[\s\S]*?"reference",[\s\S]*?"tasks",[\s\S]*?"validation",[\s\S]*?\];/,
  );
  const listMatch = adapter.match(/const PRIVATE_STATE_DIRS = \[([\s\S]*?)\];/);
  assert.ok(listMatch, "PRIVATE_STATE_DIRS list not found");
  assert.match(listMatch[1], /"reference"/);
  assert.doesNotMatch(listMatch[1], /"gold"/);
  assert.doesNotMatch(listMatch[1], /"rollout_analysis"/);
  assert.match(adapter, /for \(const relative of PRIVATE_STATE_DIRS\)/);
  assert.match(adapter, /const target = path\.join\(REPO_ROOT, relative\);/);
  assert.match(adapter, /fs\.chmodSync\(target, 0o700\);/);
  assert.match(adapter, /"--reuid", "svpgsub"/);
  assert.match(adapter, /"\/usr\/bin\/test", "!", "-r", target/);

  const setup = adapter.split("async setupProblem", 2)[1].split("async runTests", 1)[0];
  const grading = adapter.split("async runTests", 2)[1];
  for (const source of [setup, grading]) {
    assert.match(source, /sealPrivateState\(\);/);
    assert.match(source, /await assertSubmissionCannotReadPrivateState\(\);/);
  }

  assert.doesNotMatch(setup, /copyFileSync.*rollout_analysis/);
  assert.doesNotMatch(setup, /cpSync.*rollout_analysis/);
});
