/**
 * Unit tests for score aggregation (node:test, zero deps).
 *
 * Regression coverage for the status-only zeroing bug: batch/subprocess
 * runners emit TestResults with `status` but no `score`; `score ?? 0`
 * aggregated every status-only PASSING test as 0, so rollouts with hundreds
 * of passing tests persisted rolloutScore 0
 * ("Test results: 556 passed, 246 failed (score: 0%)").
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { aggregateTestResults } from "./testing.js";
import type { TestResult } from "./types.js";

function tr(partial: Partial<TestResult> & { id: string; status: TestResult["status"] }): TestResult {
  return {
    name: partial.id,
    duration: 1,
    ...partial,
  } as TestResult;
}

test("status-only results: passed counts as 1, failed as 0", () => {
  const results: TestResult[] = [
    tr({ id: "a", status: "passed" }),
    tr({ id: "b", status: "passed" }),
    tr({ id: "c", status: "failed" }),
    tr({ id: "d", status: "failed" }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 0.5);
  assert.equal(agg.weightTotal, 4);
});

test("regression: 556 status-only passes out of 802 aggregate to 556/802, not 0", () => {
  const results: TestResult[] = [];
  for (let i = 0; i < 556; i++) results.push(tr({ id: `pass-${i}`, status: "passed" }));
  for (let i = 0; i < 246; i++) results.push(tr({ id: `fail-${i}`, status: "failed" }));
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 556 / 802);
  assert.notEqual(agg.rolloutScore, 0);
});

test("explicit score always wins over status-derived score", () => {
  const results: TestResult[] = [
    // Scored tests behave exactly as before the fix.
    tr({ id: "scored-pass", status: "passed", score: 1.0 }),
    tr({ id: "scored-partial", status: "partially_passed", score: 0.5 }),
    tr({ id: "scored-zero-pass", status: "passed", score: 0 }), // e.g. clean anti-cheat
    tr({ id: "scored-fail", status: "failed", score: 0 }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, (1.0 + 0.5 + 0 + 0) / 4);
  const zeroPass = agg.contributions.find((c) => c.id === "scored-zero-pass");
  assert.equal(zeroPass?.score, 0); // explicit 0 NOT overridden by passed status
});

test("mixed status-only and scored results aggregate consistently", () => {
  const results: TestResult[] = [
    tr({ id: "status-pass", status: "passed" }), // -> 1
    tr({ id: "scored-partial", status: "partially_passed", score: 0.25 }), // -> 0.25
    tr({ id: "status-fail", status: "failed" }), // -> 0
    tr({ id: "penalty", status: "failed", score: -1, weight: 1 }), // -> -1
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, (1 + 0.25 + 0 - 1) / 4);
});

test("errored results are excluded from numerator and denominator, never zeroed in", () => {
  const results: TestResult[] = [
    tr({ id: "a", status: "passed" }),
    tr({ id: "b", status: "passed" }),
    tr({ id: "infra", status: "errored", error: "timeout connecting to grader" }),
  ];
  const agg = aggregateTestResults(results);
  // 2/2 graded tests passed; the errored case shrinks the denominator.
  assert.equal(agg.rolloutScore, 1.0);
  assert.equal(agg.weightTotal, 2);
  // Still visible in contributions for reporting.
  assert.ok(agg.contributions.some((c) => c.id === "infra"));
});

test("skipped results are excluded like errored", () => {
  const results: TestResult[] = [
    tr({ id: "a", status: "passed", score: 0.8 }),
    tr({ id: "skip", status: "skipped" }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 0.8);
  assert.equal(agg.weightTotal, 1);
});

test("weights are respected for status-only results", () => {
  const results: TestResult[] = [
    tr({ id: "heavy-pass", status: "passed", weight: 3 }),
    tr({ id: "light-fail", status: "failed", weight: 1 }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 3 / 4);
});

test("non-finite explicit score falls back to status-derived score", () => {
  const results: TestResult[] = [
    tr({ id: "nan-pass", status: "passed", score: NaN }),
    tr({ id: "nan-fail", status: "failed", score: NaN }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 0.5);
  assert.ok(Number.isFinite(agg.rolloutScore));
});

test("empty results aggregate to 0 without throwing", () => {
  const agg = aggregateTestResults([]);
  assert.equal(agg.rolloutScore, 0);
  assert.equal(agg.weightTotal, 0);
});

test("guard: throws when a status-only pass is included but the numerator is exactly 0", () => {
  // Construct the impossible-after-fix state via exact signed cancellation:
  // a status-only pass (+1) cancelled by an explicit -1 penalty.
  const results: TestResult[] = [
    tr({ id: "status-pass", status: "passed" }),
    tr({ id: "penalty", status: "failed", score: -1 }),
  ];
  assert.throws(
    () => aggregateTestResults(results),
    /status-only passed test .* numerator is 0|numerator of 0/
  );
});

test("guard: does NOT fire for all-explicit-score suites that legitimately total 0", () => {
  const results: TestResult[] = [
    tr({ id: "clean-anticheat", status: "passed", score: 0, weight: 5 }),
    tr({ id: "fail", status: "failed", score: 0 }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 0);
});

test("guard: does NOT fire when the only status-only pass has weight 0", () => {
  const results: TestResult[] = [
    tr({ id: "uncounted-pass", status: "passed", weight: 0 }),
    tr({ id: "fail", status: "failed" }),
  ];
  const agg = aggregateTestResults(results);
  assert.equal(agg.rolloutScore, 0);
});
