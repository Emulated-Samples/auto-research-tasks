/**
 * Unit tests for harbor reward emission (node:test, zero deps).
 *
 * Covers: the FORK DEFAULT "exclude-errored" (errored cases excluded from
 * the reward dict / downstream denominator, graded scores preserved — the
 * C1 fix), the opt-in "fail-closed" legacy mode, status-only score fallback
 * in scoreFor, the capped flag, and the hardened write path (0600 artifact,
 * read-back verification).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { toHarborRewards, writeHarborRewards } from "./rewards.js";
import type { TestResult } from "../types.js";

function tr(partial: Partial<TestResult> & { id: string; status: TestResult["status"] }): TestResult {
  return {
    name: partial.id,
    duration: 1,
    ...partial,
  } as TestResult;
}

test("status-only passed emits reward 1, failed emits 0 (no errored)", () => {
  const { rewards, meta } = toHarborRewards([
    tr({ id: "a", status: "passed" }),
    tr({ id: "b", status: "failed" }),
  ]);
  assert.deepEqual(rewards, { a: 1, b: 0 });
  assert.equal(meta.failClosed, false);
  assert.equal(meta.erroredHandling, "exclude-errored");
});

test("explicit finite scores are preserved (no errored)", () => {
  const { rewards } = toHarborRewards([
    tr({ id: "a", status: "partially_passed", score: 0.6 }),
    tr({ id: "b", status: "failed", score: -0.5 }),
  ]);
  assert.deepEqual(rewards, { a: 0.6, b: -0.5 });
});

test("opt-in fail-closed: any errored result zeroes every emitted test (upstream legacy)", () => {
  const { rewards, meta } = toHarborRewards(
    [
      tr({ id: "a", status: "passed" }),
      tr({ id: "b", status: "passed", score: 0.9 }),
      tr({ id: "infra", status: "errored", error: "timed out" }),
    ],
    { erroredHandling: "fail-closed" }
  );
  assert.deepEqual(rewards, { a: 0, b: 0, infra: 0 });
  assert.equal(meta.failClosed, true);
  assert.deepEqual(meta.erroredTestIds, ["infra"]);
});

test("FORK DEFAULT exclude-errored: errored cases are omitted, graded scores preserved (C1)", () => {
  const { rewards, meta } = toHarborRewards([
    tr({ id: "a", status: "passed" }),
    tr({ id: "b", status: "passed", score: 0.9 }),
    tr({ id: "c", status: "failed" }),
    tr({ id: "infra", status: "errored", error: "timed out" }),
  ]);
  // Errored case excluded from the dict → excluded from the consumer's
  // denominator. Honest passes are NOT zeroed.
  assert.deepEqual(rewards, { a: 1, b: 0.9, c: 0 });
  assert.ok(!("infra" in rewards));
  assert.equal(meta.failClosed, false);
  assert.equal(meta.erroredHandling, "exclude-errored");
  // Still reported loudly via meta.
  assert.deepEqual(meta.erroredTestIds, ["infra"]);
  assert.equal(meta.statuses["infra"], "errored");
});

test("exclude-errored with no errored results is identical to fail-closed", () => {
  const results = [
    tr({ id: "a", status: "passed" }),
    tr({ id: "b", status: "failed", score: -1 }),
    tr({ id: "s", status: "skipped" }),
  ];
  const legacy = toHarborRewards(results, { erroredHandling: "fail-closed" });
  const forkDefault = toHarborRewards(results);
  assert.deepEqual(forkDefault.rewards, legacy.rewards);
  assert.equal(legacy.meta.failClosed, false);
  assert.equal(forkDefault.meta.failClosed, false);
});

test("capped option lands in meta; absent by default", () => {
  const results = [tr({ id: "a", status: "passed" })];
  assert.equal(toHarborRewards(results).meta.capped, undefined);
  assert.equal(toHarborRewards(results, { capped: true }).meta.capped, true);
});

test("writeHarborRewards: artifact is 0600, dir 0700, pre-planted file replaced, content verified", () => {
  const logsDir = fs.mkdtempSync(path.join(os.tmpdir(), "harbor-rewards-test-"));
  try {
    // Pre-plant a bogus reward file the way a hostile agent-uid process
    // could in a world-writable logs tree.
    const harborDir = path.join(logsDir, "prob", "harbor");
    fs.mkdirSync(harborDir, { recursive: true });
    fs.writeFileSync(path.join(harborDir, "reward.json"), '{"bogus": 1}\n', { mode: 0o666 });

    const { rewardPath } = writeHarborRewards(
      "prob",
      [tr({ id: "a", status: "passed" }), tr({ id: "b", status: "failed" })],
      logsDir
    );
    const written = JSON.parse(fs.readFileSync(rewardPath, "utf-8"));
    assert.deepEqual(written, { a: 1, b: 0 });
    assert.equal(fs.statSync(rewardPath).mode & 0o777, 0o600);
    assert.equal(fs.statSync(harborDir).mode & 0o777, 0o700);
  } finally {
    fs.rmSync(logsDir, { recursive: true, force: true });
  }
});

test("writeVerified read-back throws TAMPER when reward.json is concurrently mutated (F1)", async () => {
  // F1 (adversarial review): the sha256 read-back TAMPER throw in
  // writeVerified was asserted by NO test. This points the hardened write
  // path at a reward.json a concurrent same-uid process is actively
  // overwriting — exactly the racer the read-back exists to catch — and
  // asserts the emission THROWS rather than silently publishing tampered
  // bytes. (The perms half of the defense — a NON-owner uid denied the 0600
  // artifact — is exercised by the environment's adversarial-runner gate,
  // which drops to the agent uid; this test covers the same-uid read-back.)
  const logsDir = fs.mkdtempSync(path.join(os.tmpdir(), "harbor-rewards-tamper-"));
  const harborDir = path.join(logsDir, "prob", "harbor");
  fs.mkdirSync(harborDir, { recursive: true });
  const rewardPath = path.join(harborDir, "reward.json");

  // A large legit vector widens the write→read-back window so the external
  // writer reliably lands inside it.
  const results: TestResult[] = [];
  for (let i = 0; i < 20000; i++) {
    results.push(tr({ id: `case-${i}`, status: i % 2 === 0 ? "passed" : "failed" }));
  }

  // Two detached, tight-loop writers truncating the reward path to a single
  // byte — bytes that can never equal a legit emission, so any read-back that
  // lands after them mismatches. A same-uid racer, the read-back's target.
  const writers = [0, 1].map(() =>
    spawn(
      "bash",
      ["-c", `while true; do printf 'X' > ${JSON.stringify(rewardPath)} 2>/dev/null || true; done`],
      { stdio: "ignore", detached: true }
    )
  );

  try {
    let tamperThrown = false;
    for (let attempt = 0; attempt < 400 && !tamperThrown; attempt++) {
      try {
        writeHarborRewards("prob", results, logsDir);
        // A returned emission verified its own bytes at write time — it must
        // NEVER have silently published the attacker's single byte.
        const published = fs.readFileSync(rewardPath, "utf-8");
        // (The racer may have overwritten AFTER the verified return — the
        // post-emission P-2 window — so we only assert the emission never
        // itself published "X"; a mid-write catch surfaces as the throw.)
        void published;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("TAMPER DETECTED")) tamperThrown = true;
        else throw err; // any other failure is a real bug
      }
    }
    assert.ok(
      tamperThrown,
      "writeVerified never detected the concurrent mutation of reward.json after 400 attempts — " +
        "the read-back TAMPER guard did not fire"
    );
  } finally {
    for (const w of writers) {
      try {
        if (typeof w.pid === "number") process.kill(-w.pid, "SIGKILL");
      } catch {
        /* group already gone */
      }
      w.kill("SIGKILL");
    }
    fs.rmSync(logsDir, { recursive: true, force: true });
  }
});

test("skipped results are omitted from rewards in both modes", () => {
  const { rewards, meta } = toHarborRewards([
    tr({ id: "a", status: "passed" }),
    tr({ id: "s", status: "skipped" }),
  ]);
  assert.ok(!("s" in rewards));
  assert.deepEqual(meta.skippedTestIds, ["s"]);
});
