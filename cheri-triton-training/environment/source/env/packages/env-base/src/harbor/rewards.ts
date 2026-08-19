/**
 * Harbor reward emission.
 *
 * Converts a test run's TestResult[] into Harbor's verifier reward contract:
 * /logs/verifier/reward.json is the FLAT per-test dict {"<testId>": <score>}
 * (harbor json.loads it straight into VerifierResult.rewards — an envelope
 * with nested objects fails validation; verified against harbor 0.17.1,
 * see harbor-packaging/04-harbor-spec.md).
 *
 * Everything harbor-facing is written under a `harbor/` subdirectory of the
 * problem's logs dir so it is clearly labeled and separable from hyperfocal's
 * own telemetry.
 *
 * Mapping semantics (frozen here, shared by every env — see decision D2 in
 * harbor-packaging/05-implementation-plan.md):
 *   - skipped        → omitted from the dict entirely
 *   - passed         → result.score if finite, else 1
 *   - partially_passed → result.score if finite, else 0
 *   - failed         → result.score if finite, else 0
 *   - any errored    → FAIL-CLOSED: every emitted test scores 0 (an errored
 *                      verifier must never look like a graded outcome)
 *   - weight         → recorded in rewards-meta.json, NOT pre-applied
 *                      (aggregation — min/mean/weighted — is the consumer's
 *                      choice downstream)
 */

import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import { TestResult } from "../types.js";
import { getLogsDir } from "../telemetry/index.js";

export type HarborRewards = Record<string, number>;

export interface HarborRewardsOptions {
  /**
   * How grade-time errored results (status "errored" — infrastructure
   * failures, not agent failures) affect the emitted rewards.
   *
   * - "exclude-errored" (DEFAULT on this fork branch — C1 rationale: an
   *   errored verifier is an infrastructure fact, and turning one infra
   *   hiccup into a zero vector erases every honest pass in the run; this
   *   fork exists to kill that class — "errored excludes, never zeroes").
   *   Errored results are omitted from reward.json entirely, so downstream
   *   aggregation excludes them from the denominator; graded tests keep
   *   their real scores; erroredTestIds reports them loudly.
   * - "fail-closed" (opt-in; the upstream default): if ANY result errored,
   *   every emitted test scores 0, and errored ids appear in the dict as 0
   *   so the reward key set stays stable across runs.
   */
  erroredHandling?: "fail-closed" | "exclude-errored";
  /** Marks the run as graded at a time cap; recorded in rewards-meta.json. */
  capped?: boolean;
}

export interface HarborRewardsMeta {
  /** True iff errored results caused every emitted reward to be zeroed. */
  failClosed: boolean;
  /** Which errored-handling mode produced these rewards. */
  erroredHandling: "fail-closed" | "exclude-errored";
  erroredTestIds: string[];
  skippedTestIds: string[];
  /** Present and true when the run was graded at a time cap. */
  capped?: boolean;
  /** Importance multipliers, for consumers that aggregate with weights. */
  weights: Record<string, number>;
  statuses: Record<string, TestResult["status"]>;
  generatedAt: string;
  generator: "env-base/harbor/rewards";
}

export interface HarborRewardsResult {
  rewards: HarborRewards;
  meta: HarborRewardsMeta;
}

function scoreFor(result: TestResult): number {
  if (typeof result.score === "number" && Number.isFinite(result.score)) {
    return result.score;
  }
  return result.status === "passed" ? 1 : 0;
}

export function toHarborRewards(
  results: TestResult[],
  options: HarborRewardsOptions = {}
): HarborRewardsResult {
  // Fork default: exclude-errored (C1 — see HarborRewardsOptions). The
  // upstream default is fail-closed; callers that need it must opt in.
  const erroredHandling = options.erroredHandling ?? "exclude-errored";
  const errored = results.filter((r) => r.status === "errored");
  const skipped = results.filter((r) => r.status === "skipped");
  const graded = results.filter(
    (r) => r.status !== "skipped" && r.status !== "errored"
  );

  const failClosed =
    erroredHandling === "fail-closed" && errored.length > 0;
  const rewards: HarborRewards = {};
  for (const r of graded) {
    rewards[r.id] = failClosed ? 0 : scoreFor(r);
  }
  if (erroredHandling === "fail-closed") {
    // Errored tests appear in the dict (as 0) so the reward shape is stable
    // across runs. Under "exclude-errored" they are omitted instead:
    // excluded from the consumer's denominator, reported via erroredTestIds.
    for (const r of errored) {
      rewards[r.id] = 0;
    }
  }

  const meta: HarborRewardsMeta = {
    failClosed,
    erroredHandling,
    ...(options.capped ? { capped: true } : {}),
    erroredTestIds: errored.map((r) => r.id),
    skippedTestIds: skipped.map((r) => r.id),
    weights: Object.fromEntries(
      results
        .filter((r) => r.status !== "skipped")
        .map((r) => [r.id, r.weight ?? 1])
    ),
    statuses: Object.fromEntries(results.map((r) => [r.id, r.status])),
    generatedAt: new Date().toISOString(),
    generator: "env-base/harbor/rewards",
  };

  return { rewards, meta };
}


/**
 * Hardened artifact write (fork): 0600 mode, then sha256 read-back
 * verification. Throws on any divergence; callers treat emission failure
 * as loud.
 */
function writeVerified(filePath: string, content: string): string {
  fs.rmSync(filePath, { force: true });
  fs.writeFileSync(filePath, content, { mode: 0o600 });
  const expectedHash = crypto.createHash("sha256").update(content).digest("hex");
  const readBack = crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
  if (readBack !== expectedHash) {
    throw new Error(
      `[harbor/rewards] TAMPER DETECTED: ${filePath} read back with hash ${readBack.slice(0, 12)}… ` +
        `immediately after writing ${expectedHash.slice(0, 12)}… — another process is writing this ` +
        `artifact. Refusing to publish.`
    );
  }
  return expectedHash;
}

/**
 * Write reward.json (+ rewards-meta.json) under
 * `<logsDir>/<problemId>/harbor/`. Returns the reward.json path.
 *
 * Never throws on conversion — but filesystem errors propagate so callers
 * can decide whether emission failure is fatal (the test command treats it
 * as non-fatal and logs).
 */
export function writeHarborRewards(
  problemId: string,
  results: TestResult[],
  logsDir: string = getLogsDir(),
  options: HarborRewardsOptions = {}
): { rewardPath: string; metaPath: string; result: HarborRewardsResult } {
  const harborDir = path.join(logsDir, problemId, "harbor");
  fs.mkdirSync(harborDir, { recursive: true, mode: 0o700 });
  try {
    fs.chmodSync(harborDir, 0o700);
  } catch {
    console.error(
      `[harbor/rewards] warning: could not tighten ${harborDir} to 0700 — relying on per-file defenses`
    );
  }
  try {
    fs.chmodSync(harborDir, 0o700);
  } catch {
    console.error(
      `[harbor/rewards] warning: could not tighten ${harborDir} to 0700 — relying on per-file defenses`
    );
  }

  const converted = toHarborRewards(results, options);
  // Grade-time infra failures must be loud, whichever mode is active —
  // silently degraded rewards are how false negatives hide.
  if (converted.meta.erroredTestIds.length > 0) {
    const ids = converted.meta.erroredTestIds.join(", ");
    console.error(
      `[harbor/rewards] ${converted.meta.erroredTestIds.length} test(s) ` +
        `errored at grade time for problem "${problemId}": ${ids}. ` +
        (converted.meta.failClosed
          ? "Mode fail-closed: ALL emitted rewards zeroed."
          : "Mode exclude-errored: errored tests omitted from rewards " +
            "(excluded from the denominator); graded scores preserved.")
    );
  }
  const rewardPath = path.join(harborDir, "reward.json");
  const metaPath = path.join(harborDir, "rewards-meta.json");

  writeVerified(metaPath, JSON.stringify(converted.meta, null, 2) + "\n");
  // The reward artifact is written LAST (after meta), so no observable
  // rewards state ever precedes its metadata.
  writeVerified(rewardPath, JSON.stringify(converted.rewards, null, 2) + "\n");

  return { rewardPath, metaPath, result: converted };
}
