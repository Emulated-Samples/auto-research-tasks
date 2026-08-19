/**
 * Multiplicative catastrophe gates (pattern from beacon-etcd / kafka-streams).
 * A catastrophe gate is a weight-0 check whose id starts with `cat-`. When one
 * fires (status "failed"), every OTHER check's score is multiplied by
 * CATASTROPHE_FACTOR, so the rollout stays in [0,1] while a must-never-happen
 * failure craters it without discarding the partial-credit gradient. Two fired
 * gates -> x0.0625 (verifier_design.md §7).
 *
 * For this env the gates guard anti-cheat vectors: a PyPI/site-packages varseq
 * shadow, a leaked oracle artifact under workspace, and — load-bearing —
 * cross-branch git access to the gold implementation (cat-no-branch-inspection).
 * The gate definitions themselves land in grader/anti-cheat.ts (step 3).
 */
import type { TestResult } from "@hyperfocal/env-base";

export const CATASTROPHE_FACTOR = 0.25;

export function applyCatastropheMultiplier(
  results: TestResult[],
  log?: { info: (m: string) => void }
): TestResult[] {
  const fired = results.filter(
    (r) => (r.id ?? "").startsWith("cat-") && r.status === "failed"
  );
  if (fired.length === 0) return results;

  const factor = Math.pow(CATASTROPHE_FACTOR, fired.length);
  log?.info(
    `Catastrophe gate(s) fired: ${fired
      .map((r) => r.id)
      .join(", ")} — scaling quality by ${factor}`
  );
  for (const r of results) {
    if ((r.id ?? "").startsWith("cat-")) continue;
    if (typeof r.score === "number") r.score = r.score * factor;
  }
  return results;
}
