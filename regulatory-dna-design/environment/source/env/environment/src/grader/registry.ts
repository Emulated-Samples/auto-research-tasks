import type { TestResult } from "@hyperfocal/env-base";

/**
 * Per-test category + weight for the seq-design held-out evaluation harness. Weights are relative;
 * aggregateTestResults (env-base) takes the weighted mean, so partial credit is automatic.
 *
 * There is ONE held-out-activity test: test_heldout_Dev on the DeepSTARR Dev track (weight 0.9), the
 * core signal. It is CONTINUOUS — it emits its held-out z and grader/pytest-suite.ts maps that to a
 * skill score via CONTINUOUS_METRICS. The rest of the weight (0.05) is the invalid-input loud-failure
 * block; the novelty diagnostic is informational (weight 0). Hk is not graded (its 2-way signal was a
 * shared maxpool-CNN artifact that collapses on the dilated-residual evaluator — see fixtures).
 *
 * Determinism is NOT here: it is a whole-grade catastrophe gate (grader/anti-cheat.ts::cat-determinism,
 * weight 0), which craters the grade multiplicatively rather than deducting a weight.
 */
export type TestCategory =
  | "activity" // held-out activity per track (the core signal)
  | "validity" // invalid-input loud-failure contract
  | "operational"; // diagnostics / wiring

interface TestMetadata {
  category: TestCategory;
  weight: number;
}

const DEFAULT_METADATA: TestMetadata = { category: "operational", weight: 1 };

// Keyed by BASE pytest test name (no `pytest-` prefix, no `[param]` suffix).
const BASE_METADATA: Record<string, TestMetadata> = {
  // Held-out activity (continuous). DeepSTARR Dev track, 3-way ensemble-min. Hk dropped (its 2-way
  // signal was a shared maxpool-CNN artifact that collapses on the dilated-residual evaluator).
  test_heldout_Dev: { category: "activity", weight: 0.9 },

  // Invalid-input loud-failure block (verifier §4.2): 8 parametrized cells share this base weight,
  // 8 x 0.00625 = 0.05 total. A design that returns garbage on bad args instead of raising loses 0.05.
  test_invalid_args_raise: { category: "validity", weight: 0.00625 },

  // Novelty diagnostic (spec-B B3): informational only until B3 decides whether a floor is needed.
  test_novelty_diagnostic: { category: "operational", weight: 0 },
};

/** Strip the `pytest-` prefix and any `[param]` suffix to get the base test name. */
function baseName(id: string): string {
  const noPrefix = id.startsWith("pytest-") ? id.slice("pytest-".length) : id;
  const bracket = noPrefix.indexOf("[");
  return bracket === -1 ? noPrefix : noPrefix.slice(0, bracket);
}

type AnnotatedResult = TestResult & { category?: TestCategory };

/**
 * Attach the registry weight (and category, for display) to each result. A runtime weight already
 * set on the result takes precedence. Unknown ids fall back to DEFAULT_METADATA.
 */
export function applyGraderScoring(results: TestResult[]): TestResult[] {
  return results.map((r): AnnotatedResult => {
    const meta = BASE_METADATA[baseName(r.id)] ?? DEFAULT_METADATA;
    return { ...r, category: meta.category, weight: r.weight ?? meta.weight };
  });
}
