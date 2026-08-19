// Mirrors grader/contract.py CORPUS_SCHEMA_VERSION, which the grader stamps into
// reward_detail.json. index.ts REJECTS a reward file whose schema_version differs,
// so letting these drift apart errors out every live grade.
// validation/test_aggregation_contract.py asserts they agree.
export const AGGREGATION_SCHEMA_VERSION = 8;
// Bounds of one dataset's SIGNED reward, mirroring SKILL_LO/SKILL_HI in
// grader/skill.py: a below-naive fit scores genuinely negative (that is the RL
// gradient the old floor-at-zero destroyed), so the wrapper must accept negative
// per-dataset and per-category scores while still rejecting out-of-scale ones.
// validation/test_aggregation_contract.py asserts these against the grader.
export const MIN_DATASET_REWARD = -0.5;
export const MAX_DATASET_REWARD = 1.5;

/**
 * Platform-facing affine transform of the signed native reward. Hyperfocal clamps
 * negative TestResult scores, so emitting the native scale would collapse every
 * below-naive result onto zero. This mapping preserves the complete ordering and,
 * because it is affine, commutes exactly with the category-weighted aggregation.
 * Native floor=-0.5 maps to 0, naive=0 maps to 1/3, reference=1 maps to 1,
 * and scientific headroom 1.5 maps to 4/3. Values above 1 are intentionally not
 * clamped: native remains authoritative and the transform preserves all ordering.
 */
export function toPlatformScore(nativeReward: number): number {
  return (nativeReward - MIN_DATASET_REWARD)
    / (1.0 - MIN_DATASET_REWARD);
}
// MUST equal grader/skill.py AGG_MEAN_W / AGG_TAIL_W: the wrapper recomputes the
// headline from per-category rewards with these and rejects any grade whose headline
// disagrees past 1e-9 (index.ts aggregation_ok). A drift here errors EVERY live grade
// -- it shipped exactly that way once (Python 0.6/0.4 vs TS 0.75/0.25). Now pinned by
// validation/test_aggregation_contract.py::test_wrapper_aggregation_weights_match_grader.
export const MEAN_WEIGHT = 0.6;
export const TAIL_WEIGHT = 0.4;
export const TAIL_FRACTION = 0.20;

/** Categories in the bottom tail: ceil(TAIL_FRACTION * C). Mirrors grader/skill.py. */
export function tailSize(categoryCount: number): number {
  if (categoryCount <= 0) return 0;
  return Math.max(1, Math.min(categoryCount,
    Math.ceil(TAIL_FRACTION * categoryCount)));
}

export type GradedStatus = "passed" | "partially_passed" | "failed";

/**
 * Status for a graded record. `passed` is a MASTERY claim, emitted ONLY against a
 * threshold carried in the benchmark contract. The HEADLINE record passes the real
 * grader/contract.py MASTERY_THRESHOLD (currently the frontier-ceiling anchor), so
 * it reports `passed`/`failed`. A caller may pass `null` only when the benchmark has
 * no calibrated mastery bar, yielding `partially_passed` (graded, no mastery claim).
 * Per-category records call this with `null`, because they expose continuous
 * diagnostics but have no independent mastery threshold. `failed` also covers a
 * run that broke the program contract. The old
 * `score >= 0.6 ? passed : ...` was an uncalibrated display heuristic that pass@k
 * then read as mastery.
 */
export function gradedStatus(
  score: number,
  contractOk: boolean,
  masteryThreshold: number | null,
): GradedStatus {
  if (!contractOk) return "failed";
  if (masteryThreshold === null) return "partially_passed";
  return score >= masteryThreshold ? "passed" : "failed";
}

export interface AggregationContract {
  kind: "mean_bottom_tail_blend";
  mean_weight: number;
  tail_weight: number;
  tail_fraction: number;
  method: "bottom_tail_mean";
  mean: number;
  tail_size: number;
  tail_value: number;
  headline: number;
  category_coefficients: Record<string, number>;
}

/** Minimal authenticated dataset fields needed to reconstruct category rewards. */
export interface WeightedDatasetReward {
  category: string;
  weight: number;
  reward: number;
}

/**
 * Reconstruct every category from the full-precision authenticated dataset
 * rewards. JSON numbers preserve these floats, so rounding is neither necessary
 * nor valid on this integrity path.
 */
export function datasetRewardsMatchCategories(
  datasets: WeightedDatasetReward[],
  perCategory: Record<string, number>,
  tolerance = 1e-9,
): boolean {
  const categories = Object.keys(perCategory).sort();
  const datasetCategories = [...new Set(datasets.map((dataset) => dataset.category))].sort();
  if (JSON.stringify(datasetCategories) !== JSON.stringify(categories)) return false;

  return categories.every((category) => {
    const members = datasets.filter((dataset) => dataset.category === category);
    if (members.length === 0 || members.some(
      (dataset) => !Number.isFinite(dataset.weight)
        || dataset.weight <= 0
        || !Number.isFinite(dataset.reward),
    )) return false;
    const totalWeight = members.reduce((sum, dataset) => sum + dataset.weight, 0);
    const reproduced = members.reduce(
      (sum, dataset) => sum + dataset.weight * dataset.reward,
      0,
    ) / totalWeight;
    return Number.isFinite(perCategory[category])
      && Number.isFinite(reproduced)
      && Math.abs(reproduced - perCategory[category]) <= tolerance;
  });
}

/** Validate the explicit mastery overlay against the authenticated aggregation. */
export function masteryContractError(
  masteryThreshold: number | null,
  masteryTailThreshold: number | null,
  masteryTailValue: number,
  aggregation: AggregationContract,
): string | null {
  if ((masteryThreshold === null) !== (masteryTailThreshold === null)) {
    return "both mastery thresholds must be null together";
  }
  if (!Number.isFinite(masteryTailValue)
      || masteryTailValue < MIN_DATASET_REWARD
      || masteryTailValue > MAX_DATASET_REWARD
      || Math.abs(masteryTailValue - aggregation.tail_value) > 1e-9) {
    return "mastery_tail_value is invalid or disagrees with the bottom-tail aggregate";
  }
  if (masteryThreshold !== null && (
    !Number.isFinite(masteryThreshold)
    || !Number.isFinite(masteryTailThreshold)
    || !(masteryThreshold > 0 && masteryThreshold <= MAX_DATASET_REWARD)
    || !((masteryTailThreshold as number) > 0
      && (masteryTailThreshold as number) <= MAX_DATASET_REWARD)
  )) {
    return "mastery thresholds are invalid";
  }
  return null;
}

export function categoryAggregation(
  scores: Record<string, number>,
): AggregationContract {
  const categories = Object.keys(scores);
  if (categories.length === 0) {
    return {
      kind: "mean_bottom_tail_blend",
      mean_weight: MEAN_WEIGHT,
      tail_weight: TAIL_WEIGHT,
      tail_fraction: TAIL_FRACTION,
      method: "bottom_tail_mean",
      mean: 0,
      tail_size: 0,
      tail_value: 0,
      headline: 0,
      category_coefficients: {},
    };
  }

  const coefficients = Object.fromEntries(
    categories.map((category) => [category, MEAN_WEIGHT / categories.length]),
  ) as Record<string, number>;
  const ordered = [...categories].sort(
    (left, right) => scores[left] - scores[right]
      || (left < right ? -1 : left > right ? 1 : 0),
  );
  // The tail mass is spread EVENLY over every category in the bottom tail, so the
  // headline is the documented mean/bottom-tail blend rather than a rank-weighted
  // mean of the two order statistics adjacent to a quantile position.
  const tailCount = tailSize(categories.length);
  const tail = ordered.slice(0, tailCount);
  for (const category of tail) {
    coefficients[category] += TAIL_WEIGHT / tailCount;
  }

  const mean = categories.reduce((sum, category) => sum + scores[category], 0)
    / categories.length;
  const tailValue = tail.reduce((sum, category) => sum + scores[category], 0)
    / tailCount;
  const headline = categories.reduce(
    (sum, category) => sum + coefficients[category] * scores[category],
    0,
  );
  return {
    kind: "mean_bottom_tail_blend",
    mean_weight: MEAN_WEIGHT,
    tail_weight: TAIL_WEIGHT,
    tail_fraction: TAIL_FRACTION,
    method: "bottom_tail_mean",
    mean,
    tail_size: tailCount,
    tail_value: tailValue,
    headline,
    category_coefficients: coefficients,
  };
}

export function aggregationMatches(
  scores: Record<string, number>,
  candidate: unknown,
  tolerance = 1e-9,
): candidate is AggregationContract {
  if (typeof candidate !== "object" || candidate === null) return false;
  const value = candidate as Partial<AggregationContract>;
  const expected = categoryAggregation(scores);
  if (value.kind !== expected.kind
      || value.method !== expected.method
      || value.mean_weight !== expected.mean_weight
      || value.tail_weight !== expected.tail_weight
      || value.tail_fraction !== expected.tail_fraction
      || value.tail_size !== expected.tail_size
      || !Number.isFinite(value.mean)
      || !Number.isFinite(value.tail_value)
      || !Number.isFinite(value.headline)
      || Math.abs((value.mean as number) - expected.mean) > tolerance
      || Math.abs((value.tail_value as number) - expected.tail_value) > tolerance
      || Math.abs((value.headline as number) - expected.headline) > tolerance
      || typeof value.category_coefficients !== "object"
      || value.category_coefficients === null) {
    return false;
  }
  const expectedCategories = Object.keys(expected.category_coefficients).sort();
  const candidateCategories = Object.keys(value.category_coefficients).sort();
  if (JSON.stringify(candidateCategories) !== JSON.stringify(expectedCategories)) return false;
  return expectedCategories.every((category) => {
    const coefficient = value.category_coefficients?.[category];
    return Number.isFinite(coefficient)
      && (coefficient as number) > 0
      && Math.abs((coefficient as number) - expected.category_coefficients[category]) <= tolerance;
  });
}
