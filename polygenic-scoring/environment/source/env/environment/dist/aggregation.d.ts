export declare const AGGREGATION_SCHEMA_VERSION = 8;
export declare const MIN_DATASET_REWARD = -0.5;
export declare const MAX_DATASET_REWARD = 1.5;
/**
 * Platform-facing affine transform of the signed native reward. Hyperfocal clamps
 * negative TestResult scores, so emitting the native scale would collapse every
 * below-naive result onto zero. This mapping preserves the complete ordering and,
 * because it is affine, commutes exactly with the category-weighted aggregation.
 * Native floor=-0.5 maps to 0, naive=0 maps to 1/3, reference=1 maps to 1,
 * and scientific headroom 1.5 maps to 4/3. Values above 1 are intentionally not
 * clamped: native remains authoritative and the transform preserves all ordering.
 */
export declare function toPlatformScore(nativeReward: number): number;
export declare const MEAN_WEIGHT = 0.6;
export declare const TAIL_WEIGHT = 0.4;
export declare const TAIL_FRACTION = 0.2;
/** Categories in the bottom tail: ceil(TAIL_FRACTION * C). Mirrors grader/skill.py. */
export declare function tailSize(categoryCount: number): number;
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
export declare function gradedStatus(score: number, contractOk: boolean, masteryThreshold: number | null): GradedStatus;
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
export declare function datasetRewardsMatchCategories(datasets: WeightedDatasetReward[], perCategory: Record<string, number>, tolerance?: number): boolean;
/** Validate the explicit mastery overlay against the authenticated aggregation. */
export declare function masteryContractError(masteryThreshold: number | null, masteryTailThreshold: number | null, masteryTailValue: number, aggregation: AggregationContract): string | null;
export declare function categoryAggregation(scores: Record<string, number>): AggregationContract;
export declare function aggregationMatches(scores: Record<string, number>, candidate: unknown, tolerance?: number): candidate is AggregationContract;
