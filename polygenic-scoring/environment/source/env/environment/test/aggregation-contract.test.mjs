import assert from "node:assert/strict";
import test from "node:test";

// The EMITTED JavaScript, not the TypeScript source: `node --test` only strips
// types on newer Node, and index.ts imports "./aggregation.js" at runtime, so
// dist/ is the artifact that actually grades. `npm pretest` builds it first.
import {
  AGGREGATION_SCHEMA_VERSION,
  aggregationMatches,
  categoryAggregation,
  datasetRewardsMatchCategories,
  gradedStatus,
  MAX_DATASET_REWARD,
  MEAN_WEIGHT,
  masteryContractError,
  MIN_DATASET_REWARD,
  TAIL_FRACTION,
  TAIL_WEIGHT,
  tailSize,
  toPlatformScore,
} from "../dist/aggregation.js";

function bottomTailMean(scores) {
  const ordered = Object.values(scores).sort((a, b) => a - b);
  const count = Math.ceil(TAIL_FRACTION * ordered.length);
  return ordered.slice(0, count).reduce((sum, v) => sum + v, 0) / count;
}

test("the tail term is the documented bottom-tail average", () => {
  // Must agree with grader/skill.py: the average of the lowest ceil(0.2 * C)
  // categories -- NOT a linear-interpolated Q20 over two adjacent order
  // statistics, which at C=3 skewed the coefficients to 0.40/0.35/0.25.
  for (const categoryCount of [3, 5, 14]) {
    const scores = Object.fromEntries(
      Array.from({ length: categoryCount }, (_, i) => [`c${i}`, 0.1 * i]),
    );
    const aggregation = categoryAggregation(scores);
    const tailCount = Math.ceil(TAIL_FRACTION * categoryCount);

    assert.equal(aggregation.tail_size, tailCount);
    assert.equal(tailSize(categoryCount), tailCount);
    assert.ok(Math.abs(aggregation.tail_value - bottomTailMean(scores)) <= 1e-12);

    const mean = Object.values(scores).reduce((s, v) => s + v, 0) / categoryCount;
    const expected = MEAN_WEIGHT * mean + TAIL_WEIGHT * bottomTailMean(scores);
    assert.ok(Math.abs(aggregation.headline - expected) <= 1e-12);

    // Tail mass is spread evenly across the tail; nothing outside it carries any.
    const base = MEAN_WEIGHT / categoryCount;
    const ordered = Object.keys(scores).sort((a, b) => scores[a] - scores[b]);
    for (const [index, category] of ordered.entries()) {
      const expectedCoefficient = base
        + (index < tailCount ? TAIL_WEIGHT / tailCount : 0);
      assert.ok(
        Math.abs(aggregation.category_coefficients[category] - expectedCoefficient)
          <= 1e-12,
      );
    }
  }
});

test("category coefficients reproduce the mean/lower-tail headline", () => {
  for (const scores of [
    { only: 0.7 },
    { a: 0.2, b: 0.9 },
    { a: 0.8, b: 0.1, c: 0.5 },
    { a: 0.8, b: 0.1, c: 0.5, d: 0.3, e: 1.1 },
    { z: 0.4, a: 0.4, m: 0.4, q: 0.9, b: 0.1 },
    // Below-naive categories are SIGNED: the aggregation must carry them, not
    // clamp them (a floored category is what killed the RL gradient).
    { a: -0.5, b: -0.1, c: 0.4 },
    { a: -0.25, b: 1.2 },
  ]) {
    const aggregation = categoryAggregation(scores);
    const coefficientSum = Object.values(aggregation.category_coefficients)
      .reduce((sum, weight) => sum + weight, 0);
    const weightedHeadline = Object.keys(scores).reduce(
      (sum, category) => sum
        + aggregation.category_coefficients[category] * scores[category],
      0,
    );
    assert.equal(AGGREGATION_SCHEMA_VERSION, 8);
    assert.ok(Math.abs(coefficientSum - 1) <= 1e-12);
    assert.ok(Math.abs(weightedHeadline - aggregation.headline) <= 1e-12);
    assert.equal(aggregationMatches(scores, aggregation), true);
  }
});

test("mastery is never claimed without a calibrated threshold", () => {
  // No calibration study exists, so the grader sends mastery_threshold = null and
  // NOTHING may come back "passed" -- however high the score.
  for (const score of [MIN_DATASET_REWARD, 0, 0.6, 1.0, MAX_DATASET_REWARD]) {
    assert.equal(gradedStatus(score, true, null), "partially_passed");
    assert.notEqual(gradedStatus(score, true, null), "passed");
  }
  // A contract failure is reported as such regardless of the score or threshold.
  assert.equal(gradedStatus(1.4, false, null), "failed");
  assert.equal(gradedStatus(1.4, false, 0.6), "failed");
  // With an audited threshold present, mastery becomes a real, calibrated event.
  assert.equal(gradedStatus(0.62, true, 0.6), "passed");
  assert.equal(gradedStatus(0.59, true, 0.6), "failed");
});

test("uncalibrated mastery keeps an authenticated numeric tail diagnostic", () => {
  const aggregation = categoryAggregation({ hard: 0.2, easy: 0.9 });
  assert.equal(
    masteryContractError(null, null, aggregation.tail_value, aggregation),
    null,
  );
  assert.match(
    masteryContractError(null, 0.2, aggregation.tail_value, aggregation),
    /both mastery thresholds/,
  );
  assert.match(
    masteryContractError(null, null, aggregation.tail_value + 0.01, aggregation),
    /disagrees/,
  );
  assert.equal(
    masteryContractError(0.5, 0.2, aggregation.tail_value, aggregation),
    null,
  );
});

test("the platform transform preserves native ordering and anchor landmarks", () => {
  assert.equal(toPlatformScore(MIN_DATASET_REWARD), 0);
  assert.equal(toPlatformScore(0), 1 / 3);
  assert.equal(toPlatformScore(1), 1);
  assert.equal(toPlatformScore(MAX_DATASET_REWARD), 4 / 3);
  const ordered = [-0.5, -0.2, 0, 0.4, 1, 1.5].map(toPlatformScore);
  assert.deepEqual(ordered, [...ordered].sort((a, b) => a - b));
});

test("the emitted platform results reproduce the transformed trusted headline", () => {
  // The platform's top-level score is a weighted mean over the emitted results.
  // A weight-1 headline record alongside the coefficient-weighted categories must
  // leave that mean EQUAL to the grader's trusted scalar, never skew it.
  const scores = { easy: 0.9, hard: -0.2, medium: 0.6 };
  const aggregation = categoryAggregation(scores);
  const emitted = [
    { score: toPlatformScore(aggregation.headline), weight: 1 },
    ...Object.keys(scores).map((category) => ({
      score: toPlatformScore(scores[category]),
      weight: aggregation.category_coefficients[category],
    })),
  ];
  const weightTotal = emitted.reduce((sum, result) => sum + result.weight, 0);
  const weighted = emitted.reduce(
    (sum, result) => sum + result.weight * result.score,
    0,
  ) / weightTotal;
  assert.ok(Math.abs(weighted - toPlatformScore(aggregation.headline)) <= 1e-12);
});

test("tampered aggregation contracts are rejected", () => {
  const scores = { easy: 0.9, hard: 0.2, medium: 0.6 };
  const aggregation = categoryAggregation(scores);
  assert.equal(aggregationMatches(scores, {
    ...aggregation,
    headline: aggregation.headline + 0.01,
  }), false);
  assert.equal(aggregationMatches(scores, {
    ...aggregation,
    category_coefficients: {
      ...aggregation.category_coefficients,
      easy: aggregation.category_coefficients.easy + 0.01,
    },
  }), false);
});

test("full-precision dataset rewards reproduce categories without display rounding", () => {
  const datasets = [
    { category: "dense", weight: 1, reward: 0.123456789012345 },
    { category: "dense", weight: 2, reward: 0.987654321098765 },
    { category: "sparse", weight: 1, reward: -0.234567890123456 },
    { category: "sparse", weight: 3, reward: 0.345678901234567 },
  ];
  const perCategory = {
    dense: (datasets[0].reward + 2 * datasets[1].reward) / 3,
    sparse: (datasets[2].reward + 3 * datasets[3].reward) / 4,
  };
  assert.equal(datasetRewardsMatchCategories(datasets, perCategory), true);

  // The old four-decimal serialization is too lossy for the authenticated 1e-9
  // reconstruction contract and must not be silently accepted.
  const rounded = datasets.map((dataset) => ({
    ...dataset,
    reward: Math.round(dataset.reward * 10_000) / 10_000,
  }));
  assert.equal(datasetRewardsMatchCategories(rounded, perCategory), false);
});

test("dataset/category reconstruction rejects reward and membership tampering", () => {
  const datasets = [
    { category: "a", weight: 1, reward: 0.123456789012345 },
    { category: "a", weight: 2, reward: 0.765432109876543 },
    { category: "b", weight: 1, reward: -0.111111111111111 },
  ];
  const perCategory = {
    a: (datasets[0].reward + 2 * datasets[1].reward) / 3,
    b: datasets[2].reward,
  };
  const rewardTamper = datasets.map((dataset, index) => index === 1
    ? { ...dataset, reward: dataset.reward + 1e-6 }
    : dataset);
  assert.equal(datasetRewardsMatchCategories(rewardTamper, perCategory), false);
  assert.equal(datasetRewardsMatchCategories(
    datasets.map((dataset, index) => index === 2 ? { ...dataset, category: "a" } : dataset),
    perCategory,
  ), false);
  assert.equal(datasetRewardsMatchCategories(datasets, { ...perCategory, injected: 0 }), false);
});
