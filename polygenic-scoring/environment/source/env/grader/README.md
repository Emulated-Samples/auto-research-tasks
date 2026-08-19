# The svpgsbench grader

## Outcome contract

The agent implements a from-scratch `fit`/`predict` polygenic-scoring program. The
grader judges held-out predictive outcomes, probability sanity, and program
contract compliance. It does not inspect whether the submission copied a preferred
algorithmic mechanism.

Schema v8 contains 15 capability categories with three authenticated replicates
each (45 datasets). The categories vary sparsity, effect tails, LD, annotations,
rare variants, prevalence, nonlinear and interaction structure, decoys,
suppressors, and cohort shift. Each category contributes one score regardless of
its row count or replicate count.

## Per-dataset scientific reward

The only positive-credit metric is AUC. It is normalized against two anchors:

- `naive = 0`: covariates-only logistic regression;
- `reference = 1`: the exact public NumPy training-selected best of
  `{hierarchical_eb, ridge_logistic}`, refit on all training rows and applied by
  exact locked `gold/predict` sandbox execution (not trusted in-process math).

For the oriented AUC metric,

```text
skill = (AUC_submission - AUC_naive) / (AUC_reference - AUC_naive)
```

The value is winsorized to `[-0.5, 1.5]`. A method below naive is genuinely
negative; beating the reference can score above one. Both anchor uncertainty maps
are mandatory. AUC is active only when the reference-minus-naive gap is at least
two paired bootstrap SEs and at least four naive-estimate SEs. This separately
tests whether the gap is real and whether it is large enough to be a stable skill
denominator. A dataset without an adequate scored metric cannot ship.

Brier and log loss form a continuous reference-regret multiplier, not additive
reward arms. For each proper score, worse-than-reference absolute regret is mapped
to `exp(-regret / scale)` using predeclared scales (`0.02` Brier, `0.05` log loss).
Their geometric mean multiplies positive AUC skill:

```text
factor = geometric_mean(exp(-proper_score_regret / scale))
reward = skill * factor  if skill > 0
reward = skill           if skill <= 0
```

Exact reference predictions retain score 1; a ranker compressed toward prevalence
cannot retain full AUC credit; severe overconfidence tends smoothly toward zero.
A valid below-naive ranker stays negative even when it is also miscalibrated. The
reference-relative definition cannot qualify its own anchor, so corpus construction
and the development/final shipping reports independently reject a reference whose
Brier or log-loss is worse than naive by more than the same predeclared absolute
bound. This makes reference=1 conditional on a non-vacuous calibration anchor.

Invalid execution—build failure, nonzero exit, fit or predict timeout, missing or
malformed `pred.csv`, sample-id mismatch, wrong row count, or non-finite
predictions—receives `INVALID_REWARD = -0.5`. Invalidity equals the worst valid
floor, preventing selective abstention from improving expected reward.

`accuracy` in dataset detail is the display-only nonnegative twin of `raw_skill`.
It is never used for reward, aggregation, calibration, or rollout statistics.

## Runtime contract

Runtime affects validity and diagnostics, not scientific reward:

- `build.sh`: 3,600-second hard cap;
- each dataset fit: independent 170-second hard cap;
- each dataset predict: independent 30-second hard cap.

Unused time is not transferred between phases or datasets. With 45 datasets, the
worst-case solver budget is `3600 + 45 * (170 + 30) = 12,600` seconds. The trusted
wrapper reserves another 1,800 seconds and times out at 14,400 seconds; the
platform verifier reserves another 1,800 and times out at 16,200 seconds.

`perf` reports budget headroom in `[0.5, 1]`. `efficiency` reports the submission's
speed relative to a practical runtime anchor after host calibration. Both are
diagnostic only: there is no runtime multiplier and no speed bonus. The hard caps
are the complete speed incentive.

## Category and headline aggregation

Replicates are averaged within each category using their authenticated equal
weights. The headline is

```text
0.6 * mean(all 15 category scores)
+ 0.4 * mean(the lowest ceil(0.20 * 15) category scores)
```

The bottom-tail term therefore averages the weakest three categories. It is a
bottom-k mean, not an interpolated quantile. This preserves category equality
while making collapsed capabilities load-bearing. `category_aggregation` exports
exact linear category coefficients; the wrapper recomputes the same result and
rejects any drift.

## Reference anchors

The shipped reference is not a test-set oracle and not the full development
candidate selector. For each dataset it:

1. deterministically splits training rows;
2. fits hierarchical EB and a converged ridge grid on the inner-training rows;
3. compares inner-validation AUC;
4. selects ridge only when it beats hierarchical EB by the declared `0.01` margin;
5. refits the selected method on all training rows.

The candidate family excludes marginal P+T and other diagnostic/cheat arms. Anchor
diagnostics record both candidates' bounded selection AUCs, the declared winner,
method-specific convergence evidence, total selection-plus-refit time, winner-only
refit time, finite training logits, and observed variant classes. Validation
recomputes the winner rule and rejects incomplete or inconsistent provenance.

All model fits occur before prediction features are opened. Hidden test labels are
loaded only after every prediction is complete. The exact public bytes consumed by
the reference are the bytes available to the submission.

## Native score versus platform score

Scientific detail and analysis use the signed native scale `[-0.5, 1.5]`. The
platform-facing `TestResult.score` is

```text
(native_score + 0.5) / 1.5
```

so negative scientific reward survives a platform path that clamps values below
zero. Native naive/reference values `0/1` appear as platform `1/3` and `1`. Benchmark
output records `native_score` explicitly. The affine transform commutes with the
weighted category aggregation and does not change ordering or mastery.

## Mastery

Mastery is an environment-level event, never a per-category threshold. Both the
native headline and weakest-fifth tail must clear their frozen thresholds. Schema
v8 starts uncalibrated: both constants are `None`, category records remain
continuous diagnostics, the wrapper emits no mastery claim, and pass@k refuses to
compute. After an exact-release frontier calibration of at least three retained
rollouts, both thresholds are frozen together. Reported pass@k must use a disjoint
evaluation set.

Agent-caused non-completions count at `SKILL_LO` in best-score@k and as failures in
pass@k. Infra-censored attempts are excluded from capability denominators.

## Files

- `contract.py` — schema, category grid, metric weights, timeouts, and mastery
  constants.
- `metrics.py` — AUC, Brier, and log-loss computation.
- `skill.py` — uncertainty-gated AUC skill, calibration factor, reward reasons, and
  tail-aware category aggregation.
- `perf.py` — report-only runtime headroom and host-normalized efficiency.
- `submission_runner.py` — isolated build/fit/predict execution, phase caps, and
  output validation.
- `grade.py` — authenticated corpus validation, dataset scoring, integrity detail,
  and atomic final reward artifacts.
- `preflight.py` — deployed release-lock and package integrity checks.

The grader accepts only a complete authenticated corpus and writes trusted final
reward detail only after all 45 declared datasets have been processed. Partial
snapshots are forensic artifacts and cannot become accepted scores.
