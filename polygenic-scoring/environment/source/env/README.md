# svpgsbench

`svpgsbench` is a long-horizon RL environment in which an agent builds a
polygenic-scoring (genotype → phenotype) library from scratch. The environment
measures held-out predictive outcomes across a diverse synthetic cohort matrix;
it does not require or inspect a particular modeling mechanism.

The authoritative release contract is schema v8: 15 capability categories with
three authenticated replicates each, for 45 datasets. A covariates-only model
defines native skill 0. A principled, training-only best-of-family reference
defines native skill 1.

## Benchmark design

The synthetic data in `datagen/dgp.py` span regimes where different statistical
choices matter: sparse and dense effects, heavy and light tails, structural and
rare variants, low prevalence, weak/strong/suppressor LD, soft/nonlinear/
interacting/decoy annotations, and ancestry or LD shift. Each cohort discloses
only observable family and formula-scoped metadata in `dgp.json`; hidden generator
latents are not exposed to the solver.

This breadth is intentional. SV-PGS is strong in annotation-structured,
heavy-tailed regimes, while ridge is the appropriate inductive bias for dense
near-Gaussian regimes. The benchmark therefore tests whether a solution can infer
and adapt to the cohort rather than apply one hard-coded shortcut everywhere.

## Scientific reward

Each dataset uses two anchors computed from the exact public training bytes:

- **naive = 0**: covariates-only logistic regression;
- **reference = 1**: the exact public NumPy `gold/fit` selector over
  `{hierarchical_eb, ridge_logistic}`, selected by
  deterministic training-inner-validation AUC and then refit on all training
  rows. Ridge must beat hierarchical EB by the declared margin to win. The
  builder retains the exact model.out and obtains the anchor only by executing
  locked `gold/predict` under the public 30-second sandbox cap; it scores the
  emitted pred.csv bytes and binds their hash.

The reference family deliberately excludes cheats and weak diagnostic baselines.
Selection never reads test features or hidden labels.

Positive credit comes from AUC alone:

```text
native_skill = (AUC_submission - AUC_naive)
             / (AUC_reference - AUC_naive)
```

The signed per-dataset value is winsorized to `[-0.5, 1.5]`. Below-naive models
remain negative, the reference is exactly 1, and a model that beats the reference
can score above 1. Invalid execution receives `-0.5`, so selective crashing cannot
outperform a valid poor prediction.

Brier and log loss form a continuous reference-regret multiplier, not additive
reward arms. Positive AUC skill is discounted by their geometric factor; exact
reference calibration keeps factor 1, while prevalence-compressed or severely
overconfident probabilities cannot retain full ranking credit. Negative skill is
never shrunk toward zero. Because this multiplier is reference-relative, corpus
construction and both frozen shipping reports separately require the reference's
Brier/log-loss to be no more than the same predeclared absolute bounds worse than
naive; a miscalibrated reference cannot certify itself as score 1.

Runtime is report-only. Hard build, fit, and predict caps enforce the speed
contract; runtime headroom and host-normalized efficiency are retained as
diagnostics but never multiply scientific reward and never provide a speed bonus.

## Aggregation

The three replicates are averaged within each category. The native headline is

```text
0.6 * mean(all 15 category scores)
+ 0.4 * mean(the weakest ceil(0.20 * 15) category scores)
```

The bottom-tail term therefore averages the weakest three categories. Every
category receives equal mean-component influence, while collapsed capabilities
remain load-bearing. The TypeScript wrapper independently reconstructs dataset,
category, tail, and headline values from trusted grader detail and rejects any
drift.

## Native and platform score scales

Scientific analysis uses the signed native scale `[-0.5, 1.5]`, with naive 0 and
reference 1. Platform-facing test scores use the monotone affine transform

```text
platform_score = (native_score + 0.5) / 1.5
```

so native `-0.5, 0, 1, 1.5` map to platform `0, 1/3, 1, 4/3`. This preserves
the negative learning signal through platform paths that clamp values below zero.
Benchmark output records `native_score` explicitly; native score is authoritative
for scientific analysis and mastery.

## Mastery and pass@k

Continuous reward is not a pass verdict. Schema v8 initially has no mastery
threshold: both the headline and tail thresholds are `None`, the wrapper emits no
mastery claim, and pass@k refuses to compute. Best-score@k remains reportable.

After publishing the exact uncalibrated tuple, calibration retains every attempt
from one complete `claude-opus-4-8` run on that commit, with at least three
non-infrastructure attempts and one gradable attempt. Agent failures remain in the
phase denominator and provider rate limits are censored; only gradable rows can
define the frontier maximum. `validation/mastery_calibration.py freeze` retains the
whole run content-addressed and freezes headline plus weakest-fifth tail from its
best observed gradable row. A threshold-frozen
republish then collects the complete launched-attempt set from one disjoint Opus
run, with at least three non-infrastructure attempts and one gradable attempt;
`validation/mastery_calibration.py finalize` verifies and retains both sets before it
can produce the calibrated artifact. Pass@k is emitted only by the authenticated
artifact report path and consumes every retained evaluation attempt. Rows bind the
exact ordered `run.json` membership and `totalRollouts`, so omitted siblings fail
closed. An exact completed agent failure counts at the skill floor; an exact provider
rate-limit death is censored. Unknown/error/setup/partial terminal states are rejected
rather than guessed. Retained-only packages may classify those fail-closed terminal
states, but can never supply a score or calibration threshold.

## Program contract

The agent supplies a build script plus `fit` and `predict` entry points:

- `fit` reads `covariates_train.csv`, `genotypes_train.tsv`,
  `variant_metadata.tsv`, `formula.txt`, `family.txt`, and `dgp.json`, then writes
  `model.out`;
- `predict` reads `model.out` and the test-side public files, then writes
  `pred.csv` with one finite `mean` probability per test sample in input order.

The disclosed hard caps are 3,600 seconds for build, 170 seconds for each dataset
fit, and 30 seconds for each dataset prediction. See
`tasks/from_scratch_svpgs/instruction.md` for the complete agent-facing contract.

## Repository layout

- `datagen/` — category definitions, synthetic DGP, public/truth materialization,
  and authenticated corpus construction.
- `grader/` — schema contract, metrics, signed AUC skill, calibration factor,
  report-only runtime diagnostics, submission runner, and corpus grader.
- `reference/` — shipped public-reference contract, model-zoo specifications, and the
  method-aware best-of-family reference contract.
- `validation/` — development model zoo, final shipping audit, pass@k and mastery
  calibration evidence, release gate, security checks, and regression tests.
- `cheats/` — shortcut baselines used to detect false-positive benchmark
  saturation; they are never eligible to become the shipped reference.
- `gold/` — a public-only, from-scratch best-of-family submission: deterministic
  training-inner-validation selects between hierarchical empirical Bayes and the
  exact fixed-grid ridge candidate before a full-data refit. The corpus builder
  executes these exact public bytes to define the shipped reference anchors; the
  same source is the inspectable score-1 witness and false-negative check.
- `tasks/from_scratch_svpgs/` — rendered agent-facing prompt and task metadata.
- `environment/` — Hyperfocal wrapper source, compiled runtime, and aggregation
  cross-checks.
- `corpus/` — the authenticated 45-dataset release artifact.

## Integrity and release

This is a private grader-side repository. It contains hidden labels, generator
code, authenticated anchors, and sealed verifier key material. The solver sees
only each dataset's `public/` files; it must never see `truth/`, `datagen/`, the
manifest, the corpus key, or this repository. See `SECURITY.md`.

The development report, all 45 corpus cells, final shipping audit, prompt,
scoring contract, wrapper source and compiled output, mastery-calibration artifact,
and release lock form one atomic release. Missing, stale, unexpected, tampered, or
unconverged artifacts fail closed. `validation/release_gate.py` validates and
HMAC-locks the exact tuple on a clean `main` worktree. See `RELEASE.md` for the
complete build, calibration, QA, and publishing procedure.

Heavy computation, corpus builds, full test suites, and grading run on an `hfdev`
SC box rather than a workstation. Genotype tables ship compressed and are staged
as plain `.tsv` files inside the isolated submission run directory.

## Grader entry point

The trusted invocation is:

```bash
python grader/grade.py <corpus_dir> <submission_dir> \
  --key-file <absolute-corpus-key-path> --out <reward_dir>
```

It writes an integrity-bound `reward_detail.json` containing the native headline,
tail, aggregation contract, per-category scores, per-dataset metrics, calibration
verdicts, runtime diagnostics, and completion evidence.
