# Polygenic scoring from scratch

Build a genotype-to-phenotype prediction library from scratch — NumPy, SciPy, or the C/C++/Fortran toolchain, with polygenic-scoring, GWAS, and stock penalized-regression tooling forbidden — and be graded on held-out predictive skill across fifteen hidden genetic architectures.

Rewards sit on a native skill scale: a covariates-only model earns about 0.333, the reference solution defines 1.0, a crash is exactly 0, and per-category headroom runs to 1.333, so beating the reference is possible. Models reliably build real statistics here; what separates a 1.01 from a 0.40 is whether one estimator adapts across all fifteen architectures, and whether it was ever rehearsed under the grader's time and memory limits.

## The task

The prompt is the message a colleague doing computational genetics would send: I'm building a polygenic-scoring library from scratch, here is the program contract, design any estimator that predicts held-out phenotype well and quickly — you are not being asked to reproduce a published tool. The agent ships two executables. `fit` reads a training cohort — covariates, integer genotype dosages over thousands of variants, per-variant annotations — plus a model formula it must parse rather than assume, and serializes a fitted model. `predict` emits calibrated case probabilities for held-out samples.

Grading runs 45 synthetic cohorts: fifteen capability categories, three replicates each, covering sparse and dense architectures, rare-variant signal, low prevalence, weak and shifted linkage, decoy annotations, nonlinear annotation effects, and ancestry shift. Each cohort deliberately gives fewer training samples than variants, so unregularized approaches have nothing to hold on to. The workspace contains only a 30-sample format fixture — far too small to learn from — so the agent has to reason about the estimator, not overfit a fixture.

The from-scratch constraint is real: the toolchain guarantees generic numerics (NumPy 2.1.3, SciPy 1.14.1, BLAS/LAPACK, compilers) and nothing else, offline. Each cohort gives `fit` 170 seconds and `predict` 30 seconds inside a 6 GiB address-space cap; overruns are killed and scored as invalid.

## Verifier design

We score held-out predictions on hidden cohorts the agent never sees, against baselines it cannot game.

| What we check | How |
| --- | --- |
| Predictive skill, not raw accuracy | Held-out AUC is normalised between a covariates-only baseline and a reference solution, so predicting from covariates alone — or from the base rate — earns nothing |
| The gap being scored is measurable | Per-cohort statistical gates verify the baseline-to-reference gap is resolvable above bootstrap noise before any skill is awarded on it, so no category rewards luck |
| Calibration, not just ranking | A continuous penalty on proper-scoring regret shrinks positive skill, so a good ranker that pushes probabilities to 0 and 1 loses reward instead of hiding behind AUC |
| Breadth across architectures | The headline is 60 percent an equal-weight mean over the fifteen categories and 40 percent the mean of the agent's three weakest, so one hard-coded prior that collapses somewhere is punished far beyond its share |
| Crashing is never a strategy | Invalid runs score the floor, so selectively failing on hard cohorts pays nothing |
| The corpus is tamper-evident | Every dataset and answer file is hash-verified and HMAC-signed before grading begins, and submissions run in an unprivileged sandbox with a hard memory cap |

## Trace walkthrough

Half the evaluated runs cluster between 0.89 and 1.01, ten more sit at 0.40–0.62 — real signal, far from the reference — and two crashed to 0. The split is not statistical knowledge; it is engineering discipline against the grader's constraints.

### A strong run

1. **Measure the machine before the statistics.** The winning run's first real finding was not statistical: the BLAS install defaults to 64 threads on a 4-CPU box, and pinning threads took a core matrix-vector product from 100ms to 5.5ms. It fixed that before designing anything.
2. **Distrust the fixture.** It noticed the 30-sample fixture perfectly separates on its 15 covariates and saturates predictions at 1-1e-7, and treated that as a calibration warning: it added a weakly informative prior on covariates and a probability clamp — insurance the verifier prices directly.
3. **Build one estimator that adapts, not fifteen guesses.** Per-variant association statistics feed an annotation-driven prior-variance map fit by marginal likelihood; a penalized logistic solver runs matrix-free so the dosage matrix is never squared; and cross-validation chooses the prior's scale and tail weight per cohort, so sparse-versus-dense is estimated from the data rather than assumed.
4. **Rehearse the grader's limits.** It generated its own realistic cohorts with linkage structure, measured a 20,000-sample, 15,000-variant fit at 56 of 170 seconds, caught `predict` at 25 of its 30 seconds and hardened the reader, and gave the search an internal deadline that reserves time for the final fit. Thirty-two minutes end to end; score 1.01 — above the reference — with calibration slopes of 0.92–0.96.

### A failed run

1. **Build the most ambitious system in the set.** One run spent 57 minutes on a memory-mapped C++ reader, a coordinate-descent solver with active sets and warm starts, cross-validation parallelized over the 4 CPUs, and three penalty families stacked by a meta-learner.
2. **Rehearse on toys.** It proved that pipeline end to end on the 30-row fixture and on a single self-generated cohort of 2,500 samples and 5,000 variants, where `fit` used 61 of its 170 seconds, and read that margin as safe.
3. **Meet the grader's constraints for the first time at grading.** It never once ran under the 6 GiB address-space cap and never tried a cohort larger than its own bench. All 45 graded cohorts came back invalid: score 0, on a scale where doing nothing beyond the covariates is worth 0.333.

Both runs derived real estimators. The winning run treated the grader's limits as part of the problem and rehearsed them; the 0 run treated them as fine print.

## Failure modes

These are the failure modes we saw across the evaluated runs.

| Failure mode | What goes wrong |
| --- | --- |
| Validating on the fixture | The workspace sample is 30 rows of format, not data; a pipeline that only ever proves itself there or on a small self-made bench can come back invalid on all 45 cohorts and score 0 |
| Regime monoculture | A single fixed prior (for example, ridge everywhere) holds up on average and collapses on the nonlinear-annotation and ancestry-shift categories, which the weakest-three weighting reads directly |
| Treating the time cap casually | One slow architecture times out `fit`, the kill is scored as invalid, and that category is poisoned |
| Confident nonsense | Probabilities pushed toward 0 and 1 keep their AUC but pay continuously through the calibration penalty |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/auto-research-tasks/polygenic-scoring --agent oracle -k 1 -o jobs/
```

The reference solution defines full skill on every cohort; the replay floor is 0.80. The task grades on 4 CPUs and 16 GB, CPU-only.
