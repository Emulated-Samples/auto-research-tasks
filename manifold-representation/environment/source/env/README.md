# manifold-bench

> A hard, CPU-only benchmark for discovering reusable nonlinear objects from
> sparse additive superpositions.

`manifold-bench` asks an agent to build a representation learner from scratch.
From unlabeled observations alone, it must infer model size and sparsity,
reconstruct held-out mixtures, identify which objects are present, isolate each
object's ambient contribution, and preserve the object's internal geometry.
Reconstruction is useful partial credit; it is not the task by itself.

## Visual atlas


The analytic zoo contains 18 local geometries: segments, cycles, disks,
spheres, tori, a Möbius strip, helices, a Swiss roll, annuli, cylinders, cones,
saddles, paraboloids, Lissajous and lemniscate curves, disconnected loops, and
a four-dimensional Clifford torus. Private injective sinusoidal lifts bend each
object into a higher-dimensional span before a private isometric embedding.

### Real neural-derived geometry, without inference at grading time


The repository contains a checksum-pinned 87 KB bank of reduced coordinates,
not models or raw corpora. It was built once from two revision-pinned,
CC-BY-4.0 public datasets:

- **Language:** final-token residual streams from layers 4, 8, 12, and 15 of
  `meta-llama/Llama-3.2-1B-Instruct` on LMSYS-Chat-1M, from
  [`scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations`](https://huggingface.co/datasets/scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations/tree/9682f489707fd078f5ee14730b5f5ce0c7604aed).
- **Vision:** aligned morning, noon, and afternoon drone images of forest plots
  from [`mpg-ranch/drone-lsr`](https://huggingface.co/datasets/mpg-ranch/drone-lsr/tree/e8ee1381cd8f0dd81f590884a31816e6ef47069e).
  Three coordinate families use DINOv2-base CLS embeddings; a fourth uses the
  satellite-pretrained DINOv3 morning CLS embedding.


This is a real aligned source triplet (`row 10_1`) from the pinned drone
revision, downsampled for documentation. The benchmark does not commit the
source image collection.


Source identities are split 60/20/20 *before* dimensionality reduction. All
four Llama layers share one conversation split; all four vision views share one
forest-plot split. PCA is fit on source-train rows only and applied unchanged to
the private match and score rows. Production uses four complete families; the
reference-development suites use the other four, so the sealed neural audit is
a family-level holdout, not a row reshuffle.

Grading loads these coordinates with NumPy and performs no VLM/LLM inference,
network access, or dataset download.

## The hidden causal problem

For each factor, a centered and RMS-calibrated local object is nonlinearly
lifted, embedded by a private row-orthonormal frame, and added to the other
active objects:

```text
m_i(u) = ((lift_i(gamma_i(u)) - center_i) / rms_i) V_i
x      = sum_{i in S} a_i m_i(u_i) + observation noise
```

The support size is stochastic. Training and evaluation contain shuffled null,
singleton, ordinary, and unusually dense rows; one suite shifts to much denser
test compositions. Active effects have a small nonzero norm floor, preventing
impossible hidden labels where “present” is observationally identical to
“absent,” while upper amplitude tails and signed amplitudes remain intact.
Private evaluation also includes paired observations whose latent composition
is identical except for one resampled object. Those pairs test whether a
learner localizes a causal change instead of moving unrelated features.

Nine suites vary geometry, correlation, subspace coherence, long-tailed
prevalence, train-to-test frequency and support shifts, observation noise,
signed lognormal amplitudes, sample size, and ambient dimension. The public
configuration exposes only matrix shape, resource limits, and a maximum object
budget, not the factor count, typical support size, topology, frames, or labels.

## Reward and a coherent pass event

Every metric is held-out, continuous, and scored as the fraction of a **span**
crossed: from a committed non-solution **floor** to a demonstrated full-credit
**target**. Both ends are measured on disjoint calibration suites, never on
production. Credit therefore begins where a plausible non-solution stops, which
is what the task prompt says it should.

| Category | Weight | What it rewards |
|---|---:|---|
| Reconstruction | 0.00 | additive held-out fidelity, necessary for joint quality and mastery, but a full-width non-solution reconstructs better than the reference, so reconstruction cannot honestly pay by itself |
| Factor recovery | 0.32 | one learned contribution recovering one complete true object |
| Factor discovery | 0.21 | prevalence-adjusted presence ranking |
| Structural coherence | 0.26 | gauge-free local geometry, one-object counterfactual localization, low fragmentation/merging, and match-calibrated presence informedness |
| Compression | 0.18 | empirical support entropy, low-dimensional codes, residual rate, dictionary cost, and sparse learned-object use |
| Efficiency | 0.03 | a minor soft preference inside hard CPU limits |

Compression and efficiency are multiplied by joint quality across the four
scientific axes, because a single dense blob is both maximally compact and
instant. Reconstruction is reported continuously, gates that joint quality,
and has its own mastery threshold, but has zero direct reward weight because no
observed reconstruction level separates a solution from a non-solution.

Matching is learned on a private match split and frozen for a disjoint score
split. Presence thresholds are likewise fit on match and applied unchanged to
score; the grader never chooses a cutoff from hidden score prevalence.
Structured-support MDL charges empirical support entropy, so a directional
method receives credit for predictable co-firing rather than being charged an
unfair uniform-support code.

The nine raw suite category vectors are weight-averaged first. Their semantic
suite quality is the geometric mean of reconstruction, recovery, discovery,
and structure; the mean of the worst quartile is the live breadth evidence.
The square root of that lower-tail quality multiplies every aggregate category
exactly once. Both the raw and adjusted category vectors are reported. This
keeps a continuous gradient for partial breadth while making a whole abandoned
regime visible to the headline reward: the historical eight-perfect/one-dead
archetype moves from `0.911` to approximately `0.744`, rather than failing only
an otherwise disconnected binary gate.

The binary event used for pass@k requires aggregate mastery, category floors, a
worst-quartile suite-quality floor, no catastrophic abandoned regime, and a
valid deterministic program on every included suite. Thus continuous score
provides partial credit and rollout variance, while `passed` has a coherent
scientific meaning. The bars, fixed during calibration and unchanged since, reserve pass
for broad, near-reference work:
adjusted reward `0.88`; adjusted reconstruction/recovery/discovery/structure/
compression `0.90/0.85/0.88/0.87/0.83`; and raw lower-tail quality `0.85`.
Recovery, discovery, and structure must also remain at least `0.15` on every
suite, so no regime can be nearly abandoned.

### What a frontier model actually scores, and where the ruler stops resolving

Measured, not assumed. An Opus 4.8 cohort was run on the previous
revision of the ruler (n=3 valid, one lost to a rate limit):

| rollout | turns | reward | mastery | characteristic failure |
|---:|---:|---:|---|---|
| 0 | 75 | `0.657` | fail | eight suites near-perfect, `correlated_supports` recovery **exactly 0.000** |
| 1 | 128 | `0.631` | fail | broad but under-recovering; compact |
| 2 | 152 | `0.475` | fail | weak throughout, `scaled_mixture` abandoned |

Mean `0.588`, native pass `0/3`, three distinct failure modes, no packaging
errors, and **effort anti-correlated with score**, the cheapest rollout was the
best by 0.18. `correlated_supports` has defeated four consecutive Opus cohorts
(recovery 0.000 / 0.035 / 0.194 / 0.000) while the current reference recovers
it at 0.91, because the frontier's method groups objects by co-firing and that
suite is built so distinct factors habitually co-fire.

That cohort also exposed the previous ruler's blindness: rollout 0 beat the old
reference on eight suites and saturated every scored metric there. The current
ruler answers it twice over. The reference was rebuilt to the recipe that cohort handed us , 
its witnessed targets now sit above most of the frontier's measured raw
recoveries, and the remaining resolution gap is *reported* rather than
implicit: `calibration/private_bsf.json` records, per suite, the noise-limited
achievable ceiling next to the witnessed target (`achievable_ceiling`,
`target_to_ceiling`), so how much room the ruler leaves above its own
reference is a committed number.

## Why `1.0` is genuinely achievable, and why nothing else is

Targets are never normalized to hidden oracle truth. For each regime, the
reference is run on twenty-four independent calibration seeds; the neural
calibration uses disjoint source families. A componentwise target is frozen at
the worst of those runs, relaxed further only where the metric's own cross-seed
dispersion requires a lower one-sided prediction bound, under the
deterministic current reference that bound fires only on unusually tight lower
tails, and every committed target still sits at or below a witnessed run
(machine-checked). Full credit is therefore not an extrapolation. Only then are
versioned production seeds generated and audited once. The noise-limited
ceiling of each regime is additionally reported next to its witnessed targets
(`achievable_ceiling` in the calibration record), the benchmark states how
much resolution it gives up above its own reference instead of implying that
region does not exist.

Floors are measured the same way from directional PCA, an overcomplete PCA
variant, and dense collapse on the same calibration suites. The floor uses the
strongest sampled non-solution on each scored metric, and
`check_discrimination` refuses to ship any metric whose floor and target fail
to separate.

For the current scoring:

- the learned prompt-only reference scores **1.000000 on every production
  suite** and passes the full nine-suite mastery event on the sealed draw;
- the simulator oracle scores **1.0**;
- directional PCA, overcomplete PCA, and the dense collapse score **exactly
  zero**, earned, not clipped there.

This is a constructive proof that a solver using only prompt-available
information can attain `1.0`, while a non-solution attains approximately what a
non-solution is worth. Under an earlier revision that last property did not hold: directional
PCA collected 0.433 while recovering 3% of one factor, because every metric
scored its level rather than its progress, and because a single 0.65 margin
sized for the noisiest metric had dropped the reconstruction target *below* what
PCA reached unaided. Partial credit survives, it is measured from the floor, not
from zero. Stronger solvers saturate at one and are never penalized. Production
observations did not select the current method, hyperparameters, floors, or
targets, every constant was chosen on the declared calibration seeds, and both
times a production observation motivated a change (the Opus cohorts above; the
first audit of the current ruler), the production seeds were retired and redrawn before
resealing. The complete provenance is committed in
[`calibration/private_bsf.json`](calibration/private_bsf.json).

Runtime is secondary. Commands are hard-limited to 75 seconds for fit and 15
seconds per transform (100/20 on the largest suite), on four CPU cores with 8
GiB memory. Efficiency has no pass floor and is fully credited through 85% of
the disclosed fit-plus-transform allowance.

## Repository map

- `environment/`, public task surfaces and Hyperfocal adapter.
- `grader/`, private generator, protocol, isolated runner, metrics, and score.
- `calibration/`, the current floors, targets, witnesses, ceilings, and sealed production results.
- `workspace/`, the only agent-writable task area.

This distribution ships the grading tree only. The authoring-time material , 
the prompt-only learned reference, retained rollout analyses, the dev test
suite, and the calibration/validation tooling that depends on them, is not
included; the committed `calibration/` record (notably
`calibration/private_bsf.json`) is the grade-time truth and is complete as
shipped.
