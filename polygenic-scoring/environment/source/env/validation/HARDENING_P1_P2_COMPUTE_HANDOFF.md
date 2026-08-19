# svpgsbench hardening — compute-host handoff (P1 rebuild + new-machinery categories + P2 model-zoo pilot)

> GRADER-SIDE / HIDDEN (under `validation/`, never visible to the solver). This is
> the authoritative to-do for the parts of the P1/P2 hardening that CANNOT be done
> or validated on a laptop because they need (a) the real `sv_pgs` engine and (b)
> full-size fits over many seeds. Everything here builds on what already landed:
>
> - P0 scoring: penalty-only runtime, noise-aware+winsorized skill, tail
>   aggregation (`grader/perf.py`, `grader/skill.py`, `grader/grade.py`).
> - P0 contract: explicit formula grammar, covariate-subset/decoy substrate,
>   `dgp.json` (`environment/problems.yaml`, `datagen/materialize.py`,
>   `grader/submission_runner.py`).
> - P1 code: 9-category capability matrix, load-bearing `dgp.md`, bootstrap-SE
>   skill floor + separate runtime anchor in `datagen/build_corpus.py`.
>
> The grader is backward-compatible with the CURRENTLY SHIPPED corpus (old 3
> categories) — nothing here is required for the current env to grade; it is the
> activation path to the hardened corpus.

## A. Corpus rebuild (activates P0/P1 anchor fields)

On a box with the `sv_pgs` package (`SVPGS_HOME` or sibling `../SV-PGS`) and real
CPU headroom:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd svpgsbench
python datagen/build_corpus.py ALL <per_category> <N> <P>     # per_category >= 5
# or drive per-dataset via build_corpus.sh (memory-safe subprocess-per-dataset)
```

- `per_category >= 5` seeds (spec: within-agent seed SE <= 0.02–0.03 needs >=5).
- Recommended full size: `N ~ 12000`, `P ~ 800` for the standard categories;
  `sparse_heavy_tail` overrides its own `n_samples`/`n_variants` (p>>n) inside
  `make_cfg`, so it will build larger regardless of `P`.
- The build now also writes, per dataset, `metrics_ref_naive_se`,
  `time_runtime_anchor`, and `time_runtime_calibration`; `materialize` emits
  `dgp.json` + a load-bearing `dgp.md`. `FINALIZE` will abort on any stale/missing
  dataset (provenance-guarded), so a partial rebuild can't ship.
- After the build: run `validation/e2e_test.py` (cheat separation) and confirm the
  reference still beats every cheat on every category (skill < 1.0). Any category
  where a cheat reaches the reference has lost its signal — drop or retune it.

## B. New-machinery categories (NOT yet implemented — need `datagen/dgp.py` work)

The 9 shipped categories use only existing DGP knobs. The following require new,
faithful generative code in `dgp.py` (do NOT fake them with existing knobs). Each
should be added as a `datagen/categories.py` recipe with an `architecture` dict
(the `dgp.md`/`dgp.json` disclosure phrases already exist in `materialize.py`
`_ARCH_PHRASES` for `shift: ancestry|ld` and the annotation axes — extend as
needed).

1. **nonlinear-annotation** — the log-scale model uses a U-shaped / thresholded
   function of `sv_log_length` (nonzero higher-order spline coeffs), not the
   current log-linear `length_coef_mean` term. A linear annotation-weighted
   penalty should capture only part of it; a spline/nonparametric annotation model
   wins. Disclose `annotation: nonlinear_scale`.
2. **annotation-interactions** — scale depends on `class × length` (and/or
   `MAF × class`) interactions, not additively. Add interaction terms to the scale
   model. Disclose `annotation: interaction_scale`.
3. **noisy/decoy-annotations** — ship extra annotation columns in
   `variant_metadata.tsv` that are UNCORRELATED with the true scale (pure noise or
   a plausible-but-irrelevant signal). A model that blindly trusts annotations
   should be HURT; the honest model must down-weight them. (materialize supports
   decoy COVARIATES via `extra_cov_cols`; add the analogous decoy-annotation
   emission to `variant_metadata.tsv`.) Disclose `annotation: decoy_present`.
4. **suppressor / signed-LD** — some causal variants are in NEGATIVE LD with their
   tags, so the marginal association is weak or sign-reversed while the joint
   effect is large. Current LD is positive-`rho` blocks; add signed correlations.
   Marginal/P+T should fail badly; joint decorrelation wins. Disclose
   `ld: signed` (add phrase).
5. **ancestry-shift** — the TEST cohort's ancestry (PC) distribution differs from
   train (e.g. shift the PC means / re-weight). Covariate adjustment must
   generalize across the shift; a model that overfits train ancestry transfers
   poorly. Needs a shift-aware split in `materialize._split` or a `dgp` test-cohort
   option. Disclose `shift: ancestry` (phrase already present).
6. **LD-shift** — train/test differ in the correlation structure so tag-only
   predictors don't transfer; only variant-space joint effects do. Disclose
   `shift: ld` (phrase already present).
7. **low-prevalence extreme** (0.02) and **broadened ranges** across ALL
   categories per the spec: h² 0.2–0.8, prev 0.05–0.5, n_train 2k–10k, p 2k–20k,
   n/p ~0.25–3, causal 0.1%–100%, LD weak/moderate/strong/long-range/signed. Vary
   these across seeds within a category so seed noise is real-but-bounded and the
   category still measures ONE capability.

For each: keep the reference's advantage (skill 1.0) intact, and confirm via
`e2e_test.py` + the model-zoo pilot (below).

## C. P2 — model-zoo pilot + category selection

`validation/model_zoo.py` (shipped alongside this doc) implements the model zoo
and the selection metrics. On the compute host, after the rebuild:

```
python validation/model_zoo.py --corpus corpus --out validation/model_zoo_report.json
```

It builds `S[model][category]` (accuracy skill of each model on each category,
averaged over seeds) and reports the selection metrics. Models: covariates-only,
P+T, uniform ridge, lasso, elastic net, adaptive lasso, annotation-weighted EN,
group ridge, residualized EN — plus optional ARD / horseshoe-approx / blockwise /
stacked and the real `sv_pgs` when importable.

**Select the category set** that:
- maximizes between-model variance per category (a category where all models tie
  is useless);
- minimizes pairwise category correlation of the model-score vectors (no
  redundant near-clones — this is what the old 3 categories failed);
- yields DIFFERENT model rankings across categories (some category where P+T or
  ridge tops, some where annotation-weighted/horseshoe tops);
- keeps low seed uncertainty (SE <= 0.02–0.03) and a clear reference advantage
  (reference skill = 1.0 above every non-reference model, or the category is
  flagged).

## D. Acceptance targets (pilot before shipping)

Difficulty (accuracy skill vs the frozen reference/ensemble anchor = 1.0):
- covariates-only ≈ 0
- ridge / P+T: 0.25–0.35
- plain elastic net: 0.30–0.50
- annotation-adaptive EN: 0.45–0.65
- strong Bayesian / stacked: 0.65–0.85
- frozen reference ensemble: 1.0 (an exceptional new method may exceed 1)

Agent outcomes (post-hardening, expected): median competent 0.35–0.55; p75
0.60–0.75; p90 0.75–0.90; >1 rare.

Variance: between-agent SD >= 0.15; within-agent seed SE <= 0.02–0.03;
ICC(agent) > 0.85; no two categories with baseline-score correlation > 0.8; no
single non-reference baseline best on every category.

## E. Stronger accuracy anchor (spec highest-priority #4)

Replace the single-SV-PGS accuracy anchor with a FROZEN CV ENSEMBLE of several
strong models (e.g. annotation-weighted EN + horseshoe-approx + SV-PGS, stacked
on OOF predictions). Anchors are offline, so this can be expensive. Store its
per-metric values as `metrics_reference` and keep the fast ridge as
`time_runtime_anchor` (already wired). This raises the skill=1.0 bar so beating it
requires genuine modeling, not just matching one mechanism-faithful engine.

## Definition of done

- Rebuilt corpus (>=5 seeds/category) with SE + runtime anchors + dgp.json.
- New-machinery categories implemented, reference-advantage confirmed.
- Model-zoo pilot passes the acceptance/variance targets; category set selected to
  maximize between-model variance and minimize category correlation.
- `e2e_test.py` still green (reference beats every cheat on every category).
- Frozen ensemble accuracy anchor in place.
