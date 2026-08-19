# SV-PGS: Model, Preprocessing, and Fitted-Model Artifact

Reverse-engineered from `/Users/user/SV-PGS/sv_pgs/` (READ ONLY). This document
drives the from-scratch reimplementation and the RL environment's synthetic-data
generator + prior-recovery reward. All formulas and schema are extracted from the
real source with `file:line` citations.

The model is a **hierarchical global–local shrinkage empirical-Bayes GLM**:
per-variant effects `beta_j` get a Gaussian prior whose variance is
`(global_scale * baseline_scale_j)^2 * lambda_j`, where `baseline_scale_j` is a
log-linear function of variant metadata (the "scale model") and `lambda_j` is a
per-variant local shrinkage factor from a Three-Parameter-Beta (TPB) prior whose
shape is class-specific. Covariates enter as unpenalized fixed effects. Binary
traits use a Polya-Gamma / IRLS logistic likelihood; quantitative traits a
Gaussian likelihood. One joint variational-EM (CAVI/SVI) pass fits everything.

---

## 1. Input contract

### 1.1 Genotype matrix — `RawGenotypeMatrix` (`genotype.py:686`)

- Shape **`(n_samples, n_variants)`**, samples × variants.
- **Dosage encoding** (`genotype.py:689-691`): `0` = homozygous reference,
  `1` = heterozygous, `2` = homozygous alternate, **`NaN` = missing**.
- On-disk PLINK/bitpacked backends store int8 `0/1/2` plus a missing sentinel
  `PLINK_MISSING_INT8` (`genotype.py:26`, `plink.py`); float paths use `NaN`.
- Genotypes are **not** pre-standardized on input — the model standardizes
  internally (see §2). Access is streaming via `iter_column_batches` /
  `iter_column_batches_i8`; the full dense matrix is never required.
- `fit()` also accepts a plain 2-D NumPy `(n, m)` array of dosages
  (`model.py:2258`, wrapped by `as_raw_genotype_matrix`).

### 1.2 Phenotype (`targets`)

- 1-D array, one value per sample. `TraitType.BINARY` → `{0,1}` case/control;
  `TraitType.QUANTITATIVE` → real-valued (`config.py:8-11`, `ModelConfig.trait_type`).
- Reshaped to `(n,)` float32 inside `fit`. No transformation of quantitative
  targets beyond covariate residualization.

### 1.3 Covariates

- Dense `(n_samples, p_cov)` float matrix passed to `fit(covariates=...)`.
- An **intercept column of 1.0 is prepended internally** by `_with_intercept`
  (`model.py:3468-3476`) — callers pass only real covariates.
- Covariates are **fixed effects, unpenalized** (no shrinkage prior); coefficient
  vector is `alpha` (length `p_cov + 1`, first entry = intercept).
- AoU default covariate set (`aou_runner.py:1063-1069`, README:45):
  `age_at_observation_start`, `age_squared` (= age², `aou_runner.py:841`),
  `gender_concept_id`, `race_concept_id`, `ethnicity_concept_id`, and `PC1..PC10`
  (top 10 genomic PCs; `--n-pcs`, `aou_runner.py:724-830`). The three OMOP
  categorical fields are **one-hot expanded** into `<name>_<concept_id>` columns
  (`aou_runner.py:1077-1124`, `all_of_us._add_one_hot_omop_categorical_covariates`).

### 1.4 Variant metadata schema — `VariantRecord` (`data.py:46-71`)

One record per variant column. Reserved / built-in fields:

| field | type | default | role |
|---|---|---|---|
| `variant_id` | str | — | key; joins genotype ↔ metadata |
| `variant_class` | `VariantClass` enum | — | primary prior grouping (see below) |
| `chromosome` | str | — | bookkeeping |
| `position` | int | — | bookkeeping |
| `length` | float | 1.0 | SV span; used in tie-collapse + variant routing (not a direct prior feature) |
| `allele_frequency` | float | 0.01 | **drives the MAF filter** |
| `quality` | float | 1.0 | carried through; call confidence |
| `training_support` | int \| None | None | carrier/observation support; tie-collapse + routing |
| `is_repeat` | bool | False | repeat overlap; routing / tie-collapse |
| `is_copy_number` | bool | False | CNV flag; routing / tie-collapse |
| `prior_class_members` | tuple[VariantClass] | `(variant_class,)` | fuzzy/multi-class membership |
| `prior_class_membership` | tuple[float] | `(1.0,)` | weights for the above (same length) |

`VariantClass` (10 levels, `config.py:13-23`): `snv`, `small_indel`,
`deletion_short`, `deletion_long`, `duplication_short`, `duplication_long`,
`insertion_mei`, `inversion_bnd_complex`, `str_vntr_repeat`, `other_complex_sv`.

**Custom annotation families** (arbitrary user columns; `data.py:64-69`):
`prior_binary_features` (dict[str,bool]), `prior_continuous_features`
(dict[str,float]), `prior_categorical_features` (dict[str,str]),
`prior_membership_features` (dict[str, dict[level→weight]]),
`prior_nested_features` (dict[str, path-tuple], `parent>child`),
`prior_nested_membership_features` (dict[str, dict[nested-path→weight]]).

**Reserved vs. annotation columns** (`io.py:63-78`, `VARIANT_METADATA_BASE_COLUMNS`):
the 12 reserved names above become built-in fields; **every other column** in a
`--variant-metadata` TSV becomes a user annotation, with its family **inferred
from the values** (`io.py:4228-4254`):

- all values in `{1,0,true,false,yes,no}` → **binary**
- all values `float`-parseable → **continuous**
- all values `level=weight,level=weight` → **membership** (or **nested_membership**
  if every level name contains `>`)
- all values contain `>` → **nested**
- otherwise → **categorical**

Parsing entry points: `_build_variant_records` / `_merge_variant_metadata`
(`io.py:3802-3958`). Only reserved model columns and inferred annotations are
used; **VCF INFO is not mined for annotations** (README:43-44). If no metadata
file is supplied, every record gets defaults and `prior_class_members =
(variant_class,)`, `prior_class_membership = (1.0,)` (`data.py:87-89`).

---

## 2. Preprocessing

### 2.1 Per-variant statistics — one streaming pass (`preprocessing.py:106-219`)

`compute_variant_statistics` accumulates per variant: `sums`, `non_missing_counts`,
`support_counts` (# non-zero dosages), `centered_sum_squares`, and a `dosage_like`
bound check (`0 ≤ g ≤ 2`). Produces `VariantStatistics(means, scales,
allele_frequencies, support_counts)` (`data.py:185-191`).

- **Mean** (`_means_and_scales_with_floor`, `preprocessing.py:426-463`):
  `mean_j = sum_j / non_missing_count_j` (missing entries excluded from the mean).
- **Scale** (std): `scale_j = sqrt(centered_sum_squares_j / n_samples)` — a
  population std with **ddof = 0**, divided by total `n_samples` (not the
  non-missing count).
- **Low-variance floor**: if `scale_j < config.minimum_scale` (1e-6) →
  `scale_j := 1.0` and mean held at the true column mean, so a constant column
  standardizes to exactly 0. All-missing columns → `(mean=0, scale=1)`.
- **Allele frequency** (`_allele_frequencies_from_means`, `preprocessing.py:412`):
  `af_j = clip(mean_j / 2, 0, 1)` for dosage-like columns, else `0.5`.

### 2.2 Standardization (`genotype.py:717`, `StandardizedGenotypeMatrix`)

`z_ij = (g_ij - mean_j) / scale_j`, with **missing entries mapped to 0** after
centering (i.e. imputed to the column mean). Kernel form
(`genotype.py:2290`, `model.py:3563`):
`Z[i,j] = ((float)raw - mean_j)/scale_j`, missing sentinel zeroed.
The `Preprocessor` (`preprocessing.py:475-489`) stores only `(means, scales)`;
`transform()` applies them lazily. **The training means/scales are frozen and
re-applied at scoring time** — the test cohort is standardized against the
*training* statistics, never re-standardized against itself
(`model.py:3258-3267`, `_bitpacked_scoring_fast_path` overrides subset mean/std).

### 2.3 MAF filter (`select_active_variant_indices`, `preprocessing.py:612-667`)

Keep variant `j` iff
`min(af_j, 1-af_j) >= config.minimum_minor_allele_frequency` (default **1e-2**,
`config.py:145`). `_minor_allele_frequency` treats non-finite AF as 0 (dropped).
This is the "very rare SVs will be filtered" rule (SPEC.md:4). Content-keyed
disk cache keyed on variant ids + AFs + threshold.

### 2.4 Optional marginal z-score pre-screen (`config.py:146-161`)

After MAF, an optional univariate `|z|` screen (default **0.0 = disabled**).
`compute_marginal_z_scores` (`preprocessing.py:357-409`) residualizes `y` on the
covariates and computes, per active variant,
`z_j = (X_j_std^T y_resid) / sqrt(sigma2_resid * (n - x_j'C(C'C)^-1 C' x_j))`
(`marginal_z_from_numerator`, `preprocessing.py:304-336`). Under the null z_j ~
N(0,1); binary y gives a Rao-score-style statistic. SVs can be protected from the
screen (`marginal_screen_protect_sv=True`, `config.py:161`).

### 2.5 Covariate handling / residualization (`preprocessing.py:222-276`)

`residualize_target_on_covariates` projects `y` onto `span([1 | C])` via OLS
(`lstsq`), returning `y_resid = y - C alpha_cov`, `sigma2_resid = ||y_resid||^2/n`,
and `pinv(C'C)`. This residualization is used for the **marginal screen and
z-diagnostics**. In the joint EM, covariates are carried as explicit unpenalized
columns (coefficient `alpha`) rather than projected out — genotype effects and
`alpha` are estimated jointly.

### 2.6 Tie / duplicate-column collapse (`data.py:203-259`, `preprocessing.py:810-1911`)

Variants with identical (or exactly negated) genotype columns are grouped into a
`TieMap`; the model fits one representative per group in "reduced space", then
`expand_coefficients` distributes the fitted effect back to members:
`beta_member = beta_group * weight_member * sign_member` (`data.py:234-259`). This
is a computational reduction, not a different model.

### 2.7 Holdout policy

**The fit itself uses ALL samples** — the Bayesian prior is the only regularizer;
no CV/holdout for fitting (SPEC.md:20). A deterministic **20% test split**
(`AOU_TEST_FRACTION = 0.2`, `aou_runner.py:1177`) is carved out for *evaluation
only*, via `bucket = sha256(salt|sample_id) mapped to [0,1); < 0.2 → test`
(`aou_runner.py:1012-1058`). Held-out samples feed `test_predictions.tsv.gz`;
`validation_data` passed to `fit` is monitoring-only
(`validation_is_holdout_only=True`, `model.py:2268`, `pipeline.py:168-173`).

---

## 3. The fitted-model artifact (prior recovery target)

Fitting produces a `VariationalFitResult` (`mixture_inference.py:616-638`)
exported as a `ModelArtifact` (`artifact.py:56-101`), serialized as
`arrays.npz` + `metadata.json` (`artifact.py:104-241`). This is exactly what a
"model" produces and what the reward can grade.

### 3.1 Per-variant outputs (PGS weights)
- `beta_full` `(n_variants,)` — **posterior-mean effect per variant** in
  standardized-genotype space; these are the PGS weights (`artifact.py:64`).
- `beta_reduced` — effects in tie-reduced space (`artifact.py:63`).
- `beta_variance` — posterior variance per (reduced) coefficient (`artifact.py:65`).
- `means`, `scales` `(n_variants,)` — frozen training standardization stats,
  needed to apply weights to new dosages (`artifact.py:59-60`).
- `tie_map` — reduced↔full expansion (`artifact.py:66`).

### 3.2 Covariate coefficients
- `alpha` `(p_cov+1,)` — fixed-effect coefficients incl. intercept
  (`artifact.py:62`). Applied as `C_with_intercept @ alpha`.

### 3.3 Recovered priors / hyperparameters (THE prior-recovery quantities)

These are the internal generative hyperparameters the model recovers — grade
these against the synthetic ground truth:

| artifact field | meaning | source |
|---|---|---|
| `global_scale` (σ_g) | global prior-scale multiplier, in `[1e-4, 10]` | `mixture_inference.py:622` |
| `prior_scales` `(n_reduced,)` | **per-variant effective prior variance** `τ_j²` (float64) | `mixture_inference.py:621` |
| `scale_model_coefficients` | coefficients of the **log-linear metadata → prior-scale model** (one per design feature) | `mixture_inference.py:625` |
| `scale_model_feature_names` | names of those design features (aligns 1:1 with coefficients) | `mixture_inference.py:626` |
| `class_tpb_shape_a` | dict `VariantClass → shape_a` of the local TPB prior | `mixture_inference.py:623` |
| `class_tpb_shape_b` | dict `VariantClass → shape_b` (auxiliary rate) | `mixture_inference.py:624` |
| `sigma_e2` (σ_e²) | residual noise variance (quantitative) / logistic dispersion | `mixture_inference.py:627` |
| `member_prior_variances` | per-member prior variance (tie-expanded) | `mixture_inference.py:630` |

Defaults / init values the model departs from (grade recovery relative to these):
- `DEFAULT_CLASS_LOG_BASELINE_SCALE` (`config.py:31-42`): per-class log baseline
  scale, e.g. `snv:-4.5`, `inversion_bnd_complex:-3.1` (SNVs smallest effects,
  complex SVs largest; `exp(-4.5)≈0.011`, `exp(-3.1)≈0.045`).
- `DEFAULT_CLASS_TPB_SHAPE_A` (`config.py:50-61`): e.g. `snv:1.0` (light tails),
  `inversion_bnd_complex:0.55` (heavy tails → tolerates large effects).
- `DEFAULT_CLASS_TPB_SHAPE_B` (`config.py:66-77`): e.g. `snv:0.5`,
  `inversion_bnd_complex:0.38`.

### 3.4 The prior-scale generative structure (the equations to grade)

Per-variant prior on the standardized effect:
```
beta_j ~ Normal(0, tau_j^2)
tau_j^2 = (global_scale * baseline_scale_j)^2 * lambda_j        (member_prior_variances)
```
- **Metadata baseline** (`_metadata_baseline_scales_from_coefficients`,
  `mixture_inference.py:13533-13546`):
  `baseline_scale_j = exp( clip( design_row_j · scale_model_coefficients,
  log(prior_scale_floor), log(prior_scale_ceiling) ) )`,
  with `prior_scale_floor=1e-6`, `prior_scale_ceiling=10` (`config.py:105-106`).
- **Local TPB shrinkage** `lambda_j`: recovered from the saved prior scales via
  `lambda_j = tau_j^2 / (global_scale * baseline_scale_j)^2`
  (`mixture_inference.py:970-985`). `lambda_j ≥ local_scale_floor` (1e-8).
- **Effective prior variance** used in updates
  (`_effective_prior_variances`, `mixture_inference.py:3549-3560`):
  `max(baseline_prior_variance_j * max(lambda_j, floor), 1e-8)`.

### 3.5 The scale-model **design matrix** (`_build_prior_design`, `mixture_inference.py:12649-12806`)

Rows = variants, columns = metadata-derived features; each column is **mean-centered**
(`_center_design_column`) and dropped if ~constant. Feature families
(`_compile_prior_feature_specs`, `mixture_inference.py:12696-12785`):
- `type_offset::<class>` — per-`VariantClass` intercept (from
  `prior_class_membership`); penalized by `type_offset_penalty=2.0`
  (`mixture_inference.py:13563-13571`), other features by
  `scale_model_ridge_penalty=1.0`.
- `factor_level::<col>::<level>` and `factor_interaction::<class>::…` — from
  categorical / membership annotations.
- `nested_level` / `nested_interaction` — from `parent>child` nested annotations
  (per depth).
- `continuous_spline::<col>::basis_k` and `continuous_spline_interaction::…` —
  spline bases over continuous annotations, with per-class interactions when a
  class has ≥3 members.

Built-in continuous fields (`length`, `allele_frequency`, `quality`,
`training_support`) and the booleans (`is_repeat`, `is_copy_number`) are **not**
automatically injected as scale-model features — they influence the prior only
through `variant_class` assignment and (for length/support/repeat) tie-collapse
and variant routing. To exercise length/repeat/etc. as prior drivers, supply them
as **custom annotation columns** in `--variant-metadata` (e.g. a `length` or
`is_lof` annotation column), which then become continuous/binary features.

The scale model is fit by penalized (ridge) optimization inside the EM
(`_scale_model_penalty` + up to `maximum_scale_model_iterations=8`,
`config.py:110-112`); TPB shapes updated up to `maximum_tpb_shape_iterations=8`,
bounded `[minimum_tpb_shape=0.1, maximum_tpb_shape=10]` (`config.py:116-117`).

### 3.6 Serialization details (`artifact.py:104-297`)
`arrays.npz` holds: `means, scales, alpha, beta_reduced, beta_full,
beta_variance, tie_kept_indices, tie_original_to_reduced, prior_scales,
scale_model_coefficients`. `metadata.json` holds: full `config`, all `records`
(incl. every annotation family), `tie_groups`, `sigma_e2`, `global_scale`,
`class_tpb_shape_a/b` (keyed by class value), `scale_model_feature_names`,
`objective_history`, `validation_history`, `fit_fingerprint` (SHA-256 of
inputs+config for reuse gating), and convergence diagnostics (`converged`,
`selected_iteration_count`, four `final_*_change` deltas).

---

## 4. Scoring / prediction (`model.py:3224-3295`)

### 4.1 Linear predictor
`decision_function` returns `genetic_component + covariate_component`
(`model.py:3224-3232`):
```
score_i = Σ_j beta_j * (g_ij - mean_j)/scale_j   +   [1 | c_i] · alpha
```
- `decision_components` splits the two (`model.py:3234-3282`): covariate part
  `[1|C] @ alpha`; genetic part is a matvec over **only non-zero-coefficient
  variants** (`nonzero_coefficient_indices`, typically <1% of variants),
  standardizing on the fly with the **training** means/scales.
- Prediction outputs cache both `genetic_score` and `covariate_score`.

### 4.2 Binary vs quantitative
- **Binary**: `p_i = sigmoid(score_i)` (`stable_sigmoid`); `predict_proba` returns
  `[1-p, p]`; `predict` thresholds at 0.5 (`model.py:3284-3295`).
- **Quantitative**: the linear predictor *is* the prediction (`model.py:3295`).

### 4.3 Prediction files (`pipeline.py:369-505`)
`predictions.tsv.gz` columns: `sample_id, target, genetic_score,
covariate_score, linear_predictor, probability, predicted_label` (binary) or
`… prediction` (quantitative). `test_predictions.tsv.gz` = same on the 20%
held-out set.

### 4.4 Self-evaluation metrics

**Generic benchmark** (`benchmark.py`, `BenchmarkMetrics`):
- Binary: **AUC** (`roc_auc_score`), **log_loss**, **PR-AUC**
  (`average_precision_score`) (`benchmark.py:101-107`).
- Quantitative: **R²** (`r2_score`) (`benchmark.py:129`).
- **Top-tail enrichment** (`benchmark.py:138-155`): among the top `fraction`
  (default 0.05, `config.py:322`) by score — binary:
  `mean(target[top]) / mean(target)`; quantitative:
  `(mean(target[top]) - mean(target)) / std(target)`.
- Compares an SNV-only model vs. a joint SNV+SV model (`benchmark.py:49-71`).

**AoU quasi-holdout** (`evaluate.py`): a tie-aware Mann–Whitney **AUC**
(`_compute_auc_safe`, `evaluate.py:147-179`) applied to out-of-training signals:
(1) ICD 0-code vs 1-code among training controls; (2) survey self-report vs not;
(3) the true 20% held-out set (overall / genetic-only / covariate-only / EUR-only
AUCs + dose-response by ICD count). Calibration surfaces as mean-score-by-group
diffs; no explicit calibration-slope metric in these files.

---

## 5. What an ideal synthetic dataset must contain

To exercise every input path and every prior-recovery quantity:

1. **Genotype matrix** `(n_samples, n_variants)`, integer dosages `{0,1,2}` with a
   sprinkling of missing (`NaN`/sentinel) to test mean-imputation and the
   non-missing-count mean. Include:
   - Common variants (MAF ≫ 1e-2) that survive the filter, and **very rare ones
     (MAF < 1e-2)** that must be dropped.
   - **Monomorphic / near-constant columns** (std < 1e-6) to hit the scale floor.
   - **Duplicated and exactly-negated columns** to exercise the tie-map collapse.
2. **Full variant-metadata TSV** keyed by `variant_id`, covering:
   - Multiple `variant_class` levels spanning SNV → complex SV (so per-class
     `type_offset`, baseline scale, and TPB shapes are all identifiable — put
     ≥3 members in each class so interaction features activate).
   - Built-ins: varied `length`, `allele_frequency`, `quality`,
     `training_support`, `is_repeat`, `is_copy_number`.
   - Fuzzy `prior_class_members` / `prior_class_membership` (multi-class weights)
     on some variants.
   - One column of **each annotation family** so all `io.py` inference branches
     fire: a **binary** col (`true/false`), a **continuous** col (numeric — ideally
     a genuine effect driver, e.g. a "constraint" score), a **categorical** col
     (`lof/missense/...`), a **membership** col (`enhancer=0.7,promoter=0.3`), a
     **nested** col (`protein_coding>exon`), and a **nested_membership** col
     (`a>b=0.7,a>c=0.3`).
3. **Known ground-truth generative priors** so the reward can score recovery of:
   `global_scale`, per-`class` baseline scale (via `scale_model_coefficients` on
   `type_offset` + the continuous/annotation effects), per-`class` TPB
   `shape_a`/`shape_b`, `sigma_e2`, and the per-variant `beta_j` / `prior_scales`.
   Generate `beta_j ~ Normal(0, (global_scale·baseline_scale_j)²·lambda_j)` with
   `baseline_scale_j = exp(design_row_j · true_scale_coeffs)` and `lambda_j` drawn
   from the class-specific TPB, so recovery is exactly gradeable.
4. **Covariates** `(n, p_cov)`: continuous (age, age²), a categorical to one-hot
   (sex/race), and **PC1..PC10**; give them real (non-zero) `alpha` effects so
   covariate-coefficient recovery and covariate-only AUC are meaningful.
5. **Both trait types**:
   - Binary: `y_i ~ Bernoulli(sigmoid(genetic_i + covariate_i))` — grade AUC /
     log-loss / PR-AUC / top-tail enrichment.
   - Quantitative: `y_i = genetic_i + covariate_i + Normal(0, sigma_e2)` — grade
     R² / top-tail enrichment / `sigma_e2` recovery.
6. Enough samples/variants that the joint fit is non-trivial but the true priors
   remain identifiable (class counts ≥ a few dozen each; several PCs; a modest
   missing-rate). Split 80/20 by `sha256(sample_id)` to mirror the real held-out
   evaluation.

### Key numeric defaults to bake into the generator/config
`minimum_minor_allele_frequency=1e-2`, `minimum_scale=1e-6`,
`prior_scale_floor=1e-6`, `prior_scale_ceiling=10`, `global_scale ∈ [1e-4,10]`,
`local_scale_floor=1e-8`, `sigma_error_floor=1e-3`, `tpb_shape ∈ [0.1,10]`,
`scale_model_ridge_penalty=1.0`, `type_offset_penalty=2.0`,
`top_tail_fraction=0.05`, test split `0.2`. Standardization is **ddof=0**,
mean/scale frozen at training and reused for scoring.
