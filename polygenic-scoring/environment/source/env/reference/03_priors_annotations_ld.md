# SV-PGS Reference 03 — Priors, Annotations, Marginal Screening, LD Blocks

Reverse-engineered from `/Users/user/SV-PGS` (read-only). Every claim below cites
real code. This document defines the prior/annotation/LD contract that a
from-scratch reimplementation and its "perfect-but-fair" synthetic DGP must
honor.

## 0. The one-line model this all serves

SV-PGS fits a single joint empirical-Bayes GLM (linear for quantitative,
logistic via Pólya-Gamma/IRLS for binary) on **all** variants at once. Each
variant's effect gets a **hierarchical global–local shrinkage Gaussian prior**:

```
beta_j  ~  N( 0,  sigma_j^2 )
sigma_j^2 = ( global_scale * s_j )^2  *  lambda_j
```

- `global_scale` — one scalar for the whole fit (`global_scale_floor=1e-4`,
  `global_scale_ceiling=10.0`, `config.py:107-108`).
- `s_j` = **metadata baseline scale** = `exp( clip( design_row_j @ scale_model_coefficients,
  log(prior_scale_floor), log(prior_scale_ceiling) ) )`
  (`mixture_inference.py:13533-13546`; floors `1e-6`/ceiling `10.0`,
  `config.py:105-106`). `design_row_j` is the annotation-driven feature row
  (variant class + user annotations + interactions). **This is the tool's core
  differentiator** (README/SPEC line 17).
- `lambda_j` — **per-variant local shrinkage** from a Three-Parameter-Beta /
  Generalized-Inverse-Gaussian (TPB/GIG) hyperprior with **class-specific tail
  weights** (`shape_a`, `shape_b`). Updated by CAVI GIG moments
  (`mixture_inference.py:13661-13689`).

The combination is assembled at `mixture_inference.py:982`
(`baseline_prior_variances = (global_scale * baseline_scales)**2`) and
`_effective_prior_variances` (`:13549-13560`):
`effective_var_j = max( (global_scale*s_j)^2 * lambda_j , 1e-8 )`.

This is exactly the estimator that is Bayes-optimal for a DGP whose true effect
variances factor as (global) × (per-class/annotation baseline) × (per-variant
heavy-tailed local scale). Section 5 makes that DGP concrete.

---

## 1. The full variant taxonomy (10 classes)

Enum `VariantClass` (`config.py:13-23`). Default per-class **log-baseline prior
scale** (`DEFAULT_CLASS_LOG_BASELINE_SCALE`, `config.py:31-42`) and **TPB shapes**
(`DEFAULT_CLASS_TPB_SHAPE_A` `:50-61`, `DEFAULT_CLASS_TPB_SHAPE_B` `:66-77`).
`exp(log-baseline)` is the actual effect-size scale on the standardized-genotype
coefficient axis.

| # | Class enum value | Structural? | log-baseline | **exp(log-baseline) = effect scale** | shape_a (tail) | shape_b |
|---|------------------|-------------|--------------|--------------------------------------|----------------|---------|
| 1 | `snv` | SNV | −4.5 | **0.01111** | 1.00 | 0.50 |
| 2 | `small_indel` | SNV-like | −4.2 | **0.01500** | 0.90 | 0.50 |
| 3 | `deletion_short` | **SV** | −3.8 | **0.02237** | 0.70 | 0.45 |
| 4 | `deletion_long` | **SV** | −3.3 | **0.03688** | 0.60 | 0.40 |
| 5 | `duplication_short` | **SV** | −3.7 | **0.02472** | 0.70 | 0.45 |
| 6 | `duplication_long` | **SV** | −3.3 | **0.03688** | 0.60 | 0.40 |
| 7 | `insertion_mei` | **SV** | −3.6 | **0.02732** | 0.65 | 0.42 |
| 8 | `inversion_bnd_complex` | **SV** | −3.1 | **0.04505** | 0.55 | 0.38 |
| 9 | `str_vntr_repeat` | **SV** | −3.5 | **0.03020** | 0.60 | 0.40 |
| 10 | `other_complex_sv` | **SV** | −3.3 | **0.03688** | 0.60 | 0.40 |

**Structural set** (`STRUCTURAL_VARIANT_CLASSES`, `config.py:79-88`): classes
3–10 (everything except `snv` and `small_indel`). Note `small_indel` is *not* in
the structural tuple, but the routing classifier (§3) treats it as SNV-like/dense.

Reading of the design intent, from the config comments (`config.py:26-77`):
- **More negative log-baseline ⇒ smaller expected effect.** SNV (−4.5, ~0.011)
  smallest; complex/inversion SVs (−3.1, ~0.045) largest — a ~4× spread in prior
  effect scale between the tightest and loosest class.
- **shape_a controls tail weight** of the local shrinkage: `1.0` (SNV) = moderate
  tails, most effects shrunk to ~0; `0.55` (inversion) = heavy tails, tolerant of
  a few large effects. `shape_b` shapes the auxiliary Gamma rate; smaller = heavier
  tails = more large effects allowed. SVs get systematically heavier tails
  ("they disrupt more DNA and are more likely to have individually detectable
  effects").
- Length gates class assignment: `SV_LENGTH_THRESHOLD = 1000.0` bp
  (`io.py:61`). `_structural_variant_class_from_token` (`io.py:4393-4404`) maps
  `DEL≥1kb→deletion_long` else `deletion_short`; `DUP/CNV` similarly; `INS/ME→
  insertion_mei`; `INV/BND→inversion_bnd_complex`; `STR/VNTR/REPEAT→
  str_vntr_repeat`; else `other_complex_sv`. PLINK biallelic SNP/indel:
  `_infer_plink_variant_class` (`io.py:4140-4146`): 1bp/1bp → `snv`, else
  `small_indel`.

These log-baselines are only **initial** values. `_initialize_scale_model`
(`mixture_inference.py:13184-13213`) seeds `scale_model_coefficients` by least-
squares so the design reproduces the per-class offsets, and initializes
`global_scale = clip(exp(mean_class_log_scale), floor, ceiling)`. The scale
model is then re-estimated each EM outer iteration (`_update_scale_model`,
`:13301+`). TPB shapes are likewise re-optimized within
`[minimum_tpb_shape=0.1, maximum_tpb_shape=10.0]` (`config.py:116-117`,
optimizer at `:13461-13529`).

### How a variant routes to its prior "class group"
`prior_class_members` / `prior_class_membership` on `VariantRecord`
(`data.py:70-71`). Default behavior (`data.py:87-89`): a variant with no explicit
membership gets `prior_class_members=(self.variant_class,)`,
`prior_class_membership=(1.0,)` — i.e. **hard 1-of-10 membership** in its own
class. A metadata file may instead supply soft/multi-class membership via the
reserved columns `prior_class_members` (comma list of class tokens) and
`prior_class_membership` (comma list of weights) — parsed at
`io.py:4216-4225`, assembled into the `class_membership_matrix` at
`mixture_inference.py:12666-12669`. Column `c` of that matrix becomes the
`type_offset::<class>` design feature (§2).

---

## 2. Annotation → prior mapping (the schema-based hypermodel)

Two stages: **(A)** column-type inference + parsing (`io.py`), which turns raw
metadata strings into typed `prior_*_features` on each `VariantRecord`; **(B)**
design-matrix compilation (`mixture_inference.py`), which turns those features
into centered columns of the scale-model design matrix whose linear predictor is
`log s_j`.

### 2A. Column-type inference (`io.py`)

`--variant-metadata` TSV/CSV is keyed by `variant_id`. **Reserved model columns**
(`VARIANT_METADATA_BASE_COLUMNS`, `io.py:63`; enumerated in README:81):
`variant_id, variant_class, chromosome, position, length, allele_frequency,
quality, training_support, is_repeat, is_copy_number, prior_class_members,
prior_class_membership`. Every other column is a **user annotation**, and its
kind is inferred from the column's values by `_infer_annotation_column_kinds`
(`io.py:4228-4254`), checked in this precedence order:

1. **binary** — all values in `{1,0,true,false,yes,no}` (`_is_bool_text`
   `:4263-4264`). Parsed to `bool` (`:3910-3915`) → `prior_binary_features`.
2. **continuous** — all values `float`-parseable (`_is_float_text` `:4267-4272`).
   → `prior_continuous_features[col] = float(value)` (`:3916-3917`). (This is how
   `length`-style numeric annotations enter, but note `length` itself is reserved;
   any *other* numeric column, e.g. `constraint`, is continuous.)
3. **weighted membership** `level=weight,...` — all values match
   `_is_weighted_levels_text` (`:4275-4285`: comma-separated `name=float` pairs).
   - if **every** level name additionally contains `>` (`_weighted_level_names_are_nested`
     `:4288-4290`) ⇒ kind **`nested_membership`** →
     `prior_nested_membership_features` (`:3928-3932`, parser `:4313-4319`).
   - else kind **`membership`** → `prior_membership_features`
     (`:3918-3922`, parser `_parse_weighted_levels` `:4293-4303`), a
     `{level_name: weight}` dict.
4. **nested** `parent>child[>grandchild]` — all values contain the delimiter
   `NESTED_PATH_DELIMITER = ">"` (`data.py:12`, check `io.py:4250-4251`). →
   `prior_nested_features[col] = (parent, child, ...)` tuple (`:3923-3927`,
   parser `_parse_nested_path` `:4306-4310`).
5. **categorical** — anything else (`:4252-4253`). Raw string →
   `prior_categorical_features` (`:3933-3934`).

Inference is over the **whole column** (all rows) — a single non-conforming value
demotes the column to a looser kind. Empty cells are skipped
(`_parse_string_feature_or_skip` `:4257-4260`), not treated as `false`/`0`.
Column interpretations are logged (`_log_annotation_column_kinds` `:3870-3878`).
README example (`README.md:89-95`): `coding`(binary), `constraint`(continuous),
`functional_state`(categorical: `lof`/`missense`), `regulatory_mix`
(membership: `enhancer=0.7,promoter=0.3`), `gene_context`(nested:
`protein_coding>exon`).

### 2B. Design-matrix compilation (`mixture_inference.py`)

`_build_prior_design` (`:12649-12693`) →
`_prior_annotation_tables` (`:12941-12955`) →
`_compile_prior_feature_specs` (`:12696-12785`). Every design column is
**mean-centered** (`_center_design_column` `:13177-13181`) and dropped if
constant (`append_if_nonzero`, threshold `1e-10`, `:12706-12714`). The feature
families and exactly how each annotation enters `log s_j = design_row @ coef`:

| Annotation kind | Design feature(s) | Column value for variant j | Penalty on coef |
|---|---|---|---|
| variant class membership | `type_offset::<class>` (one per class) | `class_membership_matrix[j, class]` (1.0 hard, or soft weight) | `type_offset_penalty = 2.0` |
| **binary** | `factor_level::<col>::true` and `::false` | indicator `{0,1}` for the level (`_factor_prior_annotation_weights` `:13001-13009`) | `scale_model_ridge_penalty = 1.0` |
| **categorical** | `factor_level::<col>::<level>` per observed level | one-hot indicator (`:13011-13028`) | 1.0 |
| **membership** (`level=weight`) | `factor_level::<col>::<level>` per level | the **weight** for that level (0.0 if absent) (`:13030-13047`) | 1.0 |
| **nested** (`parent>child`) | `nested_level::<col>::<path-prefix>` at each depth 0..len−1 | 1.0 for every prefix on the variant's path (`_nested_prior_annotation_weights` `:13051-13075`) | 1.0 |
| **nested_membership** (`a>b=w`) | `nested_level` at each depth | accumulated **weight** at each prefix | 1.0 |
| **continuous** | `continuous_spline` basis: 1 linear + cubic-hinge knots at quartiles | standardized value, then hinge `max(z−knot,0)^3` (`:13089-13154`) | 1.0 |

**Interactions with class.** For every factor level, nested level, and continuous
spline basis, an interaction feature with each variant class is added
(`factor_interaction`, `nested_interaction`, `continuous_spline_interaction`),
**only for classes with ≥3.0 total membership mass** (`class_totals` guard
`:12730,12753,12770`). Interaction column = `feature_column * class_membership`
(`:12834, :12851, :12866`). This lets an annotation shift the prior scale
*differently per variant class* (e.g. "LoF" widens the prior more for deletions
than for SNVs).

**Continuous splines in detail** (`_continuous_spline_feature_specs`
`:13089-13120`): standardize by column mean/std (skip if std<1e-8); emit a linear
term (`basis_kind="linear"`) plus one `cubic_hinge` term per interior knot at
quantiles `[0.25, 0.5, 0.75]` of the standardized values (`_continuous_spline_knots`
`:13123-13137`, dropping knots at the boundary or coincident). Basis eval at
`:13140-13155`: linear → `z`; cubic_hinge → `max(z − knot, 0)^3`. Needs ≥3
distinct values or no knots are emitted.

**Regularization / penalty** (`_scale_model_penalty` `:13563-13571`): ridge
`1.0` on all coefficients, **except** `type_offset::*` which gets the stronger
`type_offset_penalty = 2.0` — i.e. the per-class baseline offsets are shrunk
harder toward the shared global scale than the annotation coefficients are. The
scale-model solve is ridge-penalized least squares
(`_initialize_scale_model` `:13210-13212`;
`normal = D'D + diag(penalty)`), re-fit each outer iteration.

**Net effect on the prior scale:**
`s_j = exp( clip( sum over features [coef * centered_feature_value_j],
log 1e-6, log 10 ) )`, and the variant's prior sd is `global_scale * s_j *
sqrt(lambda_j)`. So annotations act multiplicatively on effect-size scale
through a log-linear model.

### 2C. IMPORTANT: which reserved fields do (not) enter the scale model — CONFIRMED

The only reserved field that enters the prior-scale design matrix is
**`variant_class`** (as the `type_offset::<class>` columns, via
`prior_class_members`/`prior_class_membership`). The other reserved metadata
fields — **`length`, `is_repeat`, `is_copy_number`, `allele_frequency`,
`quality`, `training_support`, `position`, `chromosome`** — do **NOT**
auto-become prior-scale features. Verified by construction:

- `_build_prior_design` (`mixture_inference.py:12649-12693`) builds design columns
  only from (a) `class_membership_matrix` (→ `type_offset`) and (b)
  `_prior_annotation_tables` (`:12941-12955`), which reads **exclusively** the six
  `prior_*_features` dicts (`_prior_annotation_feature_names:12958-12974`).
- Those dicts are populated **only** from non-reserved annotation columns:
  `_merge_variant_metadata` (`io.py:3905-3934`) iterates `annotation_kinds`, and
  `_infer_annotation_column_kinds` (`io.py:4232-4233`) **skips** any column in
  `VARIANT_METADATA_BASE_COLUMNS`. `length`, `is_repeat`, `is_copy_number`,
  `allele_frequency`, etc. are reserved (`io.py:63`), so they are parsed into
  the scalar `VariantRecord` attributes (`io.py:3939-3949`) and never into a
  `prior_*_features` dict.
- Therefore reserved scalars affect the model only **indirectly**:
  - `length` → chooses `_short` vs `_long` SV class (`SV_LENGTH_THRESHOLD=1000`,
    `io.py:4393-4404`), which then sets the `type_offset` baseline. It does not
    add a continuous length term to `log s_j`.
  - `is_repeat` / `is_copy_number` → **storage routing only** (dense vs sparse,
    `variant_routing.py:155-163`); no prior-scale effect.
  - `allele_frequency` / `training_support` → MAF filter and the per-class
    TPB/support bookkeeping; not a design column.

Consequence for the synthetic DGP: if you want the true generative prior to
depend on **length / repeat / copy-number**, you must expose them as *custom
annotation columns under different (non-reserved) names* (e.g. `sv_length_bp` as a
continuous annotation, `repeat_overlap` as a binary annotation) so they enter the
scale model — OR bake their effect entirely into the per-class `type_offset`
baseline. The README's documented annotation semantics (§2A) are the only channel
through which continuous length / repeat status reach `log s_j`. (Note this makes
README line 17's phrase "length, repeat status" slightly aspirational: those enter
the prior *only* if provided as custom annotation columns, not via the reserved
`length`/`is_repeat` fields.)

### What the synthetic metadata table must contain
To exercise every path, the synthetic `variant_metadata.tsv` should have, per
variant: `variant_id`, `variant_class` (one of the 10 tokens), `length`,
`allele_frequency`, `is_repeat`, `is_copy_number`, plus annotation columns of
each kind — e.g. `coding` (binary), `constraint` (continuous in [0,1]),
`functional_state` (categorical: lof/missense/synonymous), `regulatory_mix`
(membership `enhancer=w1,promoter=w2`), `gene_context` (nested
`protein_coding>exon`). The **true** per-variant prior scale in the DGP must be
generated as `s_j = exp(class_offset + Σ annotation_coef·feature)` using the same
feature encoding — that is what makes the model Bayes-optimal (§5).

---

## 3. Marginal screening (the optional |z| pre-screen) — and why the genuine path skips it

### Two distinct "screening" mechanisms exist; do not conflate them.

**(a) Dense/sparse *storage* routing** (`variant_routing.py`,
`sparse_screening.py`). `classify_variants` (`variant_routing.py:93-191`) sends a
variant to the sparse carrier-list representation iff it is structural-ish
(class prefix in `{deletion, duplication, mobile_element, insertion_mei,
inversion}`, or `is_copy_number`, `_is_structural_class` `:81-90`) **or** repeat-
flagged, **and** its carrier count ≤ threshold (`n_samples // 64` default,
`:135-139`). This is purely a *storage/compute* decision — the same Bayesian
model still fits every variant. `sparse_screening.compute_sparse_marginal_z`
(`sparse_screening.py:53-142`) is the O(|carriers|) marginal-z used when a rare
variant lives in the sparse rep; mathematically identical to the dense z.

**(b) The optional marginal |z| pre-screen** (`config.marginal_screen_min_abs_z`,
`config.py:145-161`; applied in `model.py:2373-2511`). **Default `0.0` ⇒ screen
disabled** — no variant is dropped for weak marginal signal. When set > 0:
- Compute per-variant covariate-residualized z (`compute_marginal_z_scores`,
  `preprocessing.py:357-409`):
  `z_j = (X_j_std^T y_resid) / sqrt( sigma2_resid * (n − x_j' C(C'C)^{-1} C' x_j) )`
  (`marginal_z_from_numerator` `:304-336`), where `y_resid` is `y` OLS-residualized
  on the covariate columns. Under the null (no association after covariate
  adjustment) `z_j ≈ N(0,1)`; for binary `y` it is a Rao-score statistic with the
  same null (`:363-372`).
- Keep variants with `|z_j| ≥ threshold` (`model.py:2497`,
  `z_pass_mask = |z| >= min_abs_z`).
- Threshold semantics (config docstring `:151-157`): `1.5` = Φ⁻¹(0.13), drops
  ~87% of pure-noise variants (common PRS "marginal-then-joint" practice);
  `2.0` ≈ per-variant p<0.05, drops ~95% of nulls (aggressive; risks losing
  small-effect signal).
- Guard: if the screen would keep **0** variants it is auto-disabled to avoid a
  covariates-only fit on a mis-calibrated z (`model.py:2499-2503`).

**SV-protection rule.** `config.marginal_screen_protect_sv = True`
(`config.py:161`) is the declared intent that structural variants be exempted
from the |z| screen — because "rare SVs and correlated-region signals can have
weak marginal z-scores but matter in the joint model" (config docstring
`:150-152`). **Important finding: in the current tree this flag is declared and
validated but NOT consulted by the screen application code** — a repo-wide grep
finds `protect_sv` only at `config.py:161` (no read site in `model.py`/
`mixture_inference.py`). So today the screen, when enabled, applies uniformly to
all classes; SV-protection is an intended-but-unwired contract. For the RL
reimplementation the *correct* behavior to encode is: when the screen is on, SVs
(the `STRUCTURAL_VARIANT_CLASSES` set) bypass the `|z|` threshold and always enter
the joint fit.

### Why single-pass joint fitting is the "genuine" path (SPEC.md)

SPEC.md line 23: **"One model, one inference pass. Every variant goes through the
same Bayesian model with the same prior structure. No two-stage pipelines, no
'background' models for some variants and 'exact' models for others... Computational
shortcuts (working sets, stochastic blocks) are optimizations that must produce
the same result as the full joint model."** SPEC line 20: **"No holdout splits or
cross-validation to do the fit itself... The Bayesian prior is the regularizer —
all samples train the model."** SPEC line 17 forbids a generic uniform-penalty
LASSO/elastic-net as the primary backend.

Consequence for the benchmark: the pre-screen is a **budget/optimization escape
hatch** (config docstring `:158-159`: "Set on the runner/CLI when the joint matrix
would otherwise exceed the GPU budget"), default off. The genuine estimator is the
**joint** global–local shrinkage fit that de-correlates LD-linked variants inside
one inference pass. A two-stage marginal-screen-then-fit, or a P+T (prune+threshold
on marginal z) pipeline, is the **cheat** the environment should be able to
distinguish from the genuine joint fit (§4, §5).

---

## 4. LD block handling

### Structure (`ld_blocks.py`, `ld_block_partition.py`)
- Embedded **Berisa–Pickrell EUR LDetect blocks**, 1703 regions, shipped as
  `sv_pgs/_data/EUR_hg38.tsv`, BED-style half-open `[start,end)` per
  `(chrom_int, start, end)` (`ld_blocks.py:1-16, 114-157`). Only `EUR`/`hg38`
  are bundled (`_resource_path` `:71-80`); config validates
  `ld_block_population ∈ {EUR,AFR,EAS,AMR}` and build `∈ {hg19,hg38}`
  (`config.py:290-299`) but non-EUR/hg38 raise `NotImplementedError`.
- `assign_ld_blocks` (`:160-236`): each variant `(chrom,pos)` is mapped by
  `searchsorted` to the rightmost block with `start ≤ pos` and `pos < end`;
  variants outside any block become **unique singleton blocks** (id ≥ n_blocks,
  one per unmapped variant) so they are never spuriously grouped. `block_partition`
  (`:239-263`) inverts to `{block_id: sorted variant indices}`.
- `LdBlockPartition` (`ld_block_partition.py:28-85`) bundles the assignment +
  partition + a SHA-256 content signature used in the fit-stage cache key.

### How blocks are *used*
`config.use_ld_blocks` is **opt-in, default False** (`config.py:188`). It is a
**computational decomposition**, not a modeling change: when True, the sample-
space genotype matvec is split into per-block matmuls sharded across all visible
CUDA GPUs (`config.py:177-192` docstring; `GPUScheduler`). Per SPEC line 23 the
result must be **identical** to the single monolithic joint solve. The joint model
already de-correlates variants regardless of blocks — blocking is only a
memory/throughput tiling of the same linear algebra. Singleton chunking for
unmapped/isolated variants is controlled by `ld_block_singleton_chunk_size=256`,
pipeline depth `2` (`config.py:191-192`). Smoke coverage:
`tests/test_ld_block_smoke.py`.

### What LD structure the synthetic genotypes need
The whole point of the joint fit is to beat a marginal / P+T cheat by resolving
correlated signal. So the synthetic genotypes must contain **real LD**:
- Group variants into blocks (mirror the Berisa–Pickrell notion: contiguous
  genomic windows). Within a block, generate genotypes with a realistic
  correlation matrix (e.g. AR(1) `corr(i,k)=rho^{|i−k|}` with `rho∈[0.6,0.95]`,
  or a haplotype/factor model), so several variants tag the same latent signal.
- Place a small number of **causal** variants inside LD blocks alongside many
  correlated non-causal tag variants. A marginal screen / P+T will mis-rank tags
  vs. causals; the joint global–local fit will assign the effect to the causal and
  shrink the tags — this is the gap the environment rewards.
- Also include **isolated** (singleton) variants and cross-class LD (an SV in LD
  with SNP tags) to exercise singleton blocks and the SV-vs-SNP de-correlation
  that motivates SV-protection (§3).
- Chromosome/position must be real coordinates so `assign_ld_blocks` maps them
  (or accept singletons). For a fair synthetic env you can define your own block
  table rather than the EUR panel, but the genotype correlation must respect it.

---

## 5. The precise "perfect-but-fair" DGP for SV-PGS

A generative process for which this exact estimator is the Bayes-optimal /
maximum-likelihood-consistent one. All numbers are grounded in the config
defaults so the model's initialization is already near-truth (fair), yet the
posterior still has to do real work (de-correlation + local shrinkage +
scale-model + TPB-shape estimation).

### 5.1 Variants, classes, MAF spectrum
- `M` variants (scale to whatever the env needs; the tool caps nothing — SPEC
  lines 15-16). Assign each a class by the empirical mix, e.g. ~85% `snv`, ~5%
  `small_indel`, and the remaining ~10% spread over the 8 SV classes, so SVs are
  a rare-but-consequential minority (mirrors AoU 700k SNP + 1.7M SV framing,
  README:22-26).
- **MAF**: draw SNV/indel MAF from a realistic rare-skewed spectrum, e.g.
  `MAF ~ Beta(0.3, 8)` truncated to `[minimum_minor_allele_frequency=1e-2, 0.5]`
  (`config.py:145`, the model's MAF floor). Give **SVs a rarer** spectrum
  (`MAF ~ Beta(0.2, 20)`, many carriers ≤ `n//64` so they route to the sparse
  rep, `variant_routing.py:135-139`) and include a tail of **very rare** SVs
  ("very rare SVs will be filtered", README line 4) so the MAF filter and SV
  carrier routing both fire.
- **Length**: SV length drives class (`SV_LENGTH_THRESHOLD=1000bp`); draw
  `log10(length) ~ Uniform(2,6)` for SVs so both `_short` (<1kb) and `_long`
  (≥1kb) classes populate.

### 5.2 LD blocks (see §4)
Partition variants into blocks of ~5–50; within-block genotype correlation
AR(1) `rho ~ Uniform(0.6, 0.95)`; ~1 causal per causal-block plus correlated
tags; include singleton variants; include occasional SV–SNP cross-class LD.
Simulate genotypes as correlated dosages then round/threshold to `{0,1,2}` at the
target MAFs (or a liability/haplotype model), then standardize as the model does.

### 5.3 True effect-size prior (the crux — matches §1–§2 exactly)
For each variant `j`, build the **true** prior scale from the same log-linear
scale model the tool uses:

```
log s_j^true = class_offset[class_j]                       # the per-class baseline
             + Σ_a  gamma_a  * f_a(annotation_{j,a})        # annotation main effects
             + Σ_a  gamma_{a,class_j} * f_a(...)            # a few class×annotation interactions
```
- `class_offset` = the config `DEFAULT_CLASS_LOG_BASELINE_SCALE` values
  (`config.py:31-42`): SNV −4.5 (exp 0.0111) … inversion −3.1 (exp 0.0451).
  Using these makes the model's initialization unbiased.
- Feature encodings `f_a` **must equal §2B**: binary → indicator; categorical →
  one-hot; membership → the weight; nested → prefix indicators at each depth;
  continuous → standardize then linear + cubic-hinge at quartile knots.
- Draw annotation coefficients `gamma_a ~ N(0, 0.5^2)` (modest, so annotations
  move `log s` by O(0.5) — a ~1.6× multiplicative effect), and a handful of
  class-interaction coefficients (only for classes with ≥3 members, matching the
  `class_totals ≥ 3` gate `:12730`).
- Clip `log s_j^true` to `[log 1e-6, log 10]` to stay inside the model's
  representable band (`config.py:105-106`).

Then draw the **local shrinkage** from the class-specific TPB/GIG so most effects
are ~0 and a few are large, per class tail weight:
```
lambda_j ~ TPB( shape_a[class_j], shape_b[class_j] )      # config.py:50-77
```
Operationally: a Three-Parameter-Beta local scale whose tail is heavier for SVs
(`shape_a` 0.55–0.70) than SNVs (`shape_a=1.0`). A faithful sampler: draw
`delta_j ~ Gamma(shape_b, 1)`, then `lambda_j ~ Gamma(shape_a, delta_j)` (the
GIG/Gamma-Gamma hierarchy the CAVI update inverts, `mixture_inference.py:13657-13689`),
giving a horseshoe-like heavy-tailed local scale.

Finally the **true effect**:
```
global_scale_true  ~  around exp(mean class_offset) ∈ [1e-4, 10]   # e.g. ~0.02
beta_j  =  global_scale_true * s_j^true * sqrt(lambda_j) * N(0,1)
```
Most `beta_j ≈ 0`; a sparse minority (heavier-tailed SV classes preferentially)
carry detectable effects — exactly what the heavy-tailed class-specific TPB prior
expects. This is the sparsity+heterogeneity structure the estimator is optimal for.

### 5.4 Covariates / ancestry-PC confounding
Mirror the AoU covariate block (README:45): `age, age^2, sex, race/ethnicity
indicators, PC1..PC10`. Generate ancestry PCs and make **both** genotype MAFs
(population structure) **and** the phenotype depend on the PCs, so there is real
confounding the covariate block must absorb (the marginal-z residualizes on
covariates, `preprocessing.py:387-389`; the joint fit conditions on them via
`alpha`). Give age/sex modest true effects. This ensures a genotype-only or
PC-naive scorer is biased, rewarding correct covariate adjustment.

### 5.5 Phenotype links
- **Quantitative**: `y = C·alpha + G_std·beta + eps`,
  `eps ~ N(0, sigma_e^2)` with `sigma_e^2 ≥ sigma_error_floor=1e-3`
  (`config.py:104`). Choose `sigma_e^2` to hit a target heritability (e.g.
  `h^2 = Var(G_std·beta)/Var(y) ∈ [0.1, 0.5]`). Linear-Gaussian likelihood is
  exactly the model's quantitative path (REML global scale).
- **Binary**: liability/logistic — `logit P(y=1) = C·alpha + G_std·beta`, sample
  Bernoulli. This is the model's Pólya-Gamma logistic path (`trait_type=BINARY`
  default, `config.py:99`). Keep case rate away from 0/1 (e.g. prevalence
  10–30%) so the logistic isn't near-separable (the calibration at
  `mixture_inference.py:13277-13281` assumes a conservative logit-scale variance
  ~0.04). Use the same `beta` and covariates as the quantitative arm for paired
  evaluation.

### 5.6 Why this is "fair" and what it discriminates
- **Bayes-optimal**: the true prior is literally `(global × metadata-baseline)²
  × TPB-local` with class-specific tails — the model's exact hierarchy. Its
  default class log-baselines and TPB shapes initialize at the truth, so the
  reimplementation is being tested on inference quality, not luck.
- **Not trivial**: LD (§5.2) forces real joint de-correlation; a marginal /
  P+T / uniform-LASSO scorer mis-assigns effects among LD tags and ignores
  class/annotation-specific priors (SPEC line 17 explicitly rules these out as a
  primary backend). The annotation-driven per-variant scale means a uniform-
  penalty method is provably sub-optimal.
- **SV-sensitive**: rare, heavy-tailed SV effects with weak marginal z are exactly
  the signals a |z| pre-screen would discard (§3) — so the "genuine" single-pass
  joint fit (screen off, SVs protected) should beat any two-stage cheat, giving
  the RL environment a clean oracle-vs-cheat gap.

---

## Key file:line index
- Classes / prior scales / TPB shapes / floors: `config.py:13-88, 99-161`.
- Effective prior variance assembly: `mixture_inference.py:982`,
  `_effective_prior_variances:13549-13560`, `_metadata_baseline_scales_from_coefficients:13533-13546`.
- Scale-model design build: `_build_prior_design:12649`,
  `_compile_prior_feature_specs:12696`, `_column_for_feature_spec:12809`,
  splines `:13089-13155`, penalty `_scale_model_penalty:13563`.
- Annotation type inference/parsing: `io.py:4228-4319`; merge `:3881-3958`;
  reserved cols `io.py:63`; `VariantRecord` `data.py:46-92`.
- Local TPB/GIG shrinkage: `mixture_inference.py:13629-13689`,
  `_build_cavi_correct_prior_precision:13692-13727`.
- Marginal screen: `config.py:145-161`; `model.py:2373-2511`;
  `preprocessing.py:304-409`; sparse-z `sparse_screening.py:53-142`.
- Storage routing: `variant_routing.py:81-191`.
- LD blocks: `ld_blocks.py`, `ld_block_partition.py`; wiring flags
  `config.py:177-192`.
- Contract: `SPEC.md` (lines 17, 20, 23), `README.md:81-95`.
