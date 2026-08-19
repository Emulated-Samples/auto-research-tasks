# SV-PGS Core Inference Engine — Reverse-Engineered Specification

Scope: the empirical-Bayes variational inference engine that fits a Bayesian
polygenic score with a **class-aware, annotation-regressed, three-parameter-beta
(TPB) local-shrinkage prior**. This document is the ground truth for the
synthetic-data generator and the anti-cheat gates of the RL environment.

Primary sources (all paths under `/Users/user/SV-PGS/sv_pgs/`, read-only):
- `mixture_inference.py` (~14.2k lines) — the engine.
- `elbo.py` — the true variational ELBO.
- `config.py` — hyperparameters, floors/ceilings, per-class priors.
- `data.py` — `VariantRecord` (the annotation carrier).
- `optimizer_helpers.py` — closed-form σ_g and GIG inverse moment.
- `anderson.py` — safeguarded Anderson(m) acceleration.

Notation used throughout:
- `n` = samples, `p` = variants (after MAF/tie reduction), `k` = covariates.
- `X` = standardized genotype matrix (columns standardized; convention
  `‖X[:,j]‖² = n`, see `elbo.py:113-114`, `_quantitative_posterior_state` trace term).
- `W` = covariate design (n×k), `α` = covariate effects, `β` = per-variant effects.
- `η = offset + Wα + Xβ` = linear predictor.

---

## 1. The exact generative / probabilistic model

The engine assumes a **Gaussian scale-mixture (global–local shrinkage) linear
model** with a log-linear regression of the local prior scale on variant
annotations. The full hierarchy, extracted from the prior-construction code and
the ELBO/objective terms:

### 1.1 Likelihood

**Quantitative trait** (`TraitType.QUANTITATIVE`), `elbo.py:136-145`,
`_quantitative_posterior_state` `mixture_inference.py:4501-4524`:

```
y_i | β, α ~ N( (Wα)_i + (Xβ)_i , σ_e² )
```

with residual variance `σ_e²` (`sigma_error2`), floored at
`config.sigma_error_floor = 1e-3`.

**Binary trait** (`TraitType.BINARY`), logistic likelihood with
**Polya-Gamma / Jaakkola-Jordan** augmentation, `elbo.py:146-166`,
`_binary_penalized_log_posterior` `mixture_inference.py:4629-4642`:

```
y_i | β, α ~ Bernoulli( σ(η_i) ),   σ(z) = 1/(1+e^{-z})
log p(y|η) = Σ_i [ y_i η_i − softplus(η_i) ]
```

For inference the logistic likelihood is lower-bounded by the JJ/PG bound with
variational parameter `ξ_i`; the ELBO uses the **tight** choice
`ξ_i² = E_q[η_i²] = μ_i² + Var_q[η_i]` so the quadratic remainder vanishes,
leaving `log σ(ξ_i) + κ_i μ_i − ξ_i/2`, `κ_i = y_i − 1/2` (`elbo.py:155-166`).
`σ_e²` is fixed at `1.0` for binary (`mixture_inference.py:4393`).

Covariates `W` (intercept, age, sex, PCs) are **not shrunk** — `α` is a free
GLS/logistic parameter; only `β` carries the prior. `α` is profiled out via a
generalized-least-squares (GLS) inner solve (§2.1).

### 1.2 Effect prior (the global–local scale mixture)

Each variant effect is a zero-mean Gaussian whose variance factorizes into a
global scale, a per-variant **annotation-predicted baseline scale**, and a
**local shrinkage factor** (`elbo.py:82-83`, `_effective_prior_variances`
`mixture_inference.py:3356-3360`, `13549-13560`):

```
β_j ~ N( 0 , τ_j² )
τ_j²  = ( σ_g · s_j )² · λ_j                       (the "reduced_prior_variance")
      = baseline_prior_variance_j · λ_j
baseline_prior_variance_j = ( σ_g · s_j )²
```

- `σ_g` — **global scale** (`global_scale`), clipped to
  `[global_scale_floor=1e-4, global_scale_ceiling=10]`.
- `s_j` — **metadata baseline scale**, `s_j = exp(clip(x_jᵀθ, log 1e-6, log 10))`
  where `x_j` is variant `j`'s annotation feature row and `θ` the scale-model
  coefficients (`_metadata_baseline_scales_from_coefficients`
  `mixture_inference.py:13533-13546`). This is the "scale model" of §3.
- `λ_j` — **local shrinkage** with the TPB prior below.
- `τ_j²` floored at `1e-8`; `λ_j` floored at `local_scale_floor=1e-8`.

### 1.3 The three-parameter-beta (TPB) local-shrinkage prior

The TPB shrinkage is implemented as a **two-level Gamma–Gamma (normal-gamma-gamma)
hierarchy** with an auxiliary rate variable `δ_j`. Read directly off the exact
prior log-density `_local_scale_prior_objective` (`mixture_inference.py:13629-13642`):

```
log p(λ,δ|a,b) = Σ_j [ a_j·log δ_j − lnΓ(a_j) + (a_j−1)·log λ_j − δ_j·λ_j ]   # Gamma(λ_j; shape a_j, rate δ_j)
               + Σ_j [ −lnΓ(b_j) + (b_j−1)·log δ_j − δ_j ]                     # Gamma(δ_j; shape b_j, rate 1)
```

i.e.

```
λ_j | δ_j ~ Gamma( shape = a_{c(j)} , rate = δ_j )
δ_j       ~ Gamma( shape = b_{c(j)} , rate = 1 )
```

Marginalizing `δ_j` yields a **Gamma mixed over its rate by a Gamma**, which is
a scaled **Beta-prime / generalized-beta-of-the-second-kind** density on `λ_j`
— this is the "three-parameter beta" (TPB) shrinkage prior of the
Armagan–Dunson–Clyde "generalized beta mixtures of Gaussians" family. Composed
with the Gaussian `β_j | λ_j`, the marginal `p(β_j)` is a global–local
shrinkage density that **interpolates the horseshoe** (`a=b=1/2`) and lighter/
heavier variants:

- **`a` (`shape_a`) controls the right tail** of `λ` (probability of *large*
  effects). Smaller `a` → heavier tail → more tolerance for large effects.
  Config defaults `DEFAULT_CLASS_TPB_SHAPE_A` (`config.py:50-61`): SNV `a=1.0`
  (light, aggressive shrink), inversion/complex SV `a=0.55` (heavy tail).
- **`b` (`shape_b`) controls the auxiliary-rate / the pole at zero** (mass of
  near-zero effects). Config defaults `DEFAULT_CLASS_TPB_SHAPE_B`
  (`config.py:66-77`): SNV `b=0.5`, SVs `b≈0.38–0.45`.

`(a,b)` are **per variant class**, expanded to per-variant via the class
membership matrix: `a_j = (M a)_j`, `b_j = (M b)_j` where `M` is the (possibly
soft/fractional) class-membership matrix (`mixture_inference.py:3560-3561`,
`1775-1776`). Membership can be fractional (a variant can belong to multiple
prior classes with weights summing over `prior_class_membership`).

### 1.4 Hyperprior on the scale model and the shapes

- Scale-model coefficients `θ` carry a **Gaussian ridge** (Normal prior),
  contributing `−½ Σ penalty_m θ_m²` to the objective (`_scale_penalty_objective`
  `mixture_inference.py:13574-13578`). Per-feature penalty is
  `scale_model_ridge_penalty=1.0`, except **type-offset (per-class intercept)
  features get `type_offset_penalty=2.0`** (`_scale_model_penalty`
  `13563-13571`). The global-scale intercept (`θ_0 = log σ_g`) is **unpenalized**
  (`penalty_vector[0]=0`, `13348`).
- Class shapes `(log a, log b)` carry a **hierarchical Gaussian shrink toward
  their cross-class mean**: `−½ (‖a-centered‖² + ‖b-centered‖²)/τ_hier` with
  `tpb_hierarchical_prior_variance = 1.0` (`_update_tpb_shape_vectors`
  `13465-13470`). Shapes bounded to `[minimum_tpb_shape=0.1, maximum_tpb_shape=10]`.

### 1.5 Covariate / annotation handling

- Covariates `W`: free, GLS-profiled; binary intercept re-calibrated each
  iteration (`_apply_binary_intercept_calibration`, `_calibrate_binary_intercept`).
- Annotations enter **only** through the scale-model design (§3), built from the
  `VariantRecord` prior-feature dicts. `VariantRecord` (`data.py:47-71`) carries
  `variant_class`, `length`, `allele_frequency`, `is_repeat`, `is_copy_number`,
  plus generic `prior_{binary,continuous,categorical,membership,nested}_features`.
  `variant_class` always yields a `type_offset` column; other annotations (e.g.
  length as continuous, repeat status as binary) are consumed if present in the
  `prior_*` dicts.

---

## 2. Variational posterior family and CAVI/SVI updates

Variational family: **mean-field factorized**

```
q(β,α,λ,δ) = N(α; α̂, ·) · N(β; m, Σ) · Π_j GIG(λ_j) · Π_j Gamma(δ_j)
```

with `Σ` treated as **diagonal** (`beta_variance` is a length-p diagonal). The
second moment `E_q[β_j²] = m_j² + Σ_jj` is the sufficient statistic threaded
through every hyper-update (`reduced_second_moment`, `mixture_inference.py:3424`).

The ELBO that every block update provably does not decrease is in `elbo.py`
(`compute_elbo`): expected log-lik + Gaussian effect-prior term
`−½Σ log τ_j² − ½Σ E[β_j²]/τ_j²` + entropy `½Σ log Σ_jj` + TPB prior term
+ scale-ridge term.

### 2.1 β mean and variance (the collapsed/GLS block)

The engine **collapses (marginalizes) β analytically** and solves in whichever
space (sample or variant) is cheaper. Sample-space covariance
(`_solve_restricted_full` `mixture_inference.py:11610-11644`):

```
V = σ_e²·I_n + X · diag(τ²) · Xᵀ            (n×n, "covariance_matrix")
α̂  = ( Wᵀ V⁻¹ W )⁻¹ Wᵀ V⁻¹ y              (GLS for covariates)
projected_targets = V⁻¹ y − V⁻¹ W α̂
m = β̂ = diag(τ²) · Xᵀ · projected_targets   (collapsed posterior mean)
```

Equivalently in variant space the posterior is
`(Xᵀ D X + diag(1/τ²)) m = Xᵀ D (y−Wα̂)` with `D = I/σ_e²` (quantitative) or the
PG working weights (binary). Posterior variance `Σ_jj` = diagonal of that inverse,
estimated by Hutchinson probes / low-rank residual correction
(`posterior_variance_probe_count=24`, `_posterior_variance_low_rank_residual_diagonal`).

**CAVI-correct precision override.** Instead of plugging in point `τ_j²`, the
engine can use the **expected inverse local scale** so the β block is the exact
CAVI update after collapsing `λ` (`_build_cavi_correct_prior_precision`
`mixture_inference.py:13692-13727`):

```
prior_precision_j = E_q[1/λ_j] / baseline_prior_variance_j
                  = E_q[1/λ_j] / (σ_g² s_j²)
```

with `E_q[1/λ_j]` from the GIG inverse-first-moment (§2.2).

### 2.2 Local shrinkage λ and auxiliary δ (GIG block)

`q(λ_j)` is **Generalized Inverse Gaussian**. Combining the Gaussian
`β_j|λ_j` factor (`λ^{-1/2} exp(−β_j²/(2 baseline_j λ_j))`) with the Gamma prior
`λ^{a-1} exp(−δ_j λ_j)` gives `GIG(p, χ, ψ)` with
(`_update_local_scales` `mixture_inference.py:13661-13689`):

```
p_j = a_j − 1/2
χ_j = E_q[β_j²] / baseline_prior_variance_j        ( = reduced_second_moment / (σ_g s_j)² )
ψ_j = 2 δ_j
```

CAVI moment updates use ratios of modified Bessel functions `K_ν` (via
exponentially-scaled `kve`; scaling cancels in the ratio):

```
E[λ_j]   = √(χ/ψ) · K_{|p+1|}(√(χψ)) / K_{|p|}(√(χψ))          (_gig_moment, 13741-13759)
E[1/λ_j] = √(ψ/χ) · K_{p-1}(√(χψ)) / K_{p}(√(χψ))              (gig_inverse_first_moment, optimizer_helpers.py:47-139)
```

`updated_local_scale = E[λ_j]` (floored at `1e-8`). The GIG `E[1/λ]` helper has
carefully coded small-`z`/large-`z` asymptotics (inverse-gamma limit
`−2p/χ` for `p<0`, etc.) to avoid `0/0` and overflow.

**Auxiliary δ** is Gamma with CAVI update `δ_j|λ_j ~ Gamma(a_j+b_j, 1+E[λ_j])`,
mean (`mixture_inference.py:13685-13688`, `3629`):

```
δ_j ← (a_j + b_j) / (1 + E[λ_j])
```

Initialized `λ_j = 1`, `δ_j = b_j` (`mixture_inference.py:1774-1777`).

### 2.3 Noise variance σ_e² (quantitative only)

Exact ELBO stationary point (`_quantitative_posterior_state`
`mixture_inference.py:4511-4515`):

```
σ_e²_new = ( ‖y − η‖²  +  tr(X Σ Xᵀ) ) / n ,   tr(X Σ Xᵀ) = n · Σ_j Σ_jj   (standardized cols)
```

floored at `sigma_error_floor`. (Comment notes the older leverage proxy was only
correct at convergence; the `RSS + trace` form is exact.)

### 2.4 Polya-Gamma weights (binary block)

`ω_i = E[PG(1, η_i)]` mean-field weights (`_binary_expected_polya_gamma_weights`
`mixture_inference.py:4615-4626`):

```
ω_i = tanh(|η_i|/2) / (2|η_i|)          (→ 1/4 as η_i → 0)
```

floored at `polya_gamma_minimum_weight=1e-4`. The binary β/α block is a
penalized-logistic Newton / PG-IRLS solve maximizing
`Σ_i[y_i η_i − softplus(η_i)] − ½ Σ_j prior_precision_j β_j²`
(`_binary_penalized_log_posterior` `4629-4642`), with `prior_precision = 1/τ²`
or the CAVI E[1/λ] override. Working response uses `κ_i = y_i − 1/2`.

### 2.5 SVI / stochastic path (large p)

When the genotype matrix is large and streamed (`_should_use_stochastic_variational_updates`),
variants are split into **disjoint deterministic blocks** (not random subsamples,
`_stochastic_variant_blocks` `1328-1343`). Each block runs a collapsed β update;
`reduced_second_moment` is blended per block with step size, then per-block λ/δ
updates run (`mixture_inference.py:1225-1252`). Because blocks are disjoint and
one epoch is one full pass, the Robbins-Monro **step size is 1.0** per epoch
(`_stochastic_step_size` `1274-1277`); damping is per-epoch. Hyperparameters
update once per epoch under the same even-iteration gate.

---

## 3. The scale-model hyper-regression

The scale model is a **log-linear (log-normal) regression of the per-variant
prior scale on variant annotations**, `s_j = exp(x_jᵀ θ)`. Its purpose: let
metadata (variant class, length, repeat status, arbitrary annotations) predict
effect magnitude, learned jointly with the effects.

### 3.1 Design matrix (`_build_prior_design`, `_compile_prior_feature_specs`)

Feature columns, all **mean-centered** (`_center_design_column` `13177-13181`);
near-constant columns dropped (`append_if_nonzero`). Built in this order
(`mixture_inference.py:12716-12785`):

1. **`type_offset::<class>`** — one column per variant class = its (fractional)
   class-membership weight. Per-class intercept of log effect scale.
2. **Factor levels** `factor_level::<src>::<level>` for binary/categorical/
   membership annotations, reference level = most-frequent (dropped,
   `_factor_levels_to_encode` `13079-13086`). Binary features expand to
   `false`/`true` indicator weights.
3. **Factor × class interactions** `factor_interaction::…` (only for classes with
   ≥3 total membership).
4. **Nested-annotation levels + interactions** `nested_level` / `nested_interaction`
   (hierarchical path annotations, per depth).
5. **Continuous splines** `continuous_spline::<src>::basis_k`: a standardized
   **linear** term plus **cubic-hinge** basis functions `max(z−knot,0)³` at
   interior quartile knots (`_continuous_spline_feature_specs` `13089-13120`,
   `_continuous_spline_basis_column` `13140-13155`), plus their class interactions.

So `x_jᵀθ` is a centered, class-aware, spline-flexible predictor of `log s_j`.
The global level is carried separately by `σ_g` (the augmented design's intercept
`θ_0 = log σ_g`).

### 3.2 Objective and Newton update (`_update_scale_model` `13301-13394`)

Let `expected_scale_j = E_q[β_j²] / λ_j` (=`reduced_second_moment/local_scale`,
floored). The engine augments the design with an intercept column
`[1 | design]`, `θ = [log σ_g ; coefficients]`, and minimizes the **expected
negative Gaussian-prior log-density in the log-scale** plus ridge:

```
F(θ) = ½ Σ_j [ expected_scale_j · exp(−2 x̃_jᵀθ) + 2 x̃_jᵀθ ]  +  ½ Σ_m penalty_m θ_m²
grad = X̃ᵀ (1 − w)            + penalty ⊙ θ ,   w_j = expected_scale_j·exp(−2 x̃_jᵀθ)
Hess = 2 X̃ᵀ diag(w) X̃       + diag(penalty)
```

Solved by **damped Newton with backtracking line search** (`maximum_scale_model_iterations=8`;
`penalty_vector[0]=0` for the intercept). `F` is convex in `θ`. A **bounded
contraction** clips the downward move of `log σ_g` to ≤0.25 nats (~22%) per outer
EM call (`13381-13392`) to stop `σ_g` collapsing to the noise floor before the
λ block identifies signal; upward moves unclipped. Zero-feature case reduces to
the closed form `log σ_g = ½ log mean(expected_scale)` with the same cap.

Ridge penalties: `scale_model_ridge_penalty=1.0` for all features,
`type_offset_penalty=2.0` for the per-class intercepts (heavier — keeps class
offsets from over-fitting), `1e-8` floor.

### 3.3 Class TPB shapes update (`_update_tpb_shape_vectors` `13435-13530`)

Empirical-Bayes on `(a,b)` by **L-BFGS-B ascent on the marginal log-likelihood
of the local scales** (the Gamma–Gamma terms of §1.3), optimizing in `log`-space
with box bounds `[log 0.1, log 10]`:

```
obj(a,b) = Σ_j [ a_j log δ_j − lnΓ(a_j) + (a_j−1) log λ_j ]
         + Σ_j [ (b_j−1) log δ_j − lnΓ(b_j) ] + hierarchical_penalty
score_a = log δ_j − ψ(a_j) + log λ_j        (ψ = digamma)
score_b = log δ_j − ψ(b_j)
```

Class-level gradients pull the per-variant scores back through `Mᵀ`; the
hierarchical Gaussian penalty shrinks classes to their common mean. The
stationary condition is `ψ(a_c) = mean_j∈c(log δ_j + log λ_j)` (up to the
hierarchical pull) — a textbook Gamma-shape MLE fixed point.

---

## 4. Empirical-Bayes loop: what is learned vs fixed

**Learned from data (this is what "prior recovery" must grade):**
- `global_scale` σ_g — Newton (§3.2). Initialized from class defaults then
  data-calibrated by a marginal-correlation screen (`_calibrate_initial_global_scale`
  `13216-13296`): matches target predictor variance to `p · mean(s²)`.
- `scale_model_coefficients` θ — Newton (§3.2). Initialized by a ridge LS fit of
  the centered per-class default log-baselines (`_initialize_scale_model` `13184-13213`).
- `tpb_shape_a_vector`, `tpb_shape_b_vector` (per class) — L-BFGS-B (§3.3).
  Initialized from `DEFAULT_CLASS_TPB_SHAPE_A/B`.
- `local_scale` λ_j, `auxiliary_delta` δ_j — GIG/Gamma CAVI (§2.2), every iter.
- `sigma_error2` σ_e² — closed form (§2.3), quantitative only.
- `α`, `β`, `beta_variance` — collapsed block every iter.

**Fixed (not updated):** all `config.py` structural constants — floors/ceilings
(`prior_scale_floor 1e-6`, `prior_scale_ceiling 10`, `global_scale_floor 1e-4`,
`global_scale_ceiling 10`, `local_scale_floor 1e-8`), the ridge penalties
(`scale_model_ridge_penalty`, `type_offset_penalty`, `tpb_hierarchical_prior_variance`),
shape bounds, and the *structure* of the class-membership / annotation design.
The `DEFAULT_CLASS_*` maps are **initializations only** ("The model updates them
during fitting", `config.py:29`).

**Update schedule / order per outer EM iteration** (`fit_variational_em`,
deterministic path `mixture_inference.py:3308-3860`):
1. Build `τ² = (σ_g s_j)² λ_j`; optionally the CAVI E[1/λ] precision override.
2. Collapsed β/α/σ_e² solve → `reduced_second_moment = m² + Σ`.
3. Record objective + true ELBO.
4. λ,δ GIG/Gamma update.
5. **Hyperparameters (σ_g, θ, a, b) on every iteration ≥2**
   (`_should_update_hyperparameters_this_iteration` `1295-1302`), with an
   **ELBO safeguard**: candidate hyperparameters are reverted if the ELBO would
   drop by > `1e-4·max(1,|ELBO|)` (`3563-3627`).
6. **Anderson(m) acceleration** on packed `(log σ_g, θ, log a, log b)`
   (`anderson_step`, memory depth `_ANDERSON_MEMORY_DEPTH`), only on hyper-update
   iterations, clipped back into bounds (`3796-3860`).

Convergence: max of scaled-RMS parameter change and hyperparameter relative
change vs `convergence_tolerance=1e-4`. Predictor and objective changes are
recorded diagnostics, not stopping gates. Upstream defaults to
`max_outer_iterations=20`; the benchmark reference pins 400 and records the
actual iteration count and all four final changes.

---

## 5. Numerical / algorithmic tricks

| Trick | Where | Problem it solves |
|---|---|---|
| **Collapsed (marginalized-β) GLS**, solve in min(n,p) space | `_solve_restricted_full/_mean_only` 11525+, sample-space Cholesky 11610 | Avoids p×p system; `V=σ²I+XTXᵀ` is n×n. Woodbury identity between the two spaces. |
| **Exact-vs-iterative solver hierarchy** | `11594-11608`, `exact_solver_matrix_limit=2048` | Direct Cholesky when small; GPU exact variant-space `XᵀWX` syrk+Cholesky when p fits VRAM; else CG. |
| **Nyström low-rank preconditioned CG** (sample space) | `_sample_space_nystrom_*`, `sample_space_preconditioner_rank=256` | Conditions the n×n CG so iteration count stays bounded; diagonal + low-rank preconditioner. |
| **Hutchinson / stochastic-probe log-det & posterior-variance** | `logdet_probe_count=12`, `logdet_lanczos_steps=20`, `posterior_variance_probe_count=24`; CG-Lanczos log-det `_sample_space_logdet_from_cg_lanczos` | Diagonal of the inverse and `log|V|` without forming the inverse; Lanczos quadrature reuses CG's tridiagonal. |
| **GIG moments via scaled Bessel `kve` + asymptotic branches** | `_gig_moment` 13741, `gig_inverse_first_moment` optimizer_helpers 47 | Stable `K_{p±1}/K_p` ratios; small-z inverse-gamma limit, large-z→1, log-space `√(ψ/χ)`, floored operands avoid `0/0`/overflow. |
| **Inexact-Newton forcing sequence** (Eisenstat-Walker style) | `_binary_newton_solver_controls` 4527, `_collapsed_posterior_solver_controls` 4572, `forcing_tolerance`/`relaxed_iteration_cap` (forcing_sequence.py) | Loose CG tolerance early in EM (gradient-proxy driven), tightening as it converges; caps CG iters. |
| **Damped Newton + backtracking line search + bounded σ_g contraction** | `_update_scale_model` 13358-13392 | Convex scale-model solve; 22%/iter cap prevents σ_g collapse to noise floor. |
| **L-BFGS-B for TPB shapes** with hierarchical penalty & bounds | `_update_tpb_shape_vectors` 13505 | Box-constrained shape MLE; falls back to previous shapes on ABNORMAL. |
| **Safeguarded Anderson(m) acceleration** (Henderson-Varadhan / GLL non-monotone) | `anderson.py`, applied to hyperparameter fixed point 3796 | Krylov-style speedup of the EB outer fixed point; SVD condition check, Tikhonov reg, damping fractions `(1,0.5,0.25)`, monotone/non-monotone objective safeguard. |
| **ELBO revert safeguard on hyper-updates** | `3563-3627` | The per-block Newton/L-BFGS optimize local objectives, not the joint ELBO; reject transient ELBO drops. |
| **Posterior working sets** (grow active variant set) | `_posterior_working_set_*`, min 65536, grow 8192, 6 passes | For huge p, iteratively grow the set of variants that get a full posterior. |
| **Stochastic disjoint-block SVI**, adaptive block size, resident GPU int8/bitpacked caches | `_stochastic_variant_blocks`, `genotype_backend="bitpacked"` | Streams genotypes that don't fit in memory; per-epoch RM step=1. |
| **CUDA graph capture / replay of the CG loop** | `_try_capture_cg_graph`/`_replay_cg_graph` 7999-8080 | Amortizes kernel-launch overhead across repeated CG solves. |
| **CAVI-correct E[1/λ] precision** instead of 1/τ² plug-in | `_build_cavi_correct_prior_precision` 13692 | Makes the β block the exact variational update after collapsing λ, not a point estimate. |
| **Tie-map reduction** of duplicate genotype columns | `TieMap`, `_expand_group_values_to_members` 13812 | Fits one representative per identical-genotype group, expands λ/τ back to members. |

---

## 6. Exploits / shortcuts and the internal quantity that distinguishes them

The environment must reward the genuine hierarchical model, not cheaper methods
that hit similar predictive accuracy by a different mechanism. For each, the
**diagnostic internal quantity** is what a from-scratch reimplementation must
reproduce and what the grader/anti-cheat should probe.

1. **Uniform ridge (single λ, class-blind)** — one global `τ²` for all variants.
   - Looks similar: on a trait dominated by many small SNV effects, prediction
     R²/AUC can match.
   - Distinguisher: the genuine model produces a **per-variant `E[1/λ_j]`
     (`_build_cavi_correct_prior_precision`) that varies over orders of
     magnitude and correlates with `E[β_j²]/(σ_g s_j)²`**; ridge has constant
     precision. Grade recovery of the `λ_j` spread and its `χ_j = E[β²]/baseline`
     dependence.

2. **Uniform LASSO / ElasticNet (sklearn)** — L1/L2 with one penalty.
   - Looks similar: sparse `β`, decent accuracy.
   - Distinguisher: LASSO gives **point estimates with no `beta_variance`,
     no `σ_e²` via `(RSS+tr(XΣXᵀ))/n`, and a soft-threshold (constant subgradient)
     rather than the GIG `E[λ]` Bessel-ratio shrinkage**. The genuine model's
     shrinkage profile `m_j/β̂_j^{OLS}` is a smooth GIG function of evidence, not
     a hard threshold; and it exposes posterior variance. Anti-cheat: require the
     posterior second moment `E[β²]=m²+Σ` and the GIG `E[λ]`/`E[1/λ]` relationship.

3. **Marginal GWAS + P+T (prune-and-threshold)** — univariate `Xᵀy` z-scores,
   clump, threshold.
   - Looks similar: standard PRS baseline, can track accuracy where LD is mild.
   - Distinguisher: P+T never solves the **joint GLS system**
     `(XᵀDX + diag(1/τ²))m` — it ignores off-diagonal `XᵀX`. The genuine engine's
     `β` differs from marginal `Xᵀy` exactly by the joint decorrelation +
     shrinkage; grade the joint-vs-marginal residual and that `α` is GLS-profiled
     (`V⁻¹`), not OLS. Note the config's own `marginal_screen_min_abs_z`
     defaults to 0 and warns marginal screening is "methodologically risky" for
     SV contribution — a cheat that pre-screens on |z| drops rare-SV signal.

4. **Class-blind horseshoe (single global-local, no annotation regression)** —
   correct GIG local shrinkage but **one shared `(a,b)` and `s_j≡1`**.
   - Looks similar: horseshoe is the `a=b=1/2` special case; on a homogeneous
     trait it matches.
   - Distinguisher: the genuine model learns **class-specific `(a_c,b_c)` tail
     weights and an annotation-regressed `s_j = exp(x_jᵀθ)`**. The load-bearing
     internal quantities are: (i) `scale_model_coefficients θ` with a nonzero
     `type_offset::<class>` spread (SV classes get larger baseline scale than
     SNVs), and (ii) the per-class shape vectors diverging (SNV `a≈1`, complex-SV
     `a≈0.55`). A class-blind horseshoe has `θ=0` and identical `(a,b)`. Grade
     recovery of the **per-class baseline-scale ordering and per-class tail
     weights**, and the fact that SV effect scales are learned larger than SNV.

5. **Plain logistic/OLS (no shrinkage)** — MLE β.
   - Distinguisher: diverges under `p≳n` / rare variants; the genuine model's
     `E[1/λ_j]` precision and the ELBO's `−½Σ E[β²]/τ²` term are absent. Any
     method with no finite prior-precision term fails the separable-binary and
     `p>n` regimes the engine is built for.

**Single most discriminative signal:** the joint pattern of
`{ s_j = exp(x_jᵀθ) (annotation-regressed, class-ordered), (a_c,b_c) per-class
tail weights, per-variant E[λ_j]/E[1/λ_j] GIG moments, posterior E[β²]=m²+Σ,
and σ_e²=(RSS+tr(XΣXᵀ))/n }`. A cheaper method may match accuracy but cannot
simultaneously reproduce all five without implementing the actual TPB
global–local hierarchy with the annotation scale-regression.

---

## Appendix: key constants (config.py)

- Per-class log-baseline init `DEFAULT_CLASS_LOG_BASELINE_SCALE`: SNV −4.5 →
  complex-SV/inversion −3.1 (SVs expected larger effects). `s_init = exp(·)`.
- Per-class `shape_a` 1.0 (SNV) … 0.55 (inversion); `shape_b` 0.5 (SNV) … 0.38.
- Floors/ceilings: `prior_scale_floor 1e-6 / ceiling 10`, `global_scale_floor
  1e-4 / ceiling 10`, `local_scale_floor 1e-8`, `sigma_error_floor 1e-3`,
  `polya_gamma_minimum_weight 1e-4`.
- Penalties: `scale_model_ridge_penalty 1.0`, `type_offset_penalty 2.0`,
  `tpb_hierarchical_prior_variance 1.0`.
- Iteration caps: `max_outer_iterations 20`, `convergence_tolerance 1e-4`,
  `maximum_scale_model_iterations 8`, `maximum_tpb_shape_iterations 8`,
  `max_inner_newton_iterations 20`, shape bounds `[0.1, 10]`.
- Solvers: `exact_solver_matrix_limit 2048`, `sample_space_preconditioner_rank
  256`, `logdet_probe_count 12`, `logdet_lanczos_steps 20`,
  `posterior_variance_probe_count 24`, CG tol `1e-6`.
- Hyper-update cadence: every outer iteration ≥ 2.
