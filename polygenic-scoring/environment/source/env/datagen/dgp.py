"""Synthetic genotype-to-phenotype DGP for the heterogeneous benchmark grid.

The base effect sampler follows the hierarchical global-local shrinkage model
used by SV-PGS (see the reference notes in this repo):

    log s_j = class_offset[class_j]                       # per-class baseline (type_offset)
            + gamma . f(annotations_j)                     # log-linear scale model (sec 3)
    beta_j  = global_scale * s_j * sqrt(lambda_j) * z_j,   z_j ~ N(0,1)
    lambda_j: TPB / Gamma-Gamma local shrinkage per class:
        delta_j  ~ Gamma(shape=b_c, rate=1)
        lambda_j ~ Gamma(shape=a_c, rate=delta_j)          # heavy tail; horseshoe at a=b=1/2
    y_i (binomial)  ~ Bernoulli(sigmoid(C_i.alpha + Gstd_i.beta))

The base hierarchy matches SV-PGS, while explicit category knobs add sparse
spikes, lighter or heavier tails, nonlinear and interacting annotation effects,
opposing-effect suppressor LD, rare causal variants, and decoy annotations. No one
estimator is asserted to be Bayes-optimal across the complete grid.

WHAT THE AGENT IS TOLD, EXACTLY. ``materialize._write_dgp_json`` writes ONLY
``{"family", "formula"}`` -- the response name, the pgs annotation columns and
their types, and the covariate names. **No regime, knob or effect family is ever
disclosed.** Which category a dataset came from, its tail temper, its heritability,
its effect family and every switch below are private, and the agent must infer what
matters from the data like any analyst.

That is deliberate and it is the correct design: a real analyst does not get the
DGP. Do not "fix" it by widening ``dgp.json``. This paragraph exists because four
comments in this file used to claim the opposite ("Each active regime is disclosed
in the materialized dgp.json", "all disclosed in dgp.json", "additional disclosed
regime knobs", "This tempered regime is DISCLOSED to the agent (fair)"), which was
false in every case and, worse, argued a FAIRNESS property from a disclosure that
does not happen. The regimes below are fair because they are learnable from the
data, not because they are announced.

Ground-truth latents are all returned in `truth` (the live binary-only grader uses
AUC, Brier score, and clipped log-loss, so these are provenance/analysis fields
rather than graded recovery targets). The bundle SEPARATES raw-scale generative
constants from
the effective (phenotype/liability-axis) coefficients that actually generated y:
  raw:       beta_raw, tau2_raw, s_j, global_scale_raw (NON-identifiable -- it
             cancels exactly under the beta/sqrt(gvar) renormalization below, so it
             affects no observable and cannot be recovered), class log-baselines,
             gamma_len/gamma_repeat; TPB shapes and local scales appear only for
             effect families that actually use them.
  effective: beta_effective, alpha_effective, tau2_effective, intercept_effective,
             heritability.

`tau2_*` is the slab variance CONDITIONAL ON CAUSAL INCLUSION. Masks (null_frac,
SNV-causal thinning, MAF ceiling) zero beta while leaving tau2 positive, and
suppressor effects are constructed analytically rather than drawn from the slab, so
`beta_effective ~ N(0, tau2_effective)` holds only on `truth["causal_mask"]`; off it
the effect is a point mass at 0.

Effect families (`cfg.effect_family`): "tpb_slab" (the heavy-tailed global-local
slab above), "dense_gaussian" (near-homoskedastic Gaussian effects on every variant,
no local scales, no spikes) and "dense_plus_spike" (an explicit dense floor plus a
heavy-tailed spike component with CONTROLLED variance shares).

Faithfulness notes from the reference docs:
  * SV-PGS's built-in reserved fields length/is_repeat/is_copy_number do NOT
    auto-enter the scale-model design -- only variant_class does. So length &
    repeat are emitted here as CUSTOM annotation columns (continuous / binary),
    which the tool's schema hypermodel DOES consume (io.py auto-typing).
  * Continuous annotations enter as standardized-linear + cubic-hinge splines at
    interior quartile knots (mixture_inference.py:13089-13155). The true scale
    model here uses the standardized-linear term (the dominant, recoverable
    component); higher spline bases are left at coefficient 0.
  * Genotype standardization matches preprocessing.py: scale = sqrt(css/n),
    ddof=0, floor 1e-6 -> 1.0.
"""
from __future__ import annotations

from grader.contract import SHIPPED_FRAC_TRAIN

import numpy as np
from scipy.special import ndtri
from dataclasses import dataclass

# ---- config.py mirrors ------------------------------------------------------
CLASS_LOG_BASELINE = {
    "snv": -4.5, "small_indel": -4.2, "deletion_short": -3.8, "deletion_long": -3.3,
    "duplication_short": -3.7, "duplication_long": -3.3, "insertion_mei": -3.6,
    "inversion_bnd_complex": -3.1, "str_vntr_repeat": -3.5, "other_complex_sv": -3.3,
}
CLASS_TPB_SHAPE_A = {
    "snv": 1.0, "small_indel": 0.9, "deletion_short": 0.7, "deletion_long": 0.6,
    "duplication_short": 0.7, "duplication_long": 0.6, "insertion_mei": 0.65,
    "inversion_bnd_complex": 0.55, "str_vntr_repeat": 0.6, "other_complex_sv": 0.6,
}
CLASS_TPB_SHAPE_B = {
    "snv": 0.5, "small_indel": 0.5, "deletion_short": 0.45, "deletion_long": 0.4,
    "duplication_short": 0.45, "duplication_long": 0.4, "insertion_mei": 0.42,
    "inversion_bnd_complex": 0.38, "str_vntr_repeat": 0.4, "other_complex_sv": 0.4,
}
CLASSES = list(CLASS_LOG_BASELINE)
SV_CLASSES = ["deletion_short", "deletion_long", "duplication_short", "duplication_long",
              "insertion_mei", "inversion_bnd_complex", "str_vntr_repeat", "other_complex_sv"]
SNV_LIKE = ["snv", "small_indel"]
PRIOR_SCALE_FLOOR, PRIOR_SCALE_CEIL = 1e-6, 10.0
MAF_FLOOR = 1e-2


@dataclass
class DGPConfig:
    n_samples: int = 4000
    n_variants: int = 3000
    n_pcs: int = 10
    # class mix: SNV-heavy, SVs a rare-but-consequential minority
    snv_frac: float = 0.72
    small_indel_frac: float = 0.10
    # (remainder split across the 8 SV classes)
    ld_block_min: int = 5
    ld_block_max: int = 40
    ld_rho_lo: float = 0.6
    ld_rho_hi: float = 0.95
    # scale-model annotation coefficients (true gamma): assigned as FIXED means
    # (length_coef_mean / repeat_coef_mean below); there is no random-draw sd knob.
    length_coef_mean: float = 0.20     # + : longer SVs => larger prior scale (recoverable sign)
    repeat_coef_mean: float = -0.60    # - : repeats mostly artefactual => downweight
    # RAW, NON-IDENTIFIABLE generative constant: multiplies exp(class_offset) to set
    # the overall genetic scale, but cancels exactly under the beta/sqrt(gvar)
    # renormalization to the phenotype axis (see generate()), so it affects NO
    # observable and is NOT recoverable. Stored only as `global_scale_raw` for
    # provenance -- never as a recovery target.
    global_scale: float = 0.9
    heritability: float = 0.4          # PRE-COVARIANCE genetic component-variance target
    # `heritability` (h) is a PRE-COVARIANCE component-variance target: the genetic
    # and covariate components are EACH scaled (to h and 1-h respectively) before
    # summation, NOT a guarantee on the realized ratio. Because the genetic and
    # covariate components share ancestry (PC1),
    # Cov(genetic, covariate) != 0 and the 2*Cov cross term is never folded in, so
    # the REALIZED Var(genetic)/Var(y) only approximates h (it deviates slightly by
    # that covariance). The logistic link supplies the residual variance.
    prevalence: float = 0.25           # binary target case rate
    ancestry_confounding: float = 0.5  # PC effect on both MAF and phenotype
    # TPB tail tempering: SV-PGS's config shape_b < 1 gives an INFINITE-MEAN local
    # scale, so a literal draw collapses all genetic variance onto one monster
    # variant (a degenerate one-SNP trait). We draw the EXACT Gamma-Gamma but shift
    # shape_b into the finite-VARIANCE regime (b_used = config_b + tail_temper >= ~2.2),
    # which preserves the per-class tail ORDERING (SVs still heavier than SNVs) and
    # yields realistic polygenicity (top variant ~3-5% of genetic var, ~130-200
    # effective variants). This tempered regime is NOT disclosed to the agent (see
    # the module docstring: dgp.json carries only family+formula). It is fair
    # because it is LEARNABLE -- fully representable by the model's own TPB family,
    # whose shapes are fitted in [0.1, 10] -- not because it is announced.
    tail_temper: float = 2.5
    seed: int = 0
    class_weights: dict | None = None
    # ---- category-specific knobs (PRIVATE; never serialized to dgp.json) ----
    # class_scale_spread multiplies each class's log-baseline DEVIATION from the
    # mean-over-present-classes, widening (>1) or shrinking (<1) the between-class
    # prior-scale heterogeneity. This is exactly a rescaling of the type_offset
    # column of the true scale model, so a class-aware model still recovers the
    # ordering while a class-blind ridge pays for ignoring it.
    class_scale_spread: float = 1.0
    # null_frac forces an extra fraction of variants to EXACTLY zero true effect
    # (a spike at 0 on top of the TPB slab), increasing sparsity so adaptive local
    # shrinkage separates further from a uniform penalty.
    null_frac: float = 0.0
    # soft (fractional multi-class) membership: a soft_frac fraction of variants
    # get 2-class membership; their true log-baseline / TPB shapes are the
    # membership-weighted mixture of the two classes' values -- exactly the
    # class_membership_matrix the model consumes (io.py:prior_class_membership).
    soft_membership: bool = False
    soft_frac: float = 0.0
    soft_weight_lo: float = 0.55
    soft_weight_hi: float = 0.80
    # Null every effect whose MAF exceeds this threshold; 0.5 disables the filter.
    causal_maf_max: float = 0.5
    # Fraction of SNV-like variants allowed to be causal; 1.0 keeps them all.
    snv_causal_frac: float = 1.0
    # ---- additional regime knobs (PRIVATE; never serialized to dgp.json) ----
    # NONLINEAR annotation scale: adds a U-shaped (quadratic) term in standardized
    # log-length to the true log-scale, so a LINEAR annotation-weighted penalty
    # captures only part of the structure and a spline/nonparametric annotation
    # model wins. 0 = inactive (pure log-linear, as shipped).
    length_nonlinear_coef: float = 0.0
    # annotation INTERACTION: length sensitivity that applies only to copy-number
    # (CNV) classes (class x length), so an additive annotation model misses it.
    # 0 = inactive.
    class_length_interaction: float = 0.0
    # SUPPRESSOR LD: in a fraction of high-positive-LD blocks, exactly one variant
    # pair receives equal-and-opposite joint effects. For pair correlation r, each
    # pair-only marginal is (1-r) times its joint effect.
    suppressor_block_frac: float = 0.0
    # DECOY / NOISY annotations: emit this many pure-noise per-variant annotation
    # numeric annotation columns that carry no effect signal. A category lists
    # them in pgs(...) so they are "available annotations"; a model that blindly
    # trusts annotations (weights them) is HURT, the honest model down-weights them.
    # 0 = inactive (no decoy columns), as shipped.
    n_decoy_annotations: int = 0
    # ---- effect-family switch (DGP-001 / DGP-002) ----
    # "tpb_slab"         : the SV-PGS global-local heavy-tailed slab (default).
    # "dense_gaussian"   : every variant carries a near-Gaussian effect drawn with a
    #                      near-HOMOSKEDASTIC scale (no TPB local scales, no spikes,
    #                      class/annotation scale variation compressed to
    #                      `dense_scale_spread`). This is a genuinely infinitesimal
    #                      architecture: dense ridge-style shrinkage is the right bias.
    # "dense_plus_spike" : beta = beta_dense + beta_spike as two EXPLICIT components
    #                      whose variance shares are controlled (`dense_var_share`),
    #                      the spikes carried by `spike_frac` of variants drawn from
    #                      the heavy-tailed slab. Neither a pure dense nor a pure
    #                      sparse prior is correct.
    effect_family: str = "tpb_slab"
    # dense arms: how strongly the (log) effect scale still tracks annotations. 0 =
    # exactly homoskedastic; 0.15 = a narrow +-15% modulation of the log scale, which
    # keeps the annotation columns weakly honest without re-introducing the heavy,
    # scale-driven heteroskedasticity that makes a "dense" label a lie.
    dense_scale_spread: float = 0.15
    # dense_plus_spike: target fraction of the REALIZED genetic variance carried by
    # the dense floor (the remainder is carried by the spikes).
    dense_var_share: float = 0.5
    spike_frac: float = 0.02
    # ---- suppressor-pair geometry (DGP-003) ----
    # Every selected block must contain a realized positive-correlation pair at or
    # above this threshold; generation aborts rather than silently weakening the DGP.
    suppressor_min_corr: float = 0.55
    # Target component-variance share Var(G beta_pair) /
    # (Var(G beta_pair) + Var(G beta_rest)). Cross-component covariance is not
    # mislabeled as belonging to either component.
    suppressor_var_share: float = 0.6
    # SIGN REVERSAL, solved for rather than hoped for. In a standardized pair with
    # realized correlation r and joint effects (b, b_partner), the target's marginal
    # covariance is
    #     Cov(G_target, y) = b + r * b_partner.
    # Equal-and-opposite effects (b, -b) give (1-r)*b: the marginal is ATTENUATED but
    # keeps the CORRECT SIGN, so a marginal screen still reads the sign right and the
    # suppressor task is only half-present. To force a reversal the partner term must
    # OVERSHOOT b rather than merely cancel it. Choosing
    #     b_partner = -(1 + m) * b / r      with m = overshoot > 0
    # gives exactly
    #     Cov(G_target, y) = b - (1 + m) * b = -m * b,
    # i.e. the marginal association is sign-REVERSED with relative margin m, which is
    # SET here (not sampled and hoped for). m is drawn per pair in
    # [suppressor_overshoot_lo, suppressor_overshoot_hi]; the lower bound is the margin
    # that has to survive finite-sample noise in the realized marginal covariance.
    suppressor_overshoot_lo: float = 0.30
    suppressor_overshoot_hi: float = 0.80
    # ---- train/test LD shift (DGP-004) ----
    # When ld_shift is set, the cohort is generated in TWO blocks of rows with
    # DIFFERENT within-block correlation matrices: the first `frac_train_hint` of rows
    # (the train cohort) use rho ~ U(ld_rho_lo, ld_rho_hi); the remaining rows (the
    # test cohort) use rho ~ U(ld_shift_rho_lo, ld_shift_rho_hi). Causal effects and
    # the per-variant allele frequencies are IDENTICAL across the two cohorts, so only
    # the tagging structure moves: a predictor that leans on tag variants (whose
    # correlation with the causal variant collapses on test) degrades, while a model
    # that puts weight on the causal variants transfers.
    ld_shift: bool = False
    ld_shift_rho_lo: float = 0.0
    ld_shift_rho_hi: float = 0.25
    frac_train_hint: float = SHIPPED_FRAC_TRAIN
    # ---- unlisted nuisance covariate columns (CONTRACT-002) ----
    # Number of per-sample columns written to covariates_*.csv but NOT listed in the
    # formula. Column 0 is a BATCH ARTIFACT: an independent latent u that ENTERS the
    # liability with coefficient +nuisance_cov_strength on the train cohort and
    # -nuisance_cov_strength on the test cohort, so it is strongly predictive of y in
    # train and its association REVERSES on the held-out cohort. A submission that
    # ignores the formula and regresses on every non-ID column learns the train sign
    # and is punished; the remaining columns are pure noise.
    #
    # The arrow points FROM u TO y (u is drawn independently, then the phenotype is
    # generated from it). No public column is ever a function of the held-out labels:
    # flipping y_test leaves every public byte identical. An earlier version of this
    # trap built the column FROM y (sign-flipped on test), which handed the hidden
    # y_test to anyone who read the covariate file -- never do that.
    nuisance_cov_cols: int = 0
    nuisance_cov_strength: float = 0.7


def _class_assignment(rng, cfg):
    if cfg.class_weights is not None:
        names = list(cfg.class_weights)
        w = np.array([cfg.class_weights[c] for c in names], float)
    else:
        names = SNV_LIKE + SV_CLASSES
        sv_each = (1.0 - cfg.snv_frac - cfg.small_indel_frac) / len(SV_CLASSES)
        w = np.array([cfg.snv_frac, cfg.small_indel_frac] + [sv_each] * len(SV_CLASSES))
    w = w / w.sum()
    idx = rng.choice(len(names), size=cfg.n_variants, p=w)
    return np.array([names[i] for i in idx]), names


def _draw_maf(rng, vclass):
    P = len(vclass)
    maf = np.empty(P)
    for j in range(P):
        if vclass[j] in SNV_LIKE:
            m = rng.beta(0.3, 8.0)
        else:  # SVs rarer
            m = rng.beta(0.2, 20.0)
        maf[j] = m
    return np.clip(maf, MAF_FLOOR, 0.5)


def _draw_length(rng, vclass):
    P = len(vclass)
    length = np.empty(P)
    for j in range(P):
        c = vclass[j]
        if c == "snv":
            length[j] = 1.0
        elif c == "small_indel":
            length[j] = float(rng.integers(2, 50))
        else:
            length[j] = float(10.0 ** rng.uniform(2.0, 6.0))  # 100 bp .. 1 Mb
    return length


def _tpb_local_scale(rng, shape_a, shape_b):
    """Faithful TPB / Gamma-Gamma local shrinkage sampler.

    delta ~ Gamma(shape_b, rate 1);  lambda ~ Gamma(shape_a, rate delta).
    Marginally a beta-prime (generalized-beta-of-the-2nd-kind) local scale -- the
    exact TPB prior SV-PGS's CAVI inverts. Mean a/(b-1) finite for b>1, variance
    finite for b>2; the caller passes tempered shape_b in [2.2, 2.5] so draws are
    heavy-tailed-but-polygenic rather than infinite-mean. Returns (lambda, delta).
    """
    n = len(shape_a)
    delta = rng.gamma(shape=shape_b, scale=1.0, size=n)          # rate 1 -> scale 1
    delta = np.maximum(delta, 1e-8)
    lam = rng.gamma(shape=shape_a, scale=1.0 / delta, size=n)    # rate delta -> scale 1/delta
    return np.maximum(lam, 1e-8), delta


def _build_membership(rng, vclass, cfg):
    """Per-variant class membership as a list of (class_token, weight) pairs.

    Hard (default): each variant is 1-of-K in its own observed class. Soft: a
    `soft_frac` subset is additionally assigned a second class (drawn uniformly
    from the other classes) with the complementary weight, so the variant's prior
    baseline/shapes are a convex mixture -- the exact fractional membership the
    SV-PGS class_membership_matrix represents. Returns (members, weights, is_soft).
    """
    P = len(vclass)
    all_classes = SNV_LIKE + SV_CLASSES
    members = [[str(vclass[j])] for j in range(P)]
    weights = [[1.0] for _ in range(P)]
    is_soft = np.zeros(P, dtype=bool)
    if cfg.soft_membership and cfg.soft_frac > 0.0:
        pick = rng.random(P) < cfg.soft_frac
        for j in np.where(pick)[0]:
            primary = str(vclass[j])
            others = [c for c in all_classes if c != primary]
            second = others[rng.integers(len(others))]
            w = float(rng.uniform(cfg.soft_weight_lo, cfg.soft_weight_hi))
            members[j] = [primary, second]
            weights[j] = [w, 1.0 - w]
            is_soft[j] = True
    return members, weights, is_soft


def _std_annotation(x):
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-12), mu, sd


def _validate_config(cfg):
    if type(cfg.n_samples) is not int or cfg.n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")
    if type(cfg.n_variants) is not int or cfg.n_variants < 2:
        raise ValueError("n_variants must be an integer >= 2")
    if cfg.effect_family not in ("tpb_slab", "dense_gaussian", "dense_plus_spike"):
        raise ValueError(f"unknown effect_family {cfg.effect_family!r}")
    fractions = {
        "suppressor_block_frac": cfg.suppressor_block_frac,
        "dense_var_share": cfg.dense_var_share,
        "spike_frac": cfg.spike_frac,
    }
    for name, value in fractions.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    for name, value in {
        "null_frac": cfg.null_frac,
        "snv_causal_frac": cfg.snv_causal_frac,
        "soft_frac": cfg.soft_frac,
    }.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if (type(cfg.n_decoy_annotations) is not int
            or cfg.n_decoy_annotations < 0):
        raise ValueError("n_decoy_annotations must be a nonnegative integer")
    if type(cfg.nuisance_cov_cols) is not int or cfg.nuisance_cov_cols < 0:
        raise ValueError("nuisance_cov_cols must be a nonnegative integer")
    if (
        not np.isfinite(cfg.nuisance_cov_strength)
        or cfg.nuisance_cov_strength < 0.0
    ):
        raise ValueError("nuisance_cov_strength must be finite and nonnegative")
    if cfg.nuisance_cov_cols > 0 and cfg.nuisance_cov_strength <= 0.0:
        raise ValueError(
            "nuisance covariates require a positive nuisance_cov_strength"
        )
    if not np.isfinite(cfg.dense_scale_spread) or cfg.dense_scale_spread < 0.0:
        raise ValueError("dense_scale_spread must be finite and nonnegative")
    if cfg.effect_family == "dense_plus_spike":
        if not 0.0 < cfg.dense_var_share < 1.0:
            raise ValueError("dense_plus_spike requires dense_var_share in (0, 1)")
        spike_count = int(round(cfg.spike_frac * cfg.n_variants))
        if spike_count < 1 or spike_count >= cfg.n_variants:
            raise ValueError("dense_plus_spike requires a nonempty proper spike set")
    if cfg.suppressor_block_frac > 0.0:
        if cfg.effect_family != "tpb_slab":
            raise ValueError("suppressor blocks require effect_family='tpb_slab'")
        if not 0.0 < cfg.suppressor_min_corr < 1.0:
            raise ValueError("suppressor_min_corr must be in (0, 1)")
        if not 0.0 < cfg.suppressor_var_share < 1.0:
            raise ValueError("suppressor_var_share must be in (0, 1)")
    rho_ranges = [("ld", cfg.ld_rho_lo, cfg.ld_rho_hi)]
    if cfg.ld_shift:
        rho_ranges.append(("ld_shift", cfg.ld_shift_rho_lo, cfg.ld_shift_rho_hi))
    if cfg.ld_shift or cfg.nuisance_cov_cols > 0:
        n_train = int(cfg.n_samples * cfg.frac_train_hint)
        if not 0.0 < cfg.frac_train_hint < 1.0 or not 0 < n_train < cfg.n_samples:
            raise ValueError(
                "cohort mechanisms require nonempty train and test cohorts"
            )
    for name, lower, upper in rho_ranges:
        if not (np.isfinite(lower) and np.isfinite(upper)
                and 0.0 <= lower <= upper < 1.0):
            raise ValueError(f"{name} rho bounds must satisfy 0 <= lower <= upper < 1")


def generate(cfg: DGPConfig):
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    N, P = cfg.n_samples, cfg.n_variants

    vclass, classnames = _class_assignment(rng, cfg)
    maf = _draw_maf(rng, vclass)
    length = _draw_length(rng, vclass)
    is_repeat = np.where(vclass == "str_vntr_repeat", 1,
                         (rng.random(P) < 0.08).astype(int))
    is_cnv = np.isin(vclass, ["deletion_short", "deletion_long",
                              "duplication_short", "duplication_long"]).astype(int)

    # ---- ancestry axis (confounds both MAF and phenotype) ----
    anc = rng.standard_normal(N)
    anc = (anc - anc.mean()) / anc.std()
    pc_loading = rng.standard_normal(P) * cfg.ancestry_confounding

    # ---- genotypes with AR(1) LD blocks ----
    # LD SHIFT (cfg.ld_shift): the rows are split into a train cohort (first n_tr) and
    # a test cohort, and each block's AR(1) correlation is drawn SEPARATELY for the two
    # cohorts (train from [ld_rho_lo, ld_rho_hi], test from [ld_shift_rho_lo,
    # ld_shift_rho_hi]). Allele frequencies, the ancestry mechanism and the causal
    # effects are identical across cohorts -- only the tagging structure moves.
    n_tr_cohort = int(N * cfg.frac_train_hint)
    cohort = np.zeros(N, dtype=int)
    # A cohort split is also required by the nuisance BATCH ARTIFACT (CONTRACT-002):
    # its sign has to differ between the train and the held-out cohort, so the split
    # must be known at generation time rather than drawn later by materialize().
    if cfg.ld_shift or cfg.nuisance_cov_cols > 0:
        cohort[n_tr_cohort:] = 1
    row_groups = ([np.arange(0, n_tr_cohort), np.arange(n_tr_cohort, N)]
                  if cfg.ld_shift else [np.arange(N)])
    G = np.zeros((N, P), dtype=np.float32)
    block_id = np.empty(P, dtype=int)
    col = 0
    blk = 0
    while col < P:
        bsize = int(rng.integers(cfg.ld_block_min, cfg.ld_block_max + 1))
        cols = np.arange(col, min(col + bsize, P))
        k = len(cols)
        rho_train = rng.uniform(cfg.ld_rho_lo, cfg.ld_rho_hi)
        rho_by_group = [rho_train]
        if cfg.ld_shift:
            rho_by_group.append(rng.uniform(cfg.ld_shift_rho_lo, cfg.ld_shift_rho_hi))
        block_id[cols] = blk
        # AR(1) latent haplotype layers -> correlated dosages
        for rows, rho in zip(row_groups, rho_by_group):
            nr = len(rows)
            for layer in range(2):
                lat = np.empty((nr, k), dtype=np.float32)
                lat[:, 0] = rng.standard_normal(nr)
                for t in range(1, k):
                    lat[:, t] = (rho * lat[:, t - 1]
                                 + np.sqrt(1 - rho ** 2) * rng.standard_normal(nr))
                for jj, j in enumerate(cols):
                    # per-variant, per-INDIVIDUAL freq shifted by ancestry (population
                    # structure): each sample gets its own allele probability p_ij and
                    # its own threshold, so allele frequency genuinely tracks the
                    # ancestry axis (PC1). Using a single mean threshold here would
                    # erase the confounding the task tells the agent to adjust for.
                    p_ij = np.clip(maf[j] + pc_loading[j] * anc[rows] * 0.03, 0.003, 0.9)
                    thr = ndtri(1.0 - p_ij)  # lat ~ N(0,1); per-sample cutpoint
                    allele = (lat[:, jj] > thr).astype(np.float32)
                    G[rows, j] += allele
        col += k
        blk += 1

    # drop monomorphic
    sd0 = G.std(axis=0)
    keep = sd0 > 1e-6
    G = G[:, keep]
    vclass = vclass[keep]
    length = length[keep]
    is_repeat = is_repeat[keep]
    is_cnv = is_cnv[keep]
    maf = maf[keep]
    block_id = block_id[keep]
    P = G.shape[1]

    # ---- class membership (hard, or fractional multi-class if configured) ----
    members, mweights, is_soft = _build_membership(rng, vclass, cfg)

    # class log-baselines, with between-class spread widened/narrowed around the
    # mean of the classes PRESENT (a pure rescale of the type_offset column).
    present = list(dict.fromkeys([str(c) for c in vclass]))
    mu_base = float(np.mean([CLASS_LOG_BASELINE[c] for c in present]))
    adj_base = {c: mu_base + cfg.class_scale_spread * (CLASS_LOG_BASELINE[c] - mu_base)
                for c in CLASSES}

    # ---- TRUE log-linear scale model (membership-weighted mixture) ----
    class_offset = np.array([sum(w * adj_base[c] for c, w in zip(members[j], mweights[j]))
                             for j in range(P)])
    log_len = np.log10(np.maximum(length, 1.0))
    log_len_std, _, _ = _std_annotation(log_len)
    # true annotation coefficients (recoverable): length (continuous, standardized-linear),
    # repeat (binary indicator). A few class interactions omitted for a clean recovery target.
    gamma_len = cfg.length_coef_mean
    gamma_repeat = cfg.repeat_coef_mean
    log_s = class_offset + gamma_len * log_len_std + gamma_repeat * is_repeat
    # NONLINEAR annotation (U-shaped/quadratic in standardized log-length; centered
    # so E[.]~0) and a class x length INTERACTION on copy-number variants. Both are
    # 0 by default (pure log-linear); when active
    # a linear annotation-weighted penalty captures only part of the true scale.
    if cfg.length_nonlinear_coef != 0.0:
        log_s = log_s + cfg.length_nonlinear_coef * (log_len_std ** 2 - 1.0)
    if cfg.class_length_interaction != 0.0:
        log_s = log_s + cfg.class_length_interaction * log_len_std * is_cnv
    log_s = np.clip(log_s, np.log(PRIOR_SCALE_FLOOR), np.log(PRIOR_SCALE_CEIL))
    s = np.exp(log_s)

    # ---- TPB local shrinkage per class (membership-weighted, tempered) ----
    # A dense Gaussian arm has no local-scale hierarchy, so do not draw or report
    # unused TPB latents for it.
    if cfg.effect_family == "dense_gaussian":
        a = b = lam = delta = None
    else:
        a = np.array([
            sum(w * CLASS_TPB_SHAPE_A[c] for c, w in zip(members[j], mweights[j]))
            for j in range(P)
        ])
        b = np.array([
            sum(w * CLASS_TPB_SHAPE_B[c] for c, w in zip(members[j], mweights[j]))
            for j in range(P)
        ]) + cfg.tail_temper
        lam, delta = _tpb_local_scale(rng, a, b)

    # ---- standardize genotypes as the model does (ddof=0, css/n) ----
    # (done BEFORE the effects because the analytic suppressor construction below
    # solves for beta against the REALIZED within-block correlations.)
    Gm = G.mean(axis=0)
    css = ((G - Gm) ** 2).sum(axis=0)
    Gsd = np.sqrt(css / N)
    Gsd = np.where(Gsd < 1e-6, 1.0, Gsd)
    Gstd = (G - Gm) / Gsd

    # ---- true effects: one of three EXPLICIT effect families (see cfg.effect_family)
    dense_arm = cfg.effect_family in ("dense_gaussian", "dense_plus_spike")
    if dense_arm and (cfg.null_frac > 0.0 or cfg.snv_causal_frac < 1.0
                      or cfg.causal_maf_max < 0.5):
        # A dense architecture has NO null set by definition; a zeroing mask on top of
        # it would reproduce exactly the "heavy-tailed slab wearing a dense label" bug.
        raise ValueError("dense effect families forbid null_frac / causal masks")
    # near-homoskedastic DENSE scale: a single baseline plus a NARROW annotation
    # modulation. No class-scale spread, no TPB local scales => no spikes.
    log_s_dense = mu_base + cfg.dense_scale_spread * (
        gamma_len * log_len_std + gamma_repeat * is_repeat)
    s_dense = np.exp(np.clip(log_s_dense, np.log(PRIOR_SCALE_FLOOR),
                             np.log(PRIOR_SCALE_CEIL)))

    beta_dense = np.zeros(P)
    beta_spike = np.zeros(P)
    spike_mask = np.zeros(P, dtype=bool)
    if cfg.effect_family == "tpb_slab":
        z = rng.standard_normal(P)
        if cfg.null_frac > 0.0:
            z[rng.random(P) < cfg.null_frac] = 0.0
        # ---- optional causal filters for sparse and rare-variant regimes ----
        if cfg.snv_causal_frac < 1.0:
            snv_mask = np.isin(vclass, SNV_LIKE)
            drop = snv_mask & (rng.random(P) >= cfg.snv_causal_frac)
            z[drop] = 0.0
        if cfg.causal_maf_max < 0.5:
            z[maf > cfg.causal_maf_max] = 0.0
        beta = cfg.global_scale * s * np.sqrt(lam) * z
        tau2 = (cfg.global_scale * s) ** 2 * lam
    elif cfg.effect_family == "dense_gaussian":
        z = rng.standard_normal(P)
        beta = cfg.global_scale * s_dense * z
        tau2 = (cfg.global_scale * s_dense) ** 2
        beta_dense = beta.copy()
    else:  # dense_plus_spike
        z = rng.standard_normal(P)
        beta_dense = cfg.global_scale * s_dense * z
        spike_count = int(round(P * cfg.spike_frac))
        if spike_count < 1 or spike_count >= P:
            raise ValueError(
                "the realized variant count makes the requested spike set empty or full"
            )
        spike_mask[rng.choice(P, size=spike_count, replace=False)] = True
        z_spike = rng.standard_normal(P) * spike_mask
        # The spike layer's heavy tail comes from the TPB LOCAL scale only. Multiplying
        # it by the wide per-class scale as well compounds two heavy factors and the
        # layer degenerates into ONE monster variant (~70% of the spike variance),
        # which is a one-SNP trait, not "a dense floor plus a few large spikes".
        beta_spike = cfg.global_scale * s_dense * np.sqrt(lam) * z_spike
        # Set the REALIZED variance shares of the two components (measured through the
        # actual genotypes, so LD is accounted for), not just their prior variances.
        var_dense = float((Gstd @ beta_dense).var()) + 1e-12
        var_spike = float((Gstd @ beta_spike).var()) + 1e-12
        beta_dense *= np.sqrt(cfg.dense_var_share / var_dense)
        beta_spike *= np.sqrt((1.0 - cfg.dense_var_share) / var_spike)
        beta = beta_dense + beta_spike
        # tau2 = slab variance of the sum of the two independent components.
        tau2 = ((cfg.global_scale * s_dense) ** 2
                * (cfg.dense_var_share / var_dense)
                + spike_mask * (cfg.global_scale * s_dense) ** 2 * lam
                * ((1.0 - cfg.dense_var_share) / var_spike))

    # ---- sparse opposing-effect SUPPRESSOR pairs -------------------------------
    # In a standardized two-variant block with positive correlation r and joint
    # effects (+b, -b), R beta = ((1-r)b, -(1-r)b). The pair's marginal effects are
    # therefore deterministically attenuated while the joint effects remain large.
    suppressor_target = np.zeros(P, dtype=bool)
    suppressor_partner = np.zeros(P, dtype=bool)
    if cfg.suppressor_block_frac > 0.0:
        blocks = np.unique(block_id)
        n_sup = int(round(cfg.suppressor_block_frac * len(blocks)))
        if n_sup < 1:
            raise ValueError("suppressor_block_frac selects no blocks")
        # Find each block's best REALIZED pair, then draw the suppressor blocks from the
        # ones that actually qualify. Drawing blindly first and aborting when a drawn
        # block happens to miss the threshold makes generation hostage to a single
        # unlucky block: at near-shipped size (N=4000, P=3000) that aborted 4 of 6
        # seeds, because with hundreds of blocks SOME block's best realized correlation
        # is always a hair under the bar. This still fails closed -- if too few blocks
        # qualify, the dataset is rejected rather than silently built with a weaker
        # mechanism.
        candidates = {}
        for blkid in blocks:
            cols = np.where(block_id == blkid)[0]
            if len(cols) < 2:
                continue
            Xb = Gstd[:, cols]
            R = Xb.T @ Xb / N
            left, right = np.triu_indices(len(cols), k=1)
            pair_correlations = R[left, right]
            best = int(np.argmax(pair_correlations))
            correlation = float(pair_correlations[best])
            if np.isfinite(correlation) and correlation >= cfg.suppressor_min_corr:
                candidates[int(blkid)] = (int(cols[left[best]]), int(cols[right[best]]),
                                          correlation)
        if len(candidates) < n_sup:
            raise ValueError(
                f"only {len(candidates)} of {len(blocks)} blocks contain a variant pair "
                f"at correlation >= {cfg.suppressor_min_corr:.6g}; "
                f"{n_sup} suppressor blocks are required"
            )
        sup_blocks = rng.choice(sorted(candidates), size=n_sup, replace=False)
        for blkid in sup_blocks:
            cols = np.where(block_id == blkid)[0]
            target, partner, correlation = candidates[int(blkid)]
            beta[cols] = 0.0
            sign = 1.0 if rng.integers(0, 2) else -1.0
            overshoot = float(rng.uniform(cfg.suppressor_overshoot_lo,
                                          cfg.suppressor_overshoot_hi))
            beta[target] = sign
            # b_partner = -(1 + m) * b / r  =>  Cov(G_target, y) = -m * b exactly:
            # the target's marginal association is REVERSED, not merely attenuated.
            beta[partner] = -sign * (1.0 + overshoot) / correlation
            suppressor_target[target] = True
            suppressor_partner[partner] = True

        # Give the pair component a controlled share of the sum of the pair and
        # remainder component variances. Cross-component covariance is deliberately
        # not attributed to either component.
        sup = suppressor_target | suppressor_partner
        rest = ~sup
        v_sup = float((Gstd[:, sup] @ beta[sup]).var())
        v_rest = float((Gstd[:, rest] @ beta[rest]).var())
        if not np.isfinite(v_sup) or v_sup <= 0.0:
            raise ValueError("suppressor-pair component has nonpositive variance")
        if not np.isfinite(v_rest) or v_rest <= 0.0:
            raise ValueError("non-suppressor component has nonpositive variance")
        share = cfg.suppressor_var_share
        beta[sup] *= np.sqrt((share / (1.0 - share)) * v_rest / v_sup)

    genetic = Gstd @ beta

    # ---- covariates: intercept, age, age^2, sex, 2 eth indicators, PC1..PC10 ----
    age = rng.normal(55, 12, N)
    age_s = (age - age.mean()) / age.std()
    sex = rng.integers(0, 2, N).astype(float)
    eth = rng.integers(0, 3, N)
    eth1 = (eth == 1).astype(float)
    eth2 = (eth == 2).astype(float)
    pcs = rng.standard_normal((N, cfg.n_pcs))
    pcs[:, 0] = anc  # PC1 = ancestry axis
    C = np.column_stack([np.ones(N), age_s, age_s ** 2, sex, eth1, eth2, pcs])
    # true covariate effects: modest age/sex, ancestry-driven PC effects (confounding)
    alpha = np.concatenate([
        [0.0],                                   # intercept (set later for prevalence)
        [0.25, 0.05, 0.15, 0.1, -0.1],           # age, age^2, sex, eth1, eth2
        rng.standard_normal(cfg.n_pcs) * cfg.ancestry_confounding,
    ])
    cov_lin = C @ alpha

    # ---- combine to target heritability ----
    h = cfg.heritability
    gvar = genetic.var() + 1e-12
    cvar = cov_lin.var() + 1e-12
    # scale genetic and covariate parts to a common liability with heritability h
    g_scaled = genetic / np.sqrt(gvar)
    c_scaled = (cov_lin - cov_lin.mean()) / np.sqrt(cvar)
    liability = np.sqrt(h) * g_scaled + np.sqrt(max(1 - h, 0.0)) * c_scaled

    # ---- unlisted nuisance covariates (CONTRACT-002) -------------------------
    # The batch artifact u is drawn INDEPENDENTLY and then ENTERS the liability with a
    # cohort-dependent sign (+ on train, - on the held-out cohort). Because the causal
    # arrow runs u -> y, no public column is a function of the hidden labels: y_test
    # can be flipped without changing a single public byte. What an all-columns parser
    # sees is a column that predicts y strongly in train and reverses out of sample.
    extra_cov_cols = {}
    nuisance_liability = np.zeros(N)
    if cfg.nuisance_cov_cols > 0:
        sign = np.where(cohort == 0, 1.0, -1.0)
        u = rng.standard_normal(N)
        nuisance_liability = cfg.nuisance_cov_strength * sign * u
        liability = liability + nuisance_liability
        extra_cov_cols["feature_1"] = u
        for k in range(1, int(cfg.nuisance_cov_cols)):
            extra_cov_cols[f"feature_{k + 1}"] = rng.standard_normal(N)

    # ---- rescale raw beta/alpha onto the effective (liability / y) axis --------
    # The generating linear predictor is
    #   eta = sqrt(h)*genetic/sqrt(gvar)
    #       + covariate_rescale*(cov_lin - mean(cov_lin)) + intercept
    # with the genetic effect-rescale k = sqrt(h)/sqrt(gvar). Storing
    # these EFFECTIVE coefficients lets  intercept_effective + C@alpha_effective +
    # Gstd@beta_effective  exactly reconstruct eta (the -covariate_rescale*mean(cov_lin)
    # centering constant is folded into intercept_effective), and makes
    # beta_effective ~ N(0, tau2_effective) self-consistent since tau2_effective =
    # k^2 * tau2_raw. When nuisance_cov_cols > 0 the generating eta additionally
    # contains truth["nuisance_liability"] (the batch artifact), which is NOT a
    # covariate of the declared model -- add it back to reconstruct eta exactly.
    eff_rescale = np.sqrt(h) / np.sqrt(gvar)        # k: raw beta -> effective axis
    beta_eff = eff_rescale * beta
    tau2_eff = (eff_rescale ** 2) * tau2
    covariate_rescale = np.sqrt(max(1 - h, 0.0)) / np.sqrt(cvar)
    alpha_eff = alpha * covariate_rescale

    # Choose the logistic intercept for the target prevalence.
    target = cfg.prevalence
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pr = 1.0 / (1.0 + np.exp(-(liability + mid)))
        if pr.mean() > target:
            hi = mid
        else:
            lo = mid
    intercept = 0.5 * (lo + hi)
    pr = 1.0 / (1.0 + np.exp(-(liability + intercept)))
    y = (rng.random(N) < pr).astype(int)
    # Fold covariate centering into the effective intercept.
    intercept_eff = intercept - covariate_rescale * cov_lin.mean()

    meta = dict(
        variant_id=np.array([f"v{j}" for j in range(P)]),
        variant_class=vclass, length=length, allele_frequency=maf,
        is_repeat=is_repeat, is_copy_number=is_cnv, block_id=block_id,
    )
    # DECOY / NOISY annotations: pure-noise per-variant columns with NO relation to
    # the true effect scale. A decoy category lists them in pgs(...) so a model that
    # blindly trusts annotations is penalized; the honest model down-weights them.
    if cfg.n_decoy_annotations > 0:
        for kdec in range(cfg.n_decoy_annotations):
            meta[f"annotation_{kdec + 1}"] = np.round(rng.standard_normal(P), 5)
    if cfg.soft_membership and cfg.soft_frac > 0.0:
        # emit the reserved SV-PGS columns (comma lists) so the model builds the
        # exact fractional class_membership_matrix used by the generator.
        meta["prior_class_members"] = np.array(
            [",".join(members[j]) for j in range(P)])
        meta["prior_class_membership"] = np.array(
            [",".join(f"{w:.4f}" for w in mweights[j]) for j in range(P)])
    # ---- causal-inclusion mask (DGP-006) ------------------------------------
    # tau2 is the SLAB variance CONDITIONAL ON INCLUSION: for a variant excluded by a
    # mask (null_frac spike-at-zero, SNV-causal thinning, MAF ceiling) the true effect
    # is EXACTLY 0 while tau2 stays at its (positive) slab value. So
    # `beta_effective ~ N(0, tau2_effective)` holds ONLY on causal_mask; off the mask
    # the effect is a point mass at zero. Suppressor-block effects are set analytically
    # (not drawn from the slab), so their tau2 is not their generating variance either
    # -- causal_mask records where the effect is nonzero, which is what an analysis
    # actually needs.
    causal_mask = beta != 0.0
    if dense_arm:
        scale_used = s_dense
        log_scale_used = log_s_dense
        class_baseline_used = {c: mu_base for c in classnames}
        gamma_len_used = cfg.dense_scale_spread * gamma_len
        gamma_repeat_used = cfg.dense_scale_spread * gamma_repeat
        class_offset_used = np.full(P, mu_base)
    else:
        scale_used = s
        log_scale_used = log_s
        class_baseline_used = {c: adj_base[c] for c in classnames}
        gamma_len_used = gamma_len
        gamma_repeat_used = gamma_repeat
        class_offset_used = class_offset
    local_scale_truth = {}
    if cfg.effect_family != "dense_gaussian":
        local_scale_truth = {
            "lam": lam,
            "delta": delta,
            "shape_a": a,
            "shape_b": b,
            "tail_temper": cfg.tail_temper,
        }
    truth = dict(
        # ---- RAW-scale generative quantities (provenance; NOT identifiable recovery
        #      targets on their own) ----
        beta_raw=beta, tau2_raw=tau2, s=scale_used, log_s=log_scale_used,
        causal_mask=causal_mask,
        effect_family=cfg.effect_family,
        beta_dense_raw=beta_dense, beta_spike_raw=beta_spike, spike_mask=spike_mask,
        suppressor_target=suppressor_target, suppressor_partner=suppressor_partner,
        cohort=cohort, ld_shift=bool(cfg.ld_shift),
        nuisance_liability=nuisance_liability,
        # global_scale_raw is NON-identifiable: it cancels exactly under the
        # beta/sqrt(gvar) renormalization, so it affects no observable and is not a
        # recovery target (kept only for provenance).
        global_scale_raw=cfg.global_scale,
        class_log_baseline=class_baseline_used,
        gamma_len=gamma_len_used, gamma_repeat=gamma_repeat_used,
        class_offset=class_offset_used,
        heritability=h,
        is_soft_membership=is_soft,
        # ---- EFFECTIVE (phenotype / liability axis) fields that actually generated
        #      y. intercept_effective + C@alpha_effective + Gstd@beta_effective
        #      reconstructs the generating linear predictor, and
        #      beta_effective ~ N(0, tau2_effective) holds (tau2_effective = k^2*tau2_raw). ----
        beta_effective=beta_eff, alpha_effective=alpha_eff,
        tau2_effective=tau2_eff, intercept_effective=intercept_eff,
        beta_dense_effective=eff_rescale * beta_dense,
        beta_spike_effective=eff_rescale * beta_spike,
        **local_scale_truth,
    )
    return dict(G=G, Gstd=Gstd, y=y, cov=C, meta=meta, truth=truth, cfg=cfg,
                classnames=classnames, cohort=cohort, extra_cov_cols=extra_cov_cols)


def _report(d):
    import collections
    m, tr = d["meta"], d["truth"]
    P = len(tr["beta_effective"])
    N = d["G"].shape[0]
    print(f"N={N} P={P} case_rate={d['y'].mean():.3f}")
    print(f"blocks={len(set(m['block_id']))}  MAF median={np.median(m['allele_frequency']):.3f}")
    print("class counts:", dict(collections.Counter(m['variant_class'])))
    # effective per-class effect concentration
    print("per-class: exp(offset)=prior scale, mean|beta_raw| among top-decile |beta|")
    for c in d["classnames"]:
        mask = m['variant_class'] == c
        if mask.sum() == 0:
            continue
        br = np.abs(tr['beta_raw'][mask])
        thr = np.quantile(br, 0.9) if mask.sum() >= 10 else 0
        top = br[br >= thr]
        print(f"  {c:22s} s~{np.exp(CLASS_LOG_BASELINE[c]):.4f} n={int(mask.sum()):4d}"
              f"  frac|b|>0.05={np.mean(br>0.05):.3f}  top|b|~{top.mean() if len(top) else 0:.3f}")
    # heritability check (realized)
    gvar = (d["Gstd"] @ tr["beta_effective"]).var()
    print(f"realized genetic var (on liability axis) ~ {gvar:.3f} (target h2={tr['heritability']})")
    # polygenicity: top-variant fraction of genetic variance + effective # variants
    b2 = tr["beta_raw"] ** 2
    contrib = b2 * d["Gstd"].var(axis=0)          # per-variant genetic-variance contribution
    frac = contrib / (contrib.sum() + 1e-12)
    eff_n = 1.0 / np.sum(frac ** 2)
    print(f"polygenicity: top-variant frac={frac.max():.3f}  top10 frac={np.sort(frac)[-10:].sum():.3f}"
          f"  effective #variants={eff_n:.1f}")


if __name__ == "__main__":
    d = generate(DGPConfig(seed=2, n_samples=3000, n_variants=1500))
    _report(d)
