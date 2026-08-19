"""Category recipes for the SV-PGS outcome-benchmark corpus.

Each category is a DGPConfig recipe (as a positional `make_cfg(replicate, N, P)`
factory), the
binary GLM `family`, and the annotation `prior_cols` the task exposes.

The canonical grid deliberately changes the winning inductive bias: dense and
sparse effects, light and heavy tails, rare variants/outcomes, weak through
signed/very-strong LD, nonlinear/interacting/decoy annotations, soft class
membership, and ancestry shift. These latent regimes are private generation and
analysis metadata. Solvers must infer useful structure from the training data.

The checked corpus may be replaced only after the full-size three-replicate shipping
gate demonstrates reference separation, useful between-model variance, and low
category redundancy.  Claims from an older three-category pilot do not define
this grid.
"""
from __future__ import annotations

from datagen.dgp import DGPConfig
from grader.contract import SHIPPED_CATEGORIES, SHIPPED_FRAC_TRAIN

# reserved/annotation columns each task exposes in variant_metadata.tsv
_STD_COLS = ("variant_class", "sv_log_length", "repeat_overlap")
_MAF_COLS = _STD_COLS + ("allele_frequency",)
_SOFT_COLS = ("variant_class", "prior_class_members", "prior_class_membership",
              "sv_log_length", "repeat_overlap")


def _cfg(seed, N, P, **kw):
    base = dict(seed=seed, n_samples=N, n_variants=P)
    base.update(kw)
    return DGPConfig(**base)


def _sparse_heavy_tail_cfg(seed, N, P):
    """Keep this regime computationally bounded while making p > training n.

    Use the full requested cohort so the sparse effects are learnable at the
    canonical 15% training split. Keep at least 1.2 variants per training row so
    the category still exercises the intended underdetermined fit.
    """
    n_samples = N
    training_count = int(SHIPPED_FRAC_TRAIN * n_samples)
    n_variants = max(P, (6 * training_count + 4) // 5)
    # Heavy-tailed BUT broadly polygenic: hundreds of causal variants with a heavy
    # (not monogenic) effect-size tail, matching how complex traits actually behave
    # (thousands of small effects, some loci larger). The earlier null_frac=0.85 /
    # snv_causal_frac=0.15 made this near-oligogenic (a handful of large effects),
    # which is not realistic for common disease; the heavy tail alone (low
    # tail_temper) already rewards the reference's per-variant adaptive shrinkage.
    return _cfg(
        seed, N, P, n_samples=n_samples, n_variants=n_variants,
        heritability=0.85, prevalence=0.3, snv_frac=0.7, small_indel_frac=0.1,
        snv_causal_frac=0.6, null_frac=0.4, causal_maf_max=0.5,
        ld_rho_lo=0.4, ld_rho_hi=0.7, class_scale_spread=2.0,
        tail_temper=1.9,
    )


# CAPABILITY MATRIX (P1 hardening, 2026-07-13). The old corpus was three
# near-clones (svld_rare_poly / svld_strong / svld_class differ only in LD range
# and class-scale spread), so one well-tuned annotation-adaptive elastic net wins
# everywhere and the env cannot separate algorithms. This matrix spans GENUINELY
# DIFFERENT statistical regimes -- each exercises a distinct capability, so
# different model families should top different categories (between-model variance
# >> seed noise). Every recipe uses faithful private DGP knobs; make_cfg
# may override n_samples / n_variants (the p/n ratio is itself a capability axis).
#
# `architecture` is deliberately private. It supports capability-matrix audits and
# shipping-gate interpretation, but is never serialized into solver-visible files.
#
# The shipping gate in validation/model_zoo.py measures between-model variance,
# category correlation, and reference advantage before this exact grid ships.
CATEGORIES = {
    # ---- (1) annotation-adaptive, strong LD, wide class spread: the class
    # annotation decides effect scale; annotation-weighted joint shrinkage wins. ----
    "svld_class": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="all variant classes causal, strong LD, wide per-class prior-scale "
             "spread; the class annotation, not marginal z, decides effect scale",
        architecture=dict(kind="annotation_adaptive", sparsity="polygenic",
                          tail="heavy", ld="strong", annotation="linear_scale",
                          shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, P, heritability=0.9, prevalence=0.3,
            snv_frac=0.55, small_indel_frac=0.05, snv_causal_frac=1.0,
            causal_maf_max=0.5, ld_rho_lo=0.7, ld_rho_hi=0.9,
            class_scale_spread=3.0, tail_temper=2.5),
    ),
    # ---- (2) very strong LD: marginal != joint; joint decorrelation wins,
    # marginal P+T double-counts. ----
    "svld_strong": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="all variant classes causal, very strong LD (0.80-0.95); marginal "
             "double-counting sinks P+T below the joint decorrelating fit",
        architecture=dict(kind="ld_decorrelation", sparsity="polygenic",
                          tail="heavy", ld="very_strong", annotation="linear_scale",
                          shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 2000), heritability=0.95, prevalence=0.3,
            snv_frac=0.55, small_indel_frac=0.05, snv_causal_frac=1.0,
            causal_maf_max=0.5, ld_rho_lo=0.8, ld_rho_hi=0.95,
            class_scale_spread=2.5, tail_temper=2.5),
    ),
    # ---- (3) sparse + heavy-tailed, p > n_train: a broad heavy-tailed signal with
    # a substantial null component; blind ridge over-shrinks the spikes. ----
    "sparse_heavy_tail": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="p > training n; a substantial null component plus heavy-tailed effects; "
             "heavy-tailed local shrinkage (horseshoe/adaptive lasso) beats ridge",
        architecture=dict(kind="sparse_spike", sparsity="sparse", tail="very_heavy",
                          ld="moderate", annotation="linear_scale", shift="none",
                          p_gt_n=True),
        make_cfg=_sparse_heavy_tail_cfg,
    ),
    # ---- (4) dense infinitesimal: every variant causal with a tiny near-Gaussian
    # effect; a plain ridge is the RIGHT tool and heavy-tailed sparsity shrinkage is
    # the WRONG one. Re-added 2026-07-18 after the raw-SV-PGS-only reference was
    # retired; the principled best-of-family reference anchors it via ridge. ----
    "dense_infinitesimal": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="every variant causal with a tiny near-Gaussian effect (no spikes); "
             "dense ridge / infinitesimal shrinkage wins, lasso-style sparsity loses",
        architecture=dict(kind="dense_infinitesimal", sparsity="dense",
                          tail="light", ld="moderate", annotation="weak_scale",
                          shift="none"),
        # This is the one capability axis where N >> P is intentional: thousands of
        # tiny effects are not identifiable in the corpus-wide p/n regime.
        make_cfg=lambda s, N, P: _cfg(
            s, N, 800, heritability=0.85, prevalence=0.3, snv_frac=0.7,
            small_indel_frac=0.1, snv_causal_frac=1.0, null_frac=0.0,
            causal_maf_max=0.5, ld_rho_lo=0.5, ld_rho_hi=0.8,
            class_scale_spread=1.0, tail_temper=3.0,
            # DGP-001: a genuinely infinitesimal architecture -- Gaussian effects on a
            # near-homoskedastic scale. No TPB local scales, no class-scale spread, no
            # spike at zero, so nothing rewards a sparsity prior.
            effect_family="dense_gaussian", dense_scale_spread=0.15),
    ),
    # ---- (5) sparse + dense mixture: every variant has a small Gaussian effect and
    # a sparse subset receives an additional heavy-tailed effect. ----
    "sparse_dense_mix": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="every variant has a small near-Gaussian effect, while a sparse subset "
             "receives an additional high-variance effect; both components carry "
             "substantial genetic variance",
        architecture=dict(kind="global_local_mixture", sparsity="mixture",
                          tail="heavy", ld="moderate", annotation="linear_scale",
                          shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 2000), heritability=0.95, prevalence=0.3, snv_frac=0.65,
            small_indel_frac=0.1, snv_causal_frac=1.0, null_frac=0.0,
            causal_maf_max=0.5, ld_rho_lo=0.5, ld_rho_hi=0.85,
            class_scale_spread=2.5, tail_temper=1.8,
            # DGP-002: TWO explicit components -- a dense floor on every variant plus
            # heavy-tailed spikes on 2% of variants -- with the realized genetic-variance
            # share split 50/50, so neither a pure dense nor a pure sparse prior is right.
            effect_family="dense_plus_spike", dense_var_share=0.5, spike_frac=0.02,
            dense_scale_spread=0.15),
    ),
    # ---- (6) rare-variant regime: only low-MAF variants are causal; MAF-aware
    # weighting / rare-variant handling wins over MAF-blind standardization. ----
    "rare_variant_maf": dict(
        family="binomial-logit", prior_cols=_MAF_COLS, ancestry_shift=0.0,
        desc="only rare (low-MAF) variants carry effects; MAF-dependent weighting "
             "beats MAF-blind standardized shrinkage",
        architecture=dict(kind="rare_variant", sparsity="polygenic", tail="heavy",
                          ld="moderate", annotation="linear_scale", shift="none",
                          maf="rare_causal"),
        # RARE but broadly POLYGENIC: only low-frequency (MAF<=0.08) variants are causal, but
        # MANY of them contribute (heavy-tailed effect sizes, yet spread across ~130
        # effective variants) -- realistic rare-variant polygenicity, NOT a handful of
        # monster rare variants (the old sparse setting concentrated the trait onto ~12
        # effective variants, which is not how complex traits work). The heavy tail +
        # rare MAF spectrum keep MAF-aware, per-variant adaptive shrinkage the rewarded
        # capability.
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 2000), heritability=0.95, prevalence=0.3, snv_frac=0.6,
            small_indel_frac=0.1, snv_causal_frac=0.9, null_frac=0.05,
            causal_maf_max=0.08, ld_rho_lo=0.5, ld_rho_hi=0.8,
            class_scale_spread=1.5, tail_temper=3.5),
    ),
    # ---- (7) low prevalence: rare binary outcome (~0.15); calibration (Brier/
    # log-loss) and class-imbalance handling are load-bearing, not just AUC ranking.
    # Re-added 2026-07-18 alongside dense_infinitesimal; the exact public
    # hierarchical-EB/ridge family can select ridge in this dense counter-regime. ----
    "low_prevalence": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="low case prevalence (~0.15); calibrated probability + imbalance "
             "handling is load-bearing, not just AUC ranking",
        architecture=dict(kind="low_prevalence", sparsity="polygenic", tail="heavy",
                          ld="strong", annotation="linear_scale", shift="none",
                          prevalence="low"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, 800, heritability=0.95, prevalence=0.15, snv_frac=0.55,
            small_indel_frac=0.05, snv_causal_frac=1.0, null_frac=0.0,
            causal_maf_max=0.5, ld_rho_lo=0.7, ld_rho_hi=0.9,
            class_scale_spread=1.0, tail_temper=3.0,
            effect_family="dense_gaussian", dense_scale_spread=0.15),
    ),
    # ---- (8) weak LD: marginal ~= joint, so P+T / marginal methods are
    # COMPETITIVE; the joint-decorrelation edge nearly vanishes -> flips the model
    # ranking vs the strong-LD categories (adds between-model variance). ----
    "weak_ld": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="weak LD (0.05-0.25): marginal association ~= joint, so simple "
             "marginal/P+T predictors are competitive and the decorrelation edge shrinks",
        architecture=dict(kind="weak_ld", sparsity="sparse", tail="heavy",
                          ld="weak", annotation="linear_scale", shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 800), heritability=0.85, prevalence=0.3, snv_frac=0.7,
            small_indel_frac=0.1, snv_causal_frac=0.25, null_frac=0.65,
            causal_maf_max=0.5, ld_rho_lo=0.05, ld_rho_hi=0.25,
            class_scale_spread=1.5, tail_temper=2.3),
    ),
    # ---- (9) soft class membership: variants belong FRACTIONALLY to classes;
    # a model that honors the soft-membership annotation prices priors right, a
    # hard 1-of-C classifier cannot. ----
    "soft_membership": dict(
        family="binomial-logit", prior_cols=_SOFT_COLS, ancestry_shift=0.0,
        desc="fractional class membership (prior_class_members/_membership); a "
             "model that blends the soft class scales beats a hard 1-of-C classifier",
        architecture=dict(kind="soft_membership", sparsity="polygenic", tail="heavy",
                          ld="strong", annotation="soft_class_scale", shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 1200), heritability=0.95, prevalence=0.3, snv_frac=0.55,
            small_indel_frac=0.05, snv_causal_frac=1.0, causal_maf_max=0.5,
            ld_rho_lo=0.7, ld_rho_hi=0.9, class_scale_spread=4.0, tail_temper=8.0,
            soft_membership=True, soft_frac=0.75),
    ),
    # ---- (10) NONLINEAR annotation: the true effect-scale is U-shaped in
    # log-length (nonzero quadratic term); a LINEAR annotation-weighted penalty
    # captures only part, a spline/nonparametric annotation model wins. ----
    "nonlinear_annotation": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="effect scale is U-shaped (nonlinear) in sv_log_length; a linear "
             "annotation-weighted penalty under-fits, a spline annotation model wins",
        architecture=dict(kind="nonlinear_annotation", sparsity="polygenic",
                          tail="heavy", ld="strong", annotation="nonlinear_scale",
                          shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 800), heritability=0.9, prevalence=0.3, snv_frac=0.55,
            small_indel_frac=0.05, snv_causal_frac=0.3, null_frac=0.75,
            causal_maf_max=0.5, ld_rho_lo=0.7, ld_rho_hi=0.9,
            class_scale_spread=2.0, tail_temper=2.3,
            length_nonlinear_coef=2.0),
    ),
    # ---- (11) annotation INTERACTION: length matters only for copy-number
    # variants (class x length); an additive annotation model misses it. ----
    "annotation_interaction": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="length modulates effect scale only for copy-number classes "
             "(class x length interaction); additive annotation models miss it",
        architecture=dict(kind="annotation_interaction", sparsity="polygenic",
                          tail="heavy", ld="strong", annotation="interaction_scale",
                          shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 1200), heritability=0.9, prevalence=0.3, snv_frac=0.5,
            small_indel_frac=0.05, snv_causal_frac=0.3, null_frac=0.75,
            causal_maf_max=0.5, ld_rho_lo=0.7, ld_rho_hi=0.9,
            class_scale_spread=2.0, tail_temper=2.3,
            class_length_interaction=2.0),
    ),
    # ---- (12) DECOY / noisy annotations: 3 extra annotation columns carrying NO
    # signal; blindly trusting annotations HURTS, the honest model down-weights. ----
    "decoy_annotations": dict(
        family="binomial-logit",
        prior_cols=_STD_COLS + ("annotation_1", "annotation_2", "annotation_3"),
        ancestry_shift=0.0,
        desc="3 candidate annotation columns carry no effect signal, and the covariate "
             "table carries UNLISTED nuisance columns (one a batch artifact whose "
             "association with the phenotype reverses on the held-out cohort); a model "
             "that assigns every declared annotation a nonzero effect-scale weight, or "
             "that swallows columns the formula does not list, is punished",
        architecture=dict(kind="decoy_annotations", sparsity="polygenic",
                          tail="heavy", ld="strong", annotation="decoy_present",
                          shift="batch_artifact"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, P, heritability=0.95, prevalence=0.3, snv_frac=0.55,
            small_indel_frac=0.05, snv_causal_frac=1.0, causal_maf_max=0.5,
            ld_rho_lo=0.7, ld_rho_hi=0.9, class_scale_spread=2.5, tail_temper=2.5,
            n_decoy_annotations=3,
            # CONTRACT-002: 2 covariate columns written to disk but NOT in the formula.
            # batch_artifact predicts y in train and REVERSES on test, so an
            # all-columns fallback parser loses measurable held-out score.
            nuisance_cov_cols=2, nuisance_cov_strength=0.7),
    ),
    # ---- (13) LD SHIFT: train and test cohorts are generated from DIFFERENT block
    # correlation matrices (causal effects and allele frequencies held fixed), so a
    # variant that tags the causal one in train stops tagging it in test. A tag-only /
    # marginal predictor degrades; weight placed on the causal variants transfers. ----
    "ld_shift": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="train and test cohorts have DIFFERENT LD (block correlation) structure "
             "with identical causal effects and allele frequencies; the causal variants "
             "are distinguishable from the variants that merely tag them only by their "
             "annotations, so a predictor that spreads weight over a block's tags "
             "transfers poorly",
        architecture=dict(kind="ld_shift", sparsity="sparse", tail="heavy",
                          ld="strong", annotation="linear_scale", shift="ld"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 1200), heritability=0.8, prevalence=0.3, snv_frac=0.65,
            small_indel_frac=0.1,
            # NO SNV-like variant is causal here. Inside a block the causal SV and the
            # SNVs that tag it are statistically INDISTINGUISHABLE in the training
            # cohort (r ~ 0.9), so the only way to tell them apart is the ANNOTATION.
            # Without this the LD shift is not a capability test at all: it just adds an
            # unlearnable penalty that happens to hurt weight-spreading joint models
            # MORE than a marginal screen (measured: gap-to-oracle widened +0.017 for
            # ridge vs +0.004 for P+T -- the opposite of the intended ranking).
            snv_causal_frac=0.0, null_frac=0.2,
            causal_maf_max=0.5, ld_rho_lo=0.80, ld_rho_hi=0.95,
            class_scale_spread=2.5, tail_temper=2.2,
            ld_shift=True, ld_shift_rho_lo=0.0, ld_shift_rho_hi=0.25,
            frac_train_hint=SHIPPED_FRAC_TRAIN),
    ),
    # ---- (14) SUPPRESSOR LD: selected high-positive-LD blocks carry one sparse
    # equal-and-opposite effect pair, attenuating both pair marginals by (1-r). ----
    "suppressor_ld": dict(
        family="binomial-logit", prior_cols=_STD_COLS, ancestry_shift=0.0,
        desc="selected high-positive-LD blocks contain a sparse opposing-effect pair "
             "whose partner OVERSHOOTS, so the pair's marginal association is reversed "
             "in sign relative to its joint effect, not merely attenuated",
        # tail is LIGHT, not heavy: the constructed pairs carry most of the genetic
        # variance and their magnitudes are set by the pair algebra, so the effect
        # distribution is bimodal with no extreme outliers (measured excess kurtosis
        # 0.17). Declaring "heavy" here would be a FALSE public disclosure -- caught by
        # test_public_prose_claims_match_the_measured_instance.
        architecture=dict(kind="suppressor_ld", sparsity="sparse", tail="light",
                          ld="opposing_effect", annotation="linear_scale", shift="none"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 1200), heritability=0.95, prevalence=0.3, snv_frac=0.6,
            small_indel_frac=0.1, snv_causal_frac=0.4, null_frac=0.6,
            causal_maf_max=0.5,
            # Strong positive correlation makes equal-and-opposite effects difficult
            # for a marginal screen while remaining identifiable to a joint fit.
            ld_rho_lo=0.95, ld_rho_hi=0.99, ld_block_min=5, ld_block_max=12,
            class_scale_spread=2.0, tail_temper=2.2,
            suppressor_block_frac=0.35, suppressor_min_corr=0.55,
            suppressor_var_share=0.8),
    ),
    # ---- (15) ANCESTRY SHIFT: test cohort enriched for the high-PC1 tail (train
    # leans low-PC1); covariate adjustment must generalize across the shift. ----
    "ancestry_shift": dict(
        family="binomial-logit", prior_cols=_STD_COLS,
        ancestry_shift=0.75,
        desc="train/test differ in ancestry (PC1) distribution; a model that "
             "overfits train ancestry transfers poorly, robust covariate adjustment wins",
        architecture=dict(kind="ancestry_shift", sparsity="polygenic", tail="heavy",
                          ld="strong", annotation="linear_scale", shift="ancestry"),
        make_cfg=lambda s, N, P: _cfg(
            s, N, min(P, 2000), heritability=0.95, prevalence=0.3, snv_frac=0.55,
            small_indel_frac=0.05, snv_causal_frac=1.0, causal_maf_max=0.5,
            ld_rho_lo=0.7, ld_rho_hi=0.9, class_scale_spread=2.5, tail_temper=2.5,
            ancestry_confounding=0.5),
    ),
}

if tuple(CATEGORIES) != SHIPPED_CATEGORIES:
    raise RuntimeError(
        "category recipes must exactly match grader.contract.SHIPPED_CATEGORIES"
    )
