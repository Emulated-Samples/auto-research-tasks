"""Single source of truth for the immutable shipped-corpus contract."""

# v8 (2026-07-18): the shipped reference becomes the exact public NumPy
# best-of-family {hierarchical_eb, ridge_logistic}; two dense regimes return to the
# grid and reference diagnostics record selection plus complete fit timing.
#
# v7 (2026-07-17): truth/anchors.json gained `metrics_naive_se` -- the bootstrap SE
# of the SINGLE naive estimate, which is the yardstick for whether an anchor's gap
# is large enough to be a skill denominator (grader/skill.py:RESOLUTION_K). v6
# anchors carry no such field and CANNOT be graded: the v6 corpus shipped five
# categories whose AUC gap was too small to divide by, and every arm from a
# 1.5-second cheat to the from-scratch gold pinned at the winsorization ceiling
# there. reference_protocol also gains `global_scale_floor`
# (reference/protocol.py:REFERENCE_PROTOCOL_VERSION = 7). See
# rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md.
#
# DELIBERATE ASYMMETRY: grader/corpus_auth.py's HMAC domain literals still read
# "schema-v6" and MUST NOT be updated to match this number. corpus_auth.py is one
# of GENERATION_PIPELINE_RELATIVE_PATHS, and that digest feeds derive_stream_seed
# and opaque_dataset_id -- so editing that file AT ALL, even a comment, changes
# every RNG stream and every dataset identity, i.e. silently regenerates the whole
# corpus under new names. The v7 change is anchor-side only; the generated data is
# byte-identical to v6, which is what makes the two directly comparable. The
# domain literal is a separator, not a claim about this constant, and the
# HMAC'd document contains schema_version anyway, so v6 and v7 anchors cannot be
# confused for each other regardless.
CORPUS_SCHEMA_VERSION = 8
MODEL_ZOO_REPORT_SCHEMA_VERSION = 8
CORPUS_BUILDER_CODE_VERSION = "build_corpus/schema-v8"
CORPUS_PURPOSE = "shipping"
# v7 temporarily dropped `dense_infinitesimal` and `low_prevalence` because raw
# SV-PGS could not anchor them. Schema v8 fixes that co-selection bias instead of
# reshaping the DGP: the exact public reference selects ridge on dense near-Gaussian
# regimes and hierarchical empirical Bayes where annotation-aware heavy-tailed
# shrinkage is appropriate. Both categories are therefore restored below. Historical evidence:
# rollout_analysis/HANDOFF_2026-07-17.md and
# rollout_analysis/DESIGN_2026-07-18_principled_best_of_family_reference.md.
SHIPPED_CATEGORIES = (
    "svld_class",
    "svld_strong",
    "sparse_heavy_tail",
    "dense_infinitesimal",
    "sparse_dense_mix",
    "rare_variant_maf",
    "low_prevalence",
    "weak_ld",
    "soft_membership",
    "nonlinear_annotation",
    "annotation_interaction",
    "decoy_annotations",
    "ld_shift",
    "suppressor_ld",
    "ancestry_shift",
)
REPLICATES_PER_CATEGORY = 3
SHIPPED_CATEGORY_COUNT = len(SHIPPED_CATEGORIES)
SHIPPED_DATASET_COUNT = SHIPPED_CATEGORY_COUNT * REPLICATES_PER_CATEGORY
# Biobank-realistic fit shape: variants outnumber training samples, matching real
# genome-wide data (e.g. All-of-Us: ~1e5-4e5 samples, ~7e5-2e6 variants). This is
# also the regime where the reference's per-variant, annotation-informed heavy-tailed
# shrinkage decisively beats a penalty-blind fit; the earlier N >> P shape both
# misrepresented real data and minimized that gap. Individual categories may still
# override the p/n ratio (it is itself a capability axis).
#
# The shape is bounded by FEASIBILITY, not only by realism. A submission gets
# FIT_TIMEOUT_S per dataset on the grading host (hyperfocal.yaml: t3.xlarge, 4 vCPU),
# so the exact public reference must finish well inside that cap -- otherwise
# skill=1.0 is unreachable by construction. The builder measures the complete
# public program (inner selection, full refit, startup, and serialization), rejects
# any fit at the cap, and the production score-1 witness must reproduce it through
# the target sandbox on the grading host before release.
# Fit cost scales with training rows, while anchor reliability scales with held-out
# rows. A 15/85 split decouples those constraints: the reference fits in the cap at
# n_train=1,800 and metrics are estimated on 10,200 held-out rows. Three replicates
# at 200 seconds preserve the exact platform budget previously used by five
# replicates at 120 seconds.
SHIPPED_REQUESTED_N = 12_000
SHIPPED_REQUESTED_P = 2_500
SHIPPED_FRAC_TRAIN = 0.15
SHIPPED_TRAIN_COUNT = int(SHIPPED_REQUESTED_N * SHIPPED_FRAC_TRAIN)
SHIPPED_TEST_COUNT = SHIPPED_REQUESTED_N - SHIPPED_TRAIN_COUNT
if not 0.0 < SHIPPED_FRAC_TRAIN < 1.0:
    raise RuntimeError("the shipped train fraction must be a proper fraction")
if SHIPPED_REQUESTED_P < SHIPPED_TRAIN_COUNT:
    raise RuntimeError("the shipped fit must remain in the P >= N regime")
DATASET_WEIGHT = 1.0
# AUC-ONLY. Discrimination is the one axis where the reference genuinely beats
# naive (gap ~0.05, 7.5x the calibration gaps); the calibration axis is
# irreducible-variance-bound (a zero-information base-rate predictor lands within
# 0.0017 Brier of naive; see grader/skill.py). Calibration is therefore a
# CONTINUOUS MULTIPLIER (calibration_factor in skill.py), NOT an additive arm --
# weighting brier/log_loss paid free credit to a constant predictor and made the
# scored objective vary per-replicate wherever the tiny brier gap flipped across the
# adequacy bar (historically active on 25/39 pre-v8 datasets under the old
# 0.6/0.2/0.2 weights).
# AUC earns the credit; reference-relative proper-score regret continuously
# discounts only positive AUC skill. Thus exact reference remains 1, naive remains
# 0, prevalence-compressed rankers cannot keep full credit, and severe
# overconfidence tends smoothly toward zero without erasing negative AUC signal.
METRIC_WEIGHTS = {"auc": 1.0}
# The dev-gate CANDIDATE SELECTOR (reference/baselines.py, zoo_protocol.py) picks a
# best all-around candidate and legitimately weighs all three metrics; that is a
# different purpose from grade-time SCORING (AUC-only above). Kept separate so the
# grader can be AUC-only without forcing the selector to choose candidates blind to
# calibration. Selector weights never touch a shipped score.
SELECTION_METRIC_WEIGHTS = {"auc": 0.60, "brier": 0.20, "log_loss": 0.20}
BUILD_TIMEOUT_S = 3600
DATASET_TIMEOUT_S = 200
# Fitting is the task; predicting is a dot product (measured well under a second).
# The old 72/48 split starved the fit to fund a predict step that never needed it, so
# it is rebalanced inside the same dataset cap -- the platform grid below is untouched.
FIT_TIMEOUT_S = 170
PREDICT_TIMEOUT_S = 30
# The feasibility invariant stated above, made ENFORCEABLE. It was stated in prose
# and checked by nothing, and the v6 corpus shipped violating it: svld_class
# d_b8d08f3cb532 (198.2 s) and soft_membership d_e15a4ad20835 (188.0 s) both
# exceed FIT_TIMEOUT_S, so on those two datasets a submission that EXACTLY
# reproduced the reference is SIGKILLed and scores INVALID_REWARD = -0.5. skill=1.0
# is unreachable there by construction -- precisely what the comment above forbids.
# See rollout_analysis/ADDENDUM_10_2026-07-17_the_speed_axis_pays_the_shortcut.md.
#
# HARD: a reference fit at or beyond the cap makes the dataset's anchor
# unreachable. The build fails rather than shipping it.
REFERENCE_FIT_FEASIBILITY_FRACTION = 1.0
# WARN: the comment above asks for "well inside", not "just inside". A reference
# burning >80% of the cap leaves a submission no room for a less-optimized
# reimplementation or a slower grading host, and the anchors are timed on the BUILD
# host while grading runs on t3.xlarge / 4 vCPU. 7/45 of the v6 corpus sat in this
# band (p90 = 147.4 s).
#
# This is deliberately a WARNING and not an error: the build host and grading host
# are different machines. Fresh per-cell build timings plus the target-sandbox
# score-1 witness determine whether the warning band is acceptable; historical
# hidden-SV-PGS timings are not evidence for this public NumPy reference.
REFERENCE_FIT_HEADROOM_FRACTION = 0.8
TRUSTED_OVERHEAD_RESERVE_S = 1800
# = BUILD_TIMEOUT_S + SHIPPED_DATASET_COUNT*DATASET_TIMEOUT_S + TRUSTED_OVERHEAD_RESERVE_S.
# 45 datasets (15 categories x 3): 3600 + 45*200 + 1800 = 14400; platform adds the reserve.
WRAPPER_GRADER_TIMEOUT_S = 14_400
PLATFORM_VERIFIER_TIMEOUT_S = 16_200

if FIT_TIMEOUT_S + PREDICT_TIMEOUT_S != DATASET_TIMEOUT_S:
    raise RuntimeError("fit and predict caps must exactly fill the dataset cap")
if (
    BUILD_TIMEOUT_S
    + SHIPPED_DATASET_COUNT * DATASET_TIMEOUT_S
    + TRUSTED_OVERHEAD_RESERVE_S
    != WRAPPER_GRADER_TIMEOUT_S
):
    raise RuntimeError("shipping grid does not fit the grader wrapper timeout")
if (
    WRAPPER_GRADER_TIMEOUT_S + TRUSTED_OVERHEAD_RESERVE_S
    != PLATFORM_VERIFIER_TIMEOUT_S
):
    raise RuntimeError("platform verifier timeout does not include its reserve")
# Headline score at or above which the benchmark is considered MASTERED -- the
# pass@k mastery event (validation/passk.py). Two thresholds, BOTH required
# (headline AND weakest-fifth), so a lopsided rollout with a collapsed tail cannot
# master on a high headline alone.
#
# RELEASE POLICY: mastery is anchored to the FRONTIER CEILING, i.e. the
# highest headline a current frontier model (Opus 4.8) reaches on the CURRENT
# (calibration-discounted AUC, tail-0.4) grader, not the demigod bar (headline
# ~1.0, which nothing reaches). pass@k is then "the fraction of rollouts that
# reach the frontier ceiling" -- a relative benchmark that is low-but-NONZERO by
# construction, rather than a flat 0 at the unreachable demigod bar. This is deliberately NOT an
# absolute reference-grade bar; it is a moving frontier reference.
#
# CALIBRATION: schema v8 changes both the category grid and its reference scale, so
# v7 frontier values are incommensurate and MUST NOT be carried forward. Keep the
# event explicitly uncalibrated for the initial scientific release, run a versioned
# Opus calibration batch on that exact release, freeze the frontier thresholds, then
# evaluate pass@k on a disjoint batch. Mastery values are deliberately excluded from
# the model-zoo scientific-contract digest so freezing them does not invalidate the
# 45-cell corpus/dev evidence.
MASTERY_THRESHOLD = None
# The weakest-fifth category mean must also clear this, so a lopsided rollout with
# a high headline but a broken capability cannot master. Production reads BOTH
# (grade.py exports both; the wrapper gates on both) -- previously only the
# headline was checked, so a 1.05/0.20 rollout falsely passed. Set just under the
# v8 calibration freezes this together with the headline threshold above.
MASTERY_TAIL_THRESHOLD = None
WEIGHT_RULE = "equal_per_dataset"
ANCHOR_HMAC_FIELD = "anchor_hmac_sha256"
MANIFEST_HMAC_FIELD = "manifest_hmac_sha256"
DEVELOPMENT_RESULT_HMAC_FIELD = "development_result_hmac_sha256"
DEVELOPMENT_REPORT_HMAC_FIELD = "development_report_hmac_sha256"
DEVELOPMENT_REPORT_RELATIVE_PATH = "validation/model_zoo_development.json"
GENERATION_CONTRACT_ITEMS = (
    ("categories", SHIPPED_CATEGORIES),
    ("replicates_per_category", REPLICATES_PER_CATEGORY),
    ("requested_n", SHIPPED_REQUESTED_N),
    ("requested_p", SHIPPED_REQUESTED_P),
    ("frac_train", SHIPPED_FRAC_TRAIN),
)
# Only sources and immutable controls that can change generated dataset bytes
# belong in this digest. Reference, grader, scoring, and corpus-orchestration
# changes have their own provenance hashes and must not silently rekey the DGP;
# stable generated data are necessary for paired protocol comparisons.
GENERATION_PIPELINE_RELATIVE_PATHS = (
    "datagen/categories.py",
    "datagen/dgp.py",
    "datagen/materialize.py",
    "grader/corpus_auth.py",
)
