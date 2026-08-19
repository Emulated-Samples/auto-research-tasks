"""Skill normalization + reward composition for the outcome env.

Per metric, skill = (m_sub - m_naive) / (m_ref - m_naive), oriented so that
higher-is-better (lower-is-better metrics are sign-flipped first). Naive scores
0, the principled best-of-family reference scores 1, and beating it scores > 1.

The active discrimination skill is AUC-only. Positive AUC skill is multiplied by
a continuous proper-score calibration factor; nonpositive AUC skill is unchanged.
A below-naive fit therefore remains genuinely negative (bounded below by
SKILL_LO), while prevalence-compressed or overconfident rankers cannot retain full
credit. A separate FLOORED value is kept for display only. Runtime is enforced by
hard fit/predict caps and reported separately; it never changes scientific reward.
benchmark = a tail-aware aggregate of the per-category rewards. Every category
receives the same mean-component weight and the weakest fifth of categories share
the bottom-tail component, so a populous category cannot dominate and a method
cannot coast on a few favorite regimes.
"""
from __future__ import annotations
import numpy as np

from grader.contract import METRIC_WEIGHTS


# Winsorization bounds for a per-metric skill BEFORE aggregation: clamp to
# [SKILL_LO, SKILL_HI] so one lucky/unlucky metric on one replicate cannot swing
# the category mean (a below-naive blowup or a huge above-reference spike is
# mostly replicate noise, not capability). Headroom above 1 is retained (a genuinely
# better method still scores >1) but bounded.
SKILL_LO, SKILL_HI = -0.5, 1.5
# Reliability multiplier: metrics whose reference-minus-naive gap is smaller
# than K_SE standard errors are excluded before the remaining metric weights are
# renormalized. The denominator itself remains the actual anchor gap, so the
# defining invariants naive=0 and reference=1 stay exact.
#
# K_SE tests whether the gap is REAL. It is NOT, and cannot be, a test of whether
# the gap is LARGE ENOUGH TO DIVIDE BY -- see RESOLUTION_K below, which is.
K_SE = 2.0

# --- ADEQUACY: an absolute floor on the anchor gap ---------------------------
#
# `skill = (sub - naive) / (ref - naive)` is a measurement only where the
# denominator is large enough to divide by. Where the gap is tiny, the ratio
# amplifies ordinary held-out noise until every arm hits a winsorization bound
# and the "score" becomes the arithmetic of SKILL_LO/SKILL_HI. Measured on the
# shipped corpus 2026-07-17 (rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md):
# five categories carry a naive->reference AUC gap of 0.0047-0.0208, and on every
# one of them a 1.5-second marginal prune+threshold cheat, a uniform ridge AND the
# from-scratch gold arm ALL pin at +1.500 AUC skill. Half of the cheat's headline
# was minted there.
#
# WHY K_SE CANNOT BE THE GATE, AND WHY RAISING IT WOULD NOT HELP.
# The stored SE is a PAIRED bootstrap of (reference - naive). As a reference
# collapses ONTO naive, the gap AND its paired SE shrink together, so their ratio
# is near scale-invariant. Measured over the 45 shipped datasets:
#     corr(gap, SE_paired) = +0.925      <- the SE shrinks WITH the gap
#     corr(gap, z)         = +0.121      <- z says almost nothing about gap size
# The SMALLEST gap in the corpus (nonlinear_annotation, +0.0019 AUC) scores
# z = 17.5 -- HIGHER than the LARGEST gap (ancestry_shift, +0.1035, z = 16.2). A
# collapsed anchor produces a precisely-measured tiny gap, which is the WORST
# case, not the safest one. No value of K_SE separates these.
#
# THE FIX. Gate on the gap in units of the metric's own SINGLE-estimate standard
# error (`se_naive`), which -- unlike the paired SE -- is a property of the data
# and the naive predictor and does NOT collapse when the reference collapses.
#
# CHOOSING RESOLUTION_K, derived and pre-registered, not tuned to a pass/fail
# split: a submission's own held-out metric carries noise of ~se_naive, so its
# skill inherits noise of ~se_naive/gap. Requiring that noise to stay at or below
# 0.25 -- i.e. the naive->reference interval resolves into at least ~4 levels,
# against a winsorized range [SKILL_LO, SKILL_HI] that is 2.0 wide -- gives
# gap >= 4 * se_naive. Below that the skill on that metric is measuring noise.
RESOLUTION_K = 4.0


def metric_skill(m_sub, m_naive, m_ref, higher_is_better, *, se, se_naive,
                 k_se=K_SE, resolution_k=RESOLUTION_K,
                 winsorize=True):
    """Normalize one metric to naive=0 / reference=1.

    Two independent exclusion tests, because "the gap is real" and "the gap is
    big enough to be a denominator" are different properties and each has its own
    yardstick:

    * RELIABILITY (``se``, the paired bootstrap SE of reference-minus-naive):
      exclude when ``gap < k_se * se``. Catches a gap that is noise.
    * ADEQUACY (``se_naive``, the bootstrap SE of the single naive estimate):
      exclude when ``gap < resolution_k * se_naive``. Catches a gap that is real
      but too small to divide by. See RESOLUTION_K.

    Either test excludes the metric and the composite renormalizes over the
    remaining metrics. For an active metric the actual gap is always the
    denominator, preserving exact naive=0 and reference=1 semantics. The result
    is winsorized to [SKILL_LO, SKILL_HI] unless disabled.

    Both uncertainty estimates are mandatory. Schema v8 rejects an anchor that
    cannot prove both reliability and denominator adequacy.
    """
    if not higher_is_better:
        m_sub, m_naive, m_ref = -m_sub, -m_naive, -m_ref
    gap = m_ref - m_naive
    if not np.isfinite(se) or se <= 0.0 or not np.isfinite(se_naive) or se_naive <= 0.0:
        raise ValueError("skill standard errors must be finite and positive")
    reliability_threshold = k_se * se
    adequacy_threshold = resolution_k * se_naive
    if gap <= 1e-9 or gap < reliability_threshold or gap < adequacy_threshold:
        # Exclude an unusable metric instead of changing what reference=1 means.
        return float("nan")
    s = (m_sub - m_naive) / gap
    if winsorize:
        s = min(SKILL_HI, max(SKILL_LO, s))
    return s


def clamp_report(per_metric):
    """Report which active scored skills hit a winsorization bound.

    The current scientific arm is AUC-only, so this normally reports whether the
    AUC ratio has saturated at ``SKILL_LO`` or ``SKILL_HI``.  It does not change
    reward; it prevents a bound-saturated value from being presented as an
    unbounded measurement.  Calibration has its own continuous factor and is not
    inserted into this per-metric skill mapping.
    """
    clamped = sorted(
        name for name, value in per_metric.items()
        if value <= SKILL_LO + 1e-12 or value >= SKILL_HI - 1e-12
    )
    return {
        "clamped_metrics": clamped,
        "clamped_fraction": (len(clamped) / len(per_metric)) if per_metric else 0.0,
        # The alarm: no active metric landed strictly inside its bounds, so the
        # composite is determined entirely by SKILL_LO/SKILL_HI and the weights.
        "all_metrics_clamped": bool(per_metric) and len(clamped) == len(per_metric),
    }


# --- calibration: continuous proper-score regret, not a weighted arm --------
#
# Measured 2026-07-15 over all 45 shipped datasets (median Brier):
#
#     base-rate constant predictor (ZERO information) : 0.20985
#     naive (covariates-only GLM)                     : 0.20816
#     historical v6 SV-PGS reference                  : 0.20225
#
# A predictor that knows LITERALLY NOTHING -- no genotypes, no covariates, just
# the training prevalence -- lands within 0.0017 Brier of naive, which is 0.25x
# the entire naive->reference gap of 0.0067. The cause is structural, not a weak
# anchor: base-rate Brier IS p(1-p), and at prevalence ~0.293 that is 0.207, so
# the whole observed Brier range [0.202, 0.210] is an ~0.008 sliver sitting on
# irreducible Bernoulli variance no model can remove. A model with no
# discrimination is automatically well calibrated, so (ref - naive) on a
# calibration axis is near-zero HOWEVER good the reference is.
#
# Calibration must not be an additive arm: a constant predictor would receive
# free reward. But the old all-or-zero gate against naive had the opposite
# exploit: a rank-perfect predictor could compress every probability arbitrarily
# close to prevalence, remain inside the loose gate, and receive FULL AUC credit.
# It also introduced a discontinuity at an arbitrary boundary.
#
# The scientific objective is AUC skill TIMES a continuous calibration factor.
# For each proper score, measure only WORSE-THAN-REFERENCE absolute regret and
# map it through exp(-regret/scale); a better proper score keeps factor 1. The
# geometric mean makes both proper scores load-bearing without paying an additive
# calibration reward. Exact reference metrics give factor 1 exactly. A ranker
# compressed toward prevalence has reference-grade AUC but naive-like Brier and
# log-loss, hence factor <1. Severe overconfidence has large proper-score regret
# and a factor tending smoothly to zero.
#
# These scales are PREDECLARED absolute proper-score units, not normalized by a
# tiny/noisy naive->reference gap and not fitted to the new corpus. Brier regret
# 0.02 is two percentage points of probability MSE; log-loss regret 0.05 is about
# a 5.1% per-observation likelihood penalty (exp(0.05)). One average scale of
# regret retains exp(-1) ~= 0.37 of positive AUC credit. These are the same
# absolute magnitudes previously used by the loose gate, now applied continuously
# relative to the score-1 reference.
CALIBRATION_METRICS = ("brier", "log_loss")
CALIBRATION_REGRET_SCALE = {"brier": 0.02, "log_loss": 0.05}


def reference_calibration_qualification(reference_metrics, naive_metrics):
    """Qualify the score-1 reference before it is allowed to define the scale.

    The submission-side factor is necessarily relative to the authenticated
    reference, so the reference itself always has factor one.  That invariant is
    useful only if the reference is not materially worse-calibrated than the
    covariates-only naive model.  This build/release qualification closes that
    otherwise-circular hole with the same predeclared absolute proper-score units
    used by the continuous factor.

    Inputs are scalar metric mappings (the exact shape stored in anchors).  Both
    proper scores are mandatory and finite.  Better-than-naive reference metrics
    have zero regret; worse metrics may use at most one declared regret scale.
    """
    if type(reference_metrics) is not dict or type(naive_metrics) is not dict:
        raise ValueError("reference calibration metrics must be scalar mappings")
    missing = [name for name in CALIBRATION_METRICS
               if name not in reference_metrics or name not in naive_metrics]
    if missing:
        raise ValueError(
            f"reference calibration qualification is missing metrics: {missing!r}")
    regrets, violations = {}, []
    for name in CALIBRATION_METRICS:
        reference_value = reference_metrics[name]
        naive_value = naive_metrics[name]
        if (type(reference_value) not in (int, float)
                or type(naive_value) not in (int, float)):
            raise ValueError(
                f"invalid reference calibration qualification input for {name!r}")
        try:
            reference = float(reference_value)
            naive = float(naive_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid reference calibration qualification input for {name!r}"
            ) from exc
        limit = float(CALIBRATION_REGRET_SCALE[name])
        if (not np.isfinite(reference) or not np.isfinite(naive)
                or not np.isfinite(limit) or limit <= 0.0):
            raise ValueError(
                f"invalid reference calibration qualification input for {name!r}")
        regret = max(0.0, reference - naive)  # proper scores are lower-is-better
        regrets[name] = float(regret)
        # Decimal thresholds such as 0.02 are not exact binary floats; tolerate
        # only roundoff at the declared boundary, not scientific slack.
        if regret > limit + 1e-12:
            violations.append({
                "metric": name,
                "regret": float(regret),
                "limit": limit,
            })
    return {
        "passed": not violations,
        "regret": regrets,
        "limits": {name: float(CALIBRATION_REGRET_SCALE[name])
                   for name in CALIBRATION_METRICS},
        "violations": violations,
    }


def calibration_factor(sub_metrics, ref_metrics):
    """Continuous multiplier for positive AUC skill from proper-score regret.

    Returns a factor in ``(0, 1]`` plus per-metric factors and regrets. Both
    calibration metrics are mandatory and finite; malformed inputs fail closed.
    The reference is exactly 1 because every regret is exactly zero.
    """
    missing = [name for name in CALIBRATION_METRICS
               if name not in sub_metrics or name not in ref_metrics]
    if missing:
        raise ValueError(f"calibration factor is missing required metrics: {missing!r}")
    factors, regrets = {}, {}
    for name in CALIBRATION_METRICS:
        for label, mapping in (("submission", sub_metrics), ("reference", ref_metrics)):
            item = mapping[name]
            if (type(item) not in (tuple, list) or len(item) != 2
                    or item[1] is not False):
                raise ValueError(
                    f"{label} calibration metric {name!r} has invalid shape/direction")
        sub = float(sub_metrics[name][0])
        ref = float(ref_metrics[name][0])
        scale = float(CALIBRATION_REGRET_SCALE[name])
        if (not np.isfinite(sub) or not np.isfinite(ref)
                or not np.isfinite(scale) or scale <= 0.0):
            raise ValueError(f"invalid calibration input for metric {name!r}")
        regret = max(0.0, sub - ref)  # proper scores are lower-is-better
        regrets[name] = float(regret)
        factors[name] = float(np.exp(-regret / scale))
    factor = float(np.exp(np.mean([
        np.log(factors[name]) for name in CALIBRATION_METRICS
    ])))
    if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("calibration factor is outside (0, 1]")
    return {"factor": factor, "per_metric_factor": factors, "regret": regrets}


def apply_calibration_factor(raw_skill, factor):
    """Discount positive AUC credit while preserving signed negative signal."""
    raw_skill, factor = float(raw_skill), float(factor)
    if (not np.isfinite(raw_skill) or not np.isfinite(factor)
            or not 0.0 < factor <= 1.0):
        raise ValueError("calibrated reward inputs are invalid")
    # Shrinking a negative value toward zero would REWARD a poor ranker for being
    # miscalibrated. Calibration can only reduce positive earned credit.
    return raw_skill if raw_skill <= 0.0 else float(raw_skill * factor)


def accuracy_skill(sub_metrics, naive_metrics, ref_metrics, ref_naive_se,
                   naive_se):
    """sub/naive/ref_metrics: dict name -> (value, higher_is_better).
    ref_naive_se: dict name -> SE of the (reference - naive) gap
    (bootstrap SE stored in anchors); used to exclude unreliable metrics before
    the remaining weights are renormalized.
    naive_se: dict name -> SE of the SINGLE naive estimate (bootstrap SE
    stored in schema-v8 anchors); used to exclude INADEQUATE metrics -- a
    gap that is real but too small to be a denominator. See RESOLUTION_K; the two
    tests are independent and neither implies the other.
    Returns (raw_weighted_skill, per_metric). `raw` is the SIGNED AUC composite
    before the positive-only calibration multiplier, in [SKILL_LO, SKILL_HI].
    The composite is a weighted mean over METRIC_WEIGHTS
    (discrimination-only), renormalized over the reliable metrics actually
    present; each per-metric skill is reliability-gated and winsorized in
    metric_skill.
    """
    if type(ref_naive_se) is not dict or type(naive_se) is not dict:
        raise TypeError("accuracy_skill requires both standard-error mappings")
    weighted = {name: weight for name, weight in METRIC_WEIGHTS.items()
                if weight > 0.0}
    if not weighted:
        raise ValueError("accuracy_skill has no positively weighted metrics")
    skills = {}
    for name in weighted:
        if name not in sub_metrics or name not in naive_metrics or name not in ref_metrics:
            raise ValueError(f"missing scored metric {name!r}")
        if name not in ref_naive_se or name not in naive_se:
            raise ValueError(f"missing standard errors for metric {name!r}")
        val, hib = sub_metrics[name]
        s = metric_skill(val, naive_metrics[name][0], ref_metrics[name][0], hib,
                         se=ref_naive_se[name], se_naive=naive_se[name])
        if not np.isnan(s):
            skills[name] = s
    if not skills:
        raise ValueError("accuracy_skill has no active positively weighted metric")
    # Unweighted proper scores are not computed into a dead normalized side
    # channel; they act only through calibration_factor.
    wsum = sum(weighted[k] for k in skills)
    raw = float(sum(weighted[k] * skills[k] for k in weighted) / wsum)
    return raw, skills


# A dataset the submission never validly scored (crash, timeout, malformed
# pred.csv, ...). Invalid output receives the same floor as the worst valid
# prediction. Anything higher would reward selective abstention: a method could
# crash on regimes where its valid prediction would score below zero.
INVALID_REWARD = SKILL_LO


def reward_reason(status, raw_skill, per_metric=None, *, calibration_factor=1.0):
    """Name WHICH failure or calibration discount a reward represents.

    ``calibration_discounted`` names a valid positive-AUC fit whose earned credit
    was continuously reduced by proper-score regret. It is not a crash or floor.

    -0.5 otherwise means three different things and a reader cannot tell them
    apart from the number:

      * ``invalid``       -- the submission crashed / timed out / emitted garbage;
      * ``valid_floor``   -- it RAN and produced usable predictions that scored at
                             or below the floor without the active AUC ratio itself
                             landing exactly on the winsorization bound;
      * ``clamped_floor`` -- as above AND the active AUC skill was pinned to its
                             lower bound, so the number is censored rather than an
                             unbounded measurement.

    The VALUES deliberately do not change. Invalid must keep scoring the
    worst-valid floor or selective abstention becomes profitable (crash wherever
    you would have scored badly), and the wrapper independently validates
    rewards against MIN_DATASET_REWARD, so moving the floor would take the env
    down rather than clarify it. The fix for an ambiguous number is to SAY which
    case it is, not to renumber the cases.
    """
    if status != "ok":
        return "invalid"
    if calibration_factor < 1.0 - 1e-12 and raw_skill > 0.0:
        return "calibration_discounted"
    if raw_skill > SKILL_LO + 1e-12:
        return "scored"
    if per_metric and clamp_report(per_metric)["all_metrics_clamped"]:
        return "clamped_floor"
    return "valid_floor"


# Headline aggregation over the per-category rewards: blend the category mean
# with a BOTTOM-TAIL MEAN so a method must be broadly competent, not just good on
# its favorite categories -- being weak on a fifth of the capability matrix is
# penalized.
#
# The tail term is the plain average of the lowest ceil(AGG_TAIL_FRACTION * C)
# category scores (a bottom-k mean, not an interpolated quantile or exact empirical
# CVaR when the requested tail mass is nonintegral). The previous
# linear-interpolated Q20 was a rank-dependent weighted mean of only the TWO
# order statistics adjacent to position (C-1)*0.20: at C=3 it gave the three
# categories coefficients 0.40/0.35/0.25, and even at C=14 the whole tail term
# rested on two categories. A bottom-k mean spreads the tail mass evenly over every
# selected weak category, and the statistic documented is exactly the statistic
# computed.
# Reweighted 2026-07-17 to 0.6/0.4 (was 0.75/0.25): weight the weakest-fifth of
# categories more heavily, so a submission strong on average but broken on its
# hardest regimes scores materially lower. This sharpens discrimination between a
# broadly-competent frontier model and a demigod, and lowers scores for uneven work.
AGG_MEAN_W = 0.6
AGG_TAIL_W = 0.4
AGG_TAIL_FRACTION = 0.20


def tail_size(category_count):
    """Number of categories in the bottom tail: ceil(AGG_TAIL_FRACTION * C),
    clamped to at least one and at most every category."""
    if category_count <= 0:
        return 0
    return max(1, min(category_count,
                      int(np.ceil(AGG_TAIL_FRACTION * category_count))))


def category_aggregation(per_dataset):
    """per_dataset: list of dicts {category, weight, reward}.
    Returns ``(benchmark, per_category_rewards, aggregation_contract)``.

    The benchmark is
    ``AGG_MEAN_W * mean(per_cat) + AGG_TAIL_W * bottom_tail_mean(per_cat)``,
    i.e. tail-aware category-equal aggregation (category-equal because each
    category contributes exactly one value to the mean regardless of how many
    seeds it holds). The returned category coefficients are an exact linear
    decomposition of the headline for the platform's per-category TestResults.
    """
    cats = {}
    for d in per_dataset:
        cats.setdefault(d["category"], []).append((d["weight"], d["reward"]))
    per_cat = {}
    for c, items in cats.items():
        w = np.array([x[0] for x in items], float)
        r = np.array([x[1] for x in items], float)
        # Guard a zero-weight category EXPLICITLY rather than by adding an epsilon
        # to the denominator: `sum(w) + 1e-12` perturbs every category score by a
        # relative 1e-12, so a category of pure floor rewards came out as
        # -0.49999999999949996 instead of -0.5. That was invisible only while an
        # invalid dataset scored exactly 0.0 (0/x == 0), and it silently denies the
        # score the exactness the platform's cross-checks compare against.
        total = float(np.sum(w))
        per_cat[c] = float(np.sum(w * r) / total) if total > 0.0 else 0.0
    category_count = len(per_cat)
    coefficients = {
        category: AGG_MEAN_W / category_count
        for category in per_cat
    } if per_cat else {}

    # The tail mass is split EVENLY across the categories in the tail. Category
    # name breaks score ties deterministically; a tie at the tail boundary changes
    # which name carries the mass but never the tail value itself.
    tail_count = tail_size(category_count)
    if per_cat:
        ordered = sorted(per_cat, key=lambda category: (per_cat[category], category))
        tail = ordered[:tail_count]
        for category in tail:
            coefficients[category] += AGG_TAIL_W / tail_count
        category_mean = float(np.mean(list(per_cat.values())))
        tail_value = float(np.mean([per_cat[category] for category in tail]))
    else:
        category_mean = 0.0
        tail_value = 0.0

    benchmark = sum(coefficients[c] * per_cat[c] for c in per_cat)
    aggregation = {
        "kind": "mean_bottom_tail_blend",
        "mean_weight": AGG_MEAN_W,
        "tail_weight": AGG_TAIL_W,
        "tail_fraction": AGG_TAIL_FRACTION,
        "method": "bottom_tail_mean",
        "mean": category_mean,
        "tail_size": tail_count,
        "tail_value": tail_value,
        "headline": float(benchmark),
        "category_coefficients": coefficients,
    }
    return float(benchmark), per_cat, aggregation


if __name__ == "__main__":
    # naive=0, ref=1 sanity
    naive = {"auc": (0.5, True), "brier": (0.21, False)}
    ref = {"auc": (0.70, True), "brier": (0.16, False)}
    for label, sub in {
        "==naive": {"auc": (0.5, True), "brier": (0.21, False)},
        "==ref": {"auc": (0.70, True), "brier": (0.16, False)},
        "half": {"auc": (0.60, True), "brier": (0.185, False)},
        "beats": {"auc": (0.75, True), "brier": (0.15, False)},
        "worse": {"auc": (0.45, True), "brier": (0.23, False)},
    }.items():
        demo_se = {name: 0.001 for name in sub}
        raw, sk = accuracy_skill(sub, naive, ref, demo_se, demo_se)
        print(f"{label:8s} raw={raw:+.3f} per={ {k:round(v,2) for k,v in sk.items()} }")
    bench, pc, agg = category_aggregation([
        {"category": "a", "weight": 1, "reward": 0.8},
        {"category": "a", "weight": 3, "reward": 0.4},
        {"category": "b", "weight": 1, "reward": 1.0},
    ])
    print("benchmark:", round(bench, 3), "per_cat:", {k: round(v, 3) for k, v in pc.items()},
          "weights:", {k: round(v, 3) for k, v in agg["category_coefficients"].items()})
