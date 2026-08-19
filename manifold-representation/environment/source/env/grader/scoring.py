"""Attainable-target continuous reward and coherent binary pass criterion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .metrics import FactorMetrics
from .targets import REQUIRED_METRICS, STRUCTURE_METRICS


CATEGORY_NAMES = (
    "reconstruction",
    "factor_recovery",
    "factor_discovery",
    "structural_coherence",
    "compression",
    "efficiency",
)

# Reconstruction carries no weight, and that is a measurement rather than a
# preference. An unmasked full-width linear projection -- a non-solution that
# discovers nothing -- reconstructs at R^2 0.9965 against the learned reference's
# own 0.9457 target. There is no level of reconstruction that distinguishes a
# solution from a non-solution, so paying for it is paying for nothing; v10 paid
# it 0.17, which a two-line identity map collected in full.
#
# It remains necessary, and is enforced where necessity belongs: it gates the
# joint-quality term multiplying compression and efficiency, and it gates the
# pass event through its own threshold. Failing it costs; passing it does not pay.
# The weight sits instead on the axes where a non-solution and the reference
# actually differ.
CATEGORY_WEIGHTS = {
    "reconstruction": 0.00,
    "factor_recovery": 0.32,
    "factor_discovery": 0.21,
    "structural_coherence": 0.26,
    "compression": 0.18,
    "efficiency": 0.03,
}

PASS_THRESHOLDS = {
    # v12's mastery ladder has three empirically anchored rungs: measured
    # non-solutions at approximately zero; an eight-perfect/one-dead frontier
    # archetype at reward 0.744 after breadth adjustment; and the prompt-only
    # production witness at approximately one. These rounded midpoint-to-full
    # bars reserve pass for broad, near-reference work while the continuous
    # reward still pays every improvement below them.
    "overall": 0.88,
    "reconstruction": 0.90,
    "factor_recovery": 0.85,
    "factor_discovery": 0.88,
    "structural_coherence": 0.87,
    "compression": 0.83,
}

# Lower-tail mastery.  Rather than requiring every suite to individually clear a
# hard floor -- which lets one noisy or marginally-missed suite dominate the
# binary event -- we require the mean of the worst quartile of per-suite
# semantic quality to clear a bar, plus a low catastrophic floor that still
# rejects a run that entirely abandons any single regime.
#
# The same lower-tail evidence now shapes the continuous reward through its
# square root. A threshold at or below 0.81 would therefore be dead: even perfect
# raw reconstruction could not clear the adjusted 0.90 reconstruction pass bar
# below that tail quality. At 0.85 the factor is sqrt(0.85)=0.922; every raw
# category requirement remains reachable (the binding reconstruction
# requirement is 0.90/0.922=0.976), while the explicit tail condition still
# changes verdicts rather than duplicating an implication of the category bars.
TAIL_QUALITY_THRESHOLD = 0.85
TAIL_FRACTION = 0.25
CATASTROPHIC_FLOORS = {
    # With nine suites and a 0.85 worst-three mean, one weak suite surrounded
    # by two perfect ones already needs each isolated core category above
    # (3*0.85 - 2)^4 = 0.0915. A floor at or below that value is dead. Fifteen
    # percent is the first simple common bar comfortably above the implication:
    # it changes verdicts only for a genuinely near-abandoned semantic axis.
    "factor_recovery": 0.15,
    "factor_discovery": 0.15,
    "structural_coherence": 0.15,
    # Compression is a scientific axis too: the parsimony prompt forbids
    # abandoning it on a single suite (near-zero active-feature / description
    # credit) while coasting on the others. Same near-abandonment bar as above;
    # the reference clears it on every suite, so the pass ceiling is unchanged.
    "compression": 0.15,
}


# A gross violation of a hard contract requirement is not partial credit -- it
# is a different decomposition than the prompt requires. Below these limits the
# soft factor still applies (numerical drift is forgiven); at or above them the
# suite is invalidated and reported as a contract fault, so a decoupled-
# reconstruction or order-dependent transform can no longer coast on near-full
# category credit while reading as contract_met. The reference sits ~1e-15 / ~0
# on both, so the pass ceiling and the reference's 1.0 are untouched.
ADDITIVE_HARD_LIMIT = 1e-3
PERMUTATION_HARD_LIMIT = 1e-3


@dataclass(frozen=True)
class Integrity:
    additive_error: float
    support_agreement: float
    permutation_error: float

    def gross_contract_breach(self) -> str | None:
        """A hard-contract fault name if a declared invariant is grossly broken."""
        if self.additive_error > ADDITIVE_HARD_LIMIT:
            return "additive_identity_violated"
        if self.permutation_error > PERMUTATION_HARD_LIMIT:
            return "permutation_equivariance_violated"
        return None

    @property
    def factor(self) -> float:
        # The additive identity and row-order determinism are hard contract
        # requirements: violating them invalidates the decomposition, so they
        # gate multiplicatively toward zero.  Presence/contribution support
        # agreement is a soft behavioral signal -- a slight mismatch should
        # penalize, not erase, otherwise-interpretable partial credit -- so it
        # is bounded below at 0.5 rather than driven to zero.
        additive = (
            1.0
            if self.additive_error <= 1e-5
            else float(np.exp(-80.0 * (self.additive_error - 1e-5)))
        )
        support = 0.5 + 0.5 * float(np.clip((self.support_agreement - 0.35) / 0.55, 0.0, 1.0))
        permutation = (
            1.0
            if self.permutation_error <= 1e-7
            else float(np.exp(-60.0 * (self.permutation_error - 1e-7)))
        )
        return additive * support * permutation


@dataclass(frozen=True)
class SuiteScore:
    categories: dict[str, float]
    reward: float
    candidate: FactorMetrics
    floor: FactorMetrics
    target: FactorMetrics
    integrity: Integrity
    candidate_seconds: float
    full_credit_seconds: float

    def to_json(self) -> dict:
        return {
            "categories": self.categories,
            "reward": self.reward,
            "candidate": asdict(self.candidate),
            "floor": asdict(self.floor),
            "target": asdict(self.target),
            "integrity": {**asdict(self.integrity), "factor": self.integrity.factor},
            "candidate_seconds": self.candidate_seconds,
            "full_credit_seconds": self.full_credit_seconds,
        }


def weighted_category_mean(categories: dict[str, float]) -> float:
    if set(categories) != set(CATEGORY_WEIGHTS):
        raise ValueError("category set does not match category weights")
    return float(sum(categories[name] * CATEGORY_WEIGHTS[name] for name in CATEGORY_NAMES))


def aggregate_categories(
    weighted_categories: list[tuple[dict[str, float], float]],
) -> dict[str, float]:
    """Weight raw per-suite categories into one raw benchmark category vector.

    Keeping this operation separate from breadth adjustment makes the single
    application point auditable.  Per-suite scores and their weighted aggregate
    remain literal measurements; only the public benchmark categories are
    multiplied by the lower-tail factor.
    """
    if not weighted_categories:
        raise ValueError("cannot aggregate an empty suite set")
    denominator = 0.0
    totals = {name: 0.0 for name in CATEGORY_NAMES}
    for categories, weight in weighted_categories:
        if set(categories) != set(CATEGORY_NAMES):
            raise ValueError("suite category set does not match benchmark categories")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("suite weights must be finite and positive")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in categories.values()
        ):
            raise ValueError("suite categories must be finite values in [0, 1]")
        denominator += weight
        for name in CATEGORY_NAMES:
            totals[name] += categories[name] * weight
    return {name: float(totals[name] / denominator) for name in CATEGORY_NAMES}


def _floor_ratio(value: float, floor: float, target: float) -> float:
    """Fraction of the floor-to-target span a candidate has actually crossed.

    ``floor`` is what a plausible non-solution reaches unaided and ``target`` is
    demonstrated full credit; both are committed per suite.  Scoring the *level*
    of a metric rather than the *progress* across this span is what let a ten-line
    principal-component baseline collect 43% of this benchmark: its adjusted
    average precision was 0.32 against a 0.37 target, so it banked 85% of the
    discovery category while recovering 3% of one factor.  Direction is handled by
    the caller passing floor/target in the metric's own orientation.
    """
    span = target - floor
    if abs(span) < 1e-9:
        raise ValueError("floor and target coincide; the metric cannot be scored")
    return float(np.clip((value - floor) / span, 0.0, 1.0))


def _target_ratio(value: float, target: float) -> float:
    """Plain fraction of a target, for a metric with no floor beneath it.

    Only reconstruction is scored this way, because only reconstruction has
    nothing below its target to measure progress from: a non-solution attains it
    better than the reference does. It therefore carries no weight, and this
    ratio exists to gate joint quality and the pass event, not to pay.
    """
    return float(np.clip(value / max(target, 1e-9), 0.0, 1.0))


def _compression_ratio(value: float, target: float) -> float:
    if value <= target * (1.0 + 1e-8):
        return 1.0
    return float(np.clip(target / max(value, 1e-6), 0.0, 1.0))


def _speed_score(candidate_seconds: float, full_credit_seconds: float) -> float:
    """Minor soft preference after a generous full-credit runtime region.

    MEASURED REACH, because the declared range is not the realized one and a
    reader would otherwise assume it is. ``full_credit_seconds`` is
    ``0.85 * (fit_timeout + transform_timeout)`` = 76.5s, while the largest
    runtime that can exist is bounded by those same timeouts at ~90s: a fit is
    killed at 75s and a transform at 15s. So the excess can never exceed
    (90 - 76.5) / 76.5 = 0.176, this returns at worst **0.965**, and the 0.80
    clip floor is unreachable -- it would need 153s, which the runner kills first.
    The reference spends 2.1s of that 90s ceiling.

    Consequence, stated plainly: with weight 0.03 the speed term can move the
    benchmark reward by at most 0.03 * 0.035 = **0.001**, so in practice
    ``efficiency == core`` for every submission that can exist. Every valid suite
    of the v12 Opus cohort scored ef exactly equal to its own joint quality.

    This is deliberate rather than broken, and the prompt is why: it promises that
    "scientific correctness and coherent factorization dominate resource use once
    the program is comfortably within the stated limits". A term that bit below
    the stated limits would contradict the prompt the agent was given, which is a
    worse defect than a small one. The hard timeouts do the real enforcing.

    So the honest reading is that ``efficiency`` is 0.03 of additional weight on
    joint quality wearing a speed label, and the clip floor is decoration. If a
    future version wants speed to actually matter it has to say so IN THE PROMPT
    first, and then set the full-credit region from what the task needs rather
    than from a fraction of the kill threshold.
    """
    if candidate_seconds <= full_credit_seconds:
        return 1.0
    excess = (candidate_seconds - full_credit_seconds) / max(full_credit_seconds, 0.25)
    return float(np.clip(1.0 - 0.20 * excess, 0.80, 1.0))


def score_suite(
    candidate: FactorMetrics,
    floor: FactorMetrics,
    target: FactorMetrics,
    integrity: Integrity,
    candidate_seconds: float,
    full_credit_seconds: float,
    scored: tuple[str, ...] = STRUCTURE_METRICS + REQUIRED_METRICS,
) -> SuiteScore:
    def higher(metric: str) -> float:
        return _floor_ratio(
            getattr(candidate, metric), getattr(floor, metric), getattr(target, metric)
        )

    reconstruction = _target_ratio(candidate.reconstruction_r2, target.reconstruction_r2)
    recovery = higher("contribution_r2")
    discovery = higher("adjusted_average_precision")
    # Within one goal, partial progress adds: a method that keeps a factor's
    # geometry but leaks a little across objects has done part of the structural
    # job.  Averaging here -- rather than the nested square roots this used to
    # take -- also stops a single near-floor term from being laundered into a
    # mid-range score by a fourth or eighth root.
    #
    # Only the terms that discriminate *in this regime* are averaged. Whether a
    # metric can tell a non-solution from the reference is a property of the
    # regime, not of the metric, and scoring a blind one pays for a coincidence.
    structure_terms = [higher(metric) for metric in STRUCTURE_METRICS if metric in scored]
    if not structure_terms:
        raise ValueError("no structural metric discriminates on this suite")
    structure = float(np.mean(structure_terms))
    # Across goals, masking must not be possible: excellent reconstruction cannot
    # stand in for finding no objects.  This joint quality gates the two
    # categories that are otherwise trivially maximized -- the one-object
    # collapse is both maximally compressed and instant.
    core = float(max(0.0, (reconstruction * recovery * discovery * structure) ** 0.25))
    compression_ratio = _compression_ratio(candidate.description_bits, target.description_bits)
    support_size_ratio = _compression_ratio(candidate.active_features, target.active_features)
    compression = np.sqrt(compression_ratio * support_size_ratio) * core
    speed = _speed_score(candidate_seconds, full_credit_seconds)
    efficiency = speed * core
    gate = integrity.factor
    categories = {
        "reconstruction": reconstruction * gate,
        "factor_recovery": recovery * gate,
        "factor_discovery": discovery * gate,
        "structural_coherence": structure * gate,
        "compression": float(compression) * gate,
        "efficiency": efficiency * gate,
    }
    reward = weighted_category_mean(categories)
    return SuiteScore(
        categories,
        reward,
        candidate,
        floor,
        target,
        integrity,
        candidate_seconds,
        full_credit_seconds,
    )


def category_pass(category_scores: dict[str, float]) -> dict[str, bool]:
    """Whether each category clears its own aggregate mastery threshold.

    Categories without a threshold (efficiency) are informational and omitted,
    so the presentation layer can show a per-category status that reflects that
    category's own criterion rather than the single joint mastery event.
    """
    if set(category_scores) != set(CATEGORY_NAMES):
        raise ValueError("category set does not match benchmark categories")
    return {
        name: float(category_scores[name]) >= floor
        for name, floor in PASS_THRESHOLDS.items()
        if name != "overall"
    }


def suite_quality(scores: dict[str, float]) -> float:
    """Geometric mean of the central scientific categories for one suite.

    A geometric mean means excellent reconstruction cannot mask near-zero
    recovery or discovery, so the per-suite quality reflects genuine object
    understanding rather than mere observation fitting.
    """
    core = (
        max(0.0, scores["reconstruction"])
        * max(0.0, scores["factor_recovery"])
        * max(0.0, scores["factor_discovery"])
        * max(0.0, scores["structural_coherence"])
    )
    return float(core**0.25)


def lower_tail_quality(qualities: list[float]) -> float:
    """Mean of the worst ``TAIL_FRACTION`` of per-suite qualities (a CVaR)."""
    if not qualities:
        return 0.0
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in qualities):
        raise ValueError("suite qualities must be finite values in [0, 1]")
    ordered = sorted(qualities)
    count = max(1, math.ceil(TAIL_FRACTION * len(ordered)))
    return float(sum(ordered[:count]) / count)


# A multiplicative factor that can reach exactly zero is not a weight, it is an
# annihilator, and it made this benchmark report a frontier solution and a
# do-nothing baseline as the same number.
#
# Measured, one rollout: nine valid suites, contract met, and raw
# recovery of 0.71-0.87 on four of them -- BETTER than the reference. On the other
# four it matched no factor at all, which zeroes `suite_quality` because that is a
# geometric mean. Three or more zeroed suites zero the worst quartile, so
# `sqrt(0) = 0`, so `raw_reward = 0.478` became `reward = 0.000` -- identical to
# directional PCA, which discovers nothing anywhere. Two of three rollouts landed
# exactly there.
#
# That is wrong three times over. It contradicts this benchmark's own goal of
# frontier scores that are low but NONZERO; it destroys the learning signal
# precisely where it is most informative, since "excellent on four regimes,
# absent on four" and "absent everywhere" are opposite results; and it makes the
# reward unattributable, which is the failure every other guard here exists to
# prevent.
#
# The floor cannot leak credit, and that is checked rather than assumed: a
# non-solution's RAW categories are ~0 on every weighted axis (reconstruction is
# the only one they win and it carries weight 0.00), so their raw reward is
# ~1e-4 and any floor multiplies it back to ~0. The measured arms stay at 0.0000.
#
# It cannot loosen the pass event either. `benchmark_pass` reads the RAW per-suite
# scores for both its breadth conditions -- `tail_mastery` and the catastrophic
# floors -- so neither sees this factor at all, and its aggregate bar of 0.88
# against a floored 0.15 would need a raw reward of 5.9 to clear.
#
# Chosen equal to CATASTROPHIC_FLOORS' 0.15 because they encode the same judgement
# from the two sides: a near-abandoned regime is worth roughly a seventh of the
# credit, and it is worth nothing only in the verdict.
BREADTH_FLOOR = 0.15


def breadth_factor(suite_category_scores: list[dict[str, float]]) -> float:
    """Continuous breadth evidence derived only from live per-suite quality.

    The square root preserves a useful gradient for imperfect broad solutions
    while making specialization expensive.  It is exactly one only when the
    entire worst quartile has semantic quality one, which in turn requires all
    four central scientific categories to be one on those suites.

    Floored at ``BREADTH_FLOOR`` so that abandoning regimes is expensive rather
    than fatal -- see the note above that constant for the measurement that forced
    it. The floor binds only below a worst-quartile quality of
    ``BREADTH_FLOOR**2 = 0.0225``, i.e. only for a submission that has essentially
    given up on a quarter of the benchmark, and the v12 archetype it was tuned on
    is untouched: eight-perfect/one-dead has tail 0.667, factor 0.816, and never
    reaches the floor. Inside the floored band the reward still moves, because it
    still scales with the raw categories.
    """
    if not suite_category_scores:
        return 0.0
    if any(set(scores) != set(CATEGORY_NAMES) for scores in suite_category_scores):
        raise ValueError("suite category set does not match benchmark categories")
    qualities = [suite_quality(scores) for scores in suite_category_scores]
    tail = lower_tail_quality(qualities)
    # The floor LIFTS partial work; it does not INVENT it. A run whose every suite
    # is quality zero -- an invalid submission, or one that failed on all nine
    # regimes -- has nothing to lift, and reporting a floored factor for it would
    # dress a total failure as narrow success. So the floor engages only once some
    # regime shows real quality. A single valid suite is enough, which is the whole
    # point: "excellent on four, absent on five" must not read as "absent
    # everywhere", and now it does not read as "invalid everywhere" either.
    if max(qualities) <= 0.0:
        return 0.0
    return float(max(BREADTH_FLOOR, math.sqrt(tail)))


def breadth_adjusted_categories(
    raw_category_scores: dict[str, float],
    suite_category_scores: list[dict[str, float]],
) -> dict[str, float]:
    """Apply the live lower-tail factor once to the raw aggregate categories."""
    if set(raw_category_scores) != set(CATEGORY_NAMES):
        raise ValueError("raw category set does not match benchmark categories")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in raw_category_scores.values()
    ):
        raise ValueError("raw categories must be finite values in [0, 1]")
    factor = breadth_factor(suite_category_scores)
    return {name: float(raw_category_scores[name] * factor) for name in CATEGORY_NAMES}


def benchmark_pass(
    category_scores: dict[str, float],
    suite_category_scores: list[dict[str, float]],
    all_valid: bool,
) -> bool:
    """Reference-reachable mastery event derived from the continuous reward.

    ``category_scores`` are the already breadth-adjusted benchmark categories;
    ``suite_category_scores`` are raw per-suite evidence. Three conditions must
    all hold: strong adjusted aggregate quality (weighted overall plus
    per-category floors), a robust lower tail (the worst-quartile mean of raw
    per-suite semantic quality), and no catastrophic abandonment of any single
    suite.  This function never applies the breadth factor again.
    """
    if (
        not all_valid
        or set(category_scores) != set(CATEGORY_NAMES)
        or not suite_category_scores
        or any(set(scores) != set(CATEGORY_NAMES) for scores in suite_category_scores)
    ):
        return False
    overall = weighted_category_mean(category_scores)
    aggregate_mastery = overall >= PASS_THRESHOLDS["overall"] and all(
        category_scores[name] >= floor
        for name, floor in PASS_THRESHOLDS.items()
        if name != "overall"
    )
    qualities = [suite_quality(scores) for scores in suite_category_scores]
    tail_mastery = lower_tail_quality(qualities) >= TAIL_QUALITY_THRESHOLD
    catastrophic_ok = all(
        scores[name] >= floor
        for scores in suite_category_scores
        for name, floor in CATASTROPHIC_FLOORS.items()
    )
    return aggregate_mastery and tail_mastery and catastrophic_ok
