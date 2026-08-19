"""Demonstrated-achievable full-credit targets and non-solution floors.

Every scored quantity is expressed on the interval between two committed
witnesses of the same suite regime:

* the **floor** -- what a plausible public-information non-solution already
  achieves without discovering any object.  Credit starts here, because the task
  prompt states outright that a method which "tiles one curved or
  multidimensional object across many unrelated scalar features may reconstruct
  well but is not a strong solution".  A grader that pays such a method a large
  fraction of full credit contradicts its own prompt.
* the **target** -- full credit, demonstrated by prompt-only reference runs on
  disjoint calibration suites.  Because it is a dispersion-aware lower envelope
  of witnessed runs, the frozen reference is a constructive witness for ``1.0``.

Both come only from calibration suites.  The simulator oracle remains an audit
and never enters normalization.
"""

from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .metrics import FactorMetrics
from .specs import SCORING_VERSION, SUITES, SuiteSpec


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_CALIBRATION = ROOT / "calibration" / "private_bsf.json"

# Cross-seed safety envelope, in standard deviations of a *new* reference draw.
#
# Earlier versions used a single multiplicative margin for every metric, which
# cannot work. Measured over twelve calibration draws of one suite, the
# reference's reconstruction R^2 has a coefficient of variation of 0.01 while
# its contribution R^2 has a CV of 0.30 -- a 2.4x spread between its best and
# worst seed. A constant wide enough for the noisy metric dropped the
# reconstruction target to 0.61, below the 0.62-0.78 that directional PCA
# reaches unaided, so the reconstruction category paid full credit to anything
# that fit at all. Each metric now gets the margin its own dispersion warrants.
#
# This is only the margin for *unseen* seeds: attainability itself rests on the
# clamp to the worst of twelve witnessed runs, not on sigma. Sigma is therefore
# bounded above by discrimination, and it binds. At 2.0 the guard rejects
# contribution_r2 on long_tail -- where the reference spreads 0.17 to 0.60
# across seeds, a 2-sigma bound lands at 0.09, inside PCA's own 0.077 -- so a
# target nobody could distinguish from doing nothing. At 1.0 the worst-witness
# clamp binds on every metric of that suite, which is the stronger claim anyway:
# full credit is not an extrapolation, it is the weakest run that actually
# happened.
#
# MEASURED, twice, because "under what condition does this fire?" changed with
# the reference. Under the v12 stochastic reference the worst-witness clamp
# decided all 117 targets and this bound decided 0 -- the paragraph above was
# describing an inert mechanism. The condition for it to fire is
# `(mean - min) / sd < sqrt(1 + 1/n)` = 1.04 at n=12: a lower tail unusually
# TIGHT relative to the spread, exactly when the clamp would be an
# over-conservative target. The v13 deterministic reference produced such
# tails -- its only run-to-run noise is the data draw -- and the bound now
# fires on a minority of metric slots, relaxing those targets marginally below
# their witnessed min. The clamp still decides the large majority, and
# attainability never depends on which term fires:
# `tests/test_calibration.py::test_targets_never_exceed_their_witnessed_minimum`
# asserts every committed target sits at or below its worst witness.
#
# The asymmetry with FLOOR_SIGMA is real and is the reason both exist: measured
# the same way, the floor's 2.0 bound binds **117 of 189** arm x metric floors
# (62%). Sigma is load-bearing on the floor side and decorative on the target
# side, because the arms are far more seed-stable than the reference is.
ENVELOPE_SIGMA = 1.0
# The floor must COVER an unseen non-solution draw, exactly as the target must be
# attainable by an unseen reference draw. It is the same statistical object,
# mirrored, and it must be sized the same way.
#
# It was not. The floor was `0.95 * max over calibration draws` -- a sample
# maximum, relaxed DOWNWARD by 5% on the theory that work which merely ties a
# non-solution should score a hair above zero rather than exactly zero. That
# theory is wrong twice over. A sample max over n draws is already exceeded by a
# fresh draw with probability 1/(n+1), which over 9 suites x 6 metrics is ~4
# expected exceedances; relaxing it downward guarantees more. And what it buys is
# precisely the thing this benchmark exists to refuse: on long_tail the
# overcomplete arm drew adjusted average precision 0.326 on production against a
# 0.296 floor (calibration min 0.218, mean 0.274, max 0.311) and was paid 0.306 of
# the discovery category and 0.105 of the suite -- a hand-built free-credit floor,
# in the one place built to prevent one.
#
# Sized at 2 sigma because the exceedance is what must be covered: that production
# draw sits 1.7 sigma above the arm's calibration mean.
FLOOR_SIGMA = 2.0
# A scored metric must leave real room between the non-solution floor and full
# credit.  Below this the metric cannot separate the two witnesses, and
# reporting it as though it could is how a benchmark silently pays for nothing.
MIN_DISCRIMINATION = 0.05

HIGHER_IS_BETTER = (
    "reconstruction_r2",
    "contribution_r2",
    "adjusted_average_precision",
    "geometry_score",
    "counterfactual_r2",
    "support_youden",
    "assignment_coherence",
)
LOWER_IS_BETTER = (
    "mean_false_positive_rate",
    "mean_false_negative_rate",
    "fragmentation",
    "merging",
    "description_bits",
    "active_features",
)
# Metrics normalized between the non-solution floor and the reference target.
#
# Four metrics are recorded but not scored directly, because none of them
# separates a non-solution from the reference on its own. Each is one side of a
# two-sided quantity, and each has a degenerate arm that maxes it out:
#
# * ``mean_false_positive_rate`` -- PCA beats the reference at it on most suites,
#   by rarely firing, while missing half of all active factors.
# * ``mean_false_negative_rate`` -- the mirror; a method that always fires wins it.
# * ``fragmentation`` -- the one-object collapse scores a perfect 0.000 on every
#   suite, because with one learned object nothing competes for a factor.
# * ``merging`` -- the mirror; one private detector per factor-fragment wins it.
#
# They are scored jointly instead, through ``support_youden`` and
# ``assignment_coherence``, which do separate. ``description_bits`` and
# ``active_features`` are likewise minimized outright by the one-object collapse,
# so a floor read off a non-solution is meaningless for them and they are gated
# by joint quality instead.
#
# ``reconstruction_r2`` is absent for the bluntest possible reason: it has no
# floor below the target to normalize against. An unmasked full-width linear
# projection reconstructs at R^2 0.9965, against the reference's own 0.9457
# target -- a non-solution reconstructs *better than the reference does*. No
# margin, weighting, or normalization rescues a metric on which the thing you
# are trying to exclude beats the thing you are trying to reward. It is scored
# as a plain target ratio, carries no reward weight, and earns its keep purely as
# a hurdle: it gates joint quality and it gates the pass event.
FLOOR_NORMALIZED = (
    "contribution_r2",
    "adjusted_average_precision",
    "geometry_score",
    "counterfactual_r2",
    "support_youden",
    "assignment_coherence",
)


def _dispersion(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _prediction_bound(witnessed: tuple[float, ...], *, lower: bool, sigma: float) -> float:
    """One-sided bound on where a *new* reference draw would land.

    The margin a target needs is exactly the uncertainty about an unseen draw of
    the same method on an unseen seed, so that is what it is computed as: a
    normal prediction bound, ``mean -/+ sigma * sd * sqrt(1 + 1/n)``, rather than
    a constant somebody picked. It is then clamped to the worst witnessed value,
    so the target can never be harder than a run that actually happened and the
    frozen reference stays a constructive witness for full credit.
    """
    count = len(witnessed)
    mean = sum(witnessed) / count
    spread = sigma * _dispersion(witnessed) * math.sqrt(1.0 + 1.0 / count)
    if lower:
        return min(min(witnessed), mean - spread)
    return max(max(witnessed), mean + spread)


RULER_DECIMALS = 6


def _ruler_quantize(value: float, *, upward: bool) -> float:
    """Quantize to the exact serialized ruler without reversing an inequality.

    The target/floor JSON stores six decimals. Nearest-rounding can nudge a
    higher-is-better target a rounding unit *above* its worst witness (the
    reference then cannot reach exactly 1.0) or a floor *below* a non-solution
    witness (a non-solution is paid a sliver of credit). Rounding strictly in
    the inequality-preserving direction closes that <=5e-7 asymmetry; it never
    moves the reference distribution, so the tight sigma envelope is unchanged.
    """
    scale = 10**RULER_DECIMALS
    units = math.ceil(value * scale) if upward else math.floor(value * scale)
    quantized = units / scale
    # Binary division can land a representable float just across the source
    # value even though the integer inequality is correct. Move one whole
    # serialized quantum, not one invisible ULP that JSON rounding would erase.
    if upward and quantized < value:
        quantized = (units + 1) / scale
    elif not upward and quantized > value:
        quantized = (units - 1) / scale
    return quantized


def build_target(references: Iterable[FactorMetrics]) -> FactorMetrics:
    """Dispersion-aware envelope of witnessed prompt-only runs."""
    values = tuple(references)
    if not values:
        raise ValueError("at least one demonstrated reference is required")
    updates: dict[str, float] = {}
    for metric in HIGHER_IS_BETTER:
        witnessed = tuple(getattr(item, metric) for item in values)
        if min(witnessed) <= 1e-6:
            raise ValueError(f"demonstrated reference leaves {metric} dead")
        target = max(_prediction_bound(witnessed, lower=True, sigma=ENVELOPE_SIGMA), 1e-3)
        updates[metric] = _ruler_quantize(target, upward=False)
    for metric in LOWER_IS_BETTER:
        witnessed = tuple(getattr(item, metric) for item in values)
        target = _prediction_bound(witnessed, lower=False, sigma=ENVELOPE_SIGMA)
        if metric in {
            "mean_false_positive_rate",
            "mean_false_negative_rate",
            "fragmentation",
            "merging",
        }:
            target = min(1.0, target)
        updates[metric] = _ruler_quantize(target, upward=True)
    return replace(values[0], **updates)


def build_floor(non_solutions: "dict[str, list[FactorMetrics]]") -> FactorMetrics:
    """Upper envelope of what plausible non-solutions achieve unaided.

    Mirrors :func:`build_target` exactly, in the opposite direction: an upper
    prediction bound on where an unseen non-solution draw would land, clamped to
    the best witnessed value so it is never below a non-solution that actually
    happened. A submission must beat what a non-solution would plausibly reach --
    not merely what the sampled ones did reach -- before a component pays.
    """
    if not non_solutions or not any(non_solutions.values()):
        raise ValueError("at least one non-solution witness is required")
    # PER ARM, then max across arms. Pooling the arms into one sample and taking
    # mean + sigma*sd of the mixture is a different statistic and a wrong one: the
    # pooled spread is dominated by how far apart the ARMS are, not by how much a
    # single arm varies between seeds. On core_balanced the three arms draw
    # adjusted average precision around 0.224, 0.333 and 0.015, so the pooled
    # bound lands at 0.466 -- above two of the three arms' best draws ever, and
    # close enough to the reference's 0.494 to make the metric look blind when it
    # is not. Each arm gets a bound on ITS OWN next draw; the floor is the
    # strongest of them.
    updates: dict[str, float] = {}
    for metric in HIGHER_IS_BETTER:
        floor = max(
            0.0,
            max(
                _prediction_bound(
                    tuple(getattr(item, metric) for item in witnesses),
                    lower=False,
                    sigma=FLOOR_SIGMA,
                )
                for witnesses in non_solutions.values()
                if witnesses
            ),
        )
        updates[metric] = _ruler_quantize(floor, upward=True)
    for metric in LOWER_IS_BETTER:
        floor = min(
            _prediction_bound(
                tuple(getattr(item, metric) for item in witnesses),
                lower=True,
                sigma=FLOOR_SIGMA,
            )
            for witnesses in non_solutions.values()
            if witnesses
        )
        updates[metric] = _ruler_quantize(floor, upward=False)
    first = next(iter(witnesses[0] for witnesses in non_solutions.values() if witnesses))
    return replace(first, **updates)


# The two claims every regime must be able to support. If a suite cannot tell a
# non-solution from the reference on whether one learned contribution recovers a
# true factor, or on whether presence is ranked better than chance, the suite is
# not measuring this task and no subset of the rest rescues it.
REQUIRED_METRICS = ("contribution_r2", "adjusted_average_precision")
# The structural terms. Blindness here is local and survivable -- the remaining
# terms carry the suite -- but at least this many must discriminate.
STRUCTURE_METRICS = (
    "geometry_score",
    "counterfactual_r2",
    "support_youden",
    "assignment_coherence",
)
# Suites differ in how many structural axes they can resolve at all, and that is
# information rather than a defect: on long_tail the reference is barely
# distinguishable from a non-solution at counterfactual localization and presence
# informedness, because training sees a steeply skewed prevalence and evaluation
# sees a nearly uniform one. One measurable structural axis keeps the category
# real; the required recovery and discovery claims carry the rest.
MIN_STRUCTURE_METRICS = 1


def discrimination_gap(floor: FactorMetrics, target: FactorMetrics, metric: str) -> float:
    """How much room a metric leaves between a non-solution and full credit."""
    floor_value = getattr(floor, metric)
    target_value = getattr(target, metric)
    if metric in HIGHER_IS_BETTER:
        return target_value - floor_value
    return floor_value - target_value


def scoreable_metrics(floor: FactorMetrics, target: FactorMetrics) -> tuple[str, ...]:
    """The metrics that actually separate a non-solution from the reference here.

    Discrimination is a property of the *regime*, not of the metric in general.
    On ``long_tail`` -- where training sees a steeply skewed prevalence and
    evaluation sees a nearly uniform one -- the reference barely beats an
    overcomplete projection at counterfactual localization, while still beating it
    comfortably at contribution recovery and assignment coherence. Scoring the
    blind metric there would pay a non-solution for a coincidence; dropping it
    everywhere would discard real signal on the eight suites where it works.
    So it is measured per suite and scored only where it discriminates.
    """
    return tuple(
        metric
        for metric in FLOOR_NORMALIZED
        if discrimination_gap(floor, target, metric) >= MIN_DISCRIMINATION
    )


def check_discrimination(floor: FactorMetrics, target: FactorMetrics, label: str) -> tuple[str, ...]:
    """Refuse to ship a suite that cannot separate a non-solution from the reference.

    This is the invariant the previous version lacked. It is checked at
    calibration time, where a violation is fixable, rather than discovered later
    in a reward distribution whose floor nobody could account for.
    """
    scoreable = scoreable_metrics(floor, target)
    for metric in REQUIRED_METRICS:
        if metric not in scoreable:
            raise RuntimeError(
                f"{label}: {metric} does not separate the non-solution floor "
                f"({getattr(floor, metric):.4f}) from full credit "
                f"({getattr(target, metric):.4f}); gap "
                f"{discrimination_gap(floor, target, metric):.4f} < {MIN_DISCRIMINATION}. "
                "This suite cannot measure the task."
            )
    structural = [metric for metric in STRUCTURE_METRICS if metric in scoreable]
    if len(structural) < MIN_STRUCTURE_METRICS:
        raise RuntimeError(
            f"{label}: only {len(structural)} structural metrics separate the "
            f"non-solution floor from full credit ({structural}); "
            f"at least {MIN_STRUCTURE_METRICS} are required."
        )
    return scoreable


def _metrics_from(raw: dict, expected_fields: set[str], label: str) -> FactorMetrics:
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise RuntimeError(f"{label} metrics are malformed")
    metrics = FactorMetrics(**{key: float(value) for key, value in raw.items()})
    if not all(math.isfinite(getattr(metrics, field)) for field in expected_fields):
        raise RuntimeError(f"{label} contains a non-finite metric")
    if any(getattr(metrics, metric) < 0.0 for metric in HIGHER_IS_BETTER):
        raise RuntimeError(f"{label} contains a negative quality metric")
    if metrics.description_bits <= 0.0:
        raise RuntimeError(f"{label} contains a degenerate description length")
    return metrics


def _envelope(key: str) -> dict[str, FactorMetrics]:
    payload = json.loads(REFERENCE_CALIBRATION.read_text())
    if payload.get("scoring_version") != SCORING_VERSION:
        raise RuntimeError(
            "private reference calibration does not match the scoring version: "
            f"{payload.get('scoring_version')!r} != {SCORING_VERSION!r}"
        )
    raw = payload.get(key)
    if not isinstance(raw, dict) or set(raw) != set(SUITES):
        raise RuntimeError(f"reference calibration {key} does not cover every production suite")
    expected_fields = {field.name for field in fields(FactorMetrics)}
    return {
        name: _metrics_from(item, expected_fields, f"{key}[{name}]") for name, item in raw.items()
    }


@lru_cache(maxsize=1)
def _targets() -> dict[str, FactorMetrics]:
    return _envelope("attainable_targets")


@lru_cache(maxsize=1)
def _floors() -> dict[str, FactorMetrics]:
    floors = _envelope("non_solution_floors")
    for name, floor in floors.items():
        committed = tuple(_scored_metrics()[name])
        derived = check_discrimination(floor, _targets()[name], f"committed calibration[{name}]")
        if committed != derived:
            raise RuntimeError(
                f"committed calibration[{name}] scored metrics {committed} disagree with "
                f"the floors and targets, which imply {derived}"
            )
    return floors


@lru_cache(maxsize=1)
def _scored_metrics() -> dict[str, tuple[str, ...]]:
    payload = json.loads(REFERENCE_CALIBRATION.read_text())
    raw = payload.get("scored_metrics")
    if not isinstance(raw, dict) or set(raw) != set(SUITES):
        raise RuntimeError("reference calibration scored_metrics does not cover every suite")
    return {name: tuple(value) for name, value in raw.items()}


def scored_metrics(spec: SuiteSpec) -> tuple[str, ...]:
    _floors()
    try:
        return _scored_metrics()[spec.name]
    except KeyError as error:
        raise RuntimeError(f"no scored metrics registered for suite {spec.name}") from error


@lru_cache(maxsize=1)
def calibration_id() -> str:
    """Semantic fingerprint of the frozen score-normalization ruler.

    Targets and floors are regenerated whenever the reference changes, which
    silently moves the scale every score is expressed in.  Reporting this
    alongside the reward lets runs be grouped by the ruler that produced them
    instead of being compared across incomparable scales.  The audit record
    also contains host-dependent runtimes and sentinel outcomes; hashing the
    whole file would assign a new ruler ID when those non-authoritative fields
    change even if every live target and floor is identical.
    """
    payload = json.loads(REFERENCE_CALIBRATION.read_text())
    authority = {
        key: payload[key]
        for key in (
            "scoring_version",
            "normalization",
            "attainable_targets",
            "non_solution_floors",
            "scored_metrics",
        )
    }
    encoded = json.dumps(
        authority,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def attainable_target(spec: SuiteSpec) -> FactorMetrics:
    try:
        return _targets()[spec.name]
    except KeyError as error:
        raise RuntimeError(f"no target is registered for suite {spec.name}") from error


def non_solution_floor(spec: SuiteSpec) -> FactorMetrics:
    try:
        return _floors()[spec.name]
    except KeyError as error:
        raise RuntimeError(f"no floor is registered for suite {spec.name}") from error
