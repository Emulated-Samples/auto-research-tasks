from __future__ import annotations

import math
import pathlib
import re

import numpy as np
import pytest

from grader.contract import CORPUS_SCHEMA_VERSION, MASTERY_THRESHOLD
from grader.grade import _build_reward
from grader.skill import (
    AGG_MEAN_W,
    AGG_TAIL_FRACTION,
    AGG_TAIL_W,
    SKILL_HI,
    SKILL_LO,
    accuracy_skill,
    category_aggregation,
    metric_skill,
    tail_size,
)

_AGGREGATION_TS = (pathlib.Path(__file__).resolve().parents[1]
                   / "environment" / "src" / "aggregation.ts")


@pytest.mark.parametrize(
    "scores",
    [
        {"only": 0.7},
        {"a": 0.2, "b": 0.9},
        {"a": 0.8, "b": 0.1, "c": 0.5},
        {"a": 0.8, "b": 0.1, "c": 0.5, "d": 0.3, "e": 1.1},
        {"z": 0.4, "a": 0.4, "m": 0.4, "q": 0.9, "b": 0.1},
        # 14 categories: the tail must average the lowest 3, not two adjacent
        # order statistics either side of a quantile position.
        {f"c{index:02d}": score for index, score in enumerate(
            [0.9, 0.1, 0.5, -0.3, 1.2, 0.4, 0.0, 0.7, 0.2, 1.0, 0.3, -0.1, 0.8, 0.6])},
    ],
)
def test_category_coefficients_exactly_decompose_headline(scores):
    rows = [
        {"category": category, "weight": 1.0, "reward": reward}
        for category, reward in scores.items()
    ]
    headline, per_category, aggregation = category_aggregation(rows)
    expected = (
        AGG_MEAN_W * float(np.mean(list(scores.values())))
        + AGG_TAIL_W * _bottom_tail_mean(scores)
    )
    coefficients = aggregation["category_coefficients"]

    assert per_category == pytest.approx(scores)
    assert sum(coefficients.values()) == pytest.approx(1.0)
    assert all(weight > 0 for weight in coefficients.values())
    # The platform's per-category TestResults rely on this exact decomposition.
    assert sum(coefficients[c] * scores[c] for c in scores) == pytest.approx(expected)
    assert headline == pytest.approx(expected)
    assert aggregation["headline"] == pytest.approx(expected)
    assert aggregation["tail_value"] == pytest.approx(_bottom_tail_mean(scores))


def _bottom_tail_mean(scores):
    """The DOCUMENTED statistic, written out independently of the implementation:
    the plain average of the lowest ceil(0.20 * C) category scores."""
    ordered = sorted(scores.values())
    count = math.ceil(AGG_TAIL_FRACTION * len(ordered))
    return float(np.mean(ordered[:count]))


@pytest.mark.parametrize("category_count", [3, 5, 14])
def test_tail_statistic_is_the_documented_bottom_tail_average(category_count):
    """CAL-006: the computed tail term must BE the documented bottom-tail mean.

    The old linear-interpolated Q20 was a rank-dependent weighted mean of the two
    order statistics adjacent to position (C-1)*0.20, so at C=3 it produced
    coefficients 0.40/0.35/0.25 and at C=14 the entire tail term rested on two
    categories."""
    scores = {f"c{index:02d}": 0.1 * index for index in range(category_count)}
    rows = [{"category": c, "weight": 1.0, "reward": s} for c, s in scores.items()]
    headline, _, aggregation = category_aggregation(rows)

    expected_tail_size = math.ceil(AGG_TAIL_FRACTION * category_count)
    assert aggregation["tail_size"] == expected_tail_size == tail_size(category_count)
    assert aggregation["tail_value"] == pytest.approx(_bottom_tail_mean(scores))
    assert headline == pytest.approx(
        AGG_MEAN_W * float(np.mean(list(scores.values())))
        + AGG_TAIL_W * _bottom_tail_mean(scores))

    # Every category in the tail carries the SAME share of the tail mass, and no
    # category outside it carries any.
    coefficients = aggregation["category_coefficients"]
    tail = sorted(scores, key=lambda c: scores[c])[:expected_tail_size]
    base = AGG_MEAN_W / category_count
    for category in scores:
        expected_coefficient = base + (
            AGG_TAIL_W / expected_tail_size if category in tail else 0.0)
        assert coefficients[category] == pytest.approx(expected_coefficient)


def test_tail_term_does_not_hinge_on_a_single_adjacent_crossing():
    """At C=14 the tail averages 3 categories, so swapping the order of two
    categories that are BOTH inside (or both outside) the tail cannot move the
    headline at all -- the old two-order-statistic term could."""
    base = {f"c{index:02d}": 0.1 * index for index in range(14)}

    def rows(scores):
        return [
            {"category": category, "weight": 1.0, "reward": reward}
            for category, reward in scores.items()
        ]

    headline, _, _ = category_aggregation(rows(base))

    # Permute two scores strictly inside the tail (the 3 lowest): no change.
    inside = dict(base)
    inside["c00"], inside["c01"] = base["c01"], base["c00"]
    assert category_aggregation(rows(inside))[0] == pytest.approx(headline)

    # A category genuinely dropping INTO the tail must lower the headline.
    worse = dict(base)
    worse["c13"] = -0.5
    assert category_aggregation(rows(worse))[0] < headline


def test_reward_detail_carries_platform_weights():
    per_dataset = [
        {"category": "strong", "weight": 1.0, "reward": 0.9},
        {"category": "weak", "weight": 1.0, "reward": 0.2},
    ]
    reward, detail = _build_reward(per_dataset, [])
    aggregation = detail["additional_data"]["aggregation"]

    # Pinned to the contract, not to a literal: the wrapper rejects a reward file
    # whose schema_version it does not expect, so a silent bump errors every grade.
    assert detail["schema_version"] == CORPUS_SCHEMA_VERSION
    assert reward["reward"] == pytest.approx(aggregation["headline"])
    assert {
        subscore["name"]: subscore["weight"]
        for subscore in detail["subscores"]
    } == pytest.approx({
        f"category:{category}": weight
        for category, weight in aggregation["category_coefficients"].items()
    })


def _typescript_constant(name):
    source = _AGGREGATION_TS.read_text(encoding="utf-8")
    match = re.search(rf"^export const {name} = (-?[\d.]+);$", source, re.MULTILINE)
    assert match, f"{name} is not declared in {_AGGREGATION_TS}"
    return float(match.group(1))


def test_wrapper_schema_version_tracks_the_grader():
    """index.ts REJECTS a reward file whose schema_version != AGGREGATION_SCHEMA_VERSION.
    When the corpus schema bumped 2 -> 3 the grader started stamping 3 while the TS
    mirror still demanded 2, which errors out EVERY live grade."""
    assert _typescript_constant("AGGREGATION_SCHEMA_VERSION") == CORPUS_SCHEMA_VERSION


def test_published_javascript_schema_version_tracks_the_source():
    """Hyperfocal loads the checked-in dist artifact, not TypeScript source.
    A source-only schema bump previously left dist rejecting every valid reward."""
    emitted = _AGGREGATION_TS.parents[1] / "dist" / "aggregation.js"
    match = re.search(
        r"^export const AGGREGATION_SCHEMA_VERSION = (\d+);$",
        emitted.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, f"schema version is not declared in {emitted}"
    assert int(match.group(1)) == CORPUS_SCHEMA_VERSION


def test_wrapper_score_bounds_track_the_grader():
    """The wrapper REJECTS a per-dataset score outside these bounds, so if the
    grader's signed skill range and the TypeScript bounds ever drift apart, a
    legitimate below-naive run would be thrown out as forged."""
    assert _typescript_constant("MIN_DATASET_REWARD") == SKILL_LO
    assert _typescript_constant("MAX_DATASET_REWARD") == SKILL_HI


def _javascript_constant(name):
    emitted = _AGGREGATION_TS.parents[1] / "dist" / "aggregation.js"
    source = emitted.read_text(encoding="utf-8")
    match = re.search(rf"^export const {name} = (-?[\d.]+);$", source, re.MULTILINE)
    assert match, f"{name} is not declared in {emitted}"
    return float(match.group(1))


def test_wrapper_aggregation_weights_match_grader():
    """The wrapper recomputes the headline from per-category rewards with these
    weights and ERRORS the grade if its headline disagrees with the grader's past
    1e-9 (index.ts aggregation_ok/reward_ok). This SHIPPED broken once: the grader
    was reweighted to AGG_MEAN_W/AGG_TAIL_W = 0.6/0.4 while the TS mirror stayed at
    0.75/0.25, so EVERY live grade errored with aggregation_ok=false. Both the TS
    source and the checked-in dist artifact (what Hyperfocal actually loads) must
    equal the grader's weights."""
    from grader.skill import AGG_MEAN_W, AGG_TAIL_W, AGG_TAIL_FRACTION
    assert _typescript_constant("MEAN_WEIGHT") == AGG_MEAN_W
    assert _typescript_constant("TAIL_WEIGHT") == AGG_TAIL_W
    assert _typescript_constant("TAIL_FRACTION") == AGG_TAIL_FRACTION
    assert _javascript_constant("MEAN_WEIGHT") == AGG_MEAN_W
    assert _javascript_constant("TAIL_WEIGHT") == AGG_TAIL_W
    assert _javascript_constant("TAIL_FRACTION") == AGG_TAIL_FRACTION


def test_mastery_threshold_state_propagates_to_the_reward_file():
    """Schema-v8 starts explicitly uncalibrated; no v7 threshold may leak across
    the changed grid/reference scale. Once a v8 calibration freezes both thresholds,
    this same path propagates them without inventing a wrapper-side display bar."""
    from grader.contract import MASTERY_TAIL_THRESHOLD
    assert (MASTERY_THRESHOLD is None) == (MASTERY_TAIL_THRESHOLD is None)
    _, detail = _build_reward(
        [{"category": "c", "weight": 1.0, "reward": 1.4}], [])
    assert detail["mastery_threshold"] == MASTERY_THRESHOLD
    assert detail["mastery_tail_threshold"] == MASTERY_TAIL_THRESHOLD
    assert detail["mastery_tail_value"] == pytest.approx(1.4)


def test_below_naive_categories_survive_aggregation():
    """A category whose datasets are worse than naive must stay negative through
    the headline: flooring it to zero is what erased the RL gradient."""
    per_dataset = [
        {"category": "strong", "weight": 1.0, "reward": 1.2},
        {"category": "catastrophic", "weight": 1.0, "reward": SKILL_LO},
    ]
    reward, detail = _build_reward(per_dataset, [])
    per_category = detail["additional_data"]["per_category"]

    assert per_category["catastrophic"] == pytest.approx(SKILL_LO)
    assert reward["reward"] < per_category["strong"]
    assert reward["reward"] == pytest.approx(
        detail["additional_data"]["aggregation"]["headline"])
    assert detail["score_bounds"] == {"min_dataset_reward": SKILL_LO,
                                      "max_dataset_reward": SKILL_HI}


def test_metric_reliability_gate_preserves_anchor_semantics():
    kwargs = {"se": 0.01, "se_naive": 0.01}
    assert np.isnan(metric_skill(0.61, 0.60, 0.61, True, **kwargs))
    assert metric_skill(0.60, 0.60, 0.70, True, **kwargs) == pytest.approx(0.0)
    assert metric_skill(0.70, 0.60, 0.70, True, **kwargs) == pytest.approx(1.0)
    assert metric_skill(0.75, 0.60, 0.70, True, **kwargs) == pytest.approx(1.5)


def test_accuracy_reference_is_one_when_noisy_metrics_are_excluded():
    naive = {
        "auc": (0.60, True),
        "brier": (0.22, False),
        "log_loss": (0.64, False),
    }
    reference = {
        "auc": (0.72, True),
        "brier": (0.219, False),
        "log_loss": (0.639, False),
    }
    se = {"auc": 0.01, "brier": 0.002, "log_loss": 0.002}

    raw, per_metric = accuracy_skill(reference, naive, reference, se, se)
    assert raw == pytest.approx(1.0)
    assert per_metric == pytest.approx({"auc": 1.0})
