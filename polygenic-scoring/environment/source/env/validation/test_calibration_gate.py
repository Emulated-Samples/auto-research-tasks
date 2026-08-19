"""Calibration must be continuous, exploit-resistant, and preserve anchor semantics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grader.skill import (  # noqa: E402
    CALIBRATION_METRICS,
    CALIBRATION_REGRET_SCALE,
    SKILL_LO,
    apply_calibration_factor,
    calibration_factor,
    reference_calibration_qualification,
    reward_reason,
)

REFERENCE = {"brier": (0.20225, False), "log_loss": (0.5793, False)}


def _sub(brier, log_loss):
    return {"brier": (brier, False), "log_loss": (log_loss, False)}


def test_reference_factor_is_exactly_one():
    result = calibration_factor(REFERENCE, REFERENCE)
    assert result["factor"] == 1.0
    assert result["per_metric_factor"] == {"brier": 1.0, "log_loss": 1.0}
    assert apply_calibration_factor(1.0, result["factor"]) == 1.0


def test_naive_auc_skill_remains_exactly_zero_for_any_valid_factor():
    factor = calibration_factor(_sub(0.20816, 0.6057), REFERENCE)["factor"]
    assert apply_calibration_factor(0.0, factor) == 0.0


def test_better_proper_scores_do_not_exceed_one():
    result = calibration_factor(_sub(0.19, 0.55), REFERENCE)
    assert result["factor"] == 1.0


def test_reference_calibration_qualification_closes_the_relative_scale_loop():
    naive = {"brier": 0.21, "log_loss": 0.61}
    qualified = reference_calibration_qualification(
        {"brier": 0.20, "log_loss": 0.59}, naive)
    assert qualified["passed"] is True
    assert qualified["violations"] == []
    at_limit = reference_calibration_qualification(
        {"brier": 0.23, "log_loss": 0.66}, naive)
    assert at_limit["passed"] is True

    # The submission factor gives the reference factor one by definition.  The
    # separate release qualification must therefore reject a reference whose own
    # proper scores are materially worse than naive.
    unqualified = reference_calibration_qualification(
        {"brier": 0.231, "log_loss": 0.61}, naive)
    assert unqualified["passed"] is False
    assert unqualified["violations"] == [{
        "metric": "brier",
        "regret": pytest.approx(0.021),
        "limit": 0.02,
    }]


def test_reference_calibration_qualification_fails_closed():
    with pytest.raises(ValueError, match="scalar mappings"):
        reference_calibration_qualification(None, {"brier": 0.21, "log_loss": 0.61})
    with pytest.raises(ValueError, match="missing metrics"):
        reference_calibration_qualification(
            {"brier": 0.20}, {"brier": 0.21, "log_loss": 0.61})
    with pytest.raises(ValueError, match="invalid reference calibration"):
        reference_calibration_qualification(
            {"brier": float("nan"), "log_loss": 0.59},
            {"brier": 0.21, "log_loss": 0.61},
        )


def test_rank_perfect_but_prevalence_compressed_does_not_get_full_credit():
    # These are naive-like proper scores from the measured prevalence regime. An
    # AUC-preserving epsilon compression can approach them without changing AUC.
    result = calibration_factor(_sub(0.20816, 0.6057), REFERENCE)
    reward = apply_calibration_factor(1.0, result["factor"])
    expected = math.exp(-0.5 * (
        (0.20816 - 0.20225) / CALIBRATION_REGRET_SCALE["brier"]
        + (0.6057 - 0.5793) / CALIBRATION_REGRET_SCALE["log_loss"]
    ))
    assert reward == pytest.approx(expected)
    assert 0.0 < reward < 0.8


@pytest.mark.parametrize("sub", [
    _sub(0.40, 1.30),
    _sub(0.2933, 10.10),
])
def test_severe_overconfidence_keeps_only_negligible_credit(sub):
    result = calibration_factor(sub, REFERENCE)
    assert result["factor"] < 0.01
    assert apply_calibration_factor(1.2, result["factor"]) < 0.012


def test_factor_is_continuous_and_monotone_in_regret():
    values = [
        calibration_factor(_sub(REFERENCE["brier"][0] + delta,
                                REFERENCE["log_loss"][0]), REFERENCE)["factor"]
        for delta in (0.0, 0.001, 0.005, 0.01, 0.02)
    ]
    assert values == sorted(values, reverse=True)
    assert all(0.0 < value <= 1.0 for value in values)


def test_negative_gradient_is_never_shrunk_toward_zero():
    factor = calibration_factor(_sub(0.40, 1.30), REFERENCE)["factor"]
    for raw in (-0.01, -0.2, SKILL_LO):
        assert apply_calibration_factor(raw, factor) == raw
    assert apply_calibration_factor(0.0, factor) == 0.0


@pytest.mark.parametrize("sub,ref", [
    ({"brier": (0.2, False)}, REFERENCE),
    (REFERENCE, {"brier": (0.2, False)}),
    (_sub(float("nan"), 0.6), REFERENCE),
])
def test_missing_or_nonfinite_calibration_inputs_fail_closed(sub, ref):
    with pytest.raises(ValueError):
        calibration_factor(sub, ref)


def test_reward_reason_reports_continuous_discount():
    assert reward_reason("ok", 0.6, {"auc": 1.0},
                         calibration_factor=0.6) == "calibration_discounted"
    assert reward_reason("ok", 1.0, {"auc": 1.0}) == "scored"
    assert reward_reason("fit_failed", SKILL_LO,
                         calibration_factor=0.2) == "invalid"


def test_grader_wires_factor_not_discontinuous_gate():
    import inspect
    from grader import grade

    source = inspect.getsource(grade.grade_one)
    assert "calibration_factor(" in source
    assert "apply_calibration_factor(" in source
    assert "calibration_gate(" not in source
    assert "min(reward, 0.0)" not in source


def test_predeclared_scales_have_physical_meaning():
    assert set(CALIBRATION_METRICS) == set(CALIBRATION_REGRET_SCALE)
    assert CALIBRATION_REGRET_SCALE == {"brier": 0.02, "log_loss": 0.05}


def test_model_zoo_uses_identical_calibrated_reward_semantics():
    from grader.metrics import auc, brier, log_loss
    from validation.model_zoo import _model_result_from_probability

    targets = np.tile(np.array([0, 1]), 100)
    naive_probability = np.full(targets.size, 0.5)
    reference_probability = np.where(targets == 1, 0.8, 0.2)
    compressed_probability = np.where(targets == 1, 0.51, 0.49)
    metrics = lambda p: {
        "auc": auc(targets, p),
        "brier": brier(targets, p),
        "log_loss": log_loss(targets, p),
    }
    naive, reference = metrics(naive_probability), metrics(reference_probability)
    result = _model_result_from_probability(
        "compressed", targets, compressed_probability, naive, reference,
        {name: 0.01 for name in naive},
        {name: 0.01 for name in naive},
    )
    assert result["auc_skill"] == pytest.approx(1.0)
    assert result["raw_skill"] == pytest.approx(
        apply_calibration_factor(result["auc_skill"], result["calibration_factor"])
    )
    assert result["raw_skill"] < result["auc_skill"]
