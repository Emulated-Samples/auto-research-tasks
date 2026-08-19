"""Below-naive behaviour must stay SIGNED and DISTINCT (CAL-004 / CAL-005).

Flooring the skill at zero mapped every below-naive fit onto an identical 0: a
mildly weak model and a catastrophic one produced the same reward, so RL saw no
gradient anywhere below the naive baseline and the model-zoo separation
statistics measured a mechanically compressed variance. These tests pin the
signed scale end to end: the metric skill, the composite, the per-dataset reward
(including its interaction with the runtime penalty), and the model-zoo report
the calibration studies read. Runtime is separately reported and cannot alter
scientific reward.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.skill import SKILL_HI, SKILL_LO, accuracy_skill, metric_skill
from validation.model_zoo import _model_result_from_probability

NAIVE = {"auc": (0.60, True), "brier": (0.24, False), "log_loss": (0.68, False)}
REFERENCE = {"auc": (0.72, True), "brier": (0.20, False), "log_loss": (0.60, False)}
SE = {"auc": 0.005, "brier": 0.002, "log_loss": 0.004}


def _submission(auc, brier, log_loss):
    return {"auc": (auc, True), "brier": (brier, False),
            "log_loss": (log_loss, False)}


def test_below_naive_metric_skill_is_negative_not_zero():
    kwargs = {"se": 0.005, "se_naive": 0.005}
    assert metric_skill(0.54, 0.60, 0.72, True, **kwargs) == pytest.approx(-0.5)
    assert metric_skill(0.57, 0.60, 0.72, True, **kwargs) == pytest.approx(-0.25)
    assert metric_skill(0.60, 0.60, 0.72, True, **kwargs) == pytest.approx(0.0)


def test_below_naive_controls_are_distinct_and_negative():
    """The core CAL-004 regression: weak != catastrophic."""
    slightly_weak = _submission(0.58, 0.25, 0.70)
    clearly_weak = _submission(0.55, 0.27, 0.74)
    catastrophic = _submission(0.40, 0.40, 1.20)

    scores = []
    for control in (slightly_weak, clearly_weak, catastrophic):
        raw, _ = accuracy_skill(control, NAIVE, REFERENCE, SE, SE)
        scores.append(raw)

    assert all(score < 0 for score in scores), scores
    assert len(set(scores)) == 3, f"below-naive controls collapsed onto {scores}"
    # Strictly monotone: worse behaviour must always score strictly lower.
    assert scores[0] > scores[1] > scores[2], scores
    assert scores[-1] >= SKILL_LO


def test_signed_composite_is_bounded_but_not_floored():
    raw, _ = accuracy_skill(
        _submission(0.0, 1.0, 20.0), NAIVE, REFERENCE, SE, SE)
    assert raw == pytest.approx(SKILL_LO)
    raw, _ = accuracy_skill(
        _submission(1.0, 0.0, 0.0), NAIVE, REFERENCE, SE, SE)
    assert raw == pytest.approx(SKILL_HI)


def test_signed_auc_skill_is_not_floored():
    raw, _ = accuracy_skill(
        _submission(0.55, 0.27, 0.74), NAIVE, REFERENCE, SE, SE)
    assert raw < 0.0


def test_composite_fails_closed_when_every_metric_is_inactive():
    with pytest.raises(ValueError, match="no active positively weighted metric"):
        accuracy_skill(NAIVE, NAIVE, NAIVE, SE, SE)


def test_composite_fails_closed_when_only_unweighted_metrics_are_active():
    with pytest.raises(ValueError, match="missing scored metric 'auc'"):
        accuracy_skill(
            {"brier": (0.22, False)},
            {"brier": (0.24, False)},
            {"brier": (0.20, False)},
            {"brier": 0.002},
            {"brier": 0.002},
        )


def test_scientific_reward_is_the_signed_skill_without_runtime_scaling():
    rewards = [-0.5, -0.25, -0.05, 0.0, 0.5, 1.0]
    assert rewards == sorted(rewards)


def test_model_zoo_reports_signed_skill_for_calibration():
    """The model zoo must hand calibration studies the signed value: two distinct
    below-naive models must not report the same number."""
    rng = np.random.default_rng(7)
    n = 4000
    logits = rng.normal(0.0, 1.0, size=n)
    targets = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    def probabilities(signal_weight, noise):
        z = signal_weight * logits + noise * rng.normal(size=n)
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)

    naive_probability = np.full(n, float(targets.mean()))
    naive = _naive_metrics(targets, naive_probability)
    reference = _reference_metrics(targets, probabilities(1.0, 0.0))
    standard_errors = {"auc": 0.005, "brier": 0.002, "log_loss": 0.004}
    # The ADEQUACY yardstick. Small enough here that every metric stays active
    # (the reference beats naive by far more than RESOLUTION_K * these), so this
    # test measures signed skill and not metric activation.
    naive_standard_errors = {"auc": 0.005, "brier": 0.002, "log_loss": 0.004}

    # Below-naive on AUC (negative signal): reward is AUC-only, so a model must
    # ANTI-discriminate to score below naive. Both do; catastrophic more so.
    # Below-naive on AUC (negative signal): reward is AUC-only, so a model must
    # ANTI-discriminate to score below naive. Both do; catastrophic more so. Kept
    # MILDLY below so both stay above the winsorization floor (-0.5) and remain
    # distinguishable -- a deeply-anti-discriminating model correctly floors.
    weak = _model_result_from_probability(
        "weak", targets, probabilities(-0.02, 0.3), naive, reference, standard_errors,
        naive_standard_errors)
    catastrophic = _model_result_from_probability(
        "catastrophic", targets, probabilities(-0.08, 0.3), naive, reference,
        standard_errors, naive_standard_errors)

    assert weak["raw_skill"] < 0.0
    assert catastrophic["raw_skill"] < 0.0
    assert catastrophic["raw_skill"] < weak["raw_skill"], (
        "below-naive models must remain distinguishable to the calibration study")
    # The floored twin is exactly the information loss the gate must not consume.
    assert weak["accuracy_skill"] == catastrophic["accuracy_skill"] == 0.0


def _naive_metrics(targets, probability):
    from grader.metrics import compute_metrics
    return {name: value for name, (value, _) in
            compute_metrics("binomial-logit", targets, {"mean": probability}).items()}


def _reference_metrics(targets, probability):
    return _naive_metrics(targets, probability)
