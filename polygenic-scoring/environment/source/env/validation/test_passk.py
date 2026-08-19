"""pass@k must be uncomputable until the mastery event is calibrated."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grader.contract import MASTERY_THRESHOLD, MASTERY_TAIL_THRESHOLD  # noqa: E402
import validation.passk as passk_module  # noqa: E402
from validation.passk import (  # noqa: E402
    MasteryNotCalibrated,
    Rollout,
    best_score_at_k,
    capability_denominator,
    is_infra_censored,
    mastered,
    pass_at_k,
    report,
)


def graded(rollout_id, headline, tail=None, ok=True):
    return Rollout(
        rollout_id=rollout_id,
        gradable=True,
        headline=headline,
        tail=headline if tail is None else tail,
        integrity_ok=ok,
        contract_ok=ok,
    )


def test_best_score_at_k_counts_agent_noncompletions_at_the_floor():
    # A model that sometimes produces no artifact (agent-failed, non-gradable) must NOT
    # look BETTER as its crash rate rises. Agent non-completions count at SKILL_LO in
    # best_score@k; only infra-censored attempts are excluded.
    from validation.passk import best_score_at_k, Rollout
    from grader.skill import SKILL_LO
    # Two decent gradable runs + two agent-caused non-completions.
    rollouts = [
        graded("g1", 0.40), graded("g2", 0.30),
        Rollout("a1", gradable=False, censor_reason="agent"),
        Rollout("a2", gradable=False, censor_reason="agent"),
    ]
    # best_score@1 = mean over the capability denominator (4 rollouts): the two floors
    # must drag it below the naive mean of only the gradable pair.
    b1 = best_score_at_k(rollouts, 1)
    assert abs(b1 - (0.40 + 0.30 + SKILL_LO + SKILL_LO) / 4) < 1e-12
    assert b1 < (0.40 + 0.30) / 2  # dragged below the gradable-only mean by the floors
    # An infra-censored death is excluded (not counted at the floor).
    with_infra = rollouts + [Rollout("i1", gradable=False, censor_reason="infra")]
    assert abs(best_score_at_k(with_infra, 1) - b1) < 1e-12
    # best_score@4 = expected max over all 4 -> the top gradable score dominates.
    assert abs(best_score_at_k(rollouts, 4) - 0.40) < 1e-12


def test_mastery_threshold_is_anchored_to_the_reference():
    # pass@k is anchored to the FRONTIER CEILING -- the highest headline/tail a
    # frontier rollout actually reached -- so it is low-but-nonzero rather than a
    # flat 0 at the unreachable demigod bar (1.0).
    #
    # The anchor is only meaningful on the scale it was measured on. Schema v8
    # changed BOTH the category grid (12 -> 15) and the reference scale (the
    # -> principled best-of-family), so the v7 anchor (headline 0.72 / tail 0.57) is
    # INCOMMENSURATE and must not be carried forward: re-asserting it here would
    # manufacture a frontier ceiling on a scale where no rollout ever ran, which is
    # the exact failure passk.py refuses to commit. Until a v8 calibration batch
    # freezes the ceiling, the event is explicitly uncalibrated and pass@k REFUSES.
    if MASTERY_THRESHOLD is None or MASTERY_TAIL_THRESHOLD is None:
        assert MASTERY_THRESHOLD is None and MASTERY_TAIL_THRESHOLD is None, (
            "half-calibrated mastery: one threshold was frozen without the other, so "
            "the pass event is defined by a bar nobody measured"
        )
        # Refusal must be total: no pass@k, at any k, from any rollout set.
        with pytest.raises(MasteryNotCalibrated):
            mastered(graded("uncalibrated", 0.9, tail=0.9))
        with pytest.raises(MasteryNotCalibrated):
            pass_at_k([graded(f"r{i}", 0.9) for i in range(5)], 1)
        # best_score@k needs no threshold, so it must STAY reportable meanwhile --
        # otherwise an uncalibrated release has no frontier statistic at all.
        assert best_score_at_k([graded(f"r{i}", 0.4 + 0.1 * i) for i in range(5)], 1)
        return
    # Calibrated branch: the frozen ceiling must sit strictly between "nobody passes"
    # and the demigod bar, and both thresholds must bind.
    assert 0.0 < MASTERY_THRESHOLD < 1.0
    assert 0.0 < MASTERY_TAIL_THRESHOLD <= MASTERY_THRESHOLD
    # a rollout at or above the anchor on headline AND tail masters
    assert mastered(graded("clean", MASTERY_THRESHOLD, tail=1.0)) is True
    # one below the anchor does not
    assert mastered(graded("almost", MASTERY_THRESHOLD - 0.01, tail=1.0)) is False
    rollouts = [graded(f"r{i}", 1.0 + i * 0.1) for i in range(5)]
    stats = pass_at_k(rollouts, 1)
    assert 0.0 <= stats["pass_at_k"] <= 1.0
    assert isinstance(report(rollouts)["pass_at_k"], dict) or "pass_at_k" in report(rollouts)


def _freeze_test_thresholds(monkeypatch, headline=1.0, tail=1.0):
    monkeypatch.setattr(passk_module, "MASTERY_THRESHOLD", headline)
    monkeypatch.setattr(passk_module, "MASTERY_TAIL_THRESHOLD", tail)


def test_public_mastery_api_rejects_caller_threshold_overrides():
    with pytest.raises(TypeError):
        mastered(graded("forged", 1.5), threshold=-0.5)
    with pytest.raises(TypeError):
        pass_at_k([graded("forged", 1.5)], 1, threshold=-0.5)


def test_infra_censored_rollouts_leave_the_capability_denominator(monkeypatch):
    _freeze_test_thresholds(monkeypatch)
    rate_limited = Rollout("r-infra", gradable=False, censor_reason="infra")
    gave_up = Rollout("r-agent", gradable=False, censor_reason="agent")
    good = graded("r-ok", 1.2)
    rollouts = [rate_limited, gave_up, good]

    assert is_infra_censored(rate_limited) is True
    assert is_infra_censored(gave_up) is False
    assert [r.rollout_id for r in capability_denominator(rollouts)] == ["r-agent", "r-ok"]

    stats = pass_at_k(rollouts, 1)
    # n counts the agent failure but NOT the rate limit; c counts the one mastery.
    assert (stats["n"], stats["c"], stats["infra_censored"]) == (2, 1, 1)
    assert stats["pass_at_k"] == pytest.approx(0.5)
    assert stats["non_infra_rate"] == pytest.approx(2 / 3)
    assert stats["gradable_rate"] == pytest.approx(1 / 3)
    lo, hi = stats["mastery_rate_wilson_95"]
    assert 0.0 <= lo < stats["c"] / stats["n"] < hi <= 1.0


def test_unclassified_death_is_an_error_not_a_silent_pass_or_failure():
    with pytest.raises(ValueError):
        capability_denominator([Rollout("r", gradable=False)])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.5001, 1.5001])
def test_rollout_rejects_nonfinite_or_out_of_contract_scores(value):
    with pytest.raises(ValueError, match="finite and within"):
        Rollout(
            "malformed", gradable=True, headline=value, tail=0.5,
            integrity_ok=True, contract_ok=True,
        )


def test_rollout_rejects_inconsistent_status_and_censor_fields():
    with pytest.raises(ValueError, match="exact boolean"):
        Rollout("truthy", gradable=1, censor_reason="agent")
    with pytest.raises(ValueError, match="cannot be censor-classified"):
        Rollout(
            "graded-censored", gradable=True, headline=0.5, tail=0.5,
            integrity_ok=True, contract_ok=True, censor_reason="infra",
        )
    with pytest.raises(ValueError, match="cannot carry scores"):
        Rollout("dead-scored", gradable=False, headline=0.5, tail=0.5,
                censor_reason="agent")
    with pytest.raises(ValueError, match="present together"):
        Rollout("half-scored", gradable=True, headline=0.5)


def test_pass_at_k_matches_the_unbiased_estimator(monkeypatch):
    _freeze_test_thresholds(monkeypatch)
    # n=4, c=1: pass@1 = 1/4, pass@2 = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5, pass@4 = 1.
    rollouts = [graded("hit", 1.5)] + [graded(f"miss{i}", 0.4) for i in range(3)]
    assert pass_at_k(rollouts, 1)["pass_at_k"] == pytest.approx(0.25)
    assert pass_at_k(rollouts, 2)["pass_at_k"] == pytest.approx(0.5)
    assert pass_at_k(rollouts, 4)["pass_at_k"] == pytest.approx(1.0)


def test_one_rollout_cannot_estimate_pass_at_k(monkeypatch):
    _freeze_test_thresholds(monkeypatch)
    with pytest.raises(ValueError):
        pass_at_k([graded("only", 1.16)], 2)


def test_mastery_requires_the_tail_not_just_the_headline(monkeypatch):
    _freeze_test_thresholds(monkeypatch, headline=1.0, tail=0.8)
    # The evaluated rollout's shape: a strong headline carried by a couple of
    # regimes while another capability is weak. A high mean must not mastery-pass.
    lopsided = graded("lopsided", headline=1.16, tail=0.30)
    broad = graded("broad", headline=1.16, tail=1.02)
    assert mastered(lopsided) is False
    assert mastered(broad) is True


def test_mastery_requires_integrity_and_contract(monkeypatch):
    _freeze_test_thresholds(monkeypatch)
    assert mastered(graded("clean", 1.5)) is True
    assert mastered(graded("dirty", 1.5, ok=False)) is False


def test_best_score_at_k_is_reportable_while_mastery_is_uncalibrated():
    rollouts = [graded(f"r{i}", h) for i, h in enumerate([0.0, 1.0])]
    # k=1 -> mean; k=2 -> the max of both draws.
    assert best_score_at_k(rollouts, 1) == pytest.approx(0.5)
    assert best_score_at_k(rollouts, 2) == pytest.approx(1.0)
    assert report(rollouts)["best_score_at_k"][2] == pytest.approx(1.0)
