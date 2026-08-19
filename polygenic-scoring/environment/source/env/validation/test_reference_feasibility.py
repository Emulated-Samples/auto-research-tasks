"""The reference's own fit must fit in the budget a SUBMISSION is given.

`grader/contract.py` has always STATED this invariant -- "a reference-fidelity fit
must finish well inside that cap, otherwise skill=1.0 is unreachable by
construction and no submission, not even SV-PGS itself, can score 1" -- and
nothing checked it. The v6 corpus shipped two datasets that violate it outright.
See rollout_analysis/ADDENDUM_10_2026-07-17_the_speed_axis_pays_the_shortcut.md.
"""
from __future__ import annotations

import pytest

from datagen.build_corpus import ReferenceFitError, _validate_reference_feasibility
from grader.contract import (
    FIT_TIMEOUT_S,
    REFERENCE_FIT_FEASIBILITY_FRACTION,
    REFERENCE_FIT_HEADROOM_FRACTION,
)

HARD = REFERENCE_FIT_FEASIBILITY_FRACTION * FIT_TIMEOUT_S
WARN = REFERENCE_FIT_HEADROOM_FRACTION * FIT_TIMEOUT_S


def test_the_two_shipped_violations_are_now_rejected():
    """The measured v6 failures, encoded as data.

    svld_class/d_b8d08f3cb532 took 198.2 s and soft_membership/d_e15a4ad20835 took
    188.0 s, against a 170 s submission fit cap. On those datasets a submission
    that EXACTLY reproduces the reference is SIGKILLed and scores
    INVALID_REWARD = -0.5, so `reference == skill 1.0` is false by construction.
    """
    for dataset_id, seconds in [("svld_class", 198.2), ("soft_membership", 188.0)]:
        assert seconds > FIT_TIMEOUT_S
        with pytest.raises(ReferenceFitError, match="UNREACHABLE ANCHOR"):
            _validate_reference_feasibility(dataset_id, seconds)


def test_a_fit_at_or_beyond_the_cap_is_rejected():
    with pytest.raises(ReferenceFitError, match="UNREACHABLE ANCHOR"):
        _validate_reference_feasibility("dataset", HARD)
    with pytest.raises(ReferenceFitError, match="UNREACHABLE ANCHOR"):
        _validate_reference_feasibility("dataset", HARD + 1.0)


def test_a_comfortable_fit_passes_silently(capsys):
    _validate_reference_feasibility("dataset", 0.5 * WARN)
    assert capsys.readouterr().out == ""


def test_the_headroom_band_warns_but_does_not_fail(capsys):
    """Deliberately a warning: the right hard margin cannot be chosen from the v6
    timings, which predate the reference's global_scale_floor."""
    _validate_reference_feasibility("dataset", WARN + 1.0)
    out = capsys.readouterr().out
    assert "[warn]" in out and "% of the" in out


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "12", None])
def test_nonsense_timings_are_rejected(bad):
    with pytest.raises(ReferenceFitError):
        _validate_reference_feasibility("dataset", bad)


def test_the_hard_limit_is_not_looser_than_the_warning():
    assert REFERENCE_FIT_HEADROOM_FRACTION <= REFERENCE_FIT_FEASIBILITY_FRACTION
    assert 0.0 < REFERENCE_FIT_HEADROOM_FRACTION <= 1.0
    assert 0.0 < REFERENCE_FIT_FEASIBILITY_FRACTION <= 1.0
