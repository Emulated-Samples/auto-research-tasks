"""The runtime penalty must be denominated in the budget the prompt DISCLOSES.

Every property here failed, or was unrepresentable, under the previous
ratio-against-`time_runtime_anchor` design. The measured damage is recorded in
`rollout_analysis/ADDENDUM_10_2026-07-17_the_speed_axis_pays_the_shortcut.md`:
the speed anchor was 0.85 s mean while the prompt advertised a 170 s fit budget,
so reward halved past ~8.5 s and the env paid a 1.1-second P+T shortcut 3.9x what
it paid a genuine annotation-aware hierarchical fit.
"""
from __future__ import annotations

import pytest

from grader.contract import (
    DATASET_TIMEOUT_S,
    FIT_TIMEOUT_S,
    PREDICT_TIMEOUT_S,
)
from grader.perf import (
    BUDGET_S,
    FLOOR,
    PARITY_FRACTION,
    efficiency_ratio,
    perf_factor,
)


def test_budget_is_the_cap_the_runner_actually_enforces():
    """The ramp's denominator must be the disclosed cap, not a private constant."""
    assert BUDGET_S == float(DATASET_TIMEOUT_S)
    assert FIT_TIMEOUT_S + PREDICT_TIMEOUT_S == DATASET_TIMEOUT_S


def test_penalty_only_and_bounded():
    """The report-only headroom factor is bounded."""
    for t in [0.01, 1.0, 5.0, 42.0, 100.0, BUDGET_S, 10 * BUDGET_S]:
        assert FLOOR <= perf_factor(t) <= 1.0


def test_monotone_nonincreasing_in_time():
    """Being slower may never help. Guards the sign of the ramp."""
    times = [0.5, 1.0, 10.0, 40.0, 50.0, 80.0, 120.0, 190.0, 200.0, 400.0]
    factors = [perf_factor(t) for t in times]
    assert factors == sorted(factors, reverse=True)


def test_using_the_advertised_budget_is_not_catastrophic():
    """THE REGRESSION THIS FILE EXISTS FOR.

    The prompt tells the agent it has DATASET_TIMEOUT_S. A submission that obeys
    the spec and spends a real fraction of it must not be halved for doing so.
    Under the old design a 66.7 s fit (`gold`) scored the 0.5 FLOOR because 66.7 s
    was 78x a 0.85 s anchor the agent could not see.
    """
    assert perf_factor(66.7) > 0.85
    # SV-PGS itself (87.6 s mean) defines skill=1.0; it must not be floored.
    assert perf_factor(87.6) > 0.75


def test_full_credit_band_then_ramp_to_floor_at_the_budget():
    assert perf_factor(PARITY_FRACTION * BUDGET_S) == 1.0
    assert perf_factor(PARITY_FRACTION * BUDGET_S * 0.5) == 1.0
    assert perf_factor(BUDGET_S) == pytest.approx(FLOOR)
    midpoint = (PARITY_FRACTION + 1.0) / 2.0 * BUDGET_S
    assert perf_factor(midpoint) == pytest.approx(1.0 - (1.0 - FLOOR) / 2.0)


def test_past_the_budget_clamps_and_never_goes_negative():
    """Unreachable in a graded run (the runner SIGKILLs first), but the function
    must not extrapolate into a negative multiplier if it is ever called."""
    assert perf_factor(2 * BUDGET_S) == FLOOR
    assert perf_factor(1000 * BUDGET_S) == FLOOR


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        perf_factor(0.0)
    with pytest.raises(ValueError):
        perf_factor(-1.0)
    with pytest.raises(ValueError):
        perf_factor(1.0, budget_s=0.0)


def test_efficiency_ratio_still_reports_against_the_runtime_anchor():
    """The runtime anchor keeps a live job -- the reported efficiency ratio -- so
    it is not dead code. It simply no longer sets reward."""
    assert efficiency_ratio(2.0, 1.0) == 0.5     # 2x slower than the anchor
    assert efficiency_ratio(0.5, 1.0) == 2.0     # 2x faster
    assert efficiency_ratio(0.0, 1.0) is None
    assert efficiency_ratio(1.0, 0.0) is None
