"""Held-out evaluation of the agent's design(): per-track activity + catastrophic contract gates.

Each track test runs design() fresh against a metered oracle, enforces the output contract (length /
count / alphabet / budget → the config scores 0 on violation), then scores the designs on the hidden
evaluator ensemble and emits the held-out z as a continuous metric (grader/pytest-suite.ts maps it to
a skill score via CONTINUOUS_METRICS). Determinism is a separate whole-grade catastrophe gate
(grader/anti-cheat.ts::cat-determinism).
"""
import warnings

import pytest


def _run_regime(design_fn, F, track_name, emit_novelty=False):
    """Run one regime, enforce the contract, emit held-out z. Failures here score the config 0."""
    cfg = F.REGIMES[track_name]
    oracle = F.build_oracle(cfg["budget"])
    try:
        designs = design_fn(oracle, cfg["track"], cfg["budget"], cfg["length"],
                            cfg["n_designs"], cfg["seed"])
    except F.BudgetExceeded as e:
        pytest.fail(f"design exceeded the oracle budget: {e}")       # gate -> score 0
    except Exception as e:  # a crashing design scores 0, it is NOT excluded (env-base F-1 lesson)
        pytest.fail(f"design() raised {type(e).__name__}: {e}")
    ok, reason = F.validate(designs, cfg["length"], cfg["n_designs"])
    assert ok, f"output-contract violation: {reason}"               # length/count/alphabet -> score 0
    # Budget is judged on the ROOT-OWNED session meter, not on the in-process counter: the latter is
    # trivially satisfied by a design() that spends nothing because it searched offline instead.
    session_used = F.session_queries()
    if session_used is None:
        # Meter absent => the scoring daemon never came up. That is an infra fault; do not convert
        # it into an agent failure, but make it loud so QA sees it (pytest prints a warnings summary
        # even for passing tests, and the metric lands in the grader's JSONL side-channel).
        F.emit_metric("session_meter_missing", 1)
        warnings.warn("session query meter unavailable (scoring daemon down?) — budget unchecked")
    else:
        F.emit_metric("session_queries", session_used)
        assert session_used <= F.SESSION_BUDGET, (
            f"session query budget: {session_used} sequences scored > {F.SESSION_BUDGET}")
    if emit_novelty:  # diagnostic (spec-B B3): reuse these designs, no extra budget spent
        F.emit_metric("min_edit_to_natural", F.min_edit_to_natural(designs))
    F.emit_metric(f"heldout_{track_name}", F.heldout_z(designs, cfg["track"]))


def test_heldout_Dev(design_fn, F):
    _run_regime(design_fn, F, "Dev", emit_novelty=True)


# --- invalid-input loud-failure block (verifier §4.2): design() must raise, not return garbage ---
@pytest.mark.parametrize("bad", [
    dict(budget=0), dict(budget=-5),
    dict(budget=2, n_designs=8),           # budget < n_designs: cannot score one design each
    dict(length=0), dict(length=248),      # length the oracle cannot take (oracle is 249bp)
    dict(track=99), dict(track=-1),
    dict(n_designs=0),
])
def test_invalid_args_raise(design_fn, F, bad):
    base = dict(track=0, budget=20000, length=249, n_designs=8, seed=0)
    base.update(bad)
    oracle = F.build_oracle(max(base["budget"], 1))
    with pytest.raises(ValueError):
        design_fn(oracle, base["track"], base["budget"], base["length"],
                  base["n_designs"], base["seed"])
