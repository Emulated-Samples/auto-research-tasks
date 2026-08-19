"""Report-only runtime headroom and host-normalized efficiency diagnostics.

The disclosed fit/predict timeouts enforce speed. `perf_factor` reports how much
budget headroom a valid run retained, and `efficiency_ratio` compares it with a
host-normalized practical anchor; neither changes scientific reward. This preserves
the exact naive=0/reference=1 accuracy scale and prevents a fast shortcut from buying
reward it did not earn.

WHAT THE RAMP IS MEASURED AGAINST, AND WHY IT CHANGED (2026-07-17)
-----------------------------------------------------------------
The penalty is a fraction of the DISCLOSED per-dataset budget: full credit up to
`PARITY_FRACTION` of it, then a linear ramp to `FLOOR` at the budget itself.

It used to be a ratio against `time_runtime_anchor` -- a fast FISTA ridge that
measured **0.85 s mean** over the shipped corpus -- with the floor at 10x that
ratio. So the reward halved past **~8.5 s** while the prompt advertised a **170 s**
fit budget, and nothing in the prompt, the workspace or the file contract stated
the exchange rate. That is a prompt/verifier mismatch: an expert who reads
"170 seconds" and spends 60 of them silently ate a 2x penalty for obeying the
spec. Measured consequences on the shipped corpus
(rollout_analysis/ADDENDUM_10_2026-07-17_the_speed_axis_pays_the_shortcut.md):

  * `cheats/marginal_pt` (1.1 s) kept perf 0.987 -> reward +0.9272, while `gold`
    (66.7 s, a genuine annotation-aware hierarchical EB) took perf 0.500 ->
    reward +0.2392. **The env paid a 1.1-second shortcut 3.9x what it paid the
    real method.**
  * The two cheats' accuracy skills are 0.9451 and 0.9423 -- a 0.003 difference --
    and their rewards differed by 2x, PURELY on speed.
  * SV-PGS itself, which *defines* skill = 1.0, takes 87.6 s mean and so scored
    perf 0.500: the env's own reference could not score 1.0 on the env's own
    reward.

The budget is the one speed quantity the agent is actually told
(`tasks/prompt_spec.py` states the fit and predict caps), so a ramp denominated
in it is inferable from the prompt: "finish well inside the cap; faster is
better". A ramp denominated in a 0.85 s anchor nobody can see is not.

This is deliberately NOT scored against `time_reference` either. Tying speed
parity to the accuracy anchor would let one implementation define both skill=1.0
and speed parity, and would silently re-scale the speed axis every time the
reference changes.

`time_runtime_anchor` remains the basis of the reported `efficiency` ratio (see
`efficiency_ratio`), which is what it is good for: a host-calibrated,
practical-model yardstick for *reporting* speed. It no longer sets reward.
"""
from __future__ import annotations

import math
import time

from grader.contract import DATASET_TIMEOUT_S

# The ramp is denominated in the DISCLOSED per-dataset budget (fit + predict),
# which is exactly the cap the prompt states and the runner enforces.
BUDGET_S = float(DATASET_TIMEOUT_S)
# Full credit while a submission uses at most this fraction of its budget. Sized
# so that a fit doing real joint work is not punished for using the budget it was
# given, while burning nearly all of it still costs something.
PARITY_FRACTION = 0.25
FLOOR = 0.5


def calibration_seconds(reps: int = 3) -> float:
    """Wall-seconds for a fixed, deterministic single-thread numeric workload.

    Run once on the reference machine during corpus build (stored in anchors as
    `time_calibration`) and once on the grading machine at grade time. Their
    ratio rescales the stored reference time to the grading host's speed, so the
    perf factor measures the SUBMISSION's algorithm rather than which CPU / BLAS
    the verifier happens to run on. The workload (fixed-size GEMM + Cholesky +
    elementwise, seeded) is representative of the linear-algebra a PGS fit does
    and is BLAS-thread-capped by the caller's env. Takes the min over `reps` to
    reject scheduler noise."""
    import numpy as np
    best = float("inf")
    for _ in range(max(1, reps)):
        rng = np.random.default_rng(12345)
        n = 500
        A = rng.standard_normal((n, n))
        t0 = time.perf_counter()
        M = A @ A.T
        M[np.diag_indices(n)] += n            # make SPD
        L = np.linalg.cholesky(M)
        s = float(np.sum(np.exp(-np.abs(L[:64, :64]))))  # keep result live
        _ = s
        best = min(best, time.perf_counter() - t0)
    return best


def normalized_reference_time(t_ref, ref_calibration, grade_calibration):
    """Rescale a stored reference fit+predict time to the grading machine's speed
    using the calibration ratio. All inputs are mandatory and positive; a corpus
    without calibration is invalid rather than silently scored on a different
    timing contract."""
    values = tuple(float(v) for v in (t_ref, ref_calibration, grade_calibration))
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"reference timing/calibrations must be positive: {values!r}")
    t_ref_value, ref_value, grade_value = values
    return t_ref_value * (grade_value / ref_value)


def perf_factor(t_sub, *, budget_s=BUDGET_S,
                parity_fraction=PARITY_FRACTION, floor=FLOOR):
    """Report-only runtime-headroom factor in [floor, 1].

    t_sub: submission fit+predict seconds, on the grading host.
    budget_s: the DISCLOSED per-dataset budget the runner enforces.

    Denominated in the budget, so `used = t_sub / budget_s` is the fraction of the
    allowance burned. Full credit while `used <= parity_fraction`; a linear ramp to
    `floor` at `used == 1.0`. Past the budget the runner has already SIGKILLed the
    phase, so `used > 1` is unreachable in a graded run and simply clamps to floor.

    Measuring the submission against its own budget is host-consistent by
    construction: the budget is a wall-clock cap on the grading host, so the same
    machine defines both the numerator and the denominator. That is why no
    calibration ratio appears here -- it would rescale the numerator against a
    denominator that is already in grading-host seconds. Calibration remains in
    `efficiency_ratio`, where a cross-host comparison is genuinely being made.

    This is a diagnostic only. There is no speed bonus and no reward multiplier.
    """
    budget = float(budget_s)
    runtime = float(t_sub)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("budget_s must be finite and positive")
    if not math.isfinite(runtime) or runtime <= 0.0:
        raise ValueError("t_sub must be finite and positive")
    if not math.isfinite(parity_fraction) or not 0.0 < parity_fraction < 1.0:
        raise ValueError("parity_fraction must lie strictly inside (0, 1)")
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("floor must lie in [0, 1]")
    used = runtime / budget  # fraction of the disclosed budget consumed
    if used <= parity_fraction:
        base = 1.0
    else:
        span = 1.0 - parity_fraction
        base = max(floor, 1.0 - (1.0 - floor) * (used - parity_fraction) / span)
    return base


def efficiency_ratio(t_sub, t_ref):
    """Report-only speed ratio (anchor_time / sub_time): >1 means faster than the
    runtime anchor, <1 slower. Recorded in the per-dataset detail so a fast
    solution stays VISIBLE even though runtime does not modify scientific reward."""
    if not t_sub or t_sub <= 0 or not t_ref or t_ref <= 0:
        return None
    return float(t_ref) / float(t_sub)


if __name__ == "__main__":
    print(f"budget = {BUDGET_S:.0f}s   full credit <= {PARITY_FRACTION*BUDGET_S:.0f}s"
          f"   floor {FLOOR} at {BUDGET_S:.0f}s")
    print("report-only runtime headroom:")
    for label, ts in {
        "1.1 s (P+T shortcut)": 1.1,
        "25% of budget": 0.25 * BUDGET_S,
        "50% of budget": 0.50 * BUDGET_S,
        "66.7 s (gold)": 66.7,
        "87.6 s (SV-PGS mean)": 87.6,
        "the whole budget": BUDGET_S,
    }.items():
        pf = perf_factor(ts)
        assert FLOOR <= pf <= 1.0, (label, pf)
        print(f"  {label:22s} -> {pf:.3f}")
