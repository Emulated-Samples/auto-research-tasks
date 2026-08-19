"""pass@k for a continuous benchmark, without inventing the pass.

The evaluated release reported "3/3 categories passed" off a hard-coded 0.6
display threshold, from ONE gradable rollout. That is not a pass rate: the
threshold was never calibrated, and n=1 cannot estimate pass@k at any k.

This module keeps the native continuous score and adds ONE auditable binary
event on top of it:

    mastered(rollout) = gradable AND integrity AND contract
                        AND headline    >= tau
                        AND bottom-k tail mean >= tau_tail

`tau`/`tau_tail` are NOT chosen here. They come from grader.contract, which
holds None until a published calibration pilot fixes them, and `pass_at_k`
REFUSES to produce a number while they are None. A fabricated threshold would
manufacture a pass@k signal that no measurement supports, which is worse than
reporting no pass@k at all.

Infrastructure attrition is not model failure. A rollout that a provider rate
limit killed before grading (as happened to the second Opus 4.8 rollout, at turn
130) is CENSORED: it leaves the capability denominator entirely and is reported
separately as a completion rate. Counting it as a failure biases pass@k down;
silently dropping it without reporting hides real evaluation cost.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grader.contract import MASTERY_THRESHOLD, MASTERY_TAIL_THRESHOLD  # noqa: E402
from grader.skill import SKILL_HI, SKILL_LO  # noqa: E402

# MASTERY_TAIL_THRESHOLD now lives in grader.contract (imported above) so the
# grader and the wrapper share one source of truth for the pass event.


class MasteryNotCalibrated(RuntimeError):
    """No published calibration fixes the mastery event, so pass@k has no meaning."""


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """Two-sided Wilson interval for the per-attempt mastery probability."""
    if trials <= 0 or not 0 <= successes <= trials:
        return None
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


@dataclass(frozen=True)
class Rollout:
    """One launched attempt. `gradable=False` means it never reached grading."""
    rollout_id: str
    gradable: bool
    headline: float | None = None
    tail: float | None = None
    integrity_ok: bool = False
    contract_ok: bool = False
    # Why a non-gradable rollout died: 'infra' (provider/harness/quota — censored,
    # excluded from the capability denominator) or 'agent' (the agent produced no
    # gradable artifact within its budget — a real capability failure, counted).
    censor_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id:
            raise ValueError("rollout_id must be a nonempty string")
        if type(self.gradable) is not bool:
            raise ValueError(f"{self.rollout_id}: gradable must be an exact boolean")
        if type(self.integrity_ok) is not bool or type(self.contract_ok) is not bool:
            raise ValueError(
                f"{self.rollout_id}: integrity_ok/contract_ok must be exact booleans")
        if not self.gradable:
            if self.censor_reason not in {"infra", "agent"}:
                raise ValueError(
                    f"{self.rollout_id}: a non-gradable rollout must declare "
                    "censor_reason 'infra' or 'agent'")
            if (
                self.headline is not None
                or self.tail is not None
                or self.integrity_ok
                or self.contract_ok
            ):
                raise ValueError(
                    f"{self.rollout_id}: a non-gradable rollout cannot carry scores "
                    "or successful integrity/contract verdicts")
            return
        if self.censor_reason is not None:
            raise ValueError(
                f"{self.rollout_id}: a gradable rollout cannot be censor-classified")
        if (self.headline is None) != (self.tail is None):
            raise ValueError(
                f"{self.rollout_id}: headline and tail must be present together")
        if self.headline is None:
            if self.integrity_ok or self.contract_ok:
                raise ValueError(
                    f"{self.rollout_id}: a scoreless rollout cannot claim integrity/contract success")
            return
        for label, value in (("headline", self.headline), ("tail", self.tail)):
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not SKILL_LO <= float(value) <= SKILL_HI
            ):
                raise ValueError(
                    f"{self.rollout_id}: {label} must be finite and within "
                    f"[{SKILL_LO}, {SKILL_HI}]")


def is_infra_censored(rollout: Rollout) -> bool:
    if rollout.gradable:
        return False
    return rollout.censor_reason == "infra"


def capability_denominator(rollouts):
    """The rollouts that could have demonstrated capability: gradable + agent-failed."""
    return [r for r in rollouts if not is_infra_censored(r)]


def mastered(rollout: Rollout) -> bool:
    """The contract-authoritative binary event; callers cannot substitute a bar."""
    if MASTERY_THRESHOLD is None or MASTERY_TAIL_THRESHOLD is None:
        raise MasteryNotCalibrated(
            "mastery thresholds are not calibrated (grader.contract.MASTERY_THRESHOLD "
            "/ MASTERY_TAIL_THRESHOLD are None). Run the frontier calibration pilot, "
            "freeze the thresholds against an INDEPENDENT rollout set, and publish "
            "them with the benchmark version before reporting pass@k."
        )
    if not (rollout.gradable and rollout.integrity_ok and rollout.contract_ok):
        return False
    if rollout.headline is None or rollout.tail is None:
        return False
    return (rollout.headline >= MASTERY_THRESHOLD
            and rollout.tail >= MASTERY_TAIL_THRESHOLD)


def pass_at_k(rollouts, k):
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k) over the capability denominator.

    Returns a dict carrying the estimator's inputs (n, c, k), the mastery event's
    provenance, and the infrastructure attrition that was excluded -- reporting a
    bare probability invites exactly the misreading this module exists to prevent.
    """
    eligible = capability_denominator(rollouts)
    n = len(eligible)
    censored = len(rollouts) - n
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(
            f"pass@{k} needs at least {k} gradable-or-agent-failed rollouts; have {n}. "
            "One rollout cannot estimate a pass rate."
        )
    c = sum(mastered(r) for r in eligible)
    if c == 0:
        estimate = 0.0
    elif n - c < k:
        estimate = 1.0
    else:
        estimate = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    return {
        "pass_at_k": estimate,
        "k": k,
        "n": n,
        "c": c,
        "infra_censored": censored,
        "non_infra_rate": n / len(rollouts) if rollouts else 0.0,
        "gradable_rate": (
            sum(1 for rollout in eligible if rollout.gradable) / len(rollouts)
            if rollouts else 0.0
        ),
        "mastery_rate_wilson_95": _wilson_interval(c, n),
        "mastery_threshold": MASTERY_THRESHOLD,
        "mastery_tail_threshold": MASTERY_TAIL_THRESHOLD,
    }


def best_score_at_k(rollouts, k):
    """E[max headline over k draws] -- the continuous statistic pass@k throws away.

    Computed over the capability denominator by the same order-statistic identity
    the pass@k estimator uses, so it needs no threshold and stays reportable while
    mastery is uncalibrated.

    An AGENT-caused non-completion (non-gradable, non-infra) counts at the skill FLOOR
    (SKILL_LO), NOT dropped: a model that sometimes produces no artifact must not look
    BETTER as its crash rate rises. Only infra-censored attempts are excluded (they are
    already out of capability_denominator).
    """
    scored = sorted(
        (r.headline if (r.gradable and r.headline is not None) else SKILL_LO)
        for r in capability_denominator(rollouts)
    )
    n = len(scored)
    if n < k or k < 1:
        raise ValueError(f"best_score@{k} needs at least {k} rollouts in the "
                         f"capability denominator; have {n}")
    total = math.comb(n, k)
    # P(max == scored[i]) = C(i, k-1) / C(n, k) for the i-th smallest (0-indexed).
    return sum(
        scored[i] * math.comb(i, k - 1) / total
        for i in range(k - 1, n)
    )


def report(rollouts, ks=(1, 2, 4, 8)):
    """Everything a release note must state together, or state as uncalibrated."""
    eligible = capability_denominator(rollouts)
    gradable_scores = [
        r.headline for r in eligible if r.gradable and r.headline is not None
    ]
    capability_scores = [
        r.headline if (r.gradable and r.headline is not None) else SKILL_LO
        for r in eligible
    ]
    out = {
        "launched": len(rollouts),
        "gradable": sum(1 for r in eligible if r.gradable),
        "infra_censored": len(rollouts) - len(eligible),
        "agent_failed": sum(1 for r in eligible if not r.gradable),
        # Never publish an ambiguous mean that silently drops agent failures.
        "gradable_score_mean": (
            sum(gradable_scores) / len(gradable_scores) if gradable_scores else None
        ),
        "gradable_score_min": min(gradable_scores) if gradable_scores else None,
        "gradable_score_max": max(gradable_scores) if gradable_scores else None,
        "capability_score_mean": (
            sum(capability_scores) / len(capability_scores)
            if capability_scores else None
        ),
        "capability_score_min": min(capability_scores) if capability_scores else None,
        "capability_score_max": max(capability_scores) if capability_scores else None,
        "pass_at_k": {},
        "best_score_at_k": {},
    }
    for k in ks:
        try:
            out["pass_at_k"][k] = pass_at_k(rollouts, k)
        except (MasteryNotCalibrated, ValueError) as exc:
            out["pass_at_k"][k] = {"unavailable": str(exc)}
        try:
            out["best_score_at_k"][k] = best_score_at_k(rollouts, k)
        except ValueError as exc:
            out["best_score_at_k"][k] = {"unavailable": str(exc)}
    return out
