"""The corpus builder and grader must activate exactly the same anchor metrics.

The gate has TWO rules and they answer different questions:

  * RELIABILITY (``gap >= K_SE * se_paired``) -- is the gap real?
  * ADEQUACY    (``gap >= RESOLUTION_K * se_naive``) -- is it large enough to be a
    skill DENOMINATOR?

Only the first existed when the v6 corpus shipped, which is how five of fifteen
categories shipped with naive->reference AUC gaps of 0.0047-0.0208: real,
precisely measured, and far too small to divide by. Those gaps carried the
corpus's HIGHEST z-scores, because the paired SE shrinks with the gap
(corr = +0.925). No value of K_SE separates them -- adequacy needs an absolute
yardstick. See rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md and
rollout_analysis/ADDENDUM_9_2026-07-17_the_anchor_is_not_a_ceiling.md.
"""
from __future__ import annotations

import pytest

from datagen.build_corpus import ReferenceFitError, _validate_anchor_quality
from grader.skill import K_SE, RESOLUTION_K

UNIT = 0.01
# Paired SE of the (reference - naive) gap, and the SE of the single naive
# estimate. Kept equal here so the two thresholds differ only by their constants.
SE = {"auc": UNIT, "brier": UNIT, "log_loss": UNIT}
NAIVE_SE = {"auc": UNIT, "brier": UNIT, "log_loss": UNIT}

# Real AND large enough to divide by.
ADEQUATE = (RESOLUTION_K + 1.0) * UNIT
# Reliable (clears K_SE * se_paired) but NOT adequate (under RESOLUTION_K *
# se_naive). This is exactly the shipped corpus's disease.
INADEQUATE = (RESOLUTION_K - 1.0) * UNIT
# Not even reliable.
TIE = 0.2 * UNIT

assert INADEQUATE >= K_SE * UNIT, "INADEQUATE must still clear the reliability bar"
assert INADEQUATE < RESOLUTION_K * UNIT
assert ADEQUATE >= RESOLUTION_K * UNIT


def _metrics(auc_gap, brier_gap, log_loss_gap):
    naive = {"auc": 0.60, "brier": 0.20, "log_loss": 0.50}
    reference = {
        "auc": naive["auc"] + auc_gap,
        "brier": naive["brier"] - brier_gap,
        "log_loss": naive["log_loss"] - log_loss_gap,
    }
    return naive, reference


@pytest.mark.parametrize(
    "gaps",
    [
        (ADEQUATE, ADEQUATE, ADEQUATE),
        (ADEQUATE, TIE, ADEQUATE),
        (ADEQUATE, -TIE, ADEQUATE),
        # A metric on which the reference reliably LOSES is excluded, exactly as
        # the grader excludes it; one adequate metric still carries the dataset.
        (ADEQUATE, -ADEQUATE, ADEQUATE),
        # One adequate metric is enough, even if the others are unusable.
        (ADEQUATE, INADEQUATE, TIE),
    ],
)
def test_any_adequate_win_is_accepted(gaps):
    naive, reference = _metrics(*gaps)
    _validate_anchor_quality("dataset", naive, reference, SE, NAIVE_SE)


@pytest.mark.parametrize(
    "gaps",
    [
        (TIE, TIE, TIE),
        (-ADEQUATE, TIE, TIE),
    ],
)
def test_no_reliable_win_is_rejected(gaps):
    naive, reference = _metrics(*gaps)
    with pytest.raises(ReferenceFitError, match="BROKEN ANCHOR"):
        _validate_anchor_quality("dataset", naive, reference, SE, NAIVE_SE)


@pytest.mark.parametrize(
    "gaps",
    [
        (INADEQUATE, INADEQUATE, INADEQUATE),
        (INADEQUATE, TIE, TIE),
    ],
)
def test_real_but_inadequate_gap_is_rejected(gaps):
    """THE REGRESSION THIS GATE EXISTS FOR.

    Every gap here is REAL and statistically significant -- it clears
    ``K_SE * se_paired`` -- and every one is too small to be a denominator. The
    old existence-only gate accepted exactly this and shipped five categories
    that minted clamped free credit. The rejection must NAME the inadequacy
    rather than blaming reliability, because the two failures need different
    fixes.
    """
    naive, reference = _metrics(*gaps)
    with pytest.raises(ReferenceFitError, match="inadequate") as excinfo:
        _validate_anchor_quality("dataset", naive, reference, SE, NAIVE_SE)
    assert "BROKEN ANCHOR" in str(excinfo.value)


def test_the_shipped_corpus_collapsed_anchors_would_now_be_rejected():
    """The measured v6 failures, encoded as data.

    `ld_shift` shipped an AUC gap of +0.0047 at a paired SE of 0.0003 -- z = 18.0,
    the HIGHEST z in the corpus and the smallest gap in it. A reliability gate
    passes it at any sane K_SE. Adequacy must not.
    """
    naive = {"auc": 0.5948, "brier": 0.20, "log_loss": 0.50}
    reference = {"auc": 0.5995, "brier": 0.20, "log_loss": 0.50}
    ref_naive_se = {"auc": 0.0003, "brier": 0.01, "log_loss": 0.01}
    naive_se = {"auc": 0.005, "brier": 0.01, "log_loss": 0.01}
    gap = reference["auc"] - naive["auc"]
    assert gap / ref_naive_se["auc"] > 15.0, "this gap is overwhelmingly significant"
    assert gap < RESOLUTION_K * naive_se["auc"], "and still too small to divide by"
    with pytest.raises(ReferenceFitError, match="BROKEN ANCHOR"):
        _validate_anchor_quality("ld_shift", naive, reference, ref_naive_se, naive_se)


def test_nonfinite_metric_data_is_rejected():
    naive, reference = _metrics(ADEQUATE, ADEQUATE, ADEQUATE)
    reference["auc"] = float("nan")
    with pytest.raises(ReferenceFitError):
        _validate_anchor_quality("dataset", naive, reference, SE, NAIVE_SE)


def test_nonpositive_naive_se_is_rejected():
    """se_naive is the adequacy denominator; a zero would make every gap look
    infinitely adequate."""
    naive, reference = _metrics(ADEQUATE, ADEQUATE, ADEQUATE)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ReferenceFitError):
            _validate_anchor_quality(
                "dataset", naive, reference, SE, dict(NAIVE_SE, auc=bad)
            )


def test_zero_paired_gap_se_is_rejected():
    """A zero paired SE is not evidence of infinite reliability.

    Bootstrap uncertainty must be strictly positive. Accepting zero turns any
    positive reference gap into an automatic reliability pass and hides a broken
    resampling pipeline.
    """
    naive, reference = _metrics(ADEQUATE, ADEQUATE, ADEQUATE)
    with pytest.raises(ReferenceFitError, match="non-finite metric data"):
        _validate_anchor_quality(
            "dataset", naive, reference, dict(SE, auc=0.0), NAIVE_SE
        )


@pytest.mark.parametrize("missing", ["auc", "brier", "log_loss"])
def test_incomplete_se_maps_are_rejected(missing):
    naive, reference = _metrics(ADEQUATE, ADEQUATE, ADEQUATE)
    with pytest.raises(ReferenceFitError, match="naive single-estimate SE keys"):
        _validate_anchor_quality(
            "dataset", naive, reference, SE,
            {k: v for k, v in NAIVE_SE.items() if k != missing},
        )
    with pytest.raises(ReferenceFitError, match="reference-gap SE keys"):
        _validate_anchor_quality(
            "dataset", naive, reference,
            {k: v for k, v in SE.items() if k != missing}, NAIVE_SE,
        )
