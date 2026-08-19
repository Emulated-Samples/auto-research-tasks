"""Held-out accuracy metrics for the binary svpgsbench outcome env.

The shipped task has one family, ``binomial-logit``: AUC (rank), Brier score,
and clipped log-loss.

Each metric reports a scalar plus a `higher_is_better` flag so the skill
normalizer (skill.py) can orient it. All are computed on HELD-OUT samples.
"""
from __future__ import annotations

import numpy as np


def auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score, float)
    order = np.argsort(score, kind="mergesort")
    # average ranks for ties
    sc = score[order]
    ranks_ord = np.arange(1, len(score) + 1, dtype=float)
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        if j > i:
            ranks_ord[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(score), float)
    ranks[order] = ranks_ord
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def brier(y, prob):
    prob = np.clip(np.asarray(prob, float), 0.0, 1.0)
    return float(np.mean((prob - np.asarray(y, float)) ** 2))


def log_loss(y, prob, eps=1e-6):
    prob = np.clip(np.asarray(prob, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def binary_metrics(y, pred_df):
    """pred_df: dict/DataFrame-like with column 'mean' (probability)."""
    prob = np.asarray(pred_df["mean"], float)
    return {
        "auc": (auc(y, prob), True),
        "brier": (brier(y, prob), False),
        "log_loss": (log_loss(y, prob), False),
    }


def compute_metrics(family, y, pred_df):
    if family != "binomial-logit":
        raise ValueError(f"unsupported family {family!r}; expected 'binomial-logit'")
    return binary_metrics(y, pred_df)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    good = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 500), 0, 1)
    bad = np.full(500, 0.3)
    print("AUC good/bad:", round(auc(y, good), 3), round(auc(y, bad), 3))
    print("Brier good/bad:", round(brier(y, good), 3), round(brier(y, bad), 3))
