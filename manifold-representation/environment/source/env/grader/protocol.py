"""Submission-output contract and behavioral integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np


MAX_PREDICTION_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Prediction:
    reconstruction: np.ndarray
    presence: np.ndarray
    contributions: np.ndarray
    bias: np.ndarray

    @property
    def n_features(self) -> int:
        return self.presence.shape[1]


def load_prediction(path: Path, x: np.ndarray, max_features: int) -> Prediction:
    """Load and strictly validate one ``predictions.npz`` artifact."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("predictions.npz must be a regular file")
    if path.stat().st_size > MAX_PREDICTION_BYTES:
        raise ValueError("predictions.npz exceeds the 512 MiB artifact limit")
    n, d = x.shape
    expected_members = {
        "reconstruction.npy",
        "presence.npy",
        "contributions.npy",
        "bias.npy",
    }
    max_uncompressed = max(
        1 << 20,
        2 * 8 * (n * d + n * max_features + n * max_features * d + d),
    )
    try:
        with zipfile.ZipFile(path) as zipped:
            infos = zipped.infolist()
            names = [info.filename for info in infos]
            # Duplicates first, and as their own message. A set comparison
            # silently collapses them, so an archive carrying two `presence.npy`
            # members passed this "exact schema" check and numpy then used the
            # second -- the value the schema check never inspected. Which member
            # wins is a property of the ZIP reader, not of the contract. Keep this
            # distinct from a wrong-names failure: they have different causes and
            # a reader who is told the wrong one goes and fixes the wrong thing.
            if len(set(names)) != len(names):
                raise ValueError("predictions.npz contains duplicate archive members")
            if set(names) != expected_members:
                raise ValueError("predictions.npz contains unexpected archive members")
            if sum(info.file_size for info in infos) > max_uncompressed:
                raise ValueError("predictions.npz expands beyond the output-shape limit")
    except zipfile.BadZipFile as error:
        raise ValueError("predictions.npz is not a valid NPZ archive") from error
    with np.load(path, allow_pickle=False) as archive:
        expected = {"reconstruction", "presence", "contributions", "bias"}
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError(f"predictions.npz keys must be exactly {sorted(expected)}")
        raw = {key: archive[key] for key in expected}
        if any(array.dtype.kind not in "biuf" for array in raw.values()):
            raise ValueError("prediction arrays must have real numeric dtypes")
        reconstruction = np.asarray(raw["reconstruction"], dtype=np.float64)
        presence = np.asarray(raw["presence"], dtype=np.float64)
        contributions = np.asarray(raw["contributions"], dtype=np.float64)
        bias = np.asarray(raw["bias"], dtype=np.float64)

    if reconstruction.shape != (n, d):
        raise ValueError(f"reconstruction must have shape {(n, d)}, got {reconstruction.shape}")
    if presence.ndim != 2 or presence.shape[0] != n:
        raise ValueError("presence must have shape (rows, learned_features)")
    g = presence.shape[1]
    if not 1 <= g <= max_features:
        raise ValueError(f"learned feature count must be in [1, {max_features}], got {g}")
    if contributions.shape != (n, g, d):
        raise ValueError(f"contributions must have shape {(n, g, d)}, got {contributions.shape}")
    if bias.shape != (d,):
        raise ValueError(f"bias must have shape {(d,)}, got {bias.shape}")
    arrays = (reconstruction, presence, contributions, bias)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("all prediction arrays must be finite")
    if any(np.max(np.abs(array), initial=0.0) > 1e12 for array in arrays):
        raise ValueError("prediction magnitudes exceed the numeric safety limit")
    if np.any(presence < 0.0):
        raise ValueError("presence scores must be nonnegative")
    return Prediction(reconstruction, presence, contributions, bias)


def additive_error(prediction: Prediction, x: np.ndarray) -> float:
    decoded = prediction.bias[None, :] + prediction.contributions.sum(axis=1)
    scale = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
    return float(np.sqrt(np.mean((prediction.reconstruction - decoded) ** 2)) / scale)


def presence_contribution_agreement(prediction: Prediction) -> float:
    """Per-row support-ranking agreement without a hidden support count."""
    norms = np.linalg.norm(prediction.contributions, axis=2)
    thresholds = np.maximum(1e-8, 1e-5 * np.max(norms, axis=1, keepdims=True))
    support = norms > thresholds
    agreements: list[float] = []
    for labels, scores in zip(support, prediction.presence, strict=True):
        positives = int(labels.sum())
        if positives == 0:
            agreements.append(float(np.max(scores, initial=0.0) <= 1e-8))
            continue
        order = np.argsort(-scores, kind="stable")
        ranked = labels[order]
        ranked_scores = np.asarray(scores)[order]
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        # Tied presence scores must not be ranked by feature index -- that is a
        # gauge, and a feature permutation would then move the integrity
        # multiplier. Every member of an equal-score run takes that run's final
        # (lowest) precision: a conservative, order-independent convention. It is
        # NOT the mean an unbiased random tie-break would give -- for one positive
        # tied with one negative that mean is 0.75 while this assigns 0.5 -- but
        # permutation-invariance is the property we need, and understating a tie
        # never lets a gauge choice buy credit.
        group_precision = precision.copy()
        start = 0
        n_ranked = len(ranked_scores)
        while start < n_ranked:
            end = start
            while end + 1 < n_ranked and ranked_scores[end + 1] == ranked_scores[start]:
                end += 1
            group_precision[start : end + 1] = precision[end]
            start = end + 1
        agreements.append(float(group_precision[ranked].sum() / positives))
    return float(np.mean(agreements))


def permutation_error(original: Prediction, permuted: Prediction, inverse: np.ndarray) -> float:
    """Maximum normalized difference after undoing an evaluation-row permutation."""
    if original.n_features != permuted.n_features:
        return float("inf")
    scale = max(1.0, float(np.max(np.abs(original.reconstruction))))
    errors = [
        np.max(np.abs(original.reconstruction - permuted.reconstruction[inverse])) / scale,
        np.max(np.abs(original.presence - permuted.presence[inverse]))
        / max(1.0, float(np.max(original.presence))),
        np.max(np.abs(original.contributions - permuted.contributions[inverse])) / scale,
        np.max(np.abs(original.bias - permuted.bias)) / scale,
    ]
    return float(max(errors))
