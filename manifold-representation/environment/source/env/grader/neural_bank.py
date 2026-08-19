"""Validated access to the compact neural activation coordinate bank.

The bank is produced offline by :mod:`tools.build_neural_bank`.  Loading and
sampling it requires only NumPy; no model runtime or network access occurs
during grading.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np


BANK_PATH = Path(__file__).resolve().parent / "data" / "neural_manifolds_v1.npz"
BANK_SHA256 = "cad81f6df96b29457fe5ce8d1f912c3510a9488b45d6ea21c32a6fa2e947e8a8"
SPLIT_INDEX = {"train": 0, "match": 1, "score": 2}
EXPECTED_KEYS = {
    "coordinates",
    "counts",
    "row_ids",
    "source_names",
    "modalities",
    "source_rows",
    "explained_variance",
}


@dataclass(frozen=True)
class NeuralBank:
    coordinates: np.ndarray
    counts: np.ndarray
    source_names: tuple[str, ...]
    modalities: tuple[str, ...]
    source_rows: np.ndarray
    explained_variance: np.ndarray

    @property
    def n_families(self) -> int:
        return len(self.source_names)

    @property
    def chart_dim(self) -> int:
        return self.coordinates.shape[-1]

    def points(self, family: int, split: str) -> np.ndarray:
        split_index = SPLIT_INDEX.get(split)
        if split_index is None:
            raise ValueError(f"unknown neural-bank split: {split}")
        if not 0 <= family < self.n_families:
            raise ValueError(f"neural-bank family out of range: {family}")
        count = int(self.counts[family, split_index])
        return self.coordinates[family, split_index, :count]

    def sample(
        self,
        family: int,
        split: str,
        rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        points = self.points(family, split)
        indices = rng.integers(0, len(points), size=rows)
        return np.asarray(points[indices], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_disjoint(
    row_ids: np.ndarray,
    counts: np.ndarray,
    modalities: tuple[str, ...],
) -> None:
    for family in range(row_ids.shape[0]):
        seen: list[set[int]] = []
        for split in range(row_ids.shape[1]):
            rows = row_ids[family, split, : int(counts[family, split])]
            seen.append(set(int(value) for value in rows))
        if any(seen[left] & seen[right] for left in range(3) for right in range(left + 1, 3)):
            raise RuntimeError(f"neural bank family {family} leaks rows across splits")
    for modality in set(modalities):
        family_indices = [index for index, value in enumerate(modalities) if value == modality]
        split_rows = []
        for split in range(3):
            split_rows.append(
                set().union(
                    *(
                        set(
                            int(value)
                            for value in row_ids[family, split, : int(counts[family, split])]
                        )
                        for family in family_indices
                    )
                )
            )
        if any(
            split_rows[left] & split_rows[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise RuntimeError(f"neural bank {modality} source leaks rows across splits")


@lru_cache(maxsize=1)
def load_neural_bank() -> NeuralBank:
    if _sha256(BANK_PATH) != BANK_SHA256:
        raise RuntimeError("neural activation bank checksum mismatch")
    with np.load(BANK_PATH, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_KEYS:
            raise RuntimeError("neural activation bank schema mismatch")
        coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
        counts = np.asarray(archive["counts"], dtype=np.int32)
        row_ids = np.asarray(archive["row_ids"], dtype=np.int32)
        source_names = tuple(str(value) for value in archive["source_names"])
        modalities = tuple(str(value) for value in archive["modalities"])
        source_rows = np.asarray(archive["source_rows"], dtype=np.int32)
        explained = np.asarray(archive["explained_variance"], dtype=np.float64)

    if coordinates.ndim != 4 or coordinates.shape[1:] != (3, 512, 4):
        raise RuntimeError(f"invalid neural activation coordinate shape: {coordinates.shape}")
    families = coordinates.shape[0]
    if (
        counts.shape != (families, 3)
        or row_ids.shape != coordinates.shape[:3]
        or source_rows.shape != (families,)
        or explained.shape != (families,)
    ):
        raise RuntimeError("invalid neural activation bank metadata shape")
    if len(source_names) != families or len(modalities) != families:
        raise RuntimeError("invalid neural activation bank family metadata")
    if not np.all((counts > 0) & (counts <= coordinates.shape[2])):
        raise RuntimeError("invalid neural activation bank split counts")
    if not np.all(source_rows >= counts.sum(axis=1)):
        raise RuntimeError("invalid neural activation source row counts")
    for family in range(families):
        for split in range(3):
            values = coordinates[family, split, : int(counts[family, split])]
            if not np.isfinite(values).all():
                raise RuntimeError("non-finite coordinate in neural activation bank")
    if not np.all((explained > 0.0) & (explained <= 1.0)):
        raise RuntimeError("invalid neural activation explained variance")
    if set(modalities) != {"llm", "vision"}:
        raise RuntimeError("neural activation bank must contain LLM and vision families")
    _validate_disjoint(row_ids, counts, modalities)
    return NeuralBank(
        coordinates,
        counts,
        source_names,
        modalities,
        source_rows,
        explained,
    )
