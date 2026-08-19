"""Strict readers for the trusted binary held-out target contract."""

from __future__ import annotations

import csv
import os
import stat
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def read_binary_truth_csv(
    path: str | Path,
) -> tuple[tuple[str, ...], np.ndarray]:
    truth_path = Path(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptor = os.open(truth_path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(
                f"held-out truth must be a plain regular file: {truth_path}"
            )
        handle = os.fdopen(descriptor, "r", encoding="utf-8", newline="")
    except Exception:
        os.close(descriptor)
        raise
    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty held-out truth file: {truth_path}") from exc
        except csv.Error as exc:
            raise ValueError(f"malformed held-out truth CSV: {truth_path}") from exc
        if header != ["sample_id", "y"]:
            raise ValueError(f"invalid held-out truth header: {truth_path}")
        sample_ids: list[str] = []
        targets: list[float] = []
        try:
            for line_number, row in enumerate(reader, start=2):
                if len(row) != 2 or not row[0]:
                    raise ValueError(
                        f"{truth_path}:{line_number} has an invalid row"
                    )
                sample_ids.append(row[0])
                try:
                    targets.append(float(row[1]))
                except ValueError as exc:
                    raise ValueError(
                        f"{truth_path}:{line_number} has a non-numeric target"
                    ) from exc
        except csv.Error as exc:
            raise ValueError(
                f"{truth_path}:{reader.line_num} has malformed CSV"
            ) from exc
    resolved_ids = tuple(sample_ids)
    target_array = np.asarray(targets, dtype=np.float64)
    if (
        not resolved_ids
        or len(set(resolved_ids)) != len(resolved_ids)
        or target_array.shape != (len(resolved_ids),)
        or set(np.unique(target_array)) != {0.0, 1.0}
    ):
        raise ValueError(
            "held-out truth must have unique sample IDs and both binary classes"
        )
    return resolved_ids, target_array


def read_aligned_binary_truth_csv(
    path: str | Path,
    expected_sample_ids: Sequence[str],
) -> np.ndarray:
    if isinstance(expected_sample_ids, str):
        raise ValueError("expected truth sample IDs must be unique non-empty strings")
    expected = tuple(expected_sample_ids)
    if (
        not expected
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in expected
        )
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("expected truth sample IDs must be unique non-empty strings")
    observed, targets = read_binary_truth_csv(path)
    if observed != expected:
        raise ValueError("held-out truth rows do not match public test-row order")
    return targets
