"""Fail-closed contract tests for the trusted held-out truth reader."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from grader.truth import read_aligned_binary_truth_csv, read_binary_truth_csv


def _write_truth(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_binary_truth_reader_preserves_row_order_and_float64_targets(
    tmp_path: Path,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\ns2,1\ns0,0\ns1,1\n",
    )

    sample_ids, targets = read_binary_truth_csv(truth_path)

    assert sample_ids == ("s2", "s0", "s1")
    assert targets.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(targets, np.array([1.0, 0.0, 1.0]))


def test_aligned_binary_truth_reader_returns_targets_for_exact_order(
    tmp_path: Path,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\ns0,0\ns1,1\n",
    )

    targets = read_aligned_binary_truth_csv(truth_path, ("s0", "s1"))

    np.testing.assert_array_equal(targets, np.array([0.0, 1.0]))


def test_aligned_binary_truth_reader_rejects_row_order_mismatch(
    tmp_path: Path,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\ns0,0\ns1,1\n",
    )

    with pytest.raises(ValueError, match="public test-row order"):
        read_aligned_binary_truth_csv(truth_path, ("s1", "s0"))


@pytest.mark.parametrize(
    "header",
    [
        "id,y",
        "y,sample_id",
        "sample_id,y,extra",
        "sample_id, y",
        "Sample_ID,y",
        "\ufeffsample_id,y",
    ],
)
def test_binary_truth_reader_requires_exact_header(
    tmp_path: Path,
    header: str,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        f"{header}\ns0,0\ns1,1\n",
    )

    with pytest.raises(ValueError, match="invalid held-out truth header"):
        read_binary_truth_csv(truth_path)


@pytest.mark.parametrize("content", ["", "\n", "   \n"])
def test_binary_truth_reader_rejects_empty_or_blank_file(
    tmp_path: Path,
    content: str,
) -> None:
    truth_path = _write_truth(tmp_path / "truth.csv", content)

    with pytest.raises(ValueError):
        read_binary_truth_csv(truth_path)


@pytest.mark.parametrize(
    "rows",
    [
        "",
        "\ns0,0\ns1,1\n",
        "s0\ns1,1\n",
        "s0,0,extra\ns1,1\n",
        "s0,0\ns1,\"1\n",
    ],
    ids=[
        "no-data-rows",
        "blank-row",
        "missing-column",
        "extra-column",
        "unterminated-quote",
    ],
)
def test_binary_truth_reader_rejects_blank_or_malformed_rows(
    tmp_path: Path,
    rows: str,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\n" + rows,
    )

    with pytest.raises(ValueError):
        read_binary_truth_csv(truth_path)


def test_binary_truth_reader_rejects_empty_sample_id(tmp_path: Path) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\n,0\ns1,1\n",
    )

    with pytest.raises(ValueError, match="invalid row"):
        read_binary_truth_csv(truth_path)


def test_binary_truth_reader_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\ns0,0\ns0,1\n",
    )

    with pytest.raises(ValueError, match="unique sample IDs"):
        read_binary_truth_csv(truth_path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("s0,nope\ns1,1\n", "non-numeric target"),
        ("s0,\ns1,1\n", "non-numeric target"),
        ("s0,nan\ns1,1\n", "both binary classes"),
        ("s0,inf\ns1,0\n", "both binary classes"),
        ("s0,-inf\ns1,1\n", "both binary classes"),
        ("s0,0\ns1,0\n", "both binary classes"),
        ("s0,1\ns1,1\n", "both binary classes"),
        ("s0,0\ns1,2\n", "both binary classes"),
        ("s0,0.5\ns1,1\n", "both binary classes"),
    ],
    ids=[
        "nonnumeric",
        "empty-target",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "single-class-zero",
        "single-class-one",
        "nonbinary-integer",
        "nonbinary-fraction",
    ],
)
def test_binary_truth_reader_rejects_invalid_targets(
    tmp_path: Path,
    rows: str,
    message: str,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\n" + rows,
    )

    with pytest.raises(ValueError, match=message):
        read_binary_truth_csv(truth_path)


@pytest.mark.parametrize(
    "expected_sample_ids",
    [
        (),
        ("", "s1"),
        ("s0", "s0"),
        ("s0", 1),
        "s0",
    ],
    ids=["empty", "empty-id", "duplicate", "non-string", "scalar-string"],
)
def test_aligned_binary_truth_reader_rejects_invalid_expected_ids(
    tmp_path: Path,
    expected_sample_ids: object,
) -> None:
    truth_path = _write_truth(
        tmp_path / "truth.csv",
        "sample_id,y\ns0,0\ns1,1\n",
    )

    with pytest.raises(
        ValueError,
        match="expected truth sample IDs must be unique non-empty strings",
    ):
        read_aligned_binary_truth_csv(
            truth_path,
            cast(Sequence[str], expected_sample_ids),
        )


def test_binary_truth_reader_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plain regular file"):
        read_binary_truth_csv(tmp_path)


def test_binary_truth_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo_path = tmp_path / "truth.fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(ValueError, match="plain regular file"):
        read_binary_truth_csv(fifo_path)


def test_binary_truth_reader_rejects_symlink(tmp_path: Path) -> None:
    target_path = _write_truth(
        tmp_path / "target.csv",
        "sample_id,y\ns0,0\ns1,1\n",
    )
    link_path = tmp_path / "truth.csv"
    link_path.symlink_to(target_path)

    with pytest.raises(OSError):
        read_binary_truth_csv(link_path)
