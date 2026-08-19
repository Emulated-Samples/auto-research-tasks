"""Transactional FINALIZE behavior with an already-published manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import datagen.build_corpus as builder
import grader.grade as grader


AUTH_KEY = bytes(range(32))


def _prepare_single_cell(monkeypatch, tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    dataset = corpus / "only" / "d_only"
    (dataset / "truth").mkdir(parents=True)
    anchor = {
        "id": "d_only",
        "path": "only/d_only",
        "category": "only",
        "family": "binomial-logit",
        "replicate": 0,
        "weight": 1.0,
        "_gap": 0.25,
    }
    (dataset / "truth" / "anchors.json").write_text(json.dumps(anchor))

    monkeypatch.setattr(builder, "CORPUS", str(corpus))
    monkeypatch.setattr(builder, "CATEGORIES", {"only": object()})
    monkeypatch.setattr(builder, "REPLICATES_PER_CATEGORY", 1)
    monkeypatch.setattr(builder, "DATASET_WEIGHT", 1.0)
    monkeypatch.setattr(builder, "SHIPPED_REQUESTED_N", 10)
    monkeypatch.setattr(builder, "SHIPPED_REQUESTED_P", 2)
    monkeypatch.setattr(builder, "_expected_datasets", lambda: [("only", 0)])
    monkeypatch.setattr(
        builder, "_dataset_path", lambda *_args, **_kwargs: "only/d_only")
    monkeypatch.setattr(
        builder, "_dataset_id", lambda *_args, **_kwargs: "d_only")
    monkeypatch.setattr(
        builder, "_provenance_matches", lambda *_args, **_kwargs: (True, "ok"))
    monkeypatch.setattr(builder, "_dataset_hashes", lambda *_args: {"x": "a" * 64})
    monkeypatch.setattr(builder, "_development_report_sha256", lambda *_args: "b" * 64)
    return corpus


def test_finalize_is_idempotent_with_an_existing_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    corpus = _prepare_single_cell(monkeypatch, tmp_path)
    validation_states = []

    def validate(corpus_dir, manifest, entries, auth_key, *, require_manifest_file=True):
        del manifest, entries, auth_key
        manifest_exists = (Path(corpus_dir) / "manifest.json").exists()
        validation_states.append((require_manifest_file, manifest_exists))
        assert manifest_exists is require_manifest_file

    monkeypatch.setattr(grader, "_validate_corpus", validate)
    builder.finalize(AUTH_KEY, pipeline_sha256="c" * 64)
    first = (corpus / "manifest.json").read_bytes()
    builder.finalize(AUTH_KEY, pipeline_sha256="c" * 64)

    assert (corpus / "manifest.json").read_bytes() == first
    assert validation_states == [
        (False, False), (True, True),
        (False, False), (True, True),
    ]
    assert not (tmp_path / ".svpgs-manifest.previous").exists()
    assert not (tmp_path / ".svpgs-manifest.tmp").exists()


def test_finalize_restores_previous_manifest_when_final_validation_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    corpus = _prepare_single_cell(monkeypatch, tmp_path)
    previous = b'{"known":"good"}\n'
    (corpus / "manifest.json").write_bytes(previous)
    calls = 0

    def validate(*_args, require_manifest_file=True, **_kwargs):
        nonlocal calls
        calls += 1
        if require_manifest_file:
            raise grader.CorpusValidationError("synthetic final failure")

    monkeypatch.setattr(grader, "_validate_corpus", validate)
    with pytest.raises(SystemExit, match="finalized validation failed"):
        builder.finalize(AUTH_KEY, pipeline_sha256="c" * 64)

    assert calls == 2
    assert (corpus / "manifest.json").read_bytes() == previous
    assert not (tmp_path / ".svpgs-manifest.previous").exists()
    assert not (tmp_path / ".svpgs-manifest.tmp").exists()


@pytest.mark.parametrize("fail_final_validation", [False, True])
def test_finalize_restores_previous_manifest_on_unexpected_exception(
    monkeypatch,
    tmp_path: Path,
    fail_final_validation: bool,
) -> None:
    corpus = _prepare_single_cell(monkeypatch, tmp_path)
    previous = b'{"known":"good"}\n'
    (corpus / "manifest.json").write_bytes(previous)

    def validate(*_args, require_manifest_file=True, **_kwargs):
        if require_manifest_file is fail_final_validation:
            raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(grader, "_validate_corpus", validate)
    with pytest.raises(RuntimeError, match="synthetic unexpected failure"):
        builder.finalize(AUTH_KEY, pipeline_sha256="c" * 64)

    assert (corpus / "manifest.json").read_bytes() == previous
    assert not (tmp_path / ".svpgs-manifest.previous").exists()
    assert not (tmp_path / ".svpgs-manifest.tmp").exists()
