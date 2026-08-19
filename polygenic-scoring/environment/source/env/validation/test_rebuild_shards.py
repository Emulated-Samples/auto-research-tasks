from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from grader.corpus_auth import corpus_key_id, json_hmac_sha256
from tools import rebuild_shards as shards


KEY = bytes(range(32))
SOURCE_SHA = "a" * 40


def _signed_manifest(*, provenance: dict, cells: list[dict], files: dict) -> dict:
    manifest = {
        "schema_version": shards.SHARD_SCHEMA_VERSION,
        "mode": "develop",
        "shard_index": 0,
        "shard_count": 1,
        "source_commit": SOURCE_SHA,
        "key_id": corpus_key_id(KEY),
        "scientific_provenance": provenance,
        "cells": cells,
        "output_files_sha256": files,
    }
    manifest["content_sha256"] = shards._content_sha256(manifest)
    manifest[shards.SHARD_HMAC_FIELD] = json_hmac_sha256(
        KEY,
        manifest,
        exclude_field=shards.SHARD_HMAC_FIELD,
        domain=shards.SHARD_HMAC_DOMAIN,
    )
    return manifest


def test_toolchain_metadata_canonicalizes_identical_linux_lib64_duplicates() -> None:
    executable = Path("/opt/svpgs-venv/bin/python")
    assert shards._canonical_distribution_pairs(
        [["numpy", "2.1.3"], ["pip", "24.3.1"], ["numpy", "2.1.3"]],
        executable=executable,
    ) == [["numpy", "2.1.3"], ["pip", "24.3.1"]]

    with pytest.raises(shards.ShardError, match="conflicting installed"):
        shards._canonical_distribution_pairs(
            [["numpy", "2.1.3"], ["numpy", "2.2.0"]],
            executable=executable,
        )


def test_rebuild_runner_preflights_before_any_persistent_mutation() -> None:
    script = (shards.ROOT / "tools" / "run_rebuild_shard.sh").read_text()
    preflight = script.index("rebuild_shards.py\" preflight")
    for mutation in ('mkdir -p "$SHARD_ROOT', "find \"$cell_work\"", "unlink \"$result\""):
        assert script.index(mutation) > preflight
    assert 'find "$SOURCE_ROOT/corpus"' not in script
    assert 'find corpus -mindepth' not in script
    assert 'SHARD_ROOT="$SHARD_BASE/$RUN_ID/${MODE}-${SHARD_INDEX}-of-${SHARD_COUNT}"' in script


def test_corpus_worker_uses_importable_cli_instead_of_exported_heredoc() -> None:
    script = (shards.ROOT / "tools" / "run_rebuild_shard.sh").read_text()
    assert "corpus-cell" in script
    assert "<<'PY'" not in script
    assert '_main()' not in script


def test_shard_tools_are_part_of_the_atomic_release_lock() -> None:
    source = (shards.ROOT / "validation" / "release_gate.py").read_text()
    assert '"tools/rebuild_shards.py"' in source
    assert '"tools/run_rebuild_shard.sh"' in source


def test_output_hashing_rejects_symlinks_and_special_entries(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "target"
    target.write_text("private")
    (output / "escape").symlink_to(target)
    with pytest.raises(shards.ShardError, match="hashed output"):
        shards._tree_file_hashes(output)


def test_authenticated_manifest_binds_provenance_and_exact_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "shard"
    output = root / "output"
    output.mkdir(parents=True)
    result = output / "d_test.json"
    result.write_text('{"ok":true}\n')
    files = {"d_test.json": hashlib.sha256(result.read_bytes()).hexdigest()}
    cell = {"category": "svld_class", "replicate": 0, "dataset_id": "d_test"}
    sealed = {**cell, "output_sha256": files["d_test.json"]}
    provenance = {"scientific": "one exact digest"}
    manifest = _signed_manifest(provenance=provenance, cells=[sealed], files=files)

    monkeypatch.setattr(shards, "_shard_cells", lambda *_args: [cell])
    monkeypatch.setattr(
        shards,
        "_validated_outputs",
        lambda *_args: ([sealed], files),
    )
    assert shards._validate_manifest(
        shard_root=root,
        manifest=manifest,
        mode="develop",
        count=1,
        source_sha=SOURCE_SHA,
        auth_key=KEY,
        provenance=provenance,
    ) == [cell]

    changed = json.loads(json.dumps(manifest))
    changed["scientific_provenance"] = {"scientific": "different"}
    with pytest.raises(shards.ShardError, match="content digest mismatch"):
        shards._validate_manifest(
            shard_root=root,
            manifest=changed,
            mode="develop",
            count=1,
            source_sha=SOURCE_SHA,
            auth_key=KEY,
            provenance=provenance,
        )

    differently_provenanced = _signed_manifest(
        provenance={"scientific": "other valid build"}, cells=[sealed], files=files)
    with pytest.raises(shards.ShardError, match="contract mismatch"):
        shards._validate_manifest(
            shard_root=root,
            manifest=differently_provenanced,
            mode="develop",
            count=1,
            source_sha=SOURCE_SHA,
            auth_key=KEY,
            provenance=provenance,
        )

    with pytest.raises(shards.ShardError, match="contract mismatch"):
        shards._validate_manifest(
            shard_root=root,
            manifest=manifest,
            mode="develop",
            count=1,
            source_sha="b" * 40,
            auth_key=KEY,
            provenance=provenance,
        )

    with pytest.raises(shards.ShardError, match="HMAC mismatch"):
        shards._validate_manifest(
            shard_root=root,
            manifest=manifest,
            mode="develop",
            count=1,
            source_sha=SOURCE_SHA,
            auth_key=b"z" * 32,
            provenance=provenance,
        )


def test_output_file_digest_changes_with_one_byte(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "artifact"
    artifact.write_bytes(b"one")
    before = shards._tree_file_hashes(output)
    artifact.write_bytes(b"two")
    after = shards._tree_file_hashes(output)
    assert before.keys() == after.keys()
    assert before != after


def test_merge_rejects_overlap_before_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "must-survive"
    marker.write_text("original")
    roots = [tmp_path / "shard0", tmp_path / "shard1"]
    for index, root in enumerate(roots):
        (root / "output").mkdir(parents=True)
        (root / shards.SHARD_MANIFEST_NAME).write_text(json.dumps({"shard_index": index}))

    duplicate = {"category": "svld_class", "replicate": 0, "dataset_id": "d_same"}
    monkeypatch.setattr(shards, "_validate_exact_source", lambda *_args: None)
    monkeypatch.setattr(shards, "load_corpus_key", lambda *_args, **_kwargs: KEY)
    monkeypatch.setattr(shards, "_scientific_provenance", lambda *_args: {})
    monkeypatch.setattr(shards, "_validate_shard_root_shape", lambda *_args: None)
    monkeypatch.setattr(
        shards,
        "_read_json",
        lambda path: {"shard_index": 0 if path.parent == roots[0] else 1},
    )
    monkeypatch.setattr(shards, "_validate_manifest", lambda **_kwargs: [duplicate])

    with pytest.raises(shards.ShardError, match="overlapping shard cell"):
        shards.merge(
            mode="develop",
            count=2,
            source_root=source,
            source_sha=SOURCE_SHA,
            key_file=tmp_path / "key",
            builder_python=tmp_path / "python",
            target=target,
            shard_roots=roots,
        )
    assert marker.read_text() == "original"


def test_atomic_merge_stage_is_validated_before_target_exchange() -> None:
    source = (shards.ROOT / "tools" / "rebuild_shards.py").read_text()
    validate = source.index("_validated_outputs(mode, stage, all_cells")
    exchange = source.index("_atomic_replace_directory(stage, target)")
    assert validate < exchange
