#!/usr/bin/env python3
"""Fail-closed, authenticated shard sealing and merge for corpus rebuilds.

This is release machinery, not a convenience copier.  A shard is useful only
after ``seal`` writes ``shard-manifest.json``.  ``merge`` authenticates every
manifest, re-hashes every output byte, proves the shards are one exact partition
of the declared grid, validates the assembled tree, and only then atomically
replaces the requested target.

Build each box with the same run id and source commit::

    bash tools/run_rebuild_shard.sh develop 0 2 v8-final <40-char-sha>
    bash tools/run_rebuild_shard.sh corpus 0 4 v8-final <40-char-sha>

After copying the complete shard roots to one box, merge them explicitly::

    python tools/rebuild_shards.py merge --mode develop --shard-count 2 \
      --source-root /hyperfocal/build --source-sha <sha> \
      --key-file /hyperfocal/corpus.key \
      --builder-python /hyperfocal/venv/bin/python \
      --target /hyperfocal/devwork/results <develop-shard-0> <develop-shard-1>

Run ``develop-finalize`` on that target but write its report outside the source
checkout (for example ``/hyperfocal/devwork/model_zoo_development.json``).  Next
merge corpus shards with ``--mode corpus --target /hyperfocal/build/corpus``.
Only after both merges have authenticated the still-clean exact source should the
development report be atomically installed at its repository path; then run the
existing corpus ``FINALIZE`` command.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datagen.categories import CATEGORIES  # noqa: E402
from grader.contract import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    DEVELOPMENT_RESULT_HMAC_FIELD,
    REPLICATES_PER_CATEGORY,
    SHIPPED_DATASET_COUNT,
)
from grader.corpus_auth import (  # noqa: E402
    DEVELOPMENT_RESULT_HMAC_DOMAIN,
    corpus_key_id,
    json_hmac_sha256,
    load_corpus_key,
    verify_json_hmac,
)
from reference.protocol import (  # noqa: E402
    REFERENCE_NUMPY_VERSION,
    REFERENCE_PYTHON_EXECUTABLE,
)


SHARD_SCHEMA_VERSION = 1
SHARD_MANIFEST_NAME = "shard-manifest.json"
SHARD_HMAC_FIELD = "shard_hmac_sha256"
SHARD_HMAC_DOMAIN = b"svpgsbench|schema-v8|rebuild-shard-v1\x00"
SHARD_CONTENT_DOMAIN = b"svpgsbench|schema-v8|rebuild-shard-content-v1\x00"
EXPECTED_SCIPY_VERSION = "1.14.1"
MODES = frozenset({"develop", "corpus"})
MANIFEST_FIELDS = frozenset({
    "schema_version",
    "mode",
    "shard_index",
    "shard_count",
    "source_commit",
    "key_id",
    "scientific_provenance",
    "cells",
    "output_files_sha256",
    "content_sha256",
    SHARD_HMAC_FIELD,
})


class ShardError(RuntimeError):
    """A fail-closed shard contract violation."""


def _plain_dir(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ShardError(f"{label} is absent: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise ShardError(f"{label} is not a plain directory: {path}")


def _plain_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ShardError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ShardError(f"{label} is not a plain regular file: {path}")


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ShardError(f"cannot securely open hashed output: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ShardError(f"hashed output is not a single-link regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _tree_file_hashes(root: Path) -> dict[str, str]:
    """Hash every regular file below root and reject links/special entries."""
    _plain_dir(root, "output root")
    hashes: dict[str, str] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for name in directories:
            directory = current_path / name
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ShardError(f"output contains a linked/special directory: {directory}")
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative in hashes:
                raise ShardError(f"duplicate output path: {relative}")
            hashes[relative] = _sha256_file(path)
    return dict(sorted(hashes.items()))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(manifest: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", SHARD_HMAC_FIELD}
    }
    return hashlib.sha256(SHARD_CONTENT_DOMAIN + _canonical_json(payload)).hexdigest()


def _validate_coordinates(mode: str, index: int, count: int) -> None:
    if mode not in MODES:
        raise ShardError(f"unknown shard mode: {mode!r}")
    if (
        type(index) is not int
        or type(count) is not int
        or count < 1
        or count > SHIPPED_DATASET_COUNT
        or not 0 <= index < count
    ):
        raise ShardError(f"invalid shard coordinates: index={index!r} count={count!r}")


def _validate_source_sha(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ShardError("source commit must be one exact lowercase 40-character Git SHA")


def _git_output(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ShardError(f"cannot verify exact source: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _validate_exact_source(source_root: Path, expected_sha: str) -> None:
    _plain_dir(source_root, "source root")
    _validate_source_sha(expected_sha)
    actual = _git_output(source_root, "rev-parse", "HEAD")
    if actual != expected_sha:
        raise ShardError(f"source commit mismatch: expected={expected_sha} actual={actual}")
    dirty = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ShardError("source checkout is not exact/clean; refusing a mixed-source shard")


def _validate_absent_path_ancestors(path: Path) -> None:
    """Reject a symlink/special component in the existing prefix of an absent path."""
    current = path.parent
    while not current.exists():
        if current.is_symlink():
            raise ShardError(f"absent output path has a symlinked ancestor: {current}")
        if current == current.parent:
            raise ShardError(f"cannot find an existing parent for output path: {path}")
        current = current.parent
    while True:
        _plain_dir(current, "output-path ancestor")
        if current == current.parent:
            break
        current = current.parent


def _canonical_distribution_pairs(
    distributions: Any,
    *,
    executable: Path,
) -> list[list[str]]:
    """Canonicalize importlib metadata without hiding version conflicts.

    Some Linux virtual environments expose the same ``*.dist-info`` directory
    through both ``lib`` and ``lib64``.  ``importlib.metadata.distributions``
    then reports the exact same installed distribution twice.  That is one
    toolchain, not an ambiguity.  Conflicting versions for the same normalized
    name remain a hard provenance failure.
    """
    if type(distributions) is not list or any(
        type(item) is not list
        or len(item) != 2
        or any(type(part) is not str or not part for part in item)
        for item in distributions
    ):
        raise ShardError(
            f"toolchain emitted invalid installed distributions: {executable}"
        )
    canonical: dict[str, str] = {}
    for name, version in distributions:
        existing = canonical.get(name)
        if existing is not None and existing != version:
            raise ShardError(
                "toolchain exposes conflicting installed distribution versions: "
                f"{executable}: {name}={existing!r}/{version!r}"
            )
        canonical[name] = version
    return [[name, canonical[name]] for name in sorted(canonical)]


def _python_versions(executable: Path) -> dict[str, Any]:
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise ShardError(f"toolchain interpreter is absent: {executable}") from exc
    _plain_file(resolved, "resolved toolchain interpreter")
    if not os.access(resolved, os.X_OK):
        raise ShardError(f"toolchain interpreter is not executable: {resolved}")
    program = (
        "import importlib.metadata as m,json,platform,numpy,scipy;"
        "print(json.dumps({'python':platform.python_version(),"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'distributions':sorted((d.metadata['Name'].lower(),d.version) "
        "for d in m.distributions())},sort_keys=True))"
    )
    completed = subprocess.run(
        # Invoke through the venv entry path so Python retains the intended
        # sys.prefix/site-packages; the resolved binary is separately hashed.
        [str(executable), "-I", "-B", "-c", program],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ShardError(
            f"cannot verify toolchain {executable}: {completed.stderr.strip()}"
        )
    try:
        versions = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ShardError(f"toolchain emitted invalid version record: {executable}") from exc
    if set(versions) != {"python", "numpy", "scipy", "distributions"}:
        raise ShardError(f"toolchain emitted incomplete version record: {executable}")
    versions["distributions"] = _canonical_distribution_pairs(
        versions["distributions"], executable=executable
    )
    return {
        **versions,
        "resolved_executable": str(resolved),
        "executable_sha256": _sha256_file(resolved),
    }


def _validated_toolchains(builder_python: Path) -> dict[str, Any]:
    builder = _python_versions(builder_python)
    reference = _python_versions(Path(REFERENCE_PYTHON_EXECUTABLE))
    if reference["numpy"] != REFERENCE_NUMPY_VERSION:
        raise ShardError(
            f"reference NumPy mismatch: expected={REFERENCE_NUMPY_VERSION} "
            f"actual={reference['numpy']}"
        )
    if reference["scipy"] != EXPECTED_SCIPY_VERSION:
        raise ShardError(
            f"reference SciPy mismatch: expected={EXPECTED_SCIPY_VERSION} "
            f"actual={reference['scipy']}"
        )
    return {
        "builder_executable": str(builder_python),
        "builder_versions": builder,
        "reference_executable": REFERENCE_PYTHON_EXECUTABLE,
        "reference_versions": reference,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
    }


def _scientific_provenance(builder_python: Path) -> dict[str, Any]:
    from datagen.build_corpus import (
        _generation_pipeline_sha256,
        _public_reference_source_sha256,
    )
    from validation.model_zoo import _implementation_hashes

    return {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "generation_pipeline_sha256": _generation_pipeline_sha256(),
        "public_reference_source_sha256": _public_reference_source_sha256(),
        "implementation_sha256": _implementation_hashes(),
        "toolchains": _validated_toolchains(builder_python),
    }


def _all_cells(mode: str, auth_key: bytes) -> list[dict[str, Any]]:
    from datagen.build_corpus import _dataset_id, _generation_pipeline_sha256
    from validation.model_zoo import _development_dataset_id, _implementation_hashes

    if mode == "corpus":
        pipeline = _generation_pipeline_sha256()
        identity = lambda category, replicate: _dataset_id(  # noqa: E731
            category, replicate, auth_key, pipeline)
    else:
        pipeline = _implementation_hashes()["generation_pipeline_sha256"]
        identity = lambda category, replicate: _development_dataset_id(  # noqa: E731
            category, replicate, auth_key, pipeline)
    cells = [
        {
            "category": category,
            "replicate": replicate,
            "dataset_id": identity(category, replicate),
        }
        for category in CATEGORIES
        for replicate in range(REPLICATES_PER_CATEGORY)
    ]
    if len(cells) != SHIPPED_DATASET_COUNT:
        raise ShardError(
            f"declared grid has {len(cells)} cells; expected {SHIPPED_DATASET_COUNT}"
        )
    return cells


def _shard_cells(mode: str, index: int, count: int, auth_key: bytes) -> list[dict[str, Any]]:
    _validate_coordinates(mode, index, count)
    return [cell for position, cell in enumerate(_all_cells(mode, auth_key))
            if position % count == index]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ShardError(f"JSON artifact is not a single-link regular file: {path}")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        value = json.loads(b"".join(chunks))
    except OSError as exc:
        raise ShardError(f"cannot securely open JSON artifact: {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ShardError(f"JSON artifact is not an object: {path}")
    return value


def _tree_digest(prefix: str, files: dict[str, str]) -> str:
    selected = {
        path[len(prefix):]: digest
        for path, digest in files.items()
        if path.startswith(prefix)
    }
    if not selected:
        raise ShardError(f"cell output contains no files: {prefix}")
    return hashlib.sha256(_canonical_json(selected)).hexdigest()


def _validate_development_output(
    output: Path,
    cells: list[dict[str, Any]],
    auth_key: bytes,
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from validation.model_zoo import DEVELOPMENT_RESULT_FIELDS, REPORT_SCHEMA_VERSION

    files = _tree_file_hashes(output)
    expected_names = {f"{cell['dataset_id']}.json" for cell in cells}
    if set(files) != expected_names:
        raise ShardError(
            "development shard output set mismatch: "
            f"missing={sorted(expected_names - set(files))}, "
            f"unexpected={sorted(set(files) - expected_names)}"
        )
    sealed_cells: list[dict[str, Any]] = []
    for cell in cells:
        name = f"{cell['dataset_id']}.json"
        result = _read_json(output / name)
        if (
            set(result) != DEVELOPMENT_RESULT_FIELDS
            or not verify_json_hmac(
                auth_key,
                result,
                field=DEVELOPMENT_RESULT_HMAC_FIELD,
                domain=DEVELOPMENT_RESULT_HMAC_DOMAIN,
            )
            or result.get("schema_version") != REPORT_SCHEMA_VERSION
            or result.get("purpose") != "development"
            or result.get("key_id") != corpus_key_id(auth_key)
            or result.get("implementation_sha256")
                != provenance["implementation_sha256"]
            or result.get("generation_pipeline_sha256")
                != provenance["generation_pipeline_sha256"]
            or result.get("id") != cell["dataset_id"]
            or result.get("category") != cell["category"]
            or result.get("replicate") != cell["replicate"]
        ):
            raise ShardError(f"invalid authenticated development result: {name}")
        sealed_cells.append({**cell, "output_sha256": files[name]})
    return sealed_cells, files


def _corpus_dataset_paths(output: Path) -> set[tuple[str, str]]:
    _plain_dir(output, "corpus shard output")
    actual: set[tuple[str, str]] = set()
    for category in sorted(output.iterdir()):
        _plain_dir(category, "corpus category")
        for dataset in sorted(category.iterdir()):
            _plain_dir(dataset, "corpus dataset")
            actual.add((category.name, dataset.name))
    return actual


def _validate_corpus_output(
    output: Path,
    cells: list[dict[str, Any]],
    auth_key: bytes,
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import datagen.build_corpus as builder

    files = _tree_file_hashes(output)
    expected = {(cell["category"], cell["dataset_id"]) for cell in cells}
    actual = _corpus_dataset_paths(output)
    if actual != expected:
        raise ShardError(
            "corpus shard output set mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    sealed_cells: list[dict[str, Any]] = []
    for cell in cells:
        dataset = output / cell["category"] / cell["dataset_id"]
        ok, why = builder._provenance_matches(
            str(dataset),
            cell["category"],
            cell["replicate"],
            auth_key,
            provenance["generation_pipeline_sha256"],
        )
        if not ok:
            raise ShardError(
                f"invalid authenticated corpus result {cell['dataset_id']}: {why}"
            )
        prefix = f"{cell['category']}/{cell['dataset_id']}/"
        sealed_cells.append({**cell, "output_sha256": _tree_digest(prefix, files)})
    return sealed_cells, files


def _validated_outputs(
    mode: str,
    output: Path,
    cells: list[dict[str, Any]],
    auth_key: bytes,
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if mode == "develop":
        return _validate_development_output(output, cells, auth_key, provenance)
    return _validate_corpus_output(output, cells, auth_key, provenance)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def preflight(
    *,
    mode: str,
    index: int,
    count: int,
    source_root: Path,
    source_sha: str,
    key_file: Path,
    builder_python: Path,
    shard_root: Path,
) -> None:
    """Validate every external input before the caller creates any output."""
    _validate_coordinates(mode, index, count)
    _validate_exact_source(source_root, source_sha)
    auth_key = load_corpus_key(str(key_file), repository_root=str(source_root))
    _scientific_provenance(builder_python)
    _shard_cells(mode, index, count, auth_key)
    if not shard_root.is_absolute():
        raise ShardError("shard root must be absolute")
    if shard_root.exists() or shard_root.is_symlink():
        raise ShardError(f"shard root must be fresh and absent: {shard_root}")
    _validate_absent_path_ancestors(shard_root)


def write_tasks(
    *, mode: str, index: int, count: int, source_root: Path, key_file: Path,
) -> None:
    auth_key = load_corpus_key(str(key_file), repository_root=str(source_root))
    for cell in _shard_cells(mode, index, count, auth_key):
        print(cell["category"], cell["replicate"], cell["dataset_id"])


def seal(
    *,
    mode: str,
    index: int,
    count: int,
    source_root: Path,
    source_sha: str,
    key_file: Path,
    builder_python: Path,
    shard_root: Path,
) -> Path:
    _validate_coordinates(mode, index, count)
    _validate_exact_source(source_root, source_sha)
    _plain_dir(shard_root, "shard root")
    auth_key = load_corpus_key(str(key_file), repository_root=str(source_root))
    provenance = _scientific_provenance(builder_python)
    cells = _shard_cells(mode, index, count, auth_key)
    sealed_cells, files = _validated_outputs(
        mode, shard_root / "output", cells, auth_key, provenance)
    manifest: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "mode": mode,
        "shard_index": index,
        "shard_count": count,
        "source_commit": source_sha,
        "key_id": corpus_key_id(auth_key),
        "scientific_provenance": provenance,
        "cells": sealed_cells,
        "output_files_sha256": files,
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    manifest[SHARD_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        manifest,
        exclude_field=SHARD_HMAC_FIELD,
        domain=SHARD_HMAC_DOMAIN,
    )
    path = shard_root / SHARD_MANIFEST_NAME
    if path.exists() or path.is_symlink():
        raise ShardError(f"shard manifest already exists: {path}")
    _write_json_atomic(path, manifest)
    return path


def _validate_shard_root_shape(shard_root: Path) -> None:
    _plain_dir(shard_root, "shard root")
    allowed_root_entries = {
        "output", "logs", "work", "tasks", SHARD_MANIFEST_NAME,
    }
    actual = {entry.name for entry in shard_root.iterdir()}
    if actual - allowed_root_entries:
        raise ShardError(
            f"unexpected shard-root entries: {sorted(actual - allowed_root_entries)}"
        )
    if not {"output", SHARD_MANIFEST_NAME}.issubset(actual):
        raise ShardError("shard root is missing output or shard-manifest.json")
    for current, directories, files in os.walk(shard_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            _plain_dir(current_path / name, "shard directory")
        for name in files:
            _plain_file(current_path / name, "shard file")


def _validate_manifest(
    *,
    shard_root: Path,
    manifest: dict[str, Any],
    mode: str,
    count: int,
    source_sha: str,
    auth_key: bytes,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    if set(manifest) != MANIFEST_FIELDS:
        raise ShardError(f"shard manifest fields are invalid: {shard_root}")
    if manifest.get("content_sha256") != _content_sha256(manifest):
        raise ShardError(f"shard manifest content digest mismatch: {shard_root}")
    if not verify_json_hmac(
        auth_key,
        manifest,
        field=SHARD_HMAC_FIELD,
        domain=SHARD_HMAC_DOMAIN,
    ):
        raise ShardError(f"shard manifest HMAC mismatch: {shard_root}")
    index = manifest.get("shard_index")
    _validate_coordinates(mode, index, count)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != SHARD_SCHEMA_VERSION
        or type(manifest.get("mode")) is not str
        or manifest.get("mode") != mode
        or type(manifest.get("shard_count")) is not int
        or manifest.get("shard_count") != count
        or manifest.get("source_commit") != source_sha
        or manifest.get("key_id") != corpus_key_id(auth_key)
        or type(manifest.get("scientific_provenance")) is not dict
        or manifest.get("scientific_provenance") != provenance
        or type(manifest.get("cells")) is not list
        or type(manifest.get("output_files_sha256")) is not dict
    ):
        raise ShardError(f"shard manifest contract mismatch: {shard_root}")
    expected = _shard_cells(mode, index, count, auth_key)
    sealed_cells, files = _validated_outputs(
        mode, shard_root / "output", expected, auth_key, provenance)
    if manifest["cells"] != sealed_cells or manifest["output_files_sha256"] != files:
        raise ShardError(f"shard output bytes disagree with manifest: {shard_root}")
    return expected


def _copy_cell(mode: str, source: Path, stage: Path, cell: dict[str, Any]) -> None:
    if mode == "develop":
        name = f"{cell['dataset_id']}.json"
        shutil.copyfile(source / name, stage / name, follow_symlinks=False)
        return
    relative = Path(cell["category"]) / cell["dataset_id"]
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / relative, destination, symlinks=False)


def _atomic_replace_directory(stage: Path, target: Path) -> None:
    """Atomically install stage; existing targets require Linux renameat2 exchange."""
    if target.is_symlink():
        raise ShardError(f"merge target may not be a symlink: {target}")
    if not target.exists():
        os.replace(stage, target)
        return
    _plain_dir(target, "existing merge target")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ShardError("atomic replacement of an existing target requires Linux renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd, os.fsencode(stage), at_fdcwd, os.fsencode(target), rename_exchange)
    if result != 0:
        error = ctypes.get_errno()
        raise ShardError(f"atomic directory exchange failed: {os.strerror(error)}")
    # The old target is now at the unique stage path.  Removing it cannot affect
    # the already-atomically-installed target.
    shutil.rmtree(stage)


def merge(
    *,
    mode: str,
    count: int,
    source_root: Path,
    source_sha: str,
    key_file: Path,
    builder_python: Path,
    target: Path,
    shard_roots: Iterable[Path],
) -> None:
    _validate_coordinates(mode, 0, count)
    _validate_exact_source(source_root, source_sha)
    auth_key = load_corpus_key(str(key_file), repository_root=str(source_root))
    provenance = _scientific_provenance(builder_python)
    roots = list(shard_roots)
    if len(roots) != count:
        raise ShardError(f"merge requires exactly {count} shard roots; received {len(roots)}")
    if len({str(path.resolve(strict=False)) for path in roots}) != len(roots):
        raise ShardError("merge received duplicate shard roots")
    by_index: dict[int, tuple[Path, list[dict[str, Any]]]] = {}
    observed_cells: set[tuple[str, int, str]] = set()
    for shard_root in roots:
        _validate_shard_root_shape(shard_root)
        manifest = _read_json(shard_root / SHARD_MANIFEST_NAME)
        cells = _validate_manifest(
            shard_root=shard_root,
            manifest=manifest,
            mode=mode,
            count=count,
            source_sha=source_sha,
            auth_key=auth_key,
            provenance=provenance,
        )
        index = manifest["shard_index"]
        if index in by_index:
            raise ShardError(f"overlapping shard index: {index}")
        for cell in cells:
            identity = (cell["category"], cell["replicate"], cell["dataset_id"])
            if identity in observed_cells:
                raise ShardError(f"overlapping shard cell: {identity}")
            observed_cells.add(identity)
        by_index[index] = (shard_root, cells)
    if set(by_index) != set(range(count)):
        raise ShardError(
            f"missing/unexpected shard indices: expected={list(range(count))} "
            f"actual={sorted(by_index)}"
        )
    all_cells = _all_cells(mode, auth_key)
    expected_cells = {
        (cell["category"], cell["replicate"], cell["dataset_id"])
        for cell in all_cells
    }
    if observed_cells != expected_cells:
        raise ShardError(
            "merged shard grid mismatch: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"unexpected={sorted(observed_cells - expected_cells)}"
        )
    expected_target = (
        source_root / "corpus"
        if mode == "corpus"
        else source_root.parent / "devwork" / "results"
    )
    if not target.is_absolute() or target != expected_target:
        raise ShardError(
            f"{mode} shards may only replace the release target {expected_target}; "
            f"received {target}"
        )
    _plain_dir(target.parent, "merge-target parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
    installed = False
    try:
        for index in range(count):
            shard_root, cells = by_index[index]
            for cell in cells:
                _copy_cell(mode, shard_root / "output", stage, cell)
        _validated_outputs(mode, stage, all_cells, auth_key, provenance)
        _atomic_replace_directory(stage, target)
        installed = True
    finally:
        if not installed and stage.exists():
            shutil.rmtree(stage)


def run_corpus_cell(
    *,
    output_root: Path,
    key_file: Path,
    action: str,
    category: str,
    replicate: int,
) -> None:
    """Run one corpus builder action against an explicit shard output root."""
    if action not in {"ONE", "CHECK"}:
        raise ShardError(f"invalid corpus-cell action: {action!r}")
    if category not in CATEGORIES or not 0 <= replicate < REPLICATES_PER_CATEGORY:
        raise ShardError(
            f"invalid corpus-cell coordinates: category={category!r} "
            f"replicate={replicate!r}"
        )
    if not output_root.is_absolute():
        raise ShardError("corpus-cell output root must be absolute")
    _plain_dir(output_root, "corpus-cell output root")

    import datagen.build_corpus as builder

    builder.CORPUS = output_root
    original_argv = sys.argv
    try:
        sys.argv = [
            builder.__file__,
            "--key-file",
            str(key_file),
            action,
            category,
            str(replicate),
        ]
        builder._main()
    finally:
        sys.argv = original_argv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def shared(command: argparse.ArgumentParser, *, coordinates: bool) -> None:
        command.add_argument("--mode", choices=sorted(MODES), required=True)
        if coordinates:
            command.add_argument("--shard-index", type=int, required=True)
        command.add_argument("--shard-count", type=int, required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--key-file", type=Path, required=True)

    preflight_parser = subparsers.add_parser("preflight")
    shared(preflight_parser, coordinates=True)
    preflight_parser.add_argument("--source-sha", required=True)
    preflight_parser.add_argument("--builder-python", type=Path, required=True)
    preflight_parser.add_argument("--shard-root", type=Path, required=True)

    tasks_parser = subparsers.add_parser("tasks")
    shared(tasks_parser, coordinates=True)

    seal_parser = subparsers.add_parser("seal")
    shared(seal_parser, coordinates=True)
    seal_parser.add_argument("--source-sha", required=True)
    seal_parser.add_argument("--builder-python", type=Path, required=True)
    seal_parser.add_argument("--shard-root", type=Path, required=True)

    merge_parser = subparsers.add_parser("merge")
    shared(merge_parser, coordinates=False)
    merge_parser.add_argument("--source-sha", required=True)
    merge_parser.add_argument("--builder-python", type=Path, required=True)
    merge_parser.add_argument("--target", type=Path, required=True)
    merge_parser.add_argument("shard_roots", type=Path, nargs="+")

    corpus_cell_parser = subparsers.add_parser("corpus-cell")
    corpus_cell_parser.add_argument("--output-root", type=Path, required=True)
    corpus_cell_parser.add_argument("--key-file", type=Path, required=True)
    corpus_cell_parser.add_argument("--action", choices=("ONE", "CHECK"), required=True)
    corpus_cell_parser.add_argument("--category", choices=tuple(CATEGORIES), required=True)
    corpus_cell_parser.add_argument("--replicate", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "corpus-cell":
        run_corpus_cell(
            output_root=arguments.output_root,
            key_file=arguments.key_file,
            action=arguments.action,
            category=arguments.category,
            replicate=arguments.replicate,
        )
        return 0
    common = {
        "mode": arguments.mode,
        "count": arguments.shard_count,
        "source_root": arguments.source_root,
        "key_file": arguments.key_file,
    }
    if arguments.command == "preflight":
        preflight(
            **common,
            index=arguments.shard_index,
            source_sha=arguments.source_sha,
            builder_python=arguments.builder_python,
            shard_root=arguments.shard_root,
        )
    elif arguments.command == "tasks":
        write_tasks(**common, index=arguments.shard_index)
    elif arguments.command == "seal":
        path = seal(
            **common,
            index=arguments.shard_index,
            source_sha=arguments.source_sha,
            builder_python=arguments.builder_python,
            shard_root=arguments.shard_root,
        )
        print(path)
    else:
        merge(
            **common,
            source_sha=arguments.source_sha,
            builder_python=arguments.builder_python,
            target=arguments.target,
            shard_roots=arguments.shard_roots,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ShardError, ValueError) as exc:
        raise SystemExit(f"[rebuild-shards] ERROR: {exc}") from exc
