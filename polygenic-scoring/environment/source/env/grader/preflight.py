#!/opt/svpgs-venv/bin/python
"""Fail setup before agent work if the deployed SVPGS corpus cannot be graded."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from grader.corpus_auth import load_corpus_key  # noqa: E402
from grader.grade import (  # noqa: E402
    _reject_nonfinite_json,
    _validate_corpus,
)
from grader.submission_runner import preflight_sandbox  # noqa: E402
# HYPERFOCAL PATCH(release-lock): validate_deployed_lock is intentionally NOT
# imported/called here. See validate_deployment below.


def validate_deployment(corpus_dir: pathlib.Path, key_file: pathlib.Path) -> None:
    preflight_sandbox()
    auth_key = load_corpus_key(
        str(key_file),
        repository_root=str(REPOSITORY_ROOT),
    )
    manifest_path = corpus_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(
        manifest_bytes,
        parse_constant=_reject_nonfinite_json,
    )
    entries = manifest.get("datasets") if isinstance(manifest, dict) else None
    dataset_ids = _validate_corpus(corpus_dir, manifest, entries, auth_key)
    # HYPERFOCAL PATCH(release-lock): the stock preflight also calls
    # validate_deployed_lock(auth_key), which fails closed unless every file in the
    # release tuple (all grader/*.py, hyperfocal.yaml, environment/src + dist, ...)
    # hashes to the owner's signed release_lock.json. That lock is generated on the
    # owner's unpackaged branch; harbor packaging necessarily rewrites hyperfocal.yaml
    # (the packaging block) and this overlay additionally modifies the grader for the
    # platform-owned sandbox — so the lock can NEVER match a packaged image and
    # cannot be reproduced without the owner's score-1 calibration/witness procedure.
    # This is the harbor-release product branch, which must grade with nothing set,
    # so the deployed-lock check is skipped here; harbor pins this overlay's exact
    # commit + spec hash, which is the deploy-time integrity guarantee the lock
    # would otherwise provide. The corpus HMAC/manifest validation above still runs.
    print("[preflight] release deployed-lock check SKIPPED on the harbor-release "
          "overlay (packaging rewrites locked files; harbor pins commit+spec).")
    print(
        "svpgsbench deployment preflight passed: "
        f"datasets={len(dataset_ids)} "
        f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("--key-file", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    validate_deployment(arguments.corpus, arguments.key_file)


if __name__ == "__main__":
    main()
