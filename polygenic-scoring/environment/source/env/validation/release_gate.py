"""Atomic, fail-closed svpgsbench release validation and release lock.

This is the single command a publisher runs after all compute artifacts exist. It
validates the authenticated corpus, regenerates both model-zoo reports from their
signed result payloads, validates the mastery-calibration overlay and authenticated
score-1 solvability witness, rejects stale prompt/runtime artifacts, and HMAC-locks
the exact source/artifact tuple. Publishing anything other than a clean, passing lock
is a release error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grader.contract import CORPUS_SCHEMA_VERSION  # noqa: E402
from grader.corpus_auth import (  # noqa: E402
    json_hmac_sha256,
    load_corpus_key,
    verify_json_hmac,
)
from grader.grade import _reject_nonfinite_json, _validate_corpus  # noqa: E402
from tasks.prompt_spec import prompt_sha256, stale_surfaces  # noqa: E402
from validation.mastery_calibration import (  # noqa: E402
    evidence_paths_from_artifact as calibration_evidence_paths,
    uncalibrated_artifact,
    validate_artifact,
)
from validation.score1_witness import (  # noqa: E402
    WITNESS_PATH,
    evidence_path_from_artifact,
    validate_artifact as validate_score1_witness,
    wrapper_sha256 as score1_wrapper_sha256,
)

LOCK_SCHEMA_VERSION = 4
LOCK_PATH = ROOT / "validation" / "release_lock.json"
CALIBRATION_PATH = ROOT / "validation" / "mastery_calibration.json"
LOCK_HMAC_FIELD = "release_lock_hmac_sha256"
LOCK_HMAC_DOMAIN = b"svpgsbench|schema-v8|release-lock\x00"


class ReleaseGateError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data, parse_constant=_reject_nonfinite_json)
    if not isinstance(value, dict):
        raise ReleaseGateError(f"{path.relative_to(ROOT)} must contain an object")
    return value, _sha256_bytes(data)


def _wrapper_sha256() -> str:
    return score1_wrapper_sha256()


def _gold_candidate_lock_bytes(witness: dict[str, Any]) -> bytes:
    """Recover the authenticated candidate embedded in retained gold evidence."""
    bundle_path = evidence_path_from_artifact(witness)
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            matches = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename == "candidate_release_lock.json"
            ]
            if len(matches) != 1:
                raise ReleaseGateError(
                    "gold evidence does not contain exactly one candidate release lock")
            return archive.read(matches[0])
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseGateError("gold evidence candidate lock is unreadable") from exc


def _assert_dist_matches_source() -> None:
    """Compile TypeScript into an isolated directory and compare shipped outputs.

    Locking source and dist hashes independently does not prove that the JavaScript
    the platform executes was generated from the reviewed TypeScript. This check is
    publisher-only: packaged/deployed validation remains compiler-free.
    """
    environment = ROOT / "environment"
    compiler = environment / "node_modules" / ".bin" / "tsc"
    if not compiler.is_file():
        raise ReleaseGateError(
            "publisher QA needs environment/node_modules/.bin/tsc to verify dist parity")
    with tempfile.TemporaryDirectory(prefix="svpgsbench-dist-") as temporary:
        completed = subprocess.run(
            [
                str(compiler), "--project", str(environment / "tsconfig.json"),
                "--outDir", temporary,
            ],
            cwd=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ReleaseGateError(f"TypeScript source does not compile: {detail}")
        generated_root = Path(temporary)
        for relative in ("aggregation.js", "index.js", "aggregation.d.ts", "index.d.ts"):
            generated = generated_root / relative
            shipped = environment / "dist" / relative
            if not generated.is_file() or not shipped.is_file():
                raise ReleaseGateError(f"wrapper dist parity is missing {relative}")
            if generated.read_bytes() != shipped.read_bytes():
                raise ReleaseGateError(
                    f"environment/dist/{relative} is stale; run npm --prefix environment run build")


def _base_locked_paths() -> list[Path]:
    paths: set[Path] = set()
    for directory in ("datagen", "grader", "reference", "tasks", "validation"):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" not in path.parts:
                paths.add(path)
    for relative in (
        ".gitmodules",
        "README.md",
        "SECURITY.md",
        "PACKAGING.md",
        "hyperfocal.yaml",
        "RELEASE.md",
        "gold/fit",
        "gold/predict",
        "gold/pgs_core.py",
        "tools/rebuild_shards.py",
        "tools/run_rebuild_shard.sh",
        "environment/README.md",
        "environment/package.json",
        "environment/package-lock.json",
        "environment/problems.yaml",
        "environment/src/aggregation.ts",
        "environment/src/index.ts",
        "environment/dist/aggregation.d.ts",
        "environment/dist/aggregation.js",
        "environment/dist/index.d.ts",
        "environment/dist/index.js",
        "environment/test/aggregation-contract.test.mjs",
        "environment/test/private-analysis.test.mjs",
        "environment/scripts/install-toolchains.sh",
        "tasks/from_scratch_svpgs/instruction.md",
        "tasks/from_scratch_svpgs/task.toml",
        "corpus/manifest.json",
        "validation/model_zoo_development.json",
        "validation/shipping_audit.json",
        "validation/mastery_calibration.json",
    ):
        paths.add(ROOT / relative)
    paths.discard(LOCK_PATH)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ReleaseGateError(
            "release tuple is missing files: "
            + ", ".join(str(path.relative_to(ROOT)) for path in sorted(missing))
        )
    return sorted(paths)


def _locked_paths(release_status: str, witness: dict[str, Any] | None = None) -> list[Path]:
    """Return the exact files for an explicit candidate or production state.

    Candidate images intentionally omit score-1 evidence: they fix the authenticated
    executable tuple before the gold solvability proof and live-harness QA. Production
    adds exactly one witness artifact and its content-addressed retained bundle.
    """
    paths = set(_base_locked_paths())
    calibration, _ = _read_json(CALIBRATION_PATH)
    paths.update(calibration_evidence_paths(calibration))
    if release_status == "production":
        if witness is None:
            raise ReleaseGateError("production release needs a score-1 witness")
        paths.add(WITNESS_PATH)
        paths.add(evidence_path_from_artifact(witness))
    elif release_status != "candidate":
        raise ReleaseGateError(f"unknown release status {release_status!r}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ReleaseGateError(
            "release tuple is missing files: "
            + ", ".join(str(path.relative_to(ROOT)) for path in sorted(missing))
        )
    return sorted(paths)


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def _executable_tuple_sha256() -> str:
    """Digest the noncircular candidate tuple (everything except lock/witness)."""
    files = _file_hashes(_base_locked_paths())
    return _sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    )


def _score1_compatibility_sha256() -> str:
    """Digest every locked byte except the narrowly defined mastery overlay.

    A score-1 witness may survive freezing/reporting mastery because mastery does
    not affect continuous reward.  It may not survive any other source, runtime,
    prompt, corpus, or evidence change.  Normalize only the two top-level threshold
    assignments and omit only the authenticated calibration artifact; changing any
    surrounding contract code (even on the same lines) fails closed.
    """
    files: dict[str, str] = {}
    calibration_relative = "validation/mastery_calibration.json"
    contract_relative = "grader/contract.py"
    for path in _base_locked_paths():
        relative = str(path.relative_to(ROOT))
        if relative == calibration_relative:
            continue
        data = path.read_bytes()
        if relative == contract_relative:
            try:
                source = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ReleaseGateError("grader/contract.py is not UTF-8") from exc
            replacements = (
                (r"(?m)^MASTERY_THRESHOLD = (?:None|[-+0-9.eE]+)$",
                 "MASTERY_THRESHOLD = <mastery-overlay>"),
                (r"(?m)^MASTERY_TAIL_THRESHOLD = (?:None|[-+0-9.eE]+)$",
                 "MASTERY_TAIL_THRESHOLD = <mastery-overlay>"),
            )
            for pattern, replacement in replacements:
                source, count = re.subn(pattern, replacement, source)
                if count != 1:
                    raise ReleaseGateError(
                        "grader mastery threshold assignment does not match the "
                        "score-1 overlay contract")
            data = source.encode("utf-8")
        files[relative] = _sha256_bytes(data)
    return _sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    )


def _scientific_tuple_sha256(
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
) -> str:
    """Stable science identity, deliberately excluding the mastery overlay.

    scoring_contract_sha256 already excludes mastery thresholds. The wrapper bytes
    implement reward integrity/aggregation and consume thresholds supplied by the
    grader; changing only calibration JSON and threshold constants leaves these bytes
    unchanged. This is the stable identity carried in rollout output; the stricter
    score1_compatibility digest separately covers every locked source/runtime byte
    and is what limits witness reuse to the mastery overlay.
    """
    return _sha256_bytes(json.dumps({
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_sha256,
    }, sort_keys=True, separators=(",", ":")).encode())


def _artifact_state(
    auth_key: bytes,
    *,
    require_witness: bool,
    expected_executable_tuple_sha256: str | None = None,
) -> dict[str, Any]:
    _assert_dist_matches_source()
    # Heavy reference/model-zoo imports stay publisher-side. Deployment preflight
    # validates the signed lock and file tuple without requiring the sibling
    # SV-PGS source tree merely to boot the grader.
    from validation.model_zoo import (
        ShippingGateError,
        _canonical_json_sha256,
        _implementation_hashes,
        _validated_development_report,
        rebuild_final_audit_report_from_evidence,
    )
    stale = stale_surfaces()
    if stale:
        raise ReleaseGateError(
            "prompt surfaces are stale: "
            + ", ".join(str(path.relative_to(ROOT)) for path in stale)
        )

    manifest, manifest_sha256 = _read_json(ROOT / "corpus" / "manifest.json")
    entries = manifest.get("datasets")
    _validate_corpus(ROOT / "corpus", manifest, entries, auth_key)

    development, development_sha256 = _read_json(
        ROOT / "validation" / "model_zoo_development.json"
    )
    frozen_development = _validated_development_report(development, auth_key)
    if manifest["meta"]["development_report_sha256"] != development_sha256:
        raise ReleaseGateError("manifest does not freeze the current development report")

    audit, audit_sha256 = _read_json(ROOT / "validation" / "shipping_audit.json")
    if audit.get("passed") is not True or audit.get("report_kind") != "final_shipping_audit":
        raise ReleaseGateError("shipping audit is not a passing final audit")
    try:
        regenerated_audit = rebuild_final_audit_report_from_evidence(
            ROOT / "corpus",
            manifest,
            manifest_sha256,
            audit.get("dataset_results", []),
            frozen_development,
            development_sha256,
            auth_key=auth_key,
        )
    except ShippingGateError as exc:
        raise ReleaseGateError(f"shipping audit does not reproduce: {exc}") from exc
    if _canonical_json_sha256(audit) != _canonical_json_sha256(regenerated_audit):
        raise ReleaseGateError("shipping audit is stale or differs from regeneration")

    implementation = _implementation_hashes()
    wrapper_sha256 = _wrapper_sha256()
    executable_tuple_sha256 = _executable_tuple_sha256()
    if (
        expected_executable_tuple_sha256 is not None
        and executable_tuple_sha256 != expected_executable_tuple_sha256
    ):
        raise ReleaseGateError(
            "production executable tuple differs from the witnessed candidate")
    calibration, calibration_sha256 = _read_json(CALIBRATION_PATH)
    validate_artifact(
        calibration,
        auth_key,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=implementation["scoring_contract_sha256"],
        wrapper_sha256=wrapper_sha256,
    )
    state = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "prompt_sha256": prompt_sha256(),
        "manifest_sha256": manifest_sha256,
        "development_report_sha256": development_sha256,
        "shipping_audit_sha256": audit_sha256,
        "mastery_calibration_sha256": calibration_sha256,
        "scoring_contract_sha256": implementation["scoring_contract_sha256"],
        "wrapper_sha256": wrapper_sha256,
        "scientific_tuple_sha256": _scientific_tuple_sha256(
            manifest_sha256,
            implementation["scoring_contract_sha256"],
            wrapper_sha256,
        ),
        "executable_tuple_sha256": executable_tuple_sha256,
        "score1_compatibility_sha256": _score1_compatibility_sha256(),
        "dataset_count": len(entries),
    }
    if require_witness:
        witness, witness_sha256 = _read_json(WITNESS_PATH)
        validate_score1_witness(
            witness,
            auth_key,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            scoring_contract_sha256=implementation["scoring_contract_sha256"],
            wrapper_contract_sha256=wrapper_sha256,
            scientific_tuple_sha256=state["scientific_tuple_sha256"],
        )
        state.update({
            "score1_witness_sha256": witness_sha256,
            "score1_witness_id": witness["witness_id"],
        })
    return state


def prepare_uncalibrated(auth_key: bytes) -> None:
    from validation.model_zoo import _implementation_hashes
    manifest, manifest_sha256 = _read_json(ROOT / "corpus" / "manifest.json")
    _validate_corpus(ROOT / "corpus", manifest, manifest.get("datasets"), auth_key)
    implementation = _implementation_hashes()
    artifact = uncalibrated_artifact(
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=implementation["scoring_contract_sha256"],
        wrapper_sha256=_wrapper_sha256(),
        auth_key=auth_key,
    )
    CALIBRATION_PATH.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")


def _git_state() -> tuple[str, str]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return branch, status


def _signed_lock(lock: dict[str, Any], auth_key: bytes) -> dict[str, Any]:
    signed = dict(lock)
    signed[LOCK_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        signed,
        exclude_field=LOCK_HMAC_FIELD,
        domain=LOCK_HMAC_DOMAIN,
    )
    return signed


def write_candidate_lock(auth_key: bytes) -> None:
    """Authenticate the exact tuple used by solvability and live-harness QA."""
    branch, status = _git_state()
    if branch != "main" or status:
        raise ReleaseGateError("write-candidate-lock requires a clean main worktree")
    state = _artifact_state(auth_key, require_witness=False)
    files = _file_hashes(_locked_paths("candidate"))
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench",
        "release_status": "candidate",
        **state,
        "files": files,
    }
    lock = _signed_lock(lock, auth_key)
    LOCK_PATH.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")


def write_production_lock(auth_key: bytes) -> None:
    """Transition an authenticated candidate to production without tuple drift."""
    branch, status = _git_state()
    if branch != "main" or status:
        raise ReleaseGateError("write-production-lock requires a clean main worktree")
    candidate_bytes = LOCK_PATH.read_bytes()
    candidate = json.loads(candidate_bytes, parse_constant=_reject_nonfinite_json)
    if not isinstance(candidate, dict) or candidate.get("release_status") != "candidate":
        raise ReleaseGateError("production transition requires the candidate release lock")
    validate_deployed_lock(auth_key)
    candidate_tuple = candidate["executable_tuple_sha256"]
    state = _artifact_state(
        auth_key,
        require_witness=True,
        expected_executable_tuple_sha256=candidate_tuple,
    )
    witness, _ = _read_json(WITNESS_PATH)
    if witness.get("evidence_kind") == "hfdev_full_bundle":
        # Only a full hfdev package claims execution. Prove its executed commit
        # contained the exact authenticated candidate lock named by the package.
        shown = subprocess.run(
            ["git", "show",
             f"{witness['executed_environment_commit']}:validation/release_lock.json"],
            cwd=ROOT, capture_output=True)
        if (shown.returncode != 0
                or _sha256_bytes(shown.stdout) != witness["candidate_lock_sha256"]):
            raise ReleaseGateError(
                "executed commit does not contain the witness candidate release lock")
        try:
            witnessed_candidate = json.loads(
                shown.stdout, parse_constant=_reject_nonfinite_json)
        except json.JSONDecodeError as exc:
            raise ReleaseGateError("executed candidate lock is invalid JSON") from exc
        expected_tuple = witness.get("witnessed_executable_tuple_sha256")
    elif witness.get("evidence_kind") == "gold_execution_bundle":
        # Trusted witness creation and validation execute the locked gold submission
        # over every dataset in the mandatory sandbox. Read its retained candidate
        # lock rather than assuming it is the current candidate: a later mastery-only
        # candidate has a distinct executable tuple while the score-1 compatibility
        # digest proves its scientific/runtime bytes unchanged.
        witnessed_candidate_bytes = _gold_candidate_lock_bytes(witness)
        if witness.get("candidate_lock_sha256") != _sha256_bytes(
            witnessed_candidate_bytes
        ):
            raise ReleaseGateError("gold witness candidate lock digest does not reproduce")
        try:
            witnessed_candidate = json.loads(
                witnessed_candidate_bytes, parse_constant=_reject_nonfinite_json)
        except json.JSONDecodeError as exc:
            raise ReleaseGateError("gold witness candidate lock is invalid JSON") from exc
        expected_tuple = witness.get("gold_executable_tuple_sha256")
    else:
        raise ReleaseGateError("unknown score-1 witness kind")
    if (
        not isinstance(witnessed_candidate, dict)
        or witnessed_candidate.get("release_status") != "candidate"
        or witnessed_candidate.get("scientific_tuple_sha256")
            != witness["scientific_tuple_sha256"]
        or witnessed_candidate.get("executable_tuple_sha256") != expected_tuple
        or witnessed_candidate.get("score1_compatibility_sha256")
            != candidate.get("score1_compatibility_sha256")
        or not verify_json_hmac(
            auth_key, witnessed_candidate,
            field=LOCK_HMAC_FIELD, domain=LOCK_HMAC_DOMAIN,
        )
    ):
        raise ReleaseGateError("witness candidate lock provenance does not reproduce")
    files = _file_hashes(_locked_paths("production", witness))
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench",
        "release_status": "production",
        "candidate_lock_sha256": _sha256_bytes(candidate_bytes),
        "candidate_executable_tuple_sha256": candidate_tuple,
        **state,
        "files": files,
    }
    lock = _signed_lock(lock, auth_key)
    LOCK_PATH.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")


def validate_deployed_lock(auth_key: bytes) -> None:
    """Validate the lock in a packaged image, where no .git directory exists."""
    lock, _ = _read_json(LOCK_PATH)
    common = {
        "schema_version", "environment_id", "corpus_schema_version",
        "prompt_sha256", "manifest_sha256", "development_report_sha256",
        "shipping_audit_sha256", "mastery_calibration_sha256",
        "scoring_contract_sha256", "wrapper_sha256", "scientific_tuple_sha256",
        "executable_tuple_sha256", "score1_compatibility_sha256",
        "dataset_count", "release_status", "files",
        LOCK_HMAC_FIELD,
    }
    candidate_required = common
    production_required = common | {
        "candidate_lock_sha256", "candidate_executable_tuple_sha256",
        "score1_witness_sha256", "score1_witness_id",
    }
    status = lock.get("release_status") if isinstance(lock, dict) else None
    expected = candidate_required if status == "candidate" else production_required
    if (
        not isinstance(lock, dict)
        or set(lock) != expected
        or lock["schema_version"] != LOCK_SCHEMA_VERSION
        or status not in {"candidate", "production"}
    ):
        raise ReleaseGateError("release lock fields do not match schema")
    if not verify_json_hmac(
        auth_key,
        lock,
        field=LOCK_HMAC_FIELD,
        domain=LOCK_HMAC_DOMAIN,
    ):
        raise ReleaseGateError("release lock HMAC mismatch")
    witness = None
    if status == "production":
        witness, _ = _read_json(WITNESS_PATH)
    current_files = _file_hashes(_locked_paths(status, witness))
    if lock["files"] != current_files:
        raise ReleaseGateError("release source/artifact tuple differs from its lock")
    executable_tuple_sha256 = _executable_tuple_sha256()
    scientific_tuple_sha256 = _scientific_tuple_sha256(
        lock["manifest_sha256"], lock["scoring_contract_sha256"],
        lock["wrapper_sha256"])
    if (
        executable_tuple_sha256 != lock["executable_tuple_sha256"]
        or scientific_tuple_sha256 != lock["scientific_tuple_sha256"]
        or _score1_compatibility_sha256() != lock["score1_compatibility_sha256"]
    ):
        raise ReleaseGateError("deployed executable tuple differs from its release lock")
    manifest, manifest_sha256 = _read_json(ROOT / "corpus" / "manifest.json")
    calibration, calibration_sha256 = _read_json(CALIBRATION_PATH)
    if calibration_sha256 != lock["mastery_calibration_sha256"]:
        raise ReleaseGateError("deployed mastery calibration differs from its release lock")
    validate_artifact(
        calibration,
        auth_key,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=lock["scoring_contract_sha256"],
        wrapper_sha256=lock["wrapper_sha256"],
    )
    if status == "candidate":
        # Explicitly deployable as the immutable environment for live-harness QA.
        # There is intentionally no score-1 witness
        # in this state; production is a separate authenticated transition below.
        return
    if (
        lock["candidate_executable_tuple_sha256"] != executable_tuple_sha256
        or not isinstance(lock["candidate_lock_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", lock["candidate_lock_sha256"]) is None
    ):
        raise ReleaseGateError("production lock does not preserve its candidate tuple")
    # Verify the witness itself even in the packaged image.  The release-lock HMAC
    # binds its bytes, while this independent domain-separated HMAC and semantic
    # validation prove those bytes are either a complete hfdev score-1 execution or
    # an independently rescored public gold proof -- not an arbitrary reward claim.
    witness, witness_sha256 = _read_json(WITNESS_PATH)
    if (
        manifest_sha256 != lock["manifest_sha256"]
        or _wrapper_sha256() != lock["wrapper_sha256"]
        or witness_sha256 != lock["score1_witness_sha256"]
        or witness.get("witness_id") != lock["score1_witness_id"]
    ):
        raise ReleaseGateError("deployed score-1 witness differs from its release lock")
    validate_score1_witness(
        witness,
        auth_key,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=lock["scoring_contract_sha256"],
        wrapper_contract_sha256=lock["wrapper_sha256"],
        scientific_tuple_sha256=lock["scientific_tuple_sha256"],
    )


def check_lock(auth_key: bytes) -> None:
    branch, status = _git_state()
    if branch != "main" or status:
        raise ReleaseGateError("release check requires a clean main worktree")
    validate_deployed_lock(auth_key)
    lock, _ = _read_json(LOCK_PATH)
    state = _artifact_state(
        auth_key,
        require_witness=lock["release_status"] == "production",
        expected_executable_tuple_sha256=lock["executable_tuple_sha256"],
    )
    for field, value in state.items():
        if lock.get(field) != value:
            raise ReleaseGateError(f"release lock {field} is stale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "prepare-uncalibrated", "write-candidate-lock",
            "write-production-lock", "check",
        ),
    )
    parser.add_argument("--key-file", required=True)
    args = parser.parse_args()
    auth_key = load_corpus_key(args.key_file, repository_root=str(ROOT))
    if args.mode == "prepare-uncalibrated":
        prepare_uncalibrated(auth_key)
    elif args.mode == "write-candidate-lock":
        write_candidate_lock(auth_key)
    elif args.mode == "write-production-lock":
        write_production_lock(auth_key)
    else:
        check_lock(auth_key)
    print(f"release gate {args.mode}: PASS")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"release gate: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
