"""Authenticated proof that native score 1 is mathematically achievable.

The proof is derived from either a complete hfdev rollout package or a public-only
gold execution request.  The latter contains only the exact inspectable submission,
candidate lock, and request metadata: publisher-supplied predictions are forbidden.
Trusted witness creation and every release validation execute those locked bytes over
all 45 datasets through the mandatory submission sandbox, retain exit/timing/model/
output hashes, and score the exact parsed pred.csv bytes the process produced.

The witness binds an executable-tuple digest from an authenticated candidate release
lock.  It deliberately does not claim that the executed commit is the later commit
which adds this evidence and the production lock; that would be circular.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grader.contract import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    FIT_TIMEOUT_S,
    PREDICT_TIMEOUT_S,
    SHIPPED_CATEGORIES,
    SHIPPED_DATASET_COUNT,
)
from grader.corpus_auth import (  # noqa: E402
    json_hmac_sha256,
    load_corpus_key,
    verify_json_hmac,
)
from grader.skill import (  # noqa: E402
    SKILL_HI,
    SKILL_LO,
    accuracy_skill,
    apply_calibration_factor,
    calibration_factor,
    category_aggregation,
)
from grader.metrics import compute_metrics  # noqa: E402
from grader.truth import read_binary_truth_csv  # noqa: E402
from grader.submission_runner import (  # noqa: E402
    build_submission,
    run_on_dataset,
    _rmtree_ro,
)

WITNESS_SCHEMA_VERSION = 4
WITNESS_PATH = ROOT / "validation" / "score1_witness.json"
EVIDENCE_DIR = ROOT / "validation" / "score1_evidence"
WITNESS_HMAC_FIELD = "score1_witness_hmac_sha256"
WITNESS_HMAC_DOMAIN = b"svpgsbench|schema-v8|score1-witness-v4\x00"
ENVIRONMENT_ID = "sc-svpgsbench"
TARGET_INSTANCE_TYPE = "t3.xlarge"
# Numerical reproduction tolerance, not a substantive score concession. A witness
# at 0.99 is not a score-1 proof; every cell, headline, and tail must reproduce one
# to floating-point accuracy.
SCORE_ONE_TOLERANCE = 1e-6

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROLLOUT_BUNDLE = re.compile(r"-(rollout-[0-9]+)(?:-retained-only)?\.zip$")
_CORE_BUNDLE_MEMBERS = {
    "run.json",
    "logs/tests.jsonl",
    "logs/agent.jsonl",
    "logs/setup.jsonl",
    "transcript.txt",
    "task_prompt.md",
    "solution.diff.json",
    "summary.txt",
}
_EVENT_FIELDS = {"timestamp", "type", "category", "message", "data"}
_RESULT_FIELDS = {
    "testId", "name", "description", "status", "duration", "score", "weight",
    "output",
}
_CATEGORY_OUTPUT_FIELDS = {
    "native_score", "platform_score", "contract_ok", "datasets",
}
_DATASET_OUTPUT_FIELDS = {
    "dataset", "status", "reward", "skill", "accuracy", "performance",
    "fit_seconds", "predict_seconds",
}
_BENCHMARK_OUTPUT_FIELDS = {
    "native_score", "platform_score", "integrity_ok", "contract_ok", "mastery",
    "mastery_threshold", "mastery_tail_threshold", "mastery_tail_value",
    "per_category",
    "release_status", "scientific_tuple_sha256", "executable_tuple_sha256",
    "manifest_sha256", "scoring_contract_sha256", "wrapper_sha256",
    "candidate_lock_sha256",
}
_GOLD_EXECUTION_METADATA_FIELDS = {
    "schema_version", "environment_id", "gold_id", "candidate_lock_sha256",
}


class ScoreOneWitnessError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ScoreOneWitnessError(f"non-finite JSON constant {value!r}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def wrapper_sha256() -> str:
    members = {}
    for relative in (
        "environment/src/aggregation.ts",
        "environment/src/index.ts",
        "environment/dist/aggregation.js",
        "environment/dist/index.js",
    ):
        members[relative] = _sha256((ROOT / relative).read_bytes())
    return _canonical_sha256(members)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _platform_score(native: float) -> float:
    return (native - SKILL_LO) / (1.0 - SKILL_LO)


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ScoreOneWitnessError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ScoreOneWitnessError(f"{label} is not a UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ScoreOneWitnessError(f"{label} must include a timezone")
    return value


def _manifest_index(manifest: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ScoreOneWitnessError("manifest must be an object")
    entries = manifest.get("datasets")
    if (
        manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
        or not isinstance(entries, list)
        or len(entries) != SHIPPED_DATASET_COUNT
    ):
        raise ScoreOneWitnessError("witness needs the exact shipping manifest")
    indexed = {
        entry.get("id"): entry for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if len(indexed) != SHIPPED_DATASET_COUNT:
        raise ScoreOneWitnessError("shipping manifest has duplicate or malformed dataset IDs")
    categories = [entry.get("category") for entry in entries if isinstance(entry, dict)]
    if set(categories) != set(SHIPPED_CATEGORIES):
        raise ScoreOneWitnessError("shipping manifest categories do not match the contract")
    return indexed


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _read_bundle(bundle_path: Path) -> tuple[bytes, dict[str, bytes], list[dict[str, Any]]]:
    bundle_bytes = Path(bundle_path).read_bytes()
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            infos = archive.infolist()
            file_infos = [info for info in infos if not info.is_dir()]
            names = [info.filename for info in file_infos]
            if len(names) != len(set(names)):
                raise ScoreOneWitnessError("retained bundle has duplicate file members")
            if any(not _safe_member(name) for name in names):
                raise ScoreOneWitnessError("retained bundle has an unsafe member path")
            if "PACKAGE_INCOMPLETE.txt" in names:
                raise ScoreOneWitnessError("score-1 proof requires a full, not retained-only, bundle")
            missing = _CORE_BUNDLE_MEMBERS - set(names)
            if missing:
                raise ScoreOneWitnessError(
                    f"full retained bundle is missing {sorted(missing)}")
            members = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ScoreOneWitnessError(f"invalid retained bundle: {exc}") from exc
    for required in _CORE_BUNDLE_MEMBERS:
        if not members[required]:
            raise ScoreOneWitnessError(f"retained bundle member {required} is empty")
    solution_files = {
        name: data for name, data in members.items()
        if name.startswith("solution/") and name != "solution/"
    }
    if not solution_files or any(not data for data in solution_files.values()):
        raise ScoreOneWitnessError("full retained bundle has no complete solution source tree")
    for executable in ("solution/fit", "solution/predict"):
        if executable not in solution_files:
            raise ScoreOneWitnessError(f"full retained bundle is missing {executable}")
    evidence_members = [
        {"path": name, "sha256": _sha256(data), "byte_count": len(data)}
        for name, data in sorted(members.items())
    ]
    return bundle_bytes, members, evidence_members


def _json_member(members: dict[str, bytes], name: str) -> Any:
    try:
        return json.loads(members[name], parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreOneWitnessError(f"retained {name} is invalid JSON") from exc


def _solution_tree_sha256(members: dict[str, bytes]) -> str:
    tree = [
        {"path": name.removeprefix("solution/"), "sha256": _sha256(data),
         "byte_count": len(data)}
        for name, data in sorted(members.items()) if name.startswith("solution/")
    ]
    return _canonical_sha256(tree)


def _gold_submission_tree_sha256(members: dict[str, bytes]) -> str:
    tree = [
        {"path": name.removeprefix("submission/"), "sha256": _sha256(data),
         "byte_count": len(data)}
        for name, data in sorted(members.items()) if name.startswith("submission/")
    ]
    return _canonical_sha256(tree)


def _prediction_from_bytes(data: bytes, dataset_id: str) -> dict[str, list[Any]]:
    """Parse retained pred.csv bytes with the shipped runner's exact CSV contract."""
    if not data or len(data) > 64 * 1024 * 1024:
        raise ScoreOneWitnessError(f"{dataset_id}: retained pred.csv size is invalid")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ScoreOneWitnessError(f"{dataset_id}: retained pred.csv is not UTF-8") from exc
    if not lines:
        raise ScoreOneWitnessError(f"{dataset_id}: retained pred.csv is empty")
    header = lines[0].split(",")
    if (
        not header or header == [""] or len(header) != len(set(header))
        or len(header) > 64 or "mean" not in header
    ):
        raise ScoreOneWitnessError(f"{dataset_id}: retained pred.csv header is invalid")
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    if not rows or len(rows) > 200_000 or any(len(row) != len(header) for row in rows):
        raise ScoreOneWitnessError(f"{dataset_id}: retained pred.csv rows are invalid")
    columns: dict[str, list[Any]] = {}
    for index, name in enumerate(header):
        values = [row[index] for row in rows]
        if name == "sample_id":
            columns[name] = values
            continue
        try:
            columns[name] = [float(value) for value in values]
        except (ValueError, OverflowError) as exc:
            raise ScoreOneWitnessError(
                f"{dataset_id}: retained pred.csv column {name!r} is nonnumeric") from exc
    mean = columns["mean"]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in mean):
        raise ScoreOneWitnessError(
            f"{dataset_id}: retained predictions are nonfinite or outside [0,1]")
    return columns


def _validate_and_normalize_gold_timings(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate retained execution timings, then remove only their nondeterminism.

    Release validation reruns the locked submission.  Exit codes and source/model/
    output hashes must reproduce byte-for-byte, while wall-clock samples naturally
    differ.  Both the retained creation sample and the fresh validation sample must
    independently clear the public phase caps before these two fields are removed
    for the stable artifact comparison.
    """
    normalized = json.loads(json.dumps(artifact))
    rows = normalized.get("dataset_results")
    if not isinstance(rows, list) or len(rows) != SHIPPED_DATASET_COUNT:
        raise ScoreOneWitnessError("gold execution result set is incomplete")
    for row in rows:
        if not isinstance(row, dict):
            raise ScoreOneWitnessError("gold execution result is malformed")
        fit_seconds = row.pop("fit_seconds", None)
        predict_seconds = row.pop("predict_seconds", None)
        if (
            not _finite(fit_seconds)
            or not 0.0 < float(fit_seconds) <= FIT_TIMEOUT_S
            or not _finite(predict_seconds)
            or not 0.0 < float(predict_seconds) <= PREDICT_TIMEOUT_S
        ):
            raise ScoreOneWitnessError("gold execution timing misses a public phase cap")
    return normalized


def _derive_gold_execution_evidence(
    bundle_path: Path,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_contract_sha256: str,
    scientific_tuple_sha256: str,
    auth_key: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Execute locked public gold in the trusted sandbox and recompute score 1."""
    bundle_bytes = Path(bundle_path).read_bytes()
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or any(not _safe_member(name) for name in names):
                raise ScoreOneWitnessError("gold execution request has duplicate/unsafe members")
            members = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ScoreOneWitnessError(f"invalid gold execution request: {exc}") from exc
    required = {
        "candidate_release_lock.json", "gold_execution.json",
        "submission/fit", "submission/predict",
    }
    missing = required - set(members)
    if missing or any(not members[name] for name in required):
        raise ScoreOneWitnessError(
            f"gold execution request is missing/empty {sorted(missing)}")
    if "reward_detail.json" in members:
        raise ScoreOneWitnessError(
            "gold execution request must not carry a publisher-authored reward_detail.json")
    submission = {name: data for name, data in members.items()
                  if name.startswith("submission/")}
    authoritative_mapping = {
        "submission/fit": ROOT / "gold" / "fit",
        "submission/predict": ROOT / "gold" / "predict",
        "submission/pgs_core.py": ROOT / "gold" / "pgs_core.py",
    }
    if set(submission) != set(authoritative_mapping) or any(
        not path.is_file() or submission[name] != path.read_bytes()
        for name, path in authoritative_mapping.items()
    ):
        raise ScoreOneWitnessError(
            "gold execution submission must byte-match the locked authoritative gold source")
    lowered_source = b"\n".join(submission.values()).lower()
    if any(token in lowered_source for token in (
        b"truth/", b"anchors.json", b"corpus-key", b"score1_witness",
        b"y_test.csv", b"calibration_evidence",
        b"from grader", b"import grader", b"from datagen", b"import datagen",
        b"from reference", b"import reference", b"from validation",
        b"import validation",
    )):
        raise ScoreOneWitnessError(
            "gold execution submission source references private benchmark material")

    metadata = _json_member(members, "gold_execution.json")
    candidate = _json_member(members, "candidate_release_lock.json")
    if not isinstance(metadata, dict) or set(metadata) != _GOLD_EXECUTION_METADATA_FIELDS:
        raise ScoreOneWitnessError("gold execution metadata fields do not match schema")
    if (
        metadata["schema_version"] != 1
        or metadata["environment_id"] != ENVIRONMENT_ID
        or not isinstance(metadata["gold_id"], str) or not metadata["gold_id"]
    ):
        raise ScoreOneWitnessError("gold execution metadata is malformed")

    candidate_bytes = members["candidate_release_lock.json"]
    candidate_sha = _sha256(candidate_bytes)
    from validation.release_gate import LOCK_SCHEMA_VERSION
    if (
        metadata["candidate_lock_sha256"] != candidate_sha
        or not isinstance(candidate, dict)
        or candidate.get("schema_version") != LOCK_SCHEMA_VERSION
        or candidate.get("environment_id") != ENVIRONMENT_ID
        or candidate.get("release_status") != "candidate"
        or not verify_json_hmac(
            auth_key, candidate, field="release_lock_hmac_sha256",
            domain=b"svpgsbench|schema-v8|release-lock\x00")
    ):
        raise ScoreOneWitnessError("gold execution candidate lock is unauthenticated")
    expected_provenance = {
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_contract_sha256,
        "scientific_tuple_sha256": scientific_tuple_sha256,
    }
    if any(candidate.get(field) != value for field, value in expected_provenance.items()):
        raise ScoreOneWitnessError("gold execution candidate science tuple is stale")
    witnessed_executable = candidate.get("executable_tuple_sha256")
    if not isinstance(witnessed_executable, str) or not _SHA256.fullmatch(witnessed_executable):
        raise ScoreOneWitnessError("gold execution candidate executable tuple is invalid")

    manifest_by_id = _manifest_index(manifest)
    prediction_names = {name for name in members if name.startswith("predictions/")}
    if prediction_names:
        raise ScoreOneWitnessError(
            "gold execution request must not contain publisher-supplied predictions")

    dataset_results: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    corpus_root = ROOT / "corpus"
    with tempfile.TemporaryDirectory(prefix="svpgs-score1-source-") as source_dir:
        source_root = Path(source_dir)
        for name, data in submission.items():
            destination = source_root / name.removeprefix("submission/")
            destination.write_bytes(data)
            os.chmod(destination, 0o755 if destination.name in {"fit", "predict"} else 0o644)
        ok, built_submission, build_detail = build_submission(
            source_root, log=lambda _message: None)
        if not ok or built_submission is None:
            raise ScoreOneWitnessError(
                f"locked gold submission failed trusted sandbox staging: {build_detail}")
        try:
            for dataset_id, entry in manifest_by_id.items():
                dataset_dir = corpus_root / entry["path"]
                try:
                    anchors = json.loads(
                        (dataset_dir / "truth" / "anchors.json").read_bytes(),
                        parse_constant=_reject_constant)
                    sample_ids, targets = read_binary_truth_csv(
                        dataset_dir / "truth" / "y_test.csv")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: cannot load authenticated gold truth") from exc
                execution = run_on_dataset(
                    built_submission,
                    dataset_dir / "public",
                    n_test_expected=len(targets),
                    family=anchors["family"],
                )
                if execution.get("status") != "ok":
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: locked gold sandbox execution failed: "
                        f"{execution.get('status')}: {execution.get('detail')}")
                prediction_bytes = execution.get("pred_bytes")
                if not isinstance(prediction_bytes, bytes):
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: sandbox retained no exact pred.csv bytes")
                prediction = _prediction_from_bytes(prediction_bytes, dataset_id)
                if len(prediction["mean"]) != len(targets):
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: executed prediction row count differs")
                if "sample_id" in prediction and tuple(prediction["sample_id"]) != sample_ids:
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: executed sample IDs are out of order")
                family = anchors["family"]
                directions = {"auc": True, "brier": False, "log_loss": False}
                submission_metrics = compute_metrics(family, targets, prediction)
                naive = {name: (value, directions[name])
                         for name, value in anchors["metrics_naive"].items()}
                reference = {name: (value, directions[name])
                             for name, value in anchors["metrics_reference"].items()}
                raw, _ = accuracy_skill(
                    submission_metrics, naive, reference,
                    dict(anchors["metrics_ref_naive_se"]),
                    dict(anchors["metrics_naive_se"]))
                calibration = calibration_factor(submission_metrics, reference)
                reward = float(apply_calibration_factor(raw, calibration["factor"]))
                if (reward < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
                        or reward > SKILL_HI + 1e-12):
                    raise ScoreOneWitnessError(
                        f"{dataset_id}: executed gold reward misses score 1")
                row = {
                    "dataset_id": dataset_id, "category": entry["category"],
                    "weight": float(entry["weight"]), "status": "ok",
                    "native_reward": reward,
                    "fit_exit_code": int(execution["fit_rc"]),
                    "predict_exit_code": int(execution["predict_rc"]),
                    "fit_seconds": float(execution["t_fit"]),
                    "predict_seconds": float(execution["t_predict"]),
                    "model_sha256": execution["model_sha256"],
                    "prediction_sha256": execution["pred_sha256"],
                    "prediction_byte_count": len(prediction_bytes),
                }
                dataset_results.append(row)
                aggregate_rows.append({
                    "category": row["category"], "weight": row["weight"],
                    "reward": reward})
        finally:
            _rmtree_ro(built_submission)
    headline, _, aggregation = category_aggregation(aggregate_rows)
    tail = float(aggregation["tail_value"])
    if (
        headline < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
        or tail < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
        or headline > SKILL_HI + 1e-12
        or tail > SKILL_HI + 1e-12
    ):
        raise ScoreOneWitnessError("gold execution headline/tail misses score 1")
    evidence_members = [
        {"path": name, "sha256": _sha256(data), "byte_count": len(data)}
        for name, data in sorted(members.items())
    ]
    return {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "benchmark_schema_version": CORPUS_SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "evidence_kind": "gold_execution_bundle",
        "witness_id": metadata["gold_id"],
        **expected_provenance,
        "gold_executable_tuple_sha256": witnessed_executable,
        "candidate_lock_sha256": candidate_sha,
        "submission_tree_sha256": _gold_submission_tree_sha256(members),
        "evidence_bundle": {
            "path": "", "sha256": _sha256(bundle_bytes), "byte_count": len(bundle_bytes)},
        "evidence_members": evidence_members,
        "dataset_count": SHIPPED_DATASET_COUNT,
        "dataset_results": dataset_results,
        "native_headline": float(headline), "native_tail": tail,
        "score_one_tolerance": SCORE_ONE_TOLERANCE,
    }, bundle_bytes


def _unique_test_events(tests_bytes: bytes) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for raw in tests_bytes.splitlines():
        try:
            event = json.loads(raw, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoreOneWitnessError("logs/tests.jsonl contains invalid JSON") from exc
        data = event.get("data") if isinstance(event, dict) else None
        test_id = data.get("testId") if isinstance(data, dict) else None
        if event.get("type") != "test_result" or not isinstance(test_id, str):
            continue
        if set(event) != _EVENT_FIELDS or event.get("category") != "tests":
            raise ScoreOneWitnessError("retained test-result event fields do not match schema")
        _timestamp(event["timestamp"], f"{test_id} event timestamp")
        if test_id in events:
            raise ScoreOneWitnessError(f"retained tests contain duplicate {test_id}")
        if set(data) != _RESULT_FIELDS:
            raise ScoreOneWitnessError(f"{test_id} TestResult fields do not match schema")
        events[test_id] = data
    expected = {"benchmark"} | {f"category:{cat}" for cat in SHIPPED_CATEGORIES}
    if set(events) != expected:
        raise ScoreOneWitnessError("retained tests do not contain the exact benchmark/category set")
    return events


def _match_run_result(retained: Any, event: dict[str, Any]) -> None:
    if not isinstance(retained, dict):
        raise ScoreOneWitnessError("run metadata has a malformed retained TestResult")
    if (
        retained.get("testName") != event["name"]
        or retained.get("status") != event["status"]
        or retained.get("durationMs") != event["duration"]
        or not _finite(retained.get("score"))
        or not _finite(event["score"])
        or abs(float(retained["score"]) - float(event["score"])) > 1e-9
    ):
        raise ScoreOneWitnessError("tests.jsonl does not belong to the packaged run metadata")


def _derive_bundle_evidence(
    bundle_path: Path,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_contract_sha256: str,
    scientific_tuple_sha256: str,
    expected_rollout_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(scientific_tuple_sha256, str) or not _SHA256.fullmatch(
        scientific_tuple_sha256
    ):
        raise ScoreOneWitnessError("candidate scientific tuple digest is invalid")
    bundle_path = Path(bundle_path)
    bundle_bytes, members, evidence_members = _read_bundle(bundle_path)
    run = _json_member(members, "run.json")
    if not isinstance(run, dict):
        raise ScoreOneWitnessError("retained run.json is not an object")
    run_id = run.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ScoreOneWitnessError("run metadata is missing runId")
    if expected_rollout_id is None:
        match = _ROLLOUT_BUNDLE.search(bundle_path.name)
        if match is None:
            raise ScoreOneWitnessError("bundle filename does not identify its rollout")
        rollout_id = f"{run_id}-{match.group(1)}"
    else:
        rollout_id = expected_rollout_id
    if re.fullmatch(re.escape(run_id) + r"-rollout-[0-9]+", rollout_id) is None:
        raise ScoreOneWitnessError("rollout id does not belong to retained run")
    rollouts = run.get("rollouts")
    matches = [item for item in rollouts or []
               if isinstance(item, dict) and item.get("id") == rollout_id]
    if len(matches) != 1:
        raise ScoreOneWitnessError("run metadata has no unique selected rollout")
    rollout = matches[0]
    if rollout.get("currentPhase") != "complete" or not isinstance(
        rollout.get("completedAt"), str
    ):
        raise ScoreOneWitnessError("score-1 rollout was not terminal when packaged")
    completed_at = _timestamp(rollout["completedAt"], "rollout completedAt")
    commit = run.get("commitSha")
    instance = run.get("instanceType")
    compute = run.get("compute")
    if run.get("environmentId") != ENVIRONMENT_ID:
        raise ScoreOneWitnessError("retained run is from the wrong environment")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ScoreOneWitnessError("retained run has no exact executed environment commit")
    if (
        instance != TARGET_INSTANCE_TYPE
        or not isinstance(compute, dict)
        or compute.get("instanceType") != TARGET_INSTANCE_TYPE
    ):
        raise ScoreOneWitnessError("retained run was not executed on the target instance type")

    events = _unique_test_events(members["logs/tests.jsonl"])
    retained_results = rollout.get("testResults")
    if not isinstance(retained_results, list):
        raise ScoreOneWitnessError("run metadata has no retained test results")
    retained_by_id = {
        row.get("testId"): row for row in retained_results
        if isinstance(row, dict) and isinstance(row.get("testId"), str)
    }
    if set(retained_by_id) != set(events) or len(retained_by_id) != len(retained_results):
        raise ScoreOneWitnessError("run metadata does not retain the exact emitted test set")
    for test_id, event in events.items():
        _match_run_result(retained_by_id[test_id], event)

    manifest_by_id = _manifest_index(manifest)
    dataset_results: list[dict[str, Any]] = []
    per_category: dict[str, float] = {}
    seen: set[str] = set()
    for category in SHIPPED_CATEGORIES:
        result = events[f"category:{category}"]
        if (
            result["testId"] != f"category:{category}"
            or result["name"] != f"{category} category"
            or result["status"] != "partially_passed"
            or not _finite(result["duration"])
            or float(result["duration"]) < 0
            or not _finite(result["score"])
            or not _finite(result["weight"])
        ):
            raise ScoreOneWitnessError(f"{category}: malformed category TestResult")
        try:
            output = json.loads(result["output"], parse_constant=_reject_constant)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ScoreOneWitnessError(f"{category}: category output is invalid JSON") from exc
        if not isinstance(output, dict) or set(output) != _CATEGORY_OUTPUT_FIELDS:
            raise ScoreOneWitnessError(f"{category}: category output fields do not match schema")
        native = output["native_score"]
        if (
            output["contract_ok"] is not True
            or not _finite(native)
            or not _finite(output["platform_score"])
            or abs(float(result["score"]) - _platform_score(float(native))) > 1e-9
            or abs(float(output["platform_score"]) - float(result["score"])) > 1e-9
        ):
            raise ScoreOneWitnessError(f"{category}: category score/provenance mismatch")
        rows = output["datasets"]
        if not isinstance(rows, list) or len(rows) != 3:
            raise ScoreOneWitnessError(f"{category}: expected exactly three dataset results")
        category_rows = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != _DATASET_OUTPUT_FIELDS:
                raise ScoreOneWitnessError("dataset output fields do not match schema")
            dataset_id = row["dataset"]
            entry = manifest_by_id.get(dataset_id)
            if entry is None or dataset_id in seen or entry.get("category") != category:
                raise ScoreOneWitnessError("dataset result is missing, duplicated, or foreign")
            if row["status"] != "ok" or any(
                not _finite(row[field]) for field in (
                    "reward", "skill", "accuracy", "performance", "fit_seconds",
                    "predict_seconds",
                )
            ):
                raise ScoreOneWitnessError(f"{dataset_id}: dataset execution was not successful")
            reward = float(row["reward"])
            if (reward < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
                    or reward > SKILL_HI + 1e-12):
                raise ScoreOneWitnessError(
                    f"{dataset_id}: native reward is not within the score-1 tolerance")
            weight = entry.get("weight")
            if not _finite(weight) or float(weight) <= 0:
                raise ScoreOneWitnessError(f"{dataset_id}: manifest weight is invalid")
            seen.add(dataset_id)
            derived = {
                "dataset_id": dataset_id,
                "category": category,
                "weight": float(weight),
                "status": "ok",
                "native_reward": reward,
            }
            dataset_results.append(derived)
            category_rows.append({"reward": reward, "weight": float(weight)})
        category_native = sum(row["reward"] * row["weight"] for row in category_rows) \
            / sum(row["weight"] for row in category_rows)
        if abs(float(native) - category_native) > 1e-9:
            raise ScoreOneWitnessError(f"{category}: native score does not reproduce datasets")
        per_category[category] = float(native)
    if seen != set(manifest_by_id) or len(dataset_results) != SHIPPED_DATASET_COUNT:
        raise ScoreOneWitnessError("retained evidence does not cover the exact manifest dataset set")

    benchmark = events["benchmark"]
    try:
        benchmark_output = json.loads(benchmark["output"], parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ScoreOneWitnessError("benchmark output is invalid JSON") from exc
    if not isinstance(benchmark_output, dict) or set(benchmark_output) != _BENCHMARK_OUTPUT_FIELDS:
        raise ScoreOneWitnessError("benchmark output fields do not match schema")
    native_headline = benchmark_output["native_score"]
    native_tail = benchmark_output["mastery_tail_value"]
    if (
        benchmark["testId"] != "benchmark"
        or benchmark["name"] != "svpgsbench headline reward"
        or benchmark["status"] != "partially_passed"
        or benchmark["duration"] != 0
        or benchmark["weight"] != 1
        or benchmark_output["integrity_ok"] is not True
        or benchmark_output["contract_ok"] is not True
        or benchmark_output["mastery"] is not None
        or benchmark_output["mastery_threshold"] is not None
        or benchmark_output["mastery_tail_threshold"] is not None
        or not all(_finite(value) for value in (
            benchmark["score"], native_headline, native_tail,
            benchmark_output["platform_score"],
        ))
        or benchmark_output["per_category"] != per_category
    ):
        raise ScoreOneWitnessError("benchmark TestResult is not an uncalibrated successful witness")
    release_fields = {
        "scientific_tuple_sha256": scientific_tuple_sha256,
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_contract_sha256,
    }
    if (
        benchmark_output["release_status"] != "candidate"
        or any(benchmark_output[field] != expected
               for field, expected in release_fields.items())
        or any(
            not isinstance(benchmark_output[field], str)
            or not _SHA256.fullmatch(benchmark_output[field])
            for field in ("executable_tuple_sha256", "candidate_lock_sha256")
        )
    ):
        raise ScoreOneWitnessError(
            "benchmark release provenance does not match the candidate science tuple")
    aggregate_rows = [
        {"category": row["category"], "weight": row["weight"],
         "reward": row["native_reward"]}
        for row in dataset_results
    ]
    recomputed_headline, _, aggregation = category_aggregation(aggregate_rows)
    for category in SHIPPED_CATEGORIES:
        if abs(
            float(events[f"category:{category}"]["weight"])
            - float(aggregation["category_coefficients"][category])
        ) > 1e-12:
            raise ScoreOneWitnessError(
                f"{category}: TestResult weight does not match trusted aggregation")
    if (
        abs(float(native_headline) - recomputed_headline) > 1e-9
        or abs(float(native_tail) - float(aggregation["tail_value"])) > 1e-9
        or abs(float(benchmark["score"]) - _platform_score(float(native_headline))) > 1e-9
        or abs(float(benchmark_output["platform_score"]) - float(benchmark["score"])) > 1e-9
        or float(native_headline) < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
        or float(native_tail) < 1.0 - SCORE_ONE_TOLERANCE - 1e-12
        or float(native_headline) > SKILL_HI + 1e-12
        or float(native_tail) > SKILL_HI + 1e-12
    ):
        raise ScoreOneWitnessError("benchmark headline/tail does not prove native score 1")

    solution_tree_sha256 = _solution_tree_sha256(members)
    artifact = {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "benchmark_schema_version": CORPUS_SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "evidence_kind": "hfdev_full_bundle",
        "witness_id": rollout_id,
        "execution_id": rollout_id,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "executed_environment_commit": commit,
        "target_instance_type": TARGET_INSTANCE_TYPE,
        "completed_at": completed_at,
        "manifest_sha256": benchmark_output["manifest_sha256"],
        "scoring_contract_sha256": benchmark_output["scoring_contract_sha256"],
        "wrapper_sha256": benchmark_output["wrapper_sha256"],
        "scientific_tuple_sha256": benchmark_output["scientific_tuple_sha256"],
        "witnessed_executable_tuple_sha256": benchmark_output["executable_tuple_sha256"],
        "candidate_lock_sha256": benchmark_output["candidate_lock_sha256"],
        "submission_tree_sha256": solution_tree_sha256,
        "evidence_bundle": {
            "path": "",  # Set only after the content-addressed copy succeeds.
            "sha256": _sha256(bundle_bytes),
            "byte_count": len(bundle_bytes),
        },
        "evidence_members": evidence_members,
        "dataset_count": SHIPPED_DATASET_COUNT,
        "dataset_results": dataset_results,
        "native_headline": float(native_headline),
        "native_tail": float(native_tail),
        "score_one_tolerance": SCORE_ONE_TOLERANCE,
    }
    return artifact, bundle_bytes


def _evidence_relative_path(bundle_sha256: str) -> str:
    return f"validation/score1_evidence/{bundle_sha256}.zip"


def evidence_path_from_artifact(artifact: Any) -> Path:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("evidence_bundle"), dict):
        raise ScoreOneWitnessError("witness has no retained evidence bundle")
    bundle = artifact["evidence_bundle"]
    digest = bundle.get("sha256")
    expected = _evidence_relative_path(digest) if isinstance(digest, str) else None
    if bundle.get("path") != expected or not isinstance(expected, str):
        raise ScoreOneWitnessError("witness evidence path is not content-addressed")
    path = ROOT / expected
    if path.resolve().parent != EVIDENCE_DIR.resolve():
        raise ScoreOneWitnessError("witness evidence path escapes its immutable directory")
    return path


def artifact_from_bundle(
    bundle_path: Path,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_contract_sha256: str,
    scientific_tuple_sha256: str,
    auth_key: bytes,
) -> dict[str, Any]:
    """Derive an hfdev execution witness or independently scored public gold proof."""
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise ScoreOneWitnessError(f"invalid score-1 evidence bundle: {exc}") from exc
    arguments = dict(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=scoring_contract_sha256,
        wrapper_contract_sha256=wrapper_contract_sha256,
        scientific_tuple_sha256=scientific_tuple_sha256,
    )
    if "gold_execution.json" in names:
        artifact, bundle_bytes = _derive_gold_execution_evidence(
            bundle_path, **arguments, auth_key=auth_key)
    else:
        artifact, bundle_bytes = _derive_bundle_evidence(bundle_path, **arguments)
    digest = artifact["evidence_bundle"]["sha256"]
    destination = EVIDENCE_DIR / f"{digest}.zip"
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != bundle_bytes:
        raise ScoreOneWitnessError("content-addressed witness destination has different bytes")
    if not destination.exists():
        temporary = destination.with_suffix(".zip.tmp")
        temporary.write_bytes(bundle_bytes)
        temporary.replace(destination)
    artifact["evidence_bundle"]["path"] = _evidence_relative_path(digest)
    artifact[WITNESS_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        artifact,
        exclude_field=WITNESS_HMAC_FIELD,
        domain=WITNESS_HMAC_DOMAIN,
    )
    return artifact


def validate_artifact(
    artifact: Any,
    auth_key: bytes,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_contract_sha256: str,
    scientific_tuple_sha256: str,
) -> None:
    if not isinstance(artifact, dict) or not verify_json_hmac(
        auth_key,
        artifact,
        field=WITNESS_HMAC_FIELD,
        domain=WITNESS_HMAC_DOMAIN,
    ):
        raise ScoreOneWitnessError("score-1 witness HMAC mismatch")
    bundle_path = evidence_path_from_artifact(artifact)
    bundle = artifact.get("evidence_bundle")
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise ScoreOneWitnessError("score-1 retained evidence bundle is missing") from exc
    if (
        _sha256(bundle_bytes) != bundle.get("sha256")
        or len(bundle_bytes) != bundle.get("byte_count")
    ):
        raise ScoreOneWitnessError("score-1 retained evidence bytes do not match artifact")
    arguments = dict(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=scoring_contract_sha256,
        wrapper_contract_sha256=wrapper_contract_sha256,
        scientific_tuple_sha256=scientific_tuple_sha256,
    )
    if artifact.get("evidence_kind") == "gold_execution_bundle":
        reparsed, _ = _derive_gold_execution_evidence(
            bundle_path, **arguments, auth_key=auth_key)
    elif artifact.get("evidence_kind") == "hfdev_full_bundle":
        reparsed, _ = _derive_bundle_evidence(
            bundle_path, **arguments,
            expected_rollout_id=artifact.get("rollout_id"))
    else:
        raise ScoreOneWitnessError("score-1 witness has an unknown evidence kind")
    reparsed["evidence_bundle"]["path"] = artifact["evidence_bundle"]["path"]
    unsigned = dict(artifact)
    unsigned.pop(WITNESS_HMAC_FIELD, None)
    if artifact.get("evidence_kind") == "gold_execution_bundle":
        unsigned = _validate_and_normalize_gold_timings(unsigned)
        reparsed = _validate_and_normalize_gold_timings(reparsed)
    if unsigned != reparsed:
        raise ScoreOneWitnessError("score-1 artifact does not reproduce retained bundle bytes")


def _candidate_scientific_tuple(lock_path: Path, auth_key: bytes) -> str:
    """Read the authenticated candidate tuple without importing release_gate."""
    try:
        lock = json.loads(Path(lock_path).read_bytes(), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreOneWitnessError("candidate release lock is missing or invalid") from exc
    from validation.release_gate import (  # Local import avoids module cycle.
        LOCK_HMAC_DOMAIN,
        LOCK_HMAC_FIELD,
        LOCK_SCHEMA_VERSION,
    )
    if (
        not isinstance(lock, dict)
        or lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("release_status") != "candidate"
        or not verify_json_hmac(
            auth_key, lock, field=LOCK_HMAC_FIELD, domain=LOCK_HMAC_DOMAIN)
    ):
        raise ScoreOneWitnessError("score-1 creation requires an authenticated candidate lock")
    value = lock.get("scientific_tuple_sha256")
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ScoreOneWitnessError("candidate lock has no scientific tuple digest")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("create", choices=("create",))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path,
                        default=ROOT / "validation" / "release_lock.json")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output", type=Path, default=WITNESS_PATH)
    args = parser.parse_args()

    manifest_path = ROOT / "corpus" / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes, parse_constant=_reject_constant)
    from validation.model_zoo import _implementation_hashes
    from grader.grade import _validate_corpus
    implementation = _implementation_hashes()
    auth_key = load_corpus_key(args.key_file, repository_root=str(ROOT))
    _validate_corpus(ROOT / "corpus", manifest, manifest.get("datasets"), auth_key)
    artifact = artifact_from_bundle(
        args.bundle,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_bytes),
        scoring_contract_sha256=implementation["scoring_contract_sha256"],
        wrapper_contract_sha256=wrapper_sha256(),
        scientific_tuple_sha256=_candidate_scientific_tuple(args.candidate_lock, auth_key),
        auth_key=auth_key,
    )
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(f"score-1 witness: PASS ({args.output})")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"score-1 witness: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
