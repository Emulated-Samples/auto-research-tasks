"""Authenticated mastery calibration artifacts built from retained hfdev evidence.

Calibration and evaluation are versioned, disjoint evidence sets. A bundle parser
extracts the native headline/tail from the exact benchmark TestResult, checks the
platform/native affine mapping, and binds the evidence to the run, rollout, env
commit, full bundle bytes, and benchmark-detail bytes. Every accepted package is
retained content-addressed under ``validation/calibration_evidence`` and reparsed
when the artifact is validated. The final JSON artifact is HMAC-authenticated with
the corpus key and must agree with grader.contract. Evaluation also binds the exact
``run.json`` launch membership in both phases: all attempts must be present, agent
non-completions remain in the capability denominator, and provider-rate-limit deaths
are infrastructure-censored. Only gradable calibration rows can define thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grader.contract import (
    CORPUS_SCHEMA_VERSION,
    MASTERY_THRESHOLD,
    MASTERY_TAIL_THRESHOLD,
    SHIPPED_CATEGORIES,
)
from grader.corpus_auth import json_hmac_sha256, load_corpus_key, verify_json_hmac
from grader.skill import (
    AGG_MEAN_W,
    AGG_TAIL_W,
    SKILL_HI,
    SKILL_LO,
    tail_size,
)

CALIBRATION_SCHEMA_VERSION = 3
CALIBRATION_HMAC_FIELD = "calibration_hmac_sha256"
CALIBRATION_HMAC_DOMAIN = b"svpgsbench|schema-v8|mastery-calibration-v3\x00"
ENVIRONMENT_ID = "sc-svpgsbench"
FRONTIER_MODEL = "claude-opus-4-8"
MIN_CAPABILITY_ATTEMPTS = 3
CALIBRATION_EVIDENCE_DIR = ROOT / "validation" / "calibration_evidence"
_ROLLOUT_SUFFIX = re.compile(r"-(rollout-[0-9]+)(?:-retained-only)?\.zip$")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
EVIDENCE_FIELDS = {
    "run_id", "rollout_id", "environment_commit", "model", "completed_at",
    "bundle_sha256", "benchmark_event_sha256", "benchmark_output_sha256",
    "native_headline", "native_tail", "observed_mastery_threshold",
    "observed_mastery_tail_threshold", "observed_mastery",
    "release_status", "scientific_tuple_sha256", "executable_tuple_sha256",
    "manifest_sha256", "scoring_contract_sha256", "wrapper_sha256",
    "candidate_lock_sha256",
    "attempt_classification", "run_total_rollouts", "run_rollout_ids",
}
FAILURE_EVIDENCE_FIELDS = {
    "run_id", "rollout_id", "environment_commit", "model", "completed_at",
    "bundle_sha256", "attempt_classification", "run_total_rollouts",
    "run_rollout_ids", "worker_status", "problem_status", "current_phase",
    "terminal_error",
}
_BENCHMARK_EVENT_FIELDS = {"timestamp", "type", "category", "message", "data"}
# The grader's tests.jsonl benchmark TestResult carries exactly these keys. Earlier
# drafts of this validator also expected "description"/"weight", but the shipped
# grader log never emits them; the headline identity is testId/name and the score is
# independently cross-checked against the rollout's retained testResults below, so the
# authenticated set is the fields the log actually writes.
_BENCHMARK_RESULT_FIELDS = {
    "testId", "name", "status", "duration", "score", "output",
}
_BENCHMARK_OUTPUT_FIELDS = {
    "native_score", "platform_score", "integrity_ok", "contract_ok", "mastery",
    "mastery_threshold", "mastery_tail_threshold", "mastery_tail_value",
    "per_category",
    "release_status", "scientific_tuple_sha256", "executable_tuple_sha256",
    "manifest_sha256", "scoring_contract_sha256", "wrapper_sha256",
    "candidate_lock_sha256",
}
_FULL_BUNDLE_MEMBERS = {
    "run.json",
    "summary.txt",
    "task_prompt.md",
    "transcript.txt",
    "solution.diff.json",
    "logs/agent.jsonl",
    "logs/setup.jsonl",
    "logs/tests.jsonl",
}


class CalibrationEvidenceError(ValueError):
    pass


def _run_membership(run: Any, rollout_suffix: str) -> tuple[str, str, dict[str, Any], int, list[str]]:
    """Validate one terminal attempt and the complete launch membership in run.json."""
    if not isinstance(run, dict):
        raise CalibrationEvidenceError("run metadata is not an object")
    run_id = run.get("runId")
    rollout_id = f"{run_id}-{rollout_suffix}"
    rollouts = run.get("rollouts")
    total = run.get("totalRollouts")
    if (
        not isinstance(run_id, str)
        or _RUN_ID.fullmatch(run_id) is None
        or not isinstance(rollouts, list)
        or type(total) is not int
        or total < 1
        or len(rollouts) != total
        or run.get("status") not in {"completed", "stopped"}
    ):
        raise CalibrationEvidenceError(
            "run metadata is not terminal or lacks a consistent "
            "runId/totalRollouts/rollouts set")
    ids = [item.get("id") if isinstance(item, dict) else None for item in rollouts]
    expected = [f"{run_id}-rollout-{index}" for index in range(total)]
    if ids != expected:
        raise CalibrationEvidenceError(
            "run rollout membership is not the exact ordered totalRollouts set")
    matches = [item for item in rollouts
               if isinstance(item, dict) and item.get("id") == rollout_id]
    if len(matches) != 1:
        raise CalibrationEvidenceError(f"run metadata does not contain exactly {rollout_id}")
    rollout = matches[0]
    if not isinstance(rollout.get("completedAt"), str):
        raise CalibrationEvidenceError("rollout was not terminal when packaged")
    return run_id, rollout_id, rollout, total, expected


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _platform_score(native: float) -> float:
    return (native - SKILL_LO) / (1.0 - SKILL_LO)


def _scientific_tuple_sha256(
    manifest_sha256: str, scoring_contract_sha256: str, wrapper_sha256: str,
) -> str:
    return _sha256(json.dumps({
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_sha256,
    }, sort_keys=True, separators=(",", ":")).encode())


def _parse_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CalibrationEvidenceError(f"{label} is not a UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CalibrationEvidenceError(f"{label} is not a UTC timestamp") from exc
    return value


def _recompute_aggregates(per_category: Any) -> tuple[float, float]:
    if (
        not isinstance(per_category, dict)
        or set(per_category) != set(SHIPPED_CATEGORIES)
        or any(not _finite(value) for value in per_category.values())
        or any(not SKILL_LO <= float(value) <= SKILL_HI
               for value in per_category.values())
    ):
        raise CalibrationEvidenceError("benchmark per_category does not match the contract")
    values = [float(per_category[category]) for category in SHIPPED_CATEGORIES]
    weakest = sorted(values)[:tail_size(len(values))]
    tail = sum(weakest) / len(weakest)
    headline = AGG_MEAN_W * (sum(values) / len(values)) + AGG_TAIL_W * tail
    return headline, tail


def evidence_from_bundle(bundle_path: Path) -> dict[str, Any]:
    """Extract one gradable rollout from a complete hfdev package.

    Retained-only/partial packages are intentionally rejected. Calibration is release
    evidence, so the exact prompt, transcript, solution diff, setup log, agent log,
    grading log, summary, and run metadata must all remain independently inspectable.
    """
    bundle_path = Path(bundle_path)
    match = _ROLLOUT_SUFFIX.search(bundle_path.name)
    if match is None:
        raise CalibrationEvidenceError("bundle filename does not identify a rollout index")
    rollout_suffix = match.group(1)
    bundle_bytes = bundle_path.read_bytes()
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            members = archive.namelist()
            names = set(members)
            required = _FULL_BUNDLE_MEMBERS
            if not required <= names:
                raise CalibrationEvidenceError(
                    f"full bundle is missing {sorted(required - names)}")
            if any(members.count(name) != 1 for name in required):
                raise CalibrationEvidenceError("full bundle has duplicate required members")
            if "PACKAGE_INCOMPLETE.txt" in names:
                raise CalibrationEvidenceError("calibration requires a complete, non-retained-only bundle")
            if any(archive.getinfo(name).file_size <= 0 for name in required):
                raise CalibrationEvidenceError("full bundle contains an empty required evidence file")
            solution_members = [
                name for name in members
                if name.startswith("solution/") and not name.endswith("/")
            ]
            if not solution_members:
                raise CalibrationEvidenceError("full bundle contains no retained solution source files")
            run = json.loads(archive.read("run.json"))
            tests_bytes = archive.read("logs/tests.jsonl")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError(f"invalid rollout bundle: {exc}") from exc

    run_id, rollout_id, rollout, run_total, run_rollout_ids = _run_membership(
        run, rollout_suffix)
    completed_at = _parse_utc_timestamp(rollout["completedAt"], "rollout completedAt")

    benchmark_events: list[tuple[dict[str, Any], bytes]] = []
    for raw_line in tests_bytes.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise CalibrationEvidenceError("tests.jsonl contains invalid JSON") from exc
        data = event.get("data") if isinstance(event, dict) else None
        if isinstance(event, dict) and event.get("type") == "test_result" \
                and isinstance(data, dict) \
                and data.get("testId") == "benchmark":
            if set(event) != _BENCHMARK_EVENT_FIELDS:
                raise CalibrationEvidenceError("benchmark event fields do not match schema")
            benchmark_events.append((event, raw_line))
    if len(benchmark_events) != 1:
        raise CalibrationEvidenceError(
            f"expected one benchmark TestResult, found {len(benchmark_events)}")
    benchmark_event, benchmark_event_bytes = benchmark_events[0]
    if (
        benchmark_event["category"] != "tests"
        or not isinstance(benchmark_event["message"], str)
    ):
        raise CalibrationEvidenceError("benchmark event metadata is malformed")
    _parse_utc_timestamp(benchmark_event["timestamp"], "benchmark event timestamp")
    benchmark = benchmark_event["data"]
    if set(benchmark) != _BENCHMARK_RESULT_FIELDS:
        raise CalibrationEvidenceError("benchmark TestResult fields do not match schema")
    if (
        benchmark["testId"] != "benchmark"
        or benchmark["name"] != "svpgsbench headline reward"
        or benchmark["duration"] != 0
        or benchmark["status"] not in {"passed", "partially_passed", "failed"}
    ):
        raise CalibrationEvidenceError("benchmark TestResult metadata is malformed")
    retained_results = rollout.get("testResults")
    if not isinstance(retained_results, list):
        raise CalibrationEvidenceError("rollout metadata has no retained test results")
    retained_benchmarks = [
        result for result in retained_results
        if isinstance(result, dict) and result.get("testId") == "benchmark"
    ]
    if len(retained_benchmarks) != 1:
        raise CalibrationEvidenceError("rollout metadata has no unique benchmark result")
    retained = retained_benchmarks[0]
    if (
        retained.get("testName") != benchmark["name"]
        or retained.get("status") != benchmark["status"]
        or retained.get("durationMs") != benchmark["duration"]
        or not _finite(retained.get("score"))
        or not _finite(benchmark["score"])
        or abs(float(retained["score"]) - float(benchmark["score"])) > 1e-9
    ):
        raise CalibrationEvidenceError(
            "benchmark event does not belong to the packaged rollout metadata")
    output_text = benchmark.get("output")
    if not isinstance(output_text, str):
        raise CalibrationEvidenceError("benchmark TestResult has no JSON output")
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise CalibrationEvidenceError("benchmark output is invalid JSON") from exc

    if not isinstance(output, dict) or set(output) != _BENCHMARK_OUTPUT_FIELDS:
        raise CalibrationEvidenceError("benchmark output fields do not match schema")
    native = output.get("native_score")
    tail = output.get("mastery_tail_value")
    platform = benchmark.get("score")
    if not all(_finite(value) for value in (native, tail, platform,
                                             output.get("platform_score"))):
        raise CalibrationEvidenceError(
            "v8 benchmark output needs finite native_score, mastery_tail_value, and score")
    native = float(native)
    tail = float(tail)
    if not SKILL_LO <= native <= SKILL_HI or not SKILL_LO <= tail <= SKILL_HI:
        raise CalibrationEvidenceError("native headline/tail is outside the scoring contract")
    if abs(float(platform) - _platform_score(native)) > 1e-9:
        raise CalibrationEvidenceError("platform score does not match the native transform")
    if abs(float(output["platform_score"]) - float(platform)) > 1e-9:
        raise CalibrationEvidenceError("benchmark output platform score disagrees with TestResult")
    if output.get("integrity_ok") is not True or output.get("contract_ok") is not True:
        raise CalibrationEvidenceError("calibration rollout failed integrity or contract")
    if (
        output.get("release_status") != "production"
        or any(not isinstance(output.get(field), str)
               or _SHA256.fullmatch(output[field]) is None
               for field in (
                   "scientific_tuple_sha256", "executable_tuple_sha256",
                   "manifest_sha256", "scoring_contract_sha256", "wrapper_sha256",
                   "candidate_lock_sha256",
               ))
    ):
        raise CalibrationEvidenceError(
            "calibration rollout is not from an authenticated production release")
    recomputed_headline, recomputed_tail = _recompute_aggregates(output["per_category"])
    if abs(native - recomputed_headline) > 1e-9 or abs(tail - recomputed_tail) > 1e-9:
        raise CalibrationEvidenceError("benchmark headline/tail do not reproduce per_category")

    observed_threshold = output["mastery_threshold"]
    observed_tail_threshold = output["mastery_tail_threshold"]
    observed_mastery = output["mastery"]
    if (observed_threshold is None) != (observed_tail_threshold is None):
        raise CalibrationEvidenceError("benchmark carries half-calibrated mastery")
    if observed_threshold is None:
        if observed_mastery is not None or benchmark["status"] != "partially_passed":
            raise CalibrationEvidenceError("uncalibrated benchmark claims mastery")
    else:
        if not all(_finite(value) for value in (observed_threshold,
                                                observed_tail_threshold)):
            raise CalibrationEvidenceError("benchmark mastery thresholds are invalid")
        expected_mastery = (
            native >= float(observed_threshold)
            and tail >= float(observed_tail_threshold)
        )
        if observed_mastery is not expected_mastery:
            raise CalibrationEvidenceError("benchmark mastery verdict disagrees with thresholds")
        expected_status = "passed" if expected_mastery else "failed"
        if benchmark["status"] != expected_status:
            raise CalibrationEvidenceError("benchmark status disagrees with mastery verdict")

    commit = run.get("commitSha")
    environment_id = run.get("environmentId")
    model = rollout.get("requestedModel", run.get("requestedModel"))
    if not isinstance(commit, str) or not _COMMIT_SHA.fullmatch(commit):
        raise CalibrationEvidenceError("run metadata has no exact environment commit")
    if environment_id != ENVIRONMENT_ID:
        raise CalibrationEvidenceError(
            f"rollout environment is {environment_id!r}, expected {ENVIRONMENT_ID!r}")
    if model != FRONTIER_MODEL:
        raise CalibrationEvidenceError(
            f"calibration model is {model!r}, expected frontier model {FRONTIER_MODEL!r}")
    return {
        "run_id": run_id,
        "rollout_id": rollout_id,
        "environment_commit": commit,
        "model": model,
        "completed_at": completed_at,
        "bundle_sha256": _sha256(bundle_bytes),
        "benchmark_event_sha256": _sha256(benchmark_event_bytes),
        "benchmark_output_sha256": _sha256(output_text.encode()),
        "native_headline": native,
        "native_tail": tail,
        "observed_mastery_threshold": observed_threshold,
        "observed_mastery_tail_threshold": observed_tail_threshold,
        "observed_mastery": observed_mastery,
        "release_status": output["release_status"],
        "scientific_tuple_sha256": output["scientific_tuple_sha256"],
        "executable_tuple_sha256": output["executable_tuple_sha256"],
        "manifest_sha256": output["manifest_sha256"],
        "scoring_contract_sha256": output["scoring_contract_sha256"],
        "wrapper_sha256": output["wrapper_sha256"],
        "candidate_lock_sha256": output["candidate_lock_sha256"],
        "attempt_classification": "gradable",
        "run_total_rollouts": run_total,
        "run_rollout_ids": run_rollout_ids,
    }


def terminal_failure_evidence_from_bundle(bundle_path: Path) -> dict[str, Any]:
    """Classify one terminal non-gradable attempt from a full or retained-only bundle.

    Provider rate-limit rejection is the sole infrastructure-censored class. Agent
    failure requires the exact stopped/fail/complete state. Unknown terminal states
    are rejected rather than guessed, so harness failures cannot bias capability.
    A failure package can never supply calibration scores or masquerade as gradable.
    """
    path = Path(bundle_path)
    match = _ROLLOUT_SUFFIX.search(path.name)
    if match is None:
        raise CalibrationEvidenceError("bundle filename does not identify a rollout index")
    bundle_bytes = path.read_bytes()
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if members.count("run.json") != 1 or archive.getinfo("run.json").file_size <= 0:
                raise CalibrationEvidenceError(
                    "terminal failure bundle needs one nonempty run.json")
            run = json.loads(archive.read("run.json"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError(f"invalid terminal failure bundle: {exc}") from exc

    run_id, rollout_id, rollout, run_total, run_rollout_ids = _run_membership(
        run, match.group(1))
    retained_results = rollout.get("testResults")
    if not isinstance(retained_results, list):
        raise CalibrationEvidenceError("terminal rollout metadata has no testResults list")
    if any(isinstance(item, dict) and item.get("testId") == "benchmark"
           for item in retained_results):
        raise CalibrationEvidenceError(
            "a rollout with a benchmark result must use complete gradable evidence")
    worker_status = rollout.get("workerStatus")
    problem_status = rollout.get("problemStatus")
    current_phase = rollout.get("currentPhase")
    error = rollout.get("error", "")
    if not isinstance(error, str):
        raise CalibrationEvidenceError(
            "non-gradable evidence has no exact terminal error metadata")
    infra = re.fullmatch(
        r"Claude Code rate limit rejected \([A-Za-z0-9_-]+\); provider reset at "
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z",
        error,
    ) is not None
    if infra:
        if worker_status != "stopped" or problem_status != "error" \
                or current_phase != "agent":
            raise CalibrationEvidenceError(
                "provider-rate-limit evidence has inconsistent terminal metadata")
    elif not (worker_status == "stopped" and problem_status == "fail"
              and current_phase == "complete"):
        raise CalibrationEvidenceError(
            "unknown non-gradable terminal state; only explicit complete agent "
            "failure or provider rate-limit censoring is accepted")
    commit = run.get("commitSha")
    environment_id = run.get("environmentId")
    model = rollout.get("requestedModel", run.get("requestedModel"))
    if not isinstance(commit, str) or _COMMIT_SHA.fullmatch(commit) is None:
        raise CalibrationEvidenceError("run metadata has no exact environment commit")
    if environment_id != ENVIRONMENT_ID:
        raise CalibrationEvidenceError(
            f"rollout environment is {environment_id!r}, expected {ENVIRONMENT_ID!r}")
    if model != FRONTIER_MODEL:
        raise CalibrationEvidenceError(
            f"evaluation model is {model!r}, expected frontier model {FRONTIER_MODEL!r}")
    return {
        "run_id": run_id,
        "rollout_id": rollout_id,
        "environment_commit": commit,
        "model": model,
        "completed_at": _parse_utc_timestamp(
            rollout["completedAt"], "rollout completedAt"),
        "bundle_sha256": _sha256(bundle_bytes),
        "attempt_classification": "infra_censored" if infra else "agent_failure",
        "run_total_rollouts": run_total,
        "run_rollout_ids": run_rollout_ids,
        "worker_status": worker_status,
        "problem_status": problem_status,
        "current_phase": current_phase,
        "terminal_error": error,
    }


def _retained_bundle_path(
    row: dict[str, Any],
    evidence_dir: Path | None = None,
) -> Path:
    root = CALIBRATION_EVIDENCE_DIR if evidence_dir is None else Path(evidence_dir)
    return root / row["bundle_sha256"] / f'{row["rollout_id"]}.zip'


def evidence_paths_from_artifact(artifact: Any) -> list[Path]:
    """Resolve the exact content-addressed bundles referenced by an artifact."""
    if not isinstance(artifact, dict):
        raise CalibrationEvidenceError("calibration artifact must be an object")
    rows: list[Any] = []
    for field in ("calibration_rollouts", "evaluation_rollouts"):
        value = artifact.get(field)
        if not isinstance(value, list):
            raise CalibrationEvidenceError(f"calibration artifact {field} must be a list")
        rows.extend(value)
    paths: list[Path] = []
    for row in rows:
        _validate_evaluation_attempt(row)
        path = _retained_bundle_path(row)
        if path.resolve().parent.parent != CALIBRATION_EVIDENCE_DIR.resolve():
            raise CalibrationEvidenceError("calibration evidence path escapes its directory")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise CalibrationEvidenceError("calibration artifact references duplicate evidence paths")
    return paths


def archive_bundle(
    bundle_path: Path,
    *,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Parse and retain one complete bundle at its content-addressed path."""
    source = Path(bundle_path)
    row = evidence_from_bundle(source)
    _validate_evidence_row(row)
    destination = _retained_bundle_path(row, evidence_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CalibrationEvidenceError(
                f"retained bundle path is not a regular file: {destination}")
        if _sha256(destination.read_bytes()) != row["bundle_sha256"]:
            raise CalibrationEvidenceError(
                f"retained bundle differs at content-addressed path: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise CalibrationEvidenceError(
                f"stale calibration-evidence temporary exists: {temporary}")
        shutil.copyfile(source, temporary)
        if _sha256(temporary.read_bytes()) != row["bundle_sha256"]:
            temporary.unlink()
            raise CalibrationEvidenceError("bundle changed while it was being archived")
        temporary.replace(destination)
    if evidence_from_bundle(destination) != row:
        raise CalibrationEvidenceError("retained bundle does not reproduce extracted evidence")
    return row


def archive_evaluation_bundle(
    bundle_path: Path,
    *,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Retain a gradable evaluation bundle or classify a terminal non-completion."""
    source = Path(bundle_path)
    try:
        row = evidence_from_bundle(source)
    except CalibrationEvidenceError as gradable_error:
        try:
            row = terminal_failure_evidence_from_bundle(source)
        except CalibrationEvidenceError as failure_error:
            raise CalibrationEvidenceError(
                "evaluation bundle is neither complete gradable evidence nor a valid "
                f"terminal failure (gradable={gradable_error}; failure={failure_error})") \
                from failure_error
    _validate_evaluation_attempt(row)
    destination = _retained_bundle_path(row, evidence_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CalibrationEvidenceError(
                f"retained bundle path is not a regular file: {destination}")
        if _sha256(destination.read_bytes()) != row["bundle_sha256"]:
            raise CalibrationEvidenceError(
                f"retained bundle differs at content-addressed path: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise CalibrationEvidenceError(
                f"stale calibration-evidence temporary exists: {temporary}")
        shutil.copyfile(source, temporary)
        if _sha256(temporary.read_bytes()) != row["bundle_sha256"]:
            temporary.unlink()
            raise CalibrationEvidenceError("bundle changed while it was being archived")
        temporary.replace(destination)
    _validate_retained_bundle(row, evidence_dir=evidence_dir)
    return row


def _validate_retained_bundle(
    row: dict[str, Any],
    *,
    evidence_dir: Path | None = None,
) -> None:
    path = _retained_bundle_path(row, evidence_dir)
    if path.is_symlink() or not path.is_file():
        raise CalibrationEvidenceError(
            f"retained calibration bundle is missing: {path}")
    if _sha256(path.read_bytes()) != row["bundle_sha256"]:
        raise CalibrationEvidenceError(
            f"retained calibration bundle digest mismatch: {path}")
    parser = (evidence_from_bundle if row.get("attempt_classification") == "gradable"
              else terminal_failure_evidence_from_bundle)
    if parser(path) != row:
        raise CalibrationEvidenceError(
            f"retained calibration bundle does not reproduce its evidence row: {path}")


def _validate_evidence_row(row: Any) -> None:
    if not isinstance(row, dict) or set(row) != EVIDENCE_FIELDS:
        raise CalibrationEvidenceError("rollout evidence fields do not match schema")
    run_id = row["run_id"]
    rollout_id = row["rollout_id"]
    if (
        not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None
        or not isinstance(rollout_id, str)
        or re.fullmatch(re.escape(run_id) + r"-rollout-[0-9]+", rollout_id) is None
        or row["model"] != FRONTIER_MODEL
        or not isinstance(row["environment_commit"], str)
        or _COMMIT_SHA.fullmatch(row["environment_commit"]) is None
        or any(not isinstance(row[field], str) or _SHA256.fullmatch(row[field]) is None
               for field in ("bundle_sha256", "benchmark_event_sha256",
                             "benchmark_output_sha256", "scientific_tuple_sha256",
                             "executable_tuple_sha256", "manifest_sha256",
                             "scoring_contract_sha256", "wrapper_sha256",
                             "candidate_lock_sha256"))
        or row["release_status"] != "production"
        or row["attempt_classification"] != "gradable"
        or type(row["run_total_rollouts"]) is not int
        or row["run_total_rollouts"] < 1
        or not isinstance(row["run_rollout_ids"], list)
        or len(row["run_rollout_ids"]) != row["run_total_rollouts"]
        or row["run_rollout_ids"] != [
            f"{run_id}-rollout-{index}"
            for index in range(row["run_total_rollouts"])
        ]
        or not all(_finite(row[field]) for field in ("native_headline", "native_tail"))
        or not all(SKILL_LO <= float(row[field]) <= SKILL_HI
                   for field in ("native_headline", "native_tail"))
    ):
        raise CalibrationEvidenceError("rollout evidence values do not match schema")
    _parse_utc_timestamp(row["completed_at"], "evidence completed_at")
    threshold = row["observed_mastery_threshold"]
    tail_threshold = row["observed_mastery_tail_threshold"]
    if (threshold is None) != (tail_threshold is None):
        raise CalibrationEvidenceError("rollout evidence is half-calibrated")
    if threshold is None:
        if row["observed_mastery"] is not None:
            raise CalibrationEvidenceError("uncalibrated evidence claims mastery")
    else:
        if (
            not _finite(threshold) or not _finite(tail_threshold)
            or not 0.0 < float(threshold) <= SKILL_HI
            or not 0.0 < float(tail_threshold) <= SKILL_HI
        ):
            raise CalibrationEvidenceError("rollout evidence thresholds are invalid")
        expected = (float(row["native_headline"]) >= float(threshold)
                    and float(row["native_tail"]) >= float(tail_threshold))
        if row["observed_mastery"] is not expected:
            raise CalibrationEvidenceError("rollout evidence mastery is inconsistent")


def _validate_evaluation_attempt(row: Any) -> None:
    """Validate one member of the complete post-freeze launched-attempt set."""
    if isinstance(row, dict) and row.get("attempt_classification") == "gradable":
        _validate_evidence_row(row)
        return
    if not isinstance(row, dict) or set(row) != FAILURE_EVIDENCE_FIELDS:
        raise CalibrationEvidenceError("evaluation failure fields do not match schema")
    run_id = row["run_id"]
    total = row["run_total_rollouts"]
    ids = row["run_rollout_ids"]
    if (
        row["attempt_classification"] not in {"agent_failure", "infra_censored"}
        or not isinstance(run_id, str)
        or _RUN_ID.fullmatch(run_id) is None
        or not isinstance(row["rollout_id"], str)
        or type(total) is not int
        or total < 1
        or not isinstance(ids, list)
        or ids != [f"{run_id}-rollout-{index}" for index in range(total)]
        or row["rollout_id"] not in ids
        or row["model"] != FRONTIER_MODEL
        or not isinstance(row["environment_commit"], str)
        or _COMMIT_SHA.fullmatch(row["environment_commit"]) is None
        or not isinstance(row["bundle_sha256"], str)
        or _SHA256.fullmatch(row["bundle_sha256"]) is None
        or row["worker_status"] != "stopped"
        or row["problem_status"] not in {"fail", "error"}
        or not isinstance(row["current_phase"], str)
        or not row["current_phase"]
        or not isinstance(row["terminal_error"], str)
    ):
        raise CalibrationEvidenceError("evaluation failure values do not match schema")
    _parse_utc_timestamp(row["completed_at"], "evidence completed_at")
    is_rate_limit = re.fullmatch(
        r"Claude Code rate limit rejected \([A-Za-z0-9_-]+\); provider reset at "
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z",
        row["terminal_error"],
    ) is not None
    expected = "infra_censored" if is_rate_limit else "agent_failure"
    if row["attempt_classification"] != expected:
        raise CalibrationEvidenceError(
            "terminal failure classification disagrees with retained run metadata")
    if expected == "infra_censored" and not (
        row["worker_status"] == "stopped"
        and row["problem_status"] == "error"
        and row["current_phase"] == "agent"
    ):
        raise CalibrationEvidenceError("infra censor metadata is not fail-closed")
    if expected == "agent_failure" and not (
        row["worker_status"] == "stopped"
        and row["problem_status"] == "fail"
        and row["current_phase"] == "complete"
    ):
        raise CalibrationEvidenceError("agent failure metadata is not fail-closed")


def _validate_complete_attempt_set(
    rows: list[dict[str, Any]], *, phase: str,
) -> None:
    """Reject omitted siblings from one complete hfdev run in either phase."""
    if not rows:
        raise CalibrationEvidenceError(f"{phase} attempt set is empty")
    for row in rows:
        _validate_evaluation_attempt(row)
    ids = [row["rollout_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise CalibrationEvidenceError(f"{phase} attempt set contains duplicate rollout ids")
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append(row)
    if len(by_run) != 1:
        raise CalibrationEvidenceError(
            f"{phase} evidence must be one complete launched hfdev run")
    for run_id, group in by_run.items():
        declared = group[0]["run_rollout_ids"]
        total = group[0]["run_total_rollouts"]
        if any(row["run_rollout_ids"] != declared
               or row["run_total_rollouts"] != total for row in group):
            raise CalibrationEvidenceError(
                f"{phase} run {run_id} has inconsistent membership claims")
        if set(row["rollout_id"] for row in group) != set(declared):
            raise CalibrationEvidenceError(
                f"{phase} run {run_id} omits launched rollout evidence")
    denominator = sum(
        row["attempt_classification"] != "infra_censored" for row in rows)
    gradable = sum(row["attempt_classification"] == "gradable" for row in rows)
    if denominator < MIN_CAPABILITY_ATTEMPTS:
        raise CalibrationEvidenceError(
            f"{phase} needs at least {MIN_CAPABILITY_ATTEMPTS} capability-denominator attempts")
    if gradable < 1:
        raise CalibrationEvidenceError(f"{phase} needs at least one gradable attempt")


def _validate_complete_calibration_attempt_set(rows: list[dict[str, Any]]) -> None:
    _validate_complete_attempt_set(rows, phase="calibration")


def _validate_complete_evaluation_attempt_set(rows: list[dict[str, Any]]) -> None:
    _validate_complete_attempt_set(rows, phase="evaluation")


def derive_thresholds(calibration_rollouts: list[dict[str, Any]]) -> tuple[float, float]:
    """Freeze the frontier row: max headline, deterministic tail/id tie breaks.

    The input is one complete launched run. Terminal failures remain retained to
    prevent success-only cherry-picking, but only gradable rows can define a score
    threshold. Agent failures remain in the phase's capability denominator and
    infrastructure failures are censored under the shared attempt contract.

    Taking the tail from the same observed frontier rollout ensures the binary event
    was achieved in calibration. Independent headline/tail maxima could synthesize an
    unobserved conjunction and manufacture a zero-pass frontier ceiling.
    """
    _validate_complete_calibration_attempt_set(calibration_rollouts)
    gradable = [row for row in calibration_rollouts
                if row["attempt_classification"] == "gradable"]
    for row in gradable:
        if row["observed_mastery_threshold"] is not None:
            raise CalibrationEvidenceError("calibration evidence must come from uncalibrated release")
    frontier = max(
        gradable,
        key=lambda row: (float(row["native_headline"]), float(row["native_tail"]),
                         row["rollout_id"]),
    )
    threshold = float(frontier["native_headline"])
    tail_threshold = float(frontier["native_tail"])
    if not (0.0 < threshold <= SKILL_HI and 0.0 < tail_threshold <= SKILL_HI):
        raise CalibrationEvidenceError("observed frontier cannot define positive mastery bars")
    return threshold, tail_threshold


def sign_artifact(artifact: dict[str, Any], auth_key: bytes) -> dict[str, Any]:
    signed = dict(artifact)
    signed[CALIBRATION_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        signed,
        exclude_field=CALIBRATION_HMAC_FIELD,
        domain=CALIBRATION_HMAC_DOMAIN,
    )
    return signed


def validate_artifact(
    artifact: Any,
    auth_key: bytes,
    *,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
) -> None:
    required = {
        "schema_version", "benchmark_schema_version", "environment_id", "status",
        "manifest_sha256", "scoring_contract_sha256", "wrapper_sha256",
        "mastery_threshold", "mastery_tail_threshold", "calibration_rollouts",
        "evaluation_rollouts", CALIBRATION_HMAC_FIELD,
    }
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise CalibrationEvidenceError("calibration artifact fields do not match schema")
    if not verify_json_hmac(
        auth_key,
        artifact,
        field=CALIBRATION_HMAC_FIELD,
        domain=CALIBRATION_HMAC_DOMAIN,
    ):
        raise CalibrationEvidenceError("calibration artifact HMAC mismatch")
    if (
        artifact["schema_version"] != CALIBRATION_SCHEMA_VERSION
        or artifact["benchmark_schema_version"] != CORPUS_SCHEMA_VERSION
        or artifact["environment_id"] != ENVIRONMENT_ID
        or artifact["manifest_sha256"] != manifest_sha256
        or artifact["scoring_contract_sha256"] != scoring_contract_sha256
        or artifact["wrapper_sha256"] != wrapper_sha256
    ):
        raise CalibrationEvidenceError("calibration artifact provenance mismatch")
    for field in ("manifest_sha256", "scoring_contract_sha256", "wrapper_sha256"):
        if not isinstance(artifact[field], str) or _SHA256.fullmatch(artifact[field]) is None:
            raise CalibrationEvidenceError("calibration artifact digest is malformed")
    calibration = artifact["calibration_rollouts"]
    evaluation = artifact["evaluation_rollouts"]
    scientific_tuple_sha256 = _scientific_tuple_sha256(
        manifest_sha256, scoring_contract_sha256, wrapper_sha256)
    if not isinstance(calibration, list) or not isinstance(evaluation, list):
        raise CalibrationEvidenceError("calibration/evaluation rollouts must be lists")
    for row in calibration:
        _validate_evaluation_attempt(row)
        _validate_retained_bundle(row)
        if row["attempt_classification"] == "gradable" and (
            row["manifest_sha256"] != manifest_sha256
            or row["scoring_contract_sha256"] != scoring_contract_sha256
            or row["wrapper_sha256"] != wrapper_sha256
            or row["scientific_tuple_sha256"] != scientific_tuple_sha256
        ):
            raise CalibrationEvidenceError(
                "rollout release provenance differs from calibration artifact")
    for row in evaluation:
        _validate_evaluation_attempt(row)
        _validate_retained_bundle(row)
        if row["attempt_classification"] == "gradable" and (
            row["manifest_sha256"] != manifest_sha256
            or row["scoring_contract_sha256"] != scoring_contract_sha256
            or row["wrapper_sha256"] != wrapper_sha256
            or row["scientific_tuple_sha256"] != scientific_tuple_sha256
        ):
            raise CalibrationEvidenceError(
                "gradable evaluation release provenance differs from artifact")
    calibration_ids = {item.get("rollout_id") for item in calibration if isinstance(item, dict)}
    evaluation_ids = {item.get("rollout_id") for item in evaluation if isinstance(item, dict)}
    if len(calibration_ids) != len(calibration) or len(evaluation_ids) != len(evaluation):
        raise CalibrationEvidenceError("rollout evidence contains duplicates or malformed rows")
    if calibration_ids & evaluation_ids:
        raise CalibrationEvidenceError("calibration and evaluation rollout sets overlap")
    all_rows = calibration + evaluation
    if (
        len({row["bundle_sha256"] for row in all_rows}) != len(all_rows)
        or len({row["benchmark_event_sha256"] for row in all_rows
                if row["attempt_classification"] == "gradable"})
        != sum(row["attempt_classification"] == "gradable" for row in all_rows)
    ):
        raise CalibrationEvidenceError("rollout evidence reuses a bundle or benchmark event")

    status = artifact["status"]
    if status == "uncalibrated":
        if (
            MASTERY_THRESHOLD is not None
            or MASTERY_TAIL_THRESHOLD is not None
            or artifact["mastery_threshold"] is not None
            or artifact["mastery_tail_threshold"] is not None
            or calibration
            or evaluation
        ):
            raise CalibrationEvidenceError("uncalibrated artifact carries mastery evidence")
    elif status in {"threshold_frozen", "calibrated"}:
        _validate_complete_calibration_attempt_set(calibration)
        if status == "threshold_frozen" and evaluation:
            raise CalibrationEvidenceError(
                "threshold-frozen artifact cannot claim evaluation evidence")
        if status == "calibrated":
            _validate_complete_evaluation_attempt_set(evaluation)
        derived_threshold, derived_tail_threshold = derive_thresholds(calibration)
        if (
            not _finite(MASTERY_THRESHOLD)
            or not _finite(MASTERY_TAIL_THRESHOLD)
            or artifact["mastery_threshold"] != MASTERY_THRESHOLD
            or artifact["mastery_tail_threshold"] != MASTERY_TAIL_THRESHOLD
            or artifact["mastery_threshold"] != derived_threshold
            or artifact["mastery_tail_threshold"] != derived_tail_threshold
        ):
            raise CalibrationEvidenceError(
                "calibrated thresholds disagree with frozen derivation or grader.contract")
        calibration_commits = {item["environment_commit"] for item in calibration}
        calibration_models = {item["model"] for item in calibration}
        gradable_calibration = [
            item for item in calibration
            if item["attempt_classification"] == "gradable"
        ]
        calibration_tuples = {
            item["executable_tuple_sha256"] for item in gradable_calibration}
        calibration_science = {
            item["scientific_tuple_sha256"] for item in gradable_calibration}
        calibration_candidate_locks = {
            item["candidate_lock_sha256"] for item in gradable_calibration}
        if (
            len(calibration_commits) != 1 or len(calibration_models) != 1
            or len(calibration_tuples) != 1 or len(calibration_science) != 1
            or len(calibration_candidate_locks) != 1
        ):
            raise CalibrationEvidenceError(
                "calibration must bind one exact production tuple and frontier model")
        if status == "calibrated":
            evaluation_commits = {item["environment_commit"] for item in evaluation}
            evaluation_models = {item["model"] for item in evaluation}
            gradable_evaluation = [
                item for item in evaluation
                if item["attempt_classification"] == "gradable"
            ]
            evaluation_tuples = {
                item["executable_tuple_sha256"] for item in gradable_evaluation}
            evaluation_science = {
                item["scientific_tuple_sha256"] for item in gradable_evaluation}
            evaluation_candidate_locks = {
                item["candidate_lock_sha256"] for item in gradable_evaluation}
            if (
                len(evaluation_commits) != 1 or len(evaluation_tuples) != 1
                or len(evaluation_candidate_locks) != 1
            ):
                raise CalibrationEvidenceError(
                    "evaluation rollouts must bind one exact threshold-frozen tuple")
            if calibration_commits == evaluation_commits:
                raise CalibrationEvidenceError(
                    "evaluation must run on the threshold-frozen release, not calibration commit")
            if evaluation_models != calibration_models:
                raise CalibrationEvidenceError(
                    "calibration and evaluation must measure one identical frontier model")
            if evaluation_tuples == calibration_tuples:
                raise CalibrationEvidenceError(
                    "evaluation must execute the distinct threshold-frozen tuple")
            if evaluation_science != calibration_science:
                raise CalibrationEvidenceError(
                    "mastery overlay changed the stable scientific tuple")
            for row in gradable_evaluation:
                if (
                    row["observed_mastery_threshold"] != derived_threshold
                    or row["observed_mastery_tail_threshold"] != derived_tail_threshold
                ):
                    raise CalibrationEvidenceError(
                        "evaluation evidence does not carry the frozen thresholds")
    else:
        raise CalibrationEvidenceError(f"unknown calibration status {status!r}")


def uncalibrated_artifact(
    *,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
    auth_key: bytes,
) -> dict[str, Any]:
    return sign_artifact({
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "benchmark_schema_version": CORPUS_SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "status": "uncalibrated",
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_sha256,
        "mastery_threshold": None,
        "mastery_tail_threshold": None,
        "calibration_rollouts": [],
        "evaluation_rollouts": [],
    }, auth_key)


def calibrated_artifact(
    *,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
    calibration_rollouts: list[dict[str, Any]],
    evaluation_rollouts: list[dict[str, Any]],
    auth_key: bytes,
) -> dict[str, Any]:
    """Build the authenticated overlay from evidence; thresholds are never caller-set."""
    threshold, tail_threshold = derive_thresholds(calibration_rollouts)
    _validate_complete_evaluation_attempt_set(evaluation_rollouts)
    calibration_ids = {row["rollout_id"] for row in calibration_rollouts}
    for row in evaluation_rollouts:
        _validate_evaluation_attempt(row)
        if row["rollout_id"] in calibration_ids:
            raise CalibrationEvidenceError("calibration and evaluation rollout sets overlap")
        if row["attempt_classification"] == "gradable" and (
            row["observed_mastery_threshold"] != threshold
            or row["observed_mastery_tail_threshold"] != tail_threshold
        ):
            raise CalibrationEvidenceError(
                "evaluation evidence does not carry the frozen thresholds")
    return sign_artifact({
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "benchmark_schema_version": CORPUS_SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "status": "calibrated",
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_sha256,
        "mastery_threshold": threshold,
        "mastery_tail_threshold": tail_threshold,
        "calibration_rollouts": list(calibration_rollouts),
        "evaluation_rollouts": list(evaluation_rollouts),
    }, auth_key)


def passk_report_from_artifact(
    artifact: Any,
    auth_key: bytes,
    *,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
) -> dict[str, Any]:
    """Report pass@k from every authenticated attempt in complete run membership."""
    validate_artifact(
        artifact,
        auth_key,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=scoring_contract_sha256,
        wrapper_sha256=wrapper_sha256,
    )
    if artifact["status"] != "calibrated":
        raise CalibrationEvidenceError(
            "pass@k report requires a calibrated artifact with disjoint evaluation")
    from validation.passk import Rollout, report
    rollouts = []
    for row in artifact["evaluation_rollouts"]:
        classification = row["attempt_classification"]
        if classification == "gradable":
            rollouts.append(Rollout(
                row["rollout_id"],
                gradable=True,
                headline=float(row["native_headline"]),
                tail=float(row["native_tail"]),
                integrity_ok=True,
                contract_ok=True,
            ))
        else:
            rollouts.append(Rollout(
                row["rollout_id"],
                gradable=False,
                censor_reason=("infra" if classification == "infra_censored"
                               else "agent"),
            ))
    result = report(rollouts, ks=(1, 2, 3))
    result["source"] = "authenticated_mastery_calibration_evaluation"
    result["calibration_schema_version"] = artifact["schema_version"]
    result["evaluation_rollout_ids"] = [row["rollout_id"]
                                         for row in artifact["evaluation_rollouts"]]
    result["evaluation_attempt_classification"] = {
        row["rollout_id"]: row["attempt_classification"]
        for row in artifact["evaluation_rollouts"]
    }
    return result


def threshold_frozen_artifact(
    *,
    manifest_sha256: str,
    scoring_contract_sha256: str,
    wrapper_sha256: str,
    calibration_rollouts: list[dict[str, Any]],
    auth_key: bytes,
) -> dict[str, Any]:
    """Intermediate release used only to collect post-freeze evaluation evidence."""
    threshold, tail_threshold = derive_thresholds(calibration_rollouts)
    return sign_artifact({
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "benchmark_schema_version": CORPUS_SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "status": "threshold_frozen",
        "manifest_sha256": manifest_sha256,
        "scoring_contract_sha256": scoring_contract_sha256,
        "wrapper_sha256": wrapper_sha256,
        "mastery_threshold": threshold,
        "mastery_tail_threshold": tail_threshold,
        "calibration_rollouts": list(calibration_rollouts),
        "evaluation_rollouts": [],
    }, auth_key)


def _current_provenance() -> tuple[str, str, str]:
    manifest_bytes = (ROOT / "corpus" / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise CalibrationEvidenceError(f"invalid corpus manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise CalibrationEvidenceError("calibration requires the current shipping manifest")
    from validation.model_zoo import _implementation_hashes
    from validation.score1_witness import wrapper_sha256
    implementation = _implementation_hashes()
    return (
        _sha256(manifest_bytes),
        implementation["scoring_contract_sha256"],
        wrapper_sha256(),
    )


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise CalibrationEvidenceError(f"stale calibration temporary exists: {temporary}")
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def _archive_many(paths: list[Path], evidence_dir: Path) -> list[dict[str, Any]]:
    rows = [archive_bundle(path, evidence_dir=evidence_dir) for path in paths]
    rollout_ids = {row["rollout_id"] for row in rows}
    if len(rollout_ids) != len(rows):
        raise CalibrationEvidenceError("bundle arguments contain duplicate rollout evidence")
    return rows


def _archive_attempt_run(
    paths: list[Path], evidence_dir: Path, *, phase: str,
) -> list[dict[str, Any]]:
    rows = [archive_evaluation_bundle(path, evidence_dir=evidence_dir) for path in paths]
    _validate_complete_attempt_set(rows, phase=phase)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, freeze, and finalize authenticated mastery calibration evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="verify and retain complete rollout bundles")
    extract.add_argument("--bundle", action="append", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser(
        "freeze", help="derive the frontier threshold from uncalibrated Opus bundles")
    freeze.add_argument("--bundle", action="append", required=True, type=Path)
    freeze.add_argument("--key-file", required=True, type=Path)
    freeze.add_argument(
        "--output", type=Path,
        default=ROOT / "validation" / "mastery_calibration.json")

    finalize = subparsers.add_parser(
        "finalize", help="add disjoint post-freeze Opus evaluation bundles")
    finalize.add_argument(
        "--calibration-bundle", action="append", required=True, type=Path)
    finalize.add_argument(
        "--evaluation-bundle", action="append", required=True, type=Path)
    finalize.add_argument("--key-file", required=True, type=Path)
    finalize.add_argument(
        "--output", type=Path,
        default=ROOT / "validation" / "mastery_calibration.json")

    report_parser = subparsers.add_parser(
        "report", help="report pass@k from validated disjoint evaluation evidence")
    report_parser.add_argument("--key-file", required=True, type=Path)
    report_parser.add_argument(
        "--artifact", type=Path,
        default=ROOT / "validation" / "mastery_calibration.json")

    arguments = parser.parse_args()
    if arguments.command == "extract":
        rows = _archive_many(arguments.bundle, CALIBRATION_EVIDENCE_DIR)
        _write_artifact(arguments.output, {
            "frontier_model": FRONTIER_MODEL,
            "rollouts": rows,
        })
        print(f"mastery evidence: retained {len(rows)} complete bundle(s)")
        return

    auth_key = load_corpus_key(
        str(arguments.key_file), repository_root=str(ROOT))
    manifest_sha256, scoring_contract_sha256, wrapper_sha256 = _current_provenance()
    if arguments.command == "freeze":
        calibration = _archive_attempt_run(
            arguments.bundle, CALIBRATION_EVIDENCE_DIR, phase="calibration")
        artifact = threshold_frozen_artifact(
            manifest_sha256=manifest_sha256,
            scoring_contract_sha256=scoring_contract_sha256,
            wrapper_sha256=wrapper_sha256,
            calibration_rollouts=calibration,
            auth_key=auth_key,
        )
        _write_artifact(arguments.output, artifact)
        print(
            "mastery calibration: threshold frozen "
            f"headline={artifact['mastery_threshold']:.17g} "
            f"tail={artifact['mastery_tail_threshold']:.17g}")
        return

    if arguments.command == "report":
        try:
            artifact = json.loads(arguments.artifact.read_bytes())
        except json.JSONDecodeError as exc:
            raise CalibrationEvidenceError(f"invalid calibration artifact: {exc}") from exc
        result = passk_report_from_artifact(
            artifact,
            auth_key,
            manifest_sha256=manifest_sha256,
            scoring_contract_sha256=scoring_contract_sha256,
            wrapper_sha256=wrapper_sha256,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return

    calibration = _archive_attempt_run(
        arguments.calibration_bundle, CALIBRATION_EVIDENCE_DIR,
        phase="calibration")
    evaluation = _archive_attempt_run(
        arguments.evaluation_bundle, CALIBRATION_EVIDENCE_DIR,
        phase="evaluation")
    artifact = calibrated_artifact(
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=scoring_contract_sha256,
        wrapper_sha256=wrapper_sha256,
        calibration_rollouts=calibration,
        evaluation_rollouts=evaluation,
        auth_key=auth_key,
    )
    validate_artifact(
        artifact,
        auth_key,
        manifest_sha256=manifest_sha256,
        scoring_contract_sha256=scoring_contract_sha256,
        wrapper_sha256=wrapper_sha256,
    )
    _write_artifact(arguments.output, artifact)
    print(
        "mastery calibration: finalized "
        f"calibration={len(calibration)} evaluation={len(evaluation)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"mastery calibration: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
