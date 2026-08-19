from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

import validation.mastery_calibration as calibration_module
from grader.contract import SHIPPED_CATEGORIES
from validation.mastery_calibration import (
    CalibrationEvidenceError,
    FRONTIER_MODEL,
    archive_bundle,
    calibrated_artifact,
    derive_thresholds,
    evidence_from_bundle,
    terminal_failure_evidence_from_bundle,
    passk_report_from_artifact,
    sign_artifact,
    threshold_frozen_artifact,
    uncalibrated_artifact,
    validate_artifact,
)

KEY = bytes(range(32))


def _per_category(native_score=0.4, tail=0.2):
    # Three weakest categories make the 15-category bottom fifth. Pick the other
    # twelve so the 0.6 mean + 0.4 tail blend reproduces native_score exactly.
    other = (native_score - 0.104) / 0.48
    return {
        category: (tail if index < 3 else other)
        for index, category in enumerate(SHIPPED_CATEGORIES)
    }


def _bundle(tmp_path, *, platform_score=0.6, native_score=0.4,
            retained_score=None, model=FRONTIER_MODEL):
    path = tmp_path / "run_test-rollout-0.zip"
    run = {
        "runId": "run_test",
        "status": "completed",
        "environmentId": "sc-svpgsbench",
        "commitSha": "a" * 40,
        "requestedModel": model,
        "totalRollouts": 1,
        "rollouts": [{
            "id": "run_test-rollout-0",
            "completedAt": "2026-07-18T00:00:00Z",
            "requestedModel": model,
            "testResults": [{
                "testId": "benchmark",
                "testName": "svpgsbench headline reward",
                "status": "partially_passed",
                "durationMs": 0,
                "executedAt": "2026-07-18T00:00:01Z",
                "score": platform_score if retained_score is None else retained_score,
            }],
        }],
    }
    output = json.dumps({
        "native_score": native_score,
        "platform_score": platform_score,
        "integrity_ok": True,
        "contract_ok": True,
        "mastery": None,
        "mastery_threshold": None,
        "mastery_tail_threshold": None,
        "mastery_tail_value": 0.2,
        "per_category": _per_category(native_score),
        "release_status": "production",
        "scientific_tuple_sha256": "d" * 64,
        "executable_tuple_sha256": "e" * 64,
        "manifest_sha256": "a" * 64,
        "scoring_contract_sha256": "b" * 64,
        "wrapper_sha256": "c" * 64,
        "candidate_lock_sha256": "f" * 64,
    })
    tests = json.dumps({
        "timestamp": "2026-07-18T00:00:01Z",
        "type": "test_result",
        "category": "tests",
        "message": "[PARTIAL] svpgsbench headline reward",
        "data": {
            "testId": "benchmark",
            "name": "svpgsbench headline reward",
            "description": "trusted headline",
            "status": "partially_passed",
            "duration": 0,
            "score": platform_score,
            "weight": 1,
            "output": output,
        },
    }) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("run.json", json.dumps(run))
        archive.writestr("summary.txt", "complete\n")
        archive.writestr("task_prompt.md", "task\n")
        archive.writestr("transcript.txt", "transcript\n")
        archive.writestr("solution.diff.json", "{}\n")
        archive.writestr("logs/agent.jsonl", "{}\n")
        archive.writestr("logs/setup.jsonl", "{}\n")
        archive.writestr("logs/tests.jsonl", tests)
        archive.writestr("solution/solution.py", "# retained source\n")
    return path


def _failure_bundle(tmp_path, *, index=1, error="Agent exited with code 1",
                    total=3):
    path = tmp_path / f"run_fail-rollout-{index}-retained-only.zip"
    rollouts = []
    for rollout_index in range(total):
        row_error = error if rollout_index == index else "agent produced no gradable artifact"
        is_rate_limit = row_error.startswith("Claude Code rate limit rejected (")
        row = {
            "id": f"run_fail-rollout-{rollout_index}",
            "completedAt": "2026-07-18T00:00:00Z",
            "requestedModel": FRONTIER_MODEL,
            "workerStatus": "stopped",
            "problemStatus": "error" if is_rate_limit else "fail",
            "currentPhase": "agent" if is_rate_limit else "complete",
            "testResults": [],
            "error": row_error,
        }
        rollouts.append(row)
    run = {
        "runId": "run_fail",
        "status": "completed",
        "environmentId": "sc-svpgsbench",
        "commitSha": "b" * 40,
        "requestedModel": FRONTIER_MODEL,
        "totalRollouts": total,
        "rollouts": rollouts,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("run.json", json.dumps(run))
        archive.writestr("PACKAGE_INCOMPLETE.txt", "terminal retained-only\n")
    return path


def _failure_evidence(index, classification, *, total=3, run_id="run_eval",
                      commit="b" * 40):
    error = ("Claude Code rate limit rejected (five_hour); provider reset at "
             "2026-07-18T01:30:00.000Z" if classification == "infra_censored"
             else "agent produced no gradable artifact")
    return {
        "run_id": run_id,
        "rollout_id": f"{run_id}-rollout-{index}",
        "environment_commit": commit,
        "model": FRONTIER_MODEL,
        "completed_at": "2026-07-18T00:00:00Z",
        "bundle_sha256": f"{index + 500:064x}",
        "attempt_classification": classification,
        "run_total_rollouts": total,
        "run_rollout_ids": [f"{run_id}-rollout-{i}" for i in range(total)],
        "worker_status": "stopped",
        "problem_status": "error" if classification == "infra_censored" else "fail",
        "current_phase": "agent" if classification == "infra_censored" else "complete",
        "terminal_error": error,
    }


def test_bundle_parser_binds_native_evidence(tmp_path):
    bundle = _bundle(tmp_path)
    evidence = evidence_from_bundle(bundle)
    assert evidence["native_headline"] == 0.4
    assert evidence["native_tail"] == 0.2
    assert evidence["bundle_sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert len(evidence["benchmark_event_sha256"]) == 64
    assert evidence["observed_mastery"] is None


def test_bundle_parser_requires_complete_frontier_model_package(tmp_path):
    with pytest.raises(CalibrationEvidenceError, match="frontier model"):
        evidence_from_bundle(_bundle(tmp_path, model="claude-haiku-4-5"))

    complete = _bundle(tmp_path)
    retained_only = tmp_path / "run_test-rollout-0-retained-only.zip"
    complete.rename(retained_only)
    # Evidence completeness is content-authenticated, not inferred from a filename.
    # A fully retained bundle remains valid if a packaging layer appends a cosmetic
    # suffix; only an actual partial package is rejected.
    evidence = evidence_from_bundle(retained_only)
    assert evidence["rollout_id"] == "run_test-rollout-0"
    with zipfile.ZipFile(retained_only, "a") as archive:
        archive.writestr("PACKAGE_INCOMPLETE.txt", "submission source not retained\n")
    with pytest.raises(CalibrationEvidenceError, match="complete, non-retained-only"):
        evidence_from_bundle(retained_only)


def test_terminal_failure_bundle_classification_is_fail_closed(tmp_path):
    agent = terminal_failure_evidence_from_bundle(_failure_bundle(tmp_path))
    assert agent["attempt_classification"] == "agent_failure"
    assert agent["run_total_rollouts"] == 3
    assert agent["run_rollout_ids"] == [
        "run_fail-rollout-0", "run_fail-rollout-1", "run_fail-rollout-2"]

    rate_limit = _failure_bundle(
        tmp_path, index=2,
        error=("Claude Code rate limit rejected (five_hour); provider reset at "
               "2026-07-18T01:30:00.000Z"))
    infra = terminal_failure_evidence_from_bundle(rate_limit)
    assert infra["attempt_classification"] == "infra_censored"


def test_terminal_failure_bundle_cannot_replace_gradable_evidence(tmp_path):
    bundle = _failure_bundle(tmp_path)
    # A duplicate authenticated metadata member is itself malformed evidence.  The
    # stdlib warns while constructing that adversarial archive; capture it here so
    # a clean QA run reports no unexplained warnings.
    with pytest.warns(UserWarning, match="Duplicate name: 'run.json'"):
        with zipfile.ZipFile(bundle, "a") as archive:
            run = json.loads(archive.read("run.json"))
            run["rollouts"][1]["testResults"] = [{"testId": "benchmark"}]
            archive.writestr("run.json", json.dumps(run))
    with pytest.raises(CalibrationEvidenceError, match="one nonempty run.json"):
        terminal_failure_evidence_from_bundle(bundle)


def test_unknown_terminal_state_is_neither_agent_failure_nor_infra(tmp_path):
    bundle = _failure_bundle(tmp_path)
    rewritten = tmp_path / "run_fail-rollout-1-retained-only-unknown.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(rewritten, "w") as target:
        run = json.loads(source.read("run.json"))
        run["rollouts"][1]["problemStatus"] = "error"
        run["rollouts"][1]["currentPhase"] = "setup"
        target.writestr("run.json", json.dumps(run))
    # Give the rewritten evidence a parseable rollout suffix.
    parseable = tmp_path / "run_fail-rollout-1-retained-only.zip"
    rewritten.replace(parseable)
    with pytest.raises(CalibrationEvidenceError, match="unknown non-gradable"):
        terminal_failure_evidence_from_bundle(parseable)


def test_archive_is_content_addressed_and_reproducible(tmp_path):
    evidence_dir = tmp_path / "calibration_evidence"
    row = archive_bundle(_bundle(tmp_path), evidence_dir=evidence_dir)
    retained = evidence_dir / row["bundle_sha256"] / f'{row["rollout_id"]}.zip'
    assert retained.is_file()
    assert evidence_from_bundle(retained) == row
    calibration_module._validate_retained_bundle(row, evidence_dir=evidence_dir)

    retained.write_bytes(retained.read_bytes() + b"tampered")
    with pytest.raises(CalibrationEvidenceError, match="digest mismatch"):
        calibration_module._validate_retained_bundle(row, evidence_dir=evidence_dir)


def test_bundle_parser_rejects_platform_transform_drift(tmp_path):
    with pytest.raises(CalibrationEvidenceError, match="platform score"):
        evidence_from_bundle(_bundle(tmp_path, platform_score=0.5))


def test_bundle_parser_rejects_event_from_a_different_rollout(tmp_path):
    with pytest.raises(CalibrationEvidenceError, match="packaged rollout metadata"):
        evidence_from_bundle(_bundle(tmp_path, retained_score=0.9))


def test_uncalibrated_artifact_is_authenticated_and_bound():
    artifact = uncalibrated_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        auth_key=KEY,
    )
    validate_artifact(
        artifact,
        KEY,
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
    )
    artifact["manifest_sha256"] = "d" * 64
    with pytest.raises(CalibrationEvidenceError, match="HMAC"):
        validate_artifact(
            artifact,
            KEY,
            manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64,
        )


def _evidence(index, *, headline, tail, commit="a" * 40,
              threshold=None, tail_threshold=None, executable_tuple=None):
    mastered = None if threshold is None else (
        headline >= threshold and tail >= tail_threshold)
    return {
        "run_id": f"run_{index}",
        "rollout_id": f"run_{index}-rollout-0",
        "environment_commit": commit,
        "model": FRONTIER_MODEL,
        "completed_at": "2026-07-18T00:00:00Z",
        "bundle_sha256": f"{index + 1:064x}",
        "benchmark_event_sha256": f"{index + 101:064x}",
        "benchmark_output_sha256": f"{index + 201:064x}",
        "native_headline": headline,
        "native_tail": tail,
        "observed_mastery_threshold": threshold,
        "observed_mastery_tail_threshold": tail_threshold,
        "observed_mastery": mastered,
        "release_status": "production",
        "scientific_tuple_sha256": calibration_module._scientific_tuple_sha256(
            "a" * 64, "b" * 64, "c" * 64),
        "executable_tuple_sha256": executable_tuple or (
            "d" * 64 if threshold is None else "e" * 64),
        "manifest_sha256": "a" * 64,
        "scoring_contract_sha256": "b" * 64,
        "wrapper_sha256": "c" * 64,
        "candidate_lock_sha256": "1" * 64 if threshold is None else "2" * 64,
        "attempt_classification": "gradable",
        "run_total_rollouts": 1,
        "run_rollout_ids": [f"run_{index}-rollout-0"],
    }


def _calibration_rows(*, offset=0):
    rows = [
        _evidence(offset, headline=0.4, tail=0.3),
        _evidence(offset + 1, headline=0.6, tail=0.2),
        _evidence(offset + 2, headline=0.5, tail=0.45),
    ]
    membership = [f"run_calibration-rollout-{index}" for index in range(3)]
    for index, row in enumerate(rows):
        row["run_id"] = "run_calibration"
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = 3
        row["run_rollout_ids"] = membership
    return rows


def _evaluation_rows(threshold, tail_threshold):
    rows = [
        _evidence(
            3, headline=0.55, tail=0.25, commit="b" * 40,
            threshold=threshold, tail_threshold=tail_threshold),
        _evidence(
            4, headline=0.61, tail=0.21, commit="b" * 40,
            threshold=threshold, tail_threshold=tail_threshold),
        _evidence(
            5, headline=0.50, tail=0.30, commit="b" * 40,
            threshold=threshold, tail_threshold=tail_threshold),
    ]
    membership = [f"run_evaluation-rollout-{index}" for index in range(len(rows))]
    for index, row in enumerate(rows):
        row["run_id"] = "run_evaluation"
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = len(rows)
        row["run_rollout_ids"] = membership
    return rows


def test_thresholds_are_derived_from_one_observed_frontier_row():
    rows = _calibration_rows()
    # Tail comes from the headline frontier row, not the independent 0.45 maximum.
    assert derive_thresholds(rows) == (0.6, 0.2)


def test_thresholds_reject_a_non_frontier_model():
    rows = _calibration_rows()
    rows[0]["model"] = "claude-haiku-4-5"
    with pytest.raises(CalibrationEvidenceError, match="values do not match schema"):
        derive_thresholds(rows)


def test_calibration_retains_complete_run_but_threshold_uses_gradable_only():
    gradable = _calibration_rows()[:2]
    membership = [f"run_calibration-rollout-{index}" for index in range(4)]
    for index, row in enumerate(gradable):
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = 4
        row["run_rollout_ids"] = membership
    calibration = gradable + [
        _failure_evidence(
            2, "agent_failure", total=4, run_id="run_calibration",
            commit="a" * 40),
        _failure_evidence(
            3, "infra_censored", total=4, run_id="run_calibration",
            commit="a" * 40),
    ]
    # Agent failure remains in the 3-attempt capability denominator; infra is
    # censored. Neither can invent a score or enter the frontier maximum.
    assert derive_thresholds(calibration) == (0.6, 0.2)


def test_calibration_rejects_omitted_launched_sibling():
    calibration = _calibration_rows()
    membership = [f"run_calibration-rollout-{index}" for index in range(4)]
    for index, row in enumerate(calibration):
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = 4
        row["run_rollout_ids"] = membership
    with pytest.raises(CalibrationEvidenceError, match="omits launched rollout"):
        derive_thresholds(calibration)


def test_calibration_rejects_success_only_retries_across_runs():
    calibration = _calibration_rows()
    calibration[2]["run_id"] = "run_retry"
    calibration[2]["rollout_id"] = "run_retry-rollout-0"
    calibration[2]["run_total_rollouts"] = 1
    calibration[2]["run_rollout_ids"] = ["run_retry-rollout-0"]
    with pytest.raises(CalibrationEvidenceError, match="one complete launched"):
        derive_thresholds(calibration)


def test_calibrated_artifact_requires_disjoint_post_freeze_evaluation(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    evaluation = _evaluation_rows(threshold, tail_threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = calibrated_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        calibration_rollouts=calibration,
        evaluation_rollouts=evaluation,
        auth_key=KEY,
    )
    validate_artifact(
        artifact, KEY, manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64,
    )

    with pytest.raises(CalibrationEvidenceError, match="evaluation"):
        calibrated_artifact(
            manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64,
            calibration_rollouts=calibration,
            evaluation_rollouts=[],
            auth_key=KEY,
        )

    with pytest.raises(CalibrationEvidenceError, match="omits launched rollout"):
        calibrated_artifact(
            manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64,
            calibration_rollouts=calibration,
            evaluation_rollouts=evaluation[:2],
            auth_key=KEY,
        )


def test_passk_report_is_derived_only_from_validated_evaluation_rows(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    evaluation = _evaluation_rows(threshold, tail_threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    import validation.passk as passk_module
    monkeypatch.setattr(passk_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(passk_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = calibrated_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        calibration_rollouts=calibration,
        evaluation_rollouts=evaluation,
        auth_key=KEY,
    )
    result = passk_report_from_artifact(
        artifact, KEY, manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64)
    assert result["source"] == "authenticated_mastery_calibration_evaluation"
    assert result["evaluation_rollout_ids"] == [row["rollout_id"] for row in evaluation]
    assert result["pass_at_k"][1]["n"] == 3
    assert result["pass_at_k"][1]["c"] == 1
    assert result["pass_at_k"][1]["pass_at_k"] == pytest.approx(1 / 3)


def test_passk_report_counts_agent_failures_and_censors_infra(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    gradable = _evaluation_rows(threshold, tail_threshold)[:2]
    membership = [f"run_eval-rollout-{index}" for index in range(4)]
    for index, row in enumerate(gradable):
        row["run_id"] = "run_eval"
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = 4
        row["run_rollout_ids"] = membership
    evaluation = gradable + [
        _failure_evidence(2, "agent_failure", total=4),
        _failure_evidence(3, "infra_censored", total=4),
    ]
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    import validation.passk as passk_module
    monkeypatch.setattr(passk_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(passk_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = calibrated_artifact(
        manifest_sha256="a" * 64, scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64, calibration_rollouts=calibration,
        evaluation_rollouts=evaluation, auth_key=KEY)
    result = passk_report_from_artifact(
        artifact, KEY, manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64)
    assert result["pass_at_k"][1]["n"] == 3
    assert result["pass_at_k"][1]["c"] == 1
    assert result["pass_at_k"][1]["infra_censored"] == 1
    assert result["evaluation_attempt_classification"] == {
        membership[0]: "gradable",
        membership[1]: "gradable",
        membership[2]: "agent_failure",
        membership[3]: "infra_censored",
    }


def test_evaluation_rejects_omitted_launched_attempt(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    evaluation = _evaluation_rows(threshold, tail_threshold)
    membership = [f"run_eval-rollout-{index}" for index in range(4)]
    for index, row in enumerate(evaluation):
        row["run_id"] = "run_eval"
        row["rollout_id"] = membership[index]
        row["run_total_rollouts"] = 4
        row["run_rollout_ids"] = membership
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    with pytest.raises(CalibrationEvidenceError, match="omits launched rollout"):
        calibrated_artifact(
            manifest_sha256="a" * 64, scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64, calibration_rollouts=calibration,
            evaluation_rollouts=evaluation, auth_key=KEY)


def test_all_infra_evaluation_has_no_capability_denominator(monkeypatch):
    calibration = _calibration_rows()
    evaluation = [
        _failure_evidence(index, "infra_censored") for index in range(3)]
    with pytest.raises(CalibrationEvidenceError, match="capability-denominator"):
        calibrated_artifact(
            manifest_sha256="a" * 64, scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64, calibration_rollouts=calibration,
            evaluation_rollouts=evaluation, auth_key=KEY)


def test_threshold_frozen_intermediate_allows_post_freeze_rollouts(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = threshold_frozen_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        calibration_rollouts=calibration,
        auth_key=KEY,
    )
    validate_artifact(
        artifact, KEY, manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64,
    )
    assert artifact["status"] == "threshold_frozen"
    assert artifact["evaluation_rollouts"] == []


def test_authenticated_artifact_rejects_missing_retained_bundle(monkeypatch, tmp_path):
    calibration = _calibration_rows(offset=10)
    threshold, tail_threshold = derive_thresholds(calibration)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    monkeypatch.setattr(calibration_module, "CALIBRATION_EVIDENCE_DIR", tmp_path)
    artifact = threshold_frozen_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        calibration_rollouts=calibration,
        auth_key=KEY,
    )
    with pytest.raises(CalibrationEvidenceError, match="bundle is missing"):
        validate_artifact(
            artifact, KEY, manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64,
        )


def test_evidence_rows_reject_extra_fields_even_with_valid_hmac(monkeypatch):
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    evaluation = _evaluation_rows(threshold, tail_threshold)
    evaluation[0]["untrusted_note"] = "ignored by old validator"
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    with pytest.raises(CalibrationEvidenceError, match="evidence fields"):
        calibrated_artifact(
            manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64,
            wrapper_sha256="c" * 64,
            calibration_rollouts=calibration,
            evaluation_rollouts=evaluation,
            auth_key=KEY,
        )


def test_calibration_phase_rejects_mixed_production_tuples(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    calibration[1]["executable_tuple_sha256"] = "f" * 64
    threshold, tail_threshold = derive_thresholds(calibration)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = threshold_frozen_artifact(
        manifest_sha256="a" * 64, scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64, calibration_rollouts=calibration, auth_key=KEY,
    )
    with pytest.raises(CalibrationEvidenceError, match="one exact production tuple"):
        validate_artifact(
            artifact, KEY, manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64)


def test_authenticated_artifact_cannot_choose_a_non_frontier_threshold(monkeypatch):
    monkeypatch.setattr(calibration_module, "_validate_retained_bundle", lambda row: None)
    calibration = _calibration_rows()
    threshold, tail_threshold = derive_thresholds(calibration)
    evaluation = _evaluation_rows(threshold, tail_threshold)
    monkeypatch.setattr(calibration_module, "MASTERY_THRESHOLD", threshold - 0.1)
    monkeypatch.setattr(calibration_module, "MASTERY_TAIL_THRESHOLD", tail_threshold)
    artifact = calibrated_artifact(
        manifest_sha256="a" * 64,
        scoring_contract_sha256="b" * 64,
        wrapper_sha256="c" * 64,
        calibration_rollouts=calibration,
        evaluation_rollouts=evaluation,
        auth_key=KEY,
    )
    artifact["mastery_threshold"] = threshold - 0.1
    artifact = sign_artifact(artifact, KEY)
    with pytest.raises(CalibrationEvidenceError, match="frozen derivation"):
        validate_artifact(
            artifact, KEY, manifest_sha256="a" * 64,
            scoring_contract_sha256="b" * 64, wrapper_sha256="c" * 64,
        )
