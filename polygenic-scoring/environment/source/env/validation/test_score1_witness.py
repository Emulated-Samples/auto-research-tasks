from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import pytest

from grader.contract import CORPUS_SCHEMA_VERSION, SHIPPED_CATEGORIES
from grader.corpus_auth import json_hmac_sha256
from grader.metrics import compute_metrics
from grader.skill import category_aggregation
import validation.score1_witness as witness_module
from validation.score1_witness import (
    SCORE_ONE_TOLERANCE,
    ScoreOneWitnessError,
    artifact_from_bundle,
    validate_artifact,
)

KEY = bytes(range(32))
MANIFEST_SHA = "a" * 64
SCORING_SHA = "b" * 64
WRAPPER_SHA = "c" * 64
TUPLE_SHA = "d" * 64
EXECUTABLE_SHA = "8" * 64
CANDIDATE_LOCK_SHA = "9" * 64
RUN_ID = "run_score1"
ROLLOUT_ID = f"{RUN_ID}-rollout-0"


def _manifest() -> dict:
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "datasets": [
            {
                "id": f"d_{category}_{replicate}",
                "category": category,
                "replicate": replicate,
                "weight": 1.0,
                "path": f"{category}/d_{category}_{replicate}",
            }
            for category in SHIPPED_CATEGORIES
            for replicate in range(3)
        ],
    }


def _result(test_id: str, name: str, score: float, output: dict, weight: float) -> dict:
    return {
        "testId": test_id,
        "name": name,
        "description": "derived test evidence",
        "status": "partially_passed",
        "duration": 0 if test_id == "benchmark" else 100,
        "score": score,
        "weight": weight,
        "output": json.dumps(output, separators=(",", ":")),
    }


def _bundle(tmp_path: Path, *, instance: str = "t3.xlarge",
            failed_dataset: bool = False, incomplete: bool = False,
            native_reward: float = 1.0) -> Path:
    manifest = _manifest()
    aggregate_rows = [
        {"category": entry["category"], "weight": entry["weight"], "reward": native_reward}
        for entry in manifest["datasets"]
    ]
    headline, _, aggregation = category_aggregation(aggregate_rows)
    platform = (headline + 0.5) / 1.5
    events = []
    emitted = []
    for category in SHIPPED_CATEGORIES:
        datasets = []
        for entry in manifest["datasets"]:
            if entry["category"] != category:
                continue
            reward = (0.5 if failed_dataset and not datasets
                      and category == SHIPPED_CATEGORIES[0] else native_reward)
            datasets.append({
                "dataset": entry["id"], "status": "ok", "reward": reward,
                "skill": reward, "accuracy": 1.0, "performance": 1.0,
                "fit_seconds": 1.0, "predict_seconds": 1.0,
            })
        native = sum(row["reward"] for row in datasets) / len(datasets)
        score = (native + 0.5) / 1.5
        result = _result(
            f"category:{category}", f"{category} category", score,
            {"native_score": native, "platform_score": score,
             "contract_ok": True, "datasets": datasets},
            aggregation["category_coefficients"][category],
        )
        events.append({
            "timestamp": "2026-07-18T18:00:00Z", "type": "test_result",
            "category": "tests", "message": "category", "data": result,
        })
        emitted.append({
            "testId": result["testId"], "testName": result["name"],
            "status": result["status"], "durationMs": result["duration"],
            "score": result["score"],
        })
    per_category = {
        category: json.loads(events[index]["data"]["output"])["native_score"]
        for index, category in enumerate(SHIPPED_CATEGORIES)
    }
    benchmark = _result(
        "benchmark", "svpgsbench headline reward", platform,
        {
            "native_score": headline, "platform_score": platform,
            "integrity_ok": True, "contract_ok": True, "mastery": None,
            "mastery_threshold": None, "mastery_tail_threshold": None,
            "mastery_tail_value": aggregation["tail_value"],
            "per_category": per_category,
            "release_status": "candidate",
            "scientific_tuple_sha256": TUPLE_SHA,
            "executable_tuple_sha256": EXECUTABLE_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "scoring_contract_sha256": SCORING_SHA,
            "wrapper_sha256": WRAPPER_SHA,
            "candidate_lock_sha256": CANDIDATE_LOCK_SHA,
        },
        1,
    )
    events.append({
        "timestamp": "2026-07-18T18:00:00Z", "type": "test_result",
        "category": "tests", "message": "benchmark", "data": benchmark,
    })
    emitted.append({
        "testId": "benchmark", "testName": benchmark["name"],
        "status": benchmark["status"], "durationMs": 0, "score": platform,
    })
    run = {
        "runId": RUN_ID,
        "environmentId": "sc-svpgsbench",
        "instanceType": instance,
        "compute": {"cloud": "aws", "instanceType": instance},
        "commitSha": "e" * 40,
        "rollouts": [{
            "id": ROLLOUT_ID, "currentPhase": "complete",
            "completedAt": "2026-07-18T18:00:01Z", "testResults": emitted,
        }],
    }
    path = tmp_path / f"{RUN_ID}-rollout-0.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("run.json", json.dumps(run))
        archive.writestr("logs/tests.jsonl", "\n".join(json.dumps(row) for row in events) + "\n")
        archive.writestr("logs/agent.jsonl", "{}\n")
        archive.writestr("logs/setup.jsonl", "{}\n")
        archive.writestr("transcript.txt", "complete transcript\n")
        archive.writestr("task_prompt.md", "task\n")
        archive.writestr("solution.diff.json", "{}\n")
        archive.writestr("summary.txt", "summary\n")
        archive.writestr("solution/fit", "#!/bin/sh\n")
        archive.writestr("solution/predict", "#!/bin/sh\n")
        archive.writestr("solution/model.py", "# reference witness\n")
        if incomplete:
            archive.writestr("PACKAGE_INCOMPLETE.txt", "source lost\n")
    return path


def _gold_execution_bundle(
    tmp_path: Path,
    *,
    supplied_prediction: bool = False,
    claimed_reward: bool = False,
) -> Path:
    manifest = _manifest()
    candidate = {
        "schema_version": 4, "environment_id": "sc-svpgsbench",
        "release_status": "candidate", "manifest_sha256": MANIFEST_SHA,
        "scoring_contract_sha256": SCORING_SHA, "wrapper_sha256": WRAPPER_SHA,
        "scientific_tuple_sha256": TUPLE_SHA,
        "executable_tuple_sha256": EXECUTABLE_SHA,
        "score1_compatibility_sha256": "7" * 64,
    }
    candidate["release_lock_hmac_sha256"] = json_hmac_sha256(
        KEY, candidate, exclude_field="release_lock_hmac_sha256",
        domain=b"svpgsbench|schema-v8|release-lock\x00")
    candidate_bytes = (json.dumps(candidate) + "\n").encode()
    replay = {
        "schema_version": 1, "environment_id": "sc-svpgsbench",
        "gold_id": "public-reference-gold-v1",
        "candidate_lock_sha256": witness_module._sha256(candidate_bytes),
    }
    path = tmp_path / "public-score1-gold.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("candidate_release_lock.json", candidate_bytes)
        archive.writestr("gold_execution.json", json.dumps(replay))
        archive.writestr("submission/fit", "#!/bin/sh\n")
        archive.writestr("submission/predict", "#!/bin/sh\n")
        archive.writestr("submission/pgs_core.py", "# public-only gold core\n")
        if supplied_prediction:
            entry = manifest["datasets"][0]
            means = [0.1, 0.9, 0.2, 0.8]
            prediction = "sample_id,mean\n" + "".join(
                f"s{row},{value}\n" for row, value in enumerate(means))
            archive.writestr(f"predictions/{entry['id']}/pred.csv", prediction)
        if claimed_reward:
            archive.writestr("reward_detail.json", json.dumps({"reward": 1.0}))
    return path


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    evidence = root / "validation" / "score1_evidence"
    monkeypatch.setattr(witness_module, "ROOT", root)
    monkeypatch.setattr(witness_module, "EVIDENCE_DIR", evidence)
    targets = [0.0, 1.0, 0.0, 1.0]
    reference_prediction = {"mean": [0.1, 0.9, 0.2, 0.8]}
    naive_prediction = {"mean": [0.5, 0.5, 0.5, 0.5]}
    gold = root / "gold"
    gold.mkdir(parents=True)
    gold.joinpath("fit").write_text("#!/bin/sh\n")
    gold.joinpath("predict").write_text("#!/bin/sh\n")
    gold.joinpath("pgs_core.py").write_text("# public-only gold core\n")
    reference_metrics = compute_metrics("binomial-logit", targets, reference_prediction)
    naive_metrics = compute_metrics("binomial-logit", targets, naive_prediction)
    for entry in _manifest()["datasets"]:
        dataset = root / "corpus" / entry["path"]
        truth = dataset / "truth"
        truth.mkdir(parents=True)
        (dataset / "public").mkdir()
        truth.joinpath("y_test.csv").write_text(
            "sample_id,y\n" + "".join(
                f"s{row},{int(value)}\n" for row, value in enumerate(targets)))
        truth.joinpath("anchors.json").write_text(json.dumps({
            "family": "binomial-logit",
            "metrics_naive": {name: value[0] for name, value in naive_metrics.items()},
            "metrics_reference": {
                name: value[0] for name, value in reference_metrics.items()},
            "metrics_ref_naive_se": {
                "auc": 0.01, "brier": 0.01, "log_loss": 0.01},
            "metrics_naive_se": {
                "auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        }))
    prediction_bytes = (
        b"sample_id,mean\n"
        b"s0,0.10000000000000001\n"
        b"s1,0.90000000000000002\n"
        b"s2,0.20000000000000001\n"
        b"s3,0.80000000000000004\n"
    )
    monkeypatch.setattr(
        witness_module, "build_submission",
        lambda source, log: (True, source, ""))
    monkeypatch.setattr(witness_module, "_rmtree_ro", lambda _path: None)
    monkeypatch.setattr(
        witness_module, "run_on_dataset",
        lambda *args, **kwargs: {
            "status": "ok",
            "pred_bytes": prediction_bytes,
            "t_fit": 1.25,
            "t_predict": 0.25,
            "fit_rc": 0,
            "predict_rc": 0,
            "model_sha256": "1" * 64,
            "pred_sha256": witness_module._sha256(prediction_bytes),
            "detail": "",
        })
    return root


def _artifact(tmp_path: Path, evidence_root: Path, **bundle_options) -> dict:
    return artifact_from_bundle(
        _bundle(tmp_path, **bundle_options),
        manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
        scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
        scientific_tuple_sha256=TUPLE_SHA, auth_key=KEY,
    )


def _validate(artifact: dict) -> None:
    validate_artifact(
        artifact, KEY, manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
        scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
        scientific_tuple_sha256=TUPLE_SHA,
    )


def test_full_target_host_bundle_is_copied_reparsed_and_authenticated(tmp_path, evidence_root):
    artifact = _artifact(tmp_path, evidence_root)
    _validate(artifact)
    retained = evidence_root / artifact["evidence_bundle"]["path"]
    assert retained.is_file()
    assert artifact["dataset_count"] == 45
    assert artifact["executed_environment_commit"] == "e" * 40
    assert artifact["target_instance_type"] == "t3.xlarge"
    assert artifact["native_headline"] == 1.0
    assert artifact["score_one_tolerance"] == SCORE_ONE_TOLERANCE
    assert {row["path"] for row in artifact["evidence_members"]} >= {
        "run.json", "logs/tests.jsonl", "logs/agent.jsonl", "solution/fit",
    }


def test_score_one_predicate_accepts_stronger_than_reference(tmp_path, evidence_root):
    artifact = _artifact(tmp_path, evidence_root, native_reward=1.1)
    _validate(artifact)
    assert artifact["native_headline"] > 1.0


def test_gold_execution_causally_recomputes_score_one(tmp_path, evidence_root):
    artifact = artifact_from_bundle(
        _gold_execution_bundle(tmp_path),
        manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
        scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
        scientific_tuple_sha256=TUPLE_SHA, auth_key=KEY,
    )
    _validate(artifact)
    assert artifact["evidence_kind"] == "gold_execution_bundle"
    assert artifact["witness_id"] == "public-reference-gold-v1"
    assert "execution_id" not in artifact
    assert "target_instance_type" not in artifact
    with zipfile.ZipFile(evidence_root / artifact["evidence_bundle"]["path"]) as archive:
        candidate_bytes = archive.read("candidate_release_lock.json")
    assert artifact["candidate_lock_sha256"] == witness_module._sha256(candidate_bytes)


def test_gold_execution_rejects_any_publisher_supplied_prediction(
    tmp_path, evidence_root,
):
    with pytest.raises(ScoreOneWitnessError, match="must not contain publisher-supplied"):
        artifact_from_bundle(
            _gold_execution_bundle(tmp_path, supplied_prediction=True),
            manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
            scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
            scientific_tuple_sha256=TUPLE_SHA, auth_key=KEY,
        )


def test_gold_execution_validation_reexecutes_and_rejects_output_hash_drift(
    tmp_path, evidence_root, monkeypatch,
):
    artifact = artifact_from_bundle(
        _gold_execution_bundle(tmp_path),
        manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
        scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
        scientific_tuple_sha256=TUPLE_SHA, auth_key=KEY,
    )
    original = witness_module.run_on_dataset

    def drifted(*args, **kwargs):
        result = dict(original(*args, **kwargs))
        result["pred_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(witness_module, "run_on_dataset", drifted)
    with pytest.raises(ScoreOneWitnessError, match="does not reproduce"):
        _validate(artifact)


def test_gold_execution_rejects_publisher_authored_reward_claim(
    tmp_path, evidence_root,
):
    with pytest.raises(ScoreOneWitnessError, match="must not carry"):
        artifact_from_bundle(
            _gold_execution_bundle(tmp_path, claimed_reward=True),
            manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
            scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
            scientific_tuple_sha256=TUPLE_SHA, auth_key=KEY,
        )


def test_rejects_wrong_target_host_derived_from_run_metadata(tmp_path, evidence_root):
    with pytest.raises(ScoreOneWitnessError, match="target instance"):
        _artifact(tmp_path, evidence_root, instance="r5.2xlarge")


def test_rejects_retained_only_bundle_or_missing_source(tmp_path, evidence_root):
    with pytest.raises(ScoreOneWitnessError, match="full, not retained-only"):
        _artifact(tmp_path, evidence_root, incomplete=True)


def test_rejects_non_reference_grade_dataset_from_category_log(tmp_path, evidence_root):
    with pytest.raises(ScoreOneWitnessError, match="score-1 tolerance"):
        _artifact(tmp_path, evidence_root, failed_dataset=True)


def test_reparse_detects_retained_bundle_byte_tampering(tmp_path, evidence_root):
    artifact = _artifact(tmp_path, evidence_root)
    retained = evidence_root / artifact["evidence_bundle"]["path"]
    retained.write_bytes(retained.read_bytes() + b"tamper")
    with pytest.raises(ScoreOneWitnessError, match="bytes do not match"):
        _validate(artifact)


def test_hmac_detects_artifact_tampering(tmp_path, evidence_root):
    artifact = _artifact(tmp_path, evidence_root)
    tampered = copy.deepcopy(artifact)
    tampered["native_headline"] = 0.5
    with pytest.raises(ScoreOneWitnessError, match="HMAC"):
        _validate(tampered)


def test_witness_is_bound_to_candidate_executable_tuple(tmp_path, evidence_root):
    artifact = _artifact(tmp_path, evidence_root)
    with pytest.raises(ScoreOneWitnessError, match="release provenance"):
        validate_artifact(
            artifact, KEY, manifest=_manifest(), manifest_sha256=MANIFEST_SHA,
            scoring_contract_sha256=SCORING_SHA, wrapper_contract_sha256=WRAPPER_SHA,
            scientific_tuple_sha256="0" * 64,
        )
