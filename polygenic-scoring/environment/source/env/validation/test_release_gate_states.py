from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from grader.corpus_auth import verify_json_hmac
import validation.release_gate as gate


KEY = bytes(range(32))
TUPLE = "a" * 64


def _state() -> dict:
    return {
        "corpus_schema_version": 8,
        "prompt_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "development_report_sha256": "3" * 64,
        "shipping_audit_sha256": "4" * 64,
        "mastery_calibration_sha256": "5" * 64,
        "scoring_contract_sha256": "6" * 64,
        "wrapper_sha256": "7" * 64,
        "scientific_tuple_sha256": "9" * 64,
        "executable_tuple_sha256": TUPLE,
        "score1_compatibility_sha256": "b" * 64,
        "dataset_count": 45,
    }


def test_candidate_lock_is_explicit_and_has_no_circular_witness_dependency(
    tmp_path, monkeypatch,
):
    lock_path = tmp_path / "release_lock.json"
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(
        gate, "_artifact_state",
        lambda _key, *, require_witness, expected_executable_tuple_sha256=None:
        (_state() if not require_witness and expected_executable_tuple_sha256 is None
         else (_ for _ in ()).throw(AssertionError("candidate requested witness"))),
    )
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})

    gate.write_candidate_lock(KEY)
    lock = json.loads(lock_path.read_text())
    assert lock["release_status"] == "candidate"
    assert lock["executable_tuple_sha256"] == TUPLE
    assert "score1_witness_sha256" not in lock
    assert verify_json_hmac(
        KEY, lock, field=gate.LOCK_HMAC_FIELD, domain=gate.LOCK_HMAC_DOMAIN)


def test_deployed_candidate_preflight_accepts_exact_tuple_without_witness(
    tmp_path, monkeypatch,
):
    lock_path = tmp_path / "release_lock.json"
    candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **_state(), "files": {},
    }, KEY)
    lock_path.write_text(json.dumps(candidate) + "\n")
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})
    monkeypatch.setattr(gate, "_executable_tuple_sha256", lambda: TUPLE)
    monkeypatch.setattr(
        gate, "_score1_compatibility_sha256",
        lambda: candidate["score1_compatibility_sha256"],
    )
    monkeypatch.setattr(
        gate, "_scientific_tuple_sha256",
        lambda manifest, scoring, wrapper: candidate["scientific_tuple_sha256"],
    )
    original_read_json = gate._read_json
    monkeypatch.setattr(gate, "_read_json", lambda path: (
        (candidate, "0" * 64) if path == lock_path
        else ({}, candidate["manifest_sha256"])
        if path == gate.ROOT / "corpus" / "manifest.json"
        else ({}, candidate["mastery_calibration_sha256"])
        if path == gate.CALIBRATION_PATH
        else original_read_json(path)
    ))
    monkeypatch.setattr(gate, "validate_artifact", lambda *args, **kwargs: None)
    gate.validate_deployed_lock(KEY)


def test_production_transition_requires_the_identical_candidate_tuple(
    tmp_path, monkeypatch,
):
    lock_path = tmp_path / "release_lock.json"
    candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench",
        "release_status": "candidate",
        **_state(),
        "files": {},
    }, KEY)
    lock_path.write_text(json.dumps(candidate) + "\n")
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(gate, "validate_deployed_lock", lambda _key: None)

    def artifact_state(_key, *, require_witness, expected_executable_tuple_sha256=None):
        assert require_witness is True
        assert expected_executable_tuple_sha256 == TUPLE
        return {
            **_state(),
            "score1_witness_sha256": "8" * 64,
            "score1_witness_id": "score1-execution",
        }

    monkeypatch.setattr(gate, "_artifact_state", artifact_state)
    executed_candidate_bytes = (json.dumps(candidate) + "\n").encode()
    witness = {
        "evidence_kind": "hfdev_full_bundle",
        "witness_id": "score1-execution",
        "execution_id": "score1-execution",
        "executed_environment_commit": "e" * 40,
        "candidate_lock_sha256": gate._sha256_bytes(executed_candidate_bytes),
        "scientific_tuple_sha256": candidate["scientific_tuple_sha256"],
        "witnessed_executable_tuple_sha256": TUPLE,
    }
    monkeypatch.setattr(gate, "_read_json", lambda path: (witness, "8" * 64))
    monkeypatch.setattr(
        gate.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=executed_candidate_bytes, stderr=b""),
    )
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})

    gate.write_production_lock(KEY)
    production = json.loads(lock_path.read_text())
    assert production["release_status"] == "production"
    assert production["candidate_executable_tuple_sha256"] == TUPLE
    assert production["executable_tuple_sha256"] == TUPLE
    assert production["candidate_lock_sha256"] == gate._sha256_bytes(
        (json.dumps(candidate) + "\n").encode())
    assert verify_json_hmac(
        KEY, production, field=gate.LOCK_HMAC_FIELD, domain=gate.LOCK_HMAC_DOMAIN)


def test_gold_solvability_proof_makes_no_execution_claim(tmp_path, monkeypatch):
    lock_path = tmp_path / "release_lock.json"
    candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **_state(), "files": {},
    }, KEY)
    candidate_bytes = (json.dumps(candidate) + "\n").encode()
    lock_path.write_bytes(candidate_bytes)
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(gate, "validate_deployed_lock", lambda _key: None)
    monkeypatch.setattr(gate, "_artifact_state", lambda *args, **kwargs: {
        **_state(), "score1_witness_sha256": "8" * 64,
        "score1_witness_id": "public-gold-v1",
    })
    witness = {
        "evidence_kind": "gold_execution_bundle",
        "witness_id": "public-gold-v1",
        "candidate_lock_sha256": gate._sha256_bytes(candidate_bytes),
        "scientific_tuple_sha256": candidate["scientific_tuple_sha256"],
        "gold_executable_tuple_sha256": TUPLE,
    }
    monkeypatch.setattr(gate, "_read_json", lambda path: (witness, "8" * 64))
    monkeypatch.setattr(
        gate.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("gold proof must not invoke git-show execution provenance")),
    )
    monkeypatch.setattr(
        gate, "_gold_candidate_lock_bytes", lambda _witness: candidate_bytes)
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})
    gate.write_production_lock(KEY)
    assert json.loads(lock_path.read_text())["score1_witness_id"] == "public-gold-v1"


def test_gold_witness_reuse_accepts_only_a_mastery_overlay(tmp_path, monkeypatch):
    lock_path = tmp_path / "release_lock.json"
    witnessed_candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **_state(), "files": {},
    }, KEY)
    witnessed_bytes = (json.dumps(witnessed_candidate) + "\n").encode()

    frozen_tuple = "c" * 64
    frozen_state = {
        **_state(),
        "mastery_calibration_sha256": "d" * 64,
        "executable_tuple_sha256": frozen_tuple,
    }
    frozen_candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **frozen_state, "files": {},
    }, KEY)
    lock_path.write_text(json.dumps(frozen_candidate) + "\n")

    witness = {
        "evidence_kind": "gold_execution_bundle",
        "witness_id": "public-gold-v1",
        "candidate_lock_sha256": gate._sha256_bytes(witnessed_bytes),
        "scientific_tuple_sha256": witnessed_candidate["scientific_tuple_sha256"],
        "gold_executable_tuple_sha256": TUPLE,
    }
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(gate, "validate_deployed_lock", lambda _key: None)

    def artifact_state(_key, *, require_witness, expected_executable_tuple_sha256=None):
        assert require_witness is True
        assert expected_executable_tuple_sha256 == frozen_tuple
        return {
            **frozen_state, "score1_witness_sha256": "8" * 64,
            "score1_witness_id": "public-gold-v1",
        }

    monkeypatch.setattr(gate, "_artifact_state", artifact_state)
    monkeypatch.setattr(gate, "_read_json", lambda path: (witness, "8" * 64))
    monkeypatch.setattr(
        gate, "_gold_candidate_lock_bytes", lambda _witness: witnessed_bytes)
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})

    gate.write_production_lock(KEY)
    production = json.loads(lock_path.read_text())
    assert production["release_status"] == "production"
    assert production["candidate_executable_tuple_sha256"] == frozen_tuple
    assert production["score1_witness_id"] == "public-gold-v1"


def test_gold_witness_reuse_rejects_non_mastery_drift(tmp_path, monkeypatch):
    lock_path = tmp_path / "release_lock.json"
    witnessed_candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **_state(), "files": {},
    }, KEY)
    witnessed_bytes = (json.dumps(witnessed_candidate) + "\n").encode()
    drifted_state = {
        **_state(),
        "executable_tuple_sha256": "c" * 64,
        "score1_compatibility_sha256": "d" * 64,
    }
    current = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **drifted_state, "files": {},
    }, KEY)
    lock_path.write_text(json.dumps(current) + "\n")
    witness = {
        "evidence_kind": "gold_execution_bundle",
        "witness_id": "public-gold-v1",
        "candidate_lock_sha256": gate._sha256_bytes(witnessed_bytes),
        "scientific_tuple_sha256": witnessed_candidate["scientific_tuple_sha256"],
        "gold_executable_tuple_sha256": TUPLE,
    }
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(gate, "validate_deployed_lock", lambda _key: None)
    monkeypatch.setattr(gate, "_artifact_state", lambda *args, **kwargs: {
        **drifted_state, "score1_witness_sha256": "8" * 64,
        "score1_witness_id": "public-gold-v1",
    })
    monkeypatch.setattr(gate, "_read_json", lambda path: (witness, "8" * 64))
    monkeypatch.setattr(
        gate, "_gold_candidate_lock_bytes", lambda _witness: witnessed_bytes)
    monkeypatch.setattr(gate, "_locked_paths", lambda status, witness=None: [])
    monkeypatch.setattr(gate, "_file_hashes", lambda paths: {})

    with pytest.raises(gate.ReleaseGateError, match="provenance"):
        gate.write_production_lock(KEY)


def test_production_rejects_witness_reuse_after_non_mastery_tuple_drift(
    tmp_path, monkeypatch,
):
    lock_path = tmp_path / "release_lock.json"
    candidate = gate._signed_lock({
        "schema_version": gate.LOCK_SCHEMA_VERSION,
        "environment_id": "sc-svpgsbench", "release_status": "candidate",
        **_state(), "files": {},
    }, KEY)
    lock_path.write_text(json.dumps(candidate) + "\n")
    monkeypatch.setattr(gate, "LOCK_PATH", lock_path)
    monkeypatch.setattr(gate, "_git_state", lambda: ("main", ""))
    monkeypatch.setattr(gate, "validate_deployed_lock", lambda _key: None)
    monkeypatch.setattr(gate, "_artifact_state", lambda *args, **kwargs: {
        **_state(), "score1_witness_sha256": "8" * 64,
        "score1_witness_id": "score1-execution",
    })
    executed = gate._signed_lock({
        **candidate,
        "score1_compatibility_sha256": "c" * 64,
    }, KEY)
    executed_bytes = (json.dumps(executed) + "\n").encode()
    witness = {
        "evidence_kind": "hfdev_full_bundle",
        "witness_id": "score1-execution",
        "execution_id": "score1-execution",
        "executed_environment_commit": "e" * 40,
        "candidate_lock_sha256": gate._sha256_bytes(executed_bytes),
        "scientific_tuple_sha256": candidate["scientific_tuple_sha256"],
        "witnessed_executable_tuple_sha256": candidate["executable_tuple_sha256"],
    }
    monkeypatch.setattr(gate, "_read_json", lambda path: (witness, "8" * 64))
    monkeypatch.setattr(
        gate.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=executed_bytes, stderr=b""),
    )
    with pytest.raises(gate.ReleaseGateError, match="provenance"):
        gate.write_production_lock(KEY)


def test_score1_compatibility_normalizes_only_the_mastery_overlay(
    tmp_path, monkeypatch,
):
    contract = tmp_path / "grader" / "contract.py"
    calibration = tmp_path / "validation" / "mastery_calibration.json"
    runtime = tmp_path / "grader" / "grade.py"
    contract.parent.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    contract.write_text(
        "MASTERY_THRESHOLD = None\nMASTERY_TAIL_THRESHOLD = None\nVALUE = 1\n")
    calibration.write_text('{"status":"uncalibrated"}\n')
    runtime.write_text("REWARD = 1\n")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(
        gate, "_base_locked_paths", lambda: [contract, calibration, runtime])

    initial = gate._score1_compatibility_sha256()
    contract.write_text(
        "MASTERY_THRESHOLD = 0.4\nMASTERY_TAIL_THRESHOLD = 0.2\nVALUE = 1\n")
    calibration.write_text('{"status":"calibrated","evidence":[1]}\n')
    assert gate._score1_compatibility_sha256() == initial

    runtime.write_text("REWARD = 0\n")
    assert gate._score1_compatibility_sha256() != initial
