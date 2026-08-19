"""Fixed, non-transferable grading caps and fail-closed completion.

There is no global or category time bank. Build, fit, and predict each own an
independent cap, so manifest order cannot move time from one dataset to another.
The outer wrapper is only a hard interruption boundary: it never turns an
incomplete partial snapshot into a trusted score.
"""
from __future__ import annotations

import inspect
import itertools
import json
import os
from pathlib import Path

import pytest

from grader import grade
from grader import submission_runner as runner
from grader.contract import (
    BUILD_TIMEOUT_S,
    DATASET_TIMEOUT_S,
    FIT_TIMEOUT_S,
    PLATFORM_VERIFIER_TIMEOUT_S,
    PREDICT_TIMEOUT_S,
    REPLICATES_PER_CATEGORY,
    SHIPPED_CATEGORY_COUNT,
    SHIPPED_DATASET_COUNT,
    TRUSTED_OVERHEAD_RESERVE_S,
    WRAPPER_GRADER_TIMEOUT_S,
)
from grader.skill import INVALID_REWARD


CATEGORIES = ("alpha", "beta", "gamma")
REPLICATES = (0, 1)
FAST_SECONDS = 10.0
SLOW_DATASET = "beta__r1"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _entries(category_order: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "id": f"{category}__r{replicate}",
            "path": f"{category}/{category}__r{replicate}",
            "category": category,
            "replicate": replicate,
            "family": "binomial-logit",
            "weight": 1.0,
        }
        for category in category_order
        for replicate in REPLICATES
    ]


def _ok_record(dataset_id: str, category: str) -> tuple[dict, dict]:
    aggregate = {"category": category, "weight": 1.0, "reward": 1.0}
    detail = {
        "dataset": dataset_id,
        "category": category,
        "weight": 1.0,
        "status": "ok",
        "reward": 1.0,
        "accuracy": 1.0,
        "raw_skill": 1.0,
        "perf": 1.0,
        "t_fit": FAST_SECONDS,
        "t_predict": 0.0,
    }
    return aggregate, detail


def _timeout_record(dataset_id: str, category: str) -> tuple[dict, dict]:
    aggregate = {"category": category, "weight": 1.0, "reward": INVALID_REWARD}
    detail = {
        "dataset": dataset_id,
        "category": category,
        "weight": 1.0,
        "status": "fit_failed",
        "reward": INVALID_REWARD,
        "accuracy": 0.0,
        "raw_skill": INVALID_REWARD,
        "perf": 0.0,
        "t_fit": FIT_TIMEOUT_S,
        "t_predict": 0.0,
        "note": "rc=124 TIMEOUT",
    }
    return aggregate, detail


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category_order: tuple[str, ...],
    *,
    build_seconds: float = 0.0,
) -> tuple[dict, dict, _FakeClock, list[str]]:
    entries = _entries(category_order)
    clock = _FakeClock()
    calls: list[str] = []

    def fake_build(*_args, **_kwargs):
        clock.advance(build_seconds)
        return True, "/built", ""

    def fake_grade_one(dataset_dir, _submission, *, grade_calibration):
        assert grade_calibration == 1.0
        dataset_id = os.path.basename(dataset_dir)
        category = os.path.basename(os.path.dirname(dataset_dir))
        calls.append(dataset_id)
        if dataset_id == SLOW_DATASET:
            clock.advance(FIT_TIMEOUT_S)
            return _timeout_record(dataset_id, category)
        clock.advance(FAST_SECONDS)
        return _ok_record(dataset_id, category)

    monkeypatch.setattr(grade, "load_corpus_key", lambda *_a, **_k: b"x" * 32)
    monkeypatch.setattr(
        grade,
        "_validate_corpus",
        lambda *_args: [str(entry["id"]) for entry in entries],
    )
    monkeypatch.setattr(grade, "calibration_seconds", lambda: 1.0)
    monkeypatch.setattr(grade, "build_submission", fake_build)
    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(grade, "_rmtree_ro", lambda _path: None)
    monkeypatch.setenv("SVPGSBENCH_GRADER_NONCE", "b" * 32)

    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps({"datasets": entries, "meta": {"n_datasets": len(entries)}}),
        encoding="utf-8",
    )
    reward, detail = grade.grade_corpus(
        str(corpus),
        "/submission",
        key_file="/private/test-corpus.key",
        out_dir=str(out),
        log=lambda *_args: None,
    )
    return reward, detail, clock, calls


def test_timeout_contract_algebra_is_exact() -> None:
    assert BUILD_TIMEOUT_S == runner.BUILD_TIMEOUT_S == 3600
    assert DATASET_TIMEOUT_S == 200
    assert FIT_TIMEOUT_S == runner.FIT_TIMEOUT_S == 170
    assert PREDICT_TIMEOUT_S == runner.PREDICT_TIMEOUT_S == 30
    assert FIT_TIMEOUT_S + PREDICT_TIMEOUT_S == DATASET_TIMEOUT_S
    assert REPLICATES_PER_CATEGORY == 3
    # 15 shipped categories x 3 replicates == 45 datasets. The wrapper timeout
    # was re-derived for the 45-dataset grid below.
    assert SHIPPED_CATEGORY_COUNT == 15
    assert SHIPPED_DATASET_COUNT == 45
    assert SHIPPED_CATEGORY_COUNT * REPLICATES_PER_CATEGORY == SHIPPED_DATASET_COUNT
    assert (
        BUILD_TIMEOUT_S
        + SHIPPED_DATASET_COUNT * DATASET_TIMEOUT_S
        + TRUSTED_OVERHEAD_RESERVE_S
        == WRAPPER_GRADER_TIMEOUT_S
        == 14_400
    )
    assert (
        WRAPPER_GRADER_TIMEOUT_S + TRUSTED_OVERHEAD_RESERVE_S
        == PLATFORM_VERIFIER_TIMEOUT_S
        == 16_200
    )


def test_packaged_outer_timeouts_match_python_contract() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    wrapper_source = (repository_root / "environment/src/index.ts").read_text(
        encoding="utf-8"
    )
    task_toml = (
        repository_root / "tasks/from_scratch_svpgs/task.toml"
    ).read_text(encoding="utf-8")
    expected_wrapper = (
        f"const WRAPPER_GRADER_TIMEOUT_MS = {WRAPPER_GRADER_TIMEOUT_S:_} * 1000;"
    )
    assert expected_wrapper in wrapper_source
    assert "timeout: WRAPPER_GRADER_TIMEOUT_MS" in wrapper_source
    assert f"timeout_sec = {PLATFORM_VERIFIER_TIMEOUT_S}" in task_toml


def test_budget_override_and_time_bank_apis_do_not_exist() -> None:
    assert "budget_s" not in inspect.signature(grade.grade_one).parameters
    assert "budget_s" not in inspect.signature(grade.grade_corpus).parameters
    assert "fit_timeout" not in inspect.signature(runner.run_on_dataset).parameters
    assert "predict_timeout" not in inspect.signature(runner.run_on_dataset).parameters
    for dead_name in (
        "DEFAULT_GRADE_BUDGET_S",
        "MIN_DATASET_BUDGET_S",
        "BUDGET_ALLOCATION",
        "_round_robin",
    ):
        assert not hasattr(grade, dead_name)


def test_manifest_permutations_do_not_change_rewards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headlines = set()
    for index, category_order in enumerate(itertools.permutations(CATEGORIES)):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        reward, detail, _, calls = _run(run_dir, monkeypatch, category_order)
        headlines.add(round(reward["reward"], 12))
        expected_order = [str(entry["id"]) for entry in _entries(category_order)]
        assert calls == expected_order
        assert [row["dataset"] for row in detail["datasets"]] == expected_order
        rewards = {row["dataset"]: row["reward"] for row in detail["datasets"]}
        assert rewards[SLOW_DATASET] == INVALID_REWARD
        assert all(
            value == 1.0
            for dataset_id, value in rewards.items()
            if dataset_id != SLOW_DATASET
        )
    assert len(headlines) == 1


def test_near_cap_build_does_not_shrink_any_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, detail, clock, _ = _run(
        tmp_path,
        monkeypatch,
        CATEGORIES,
        build_seconds=BUILD_TIMEOUT_S - 0.001,
    )
    slow = next(row for row in detail["datasets"] if row["dataset"] == SLOW_DATASET)
    assert slow["t_fit"] == FIT_TIMEOUT_S
    assert slow["t_predict"] == 0.0
    assert clock.now == pytest.approx(
        BUILD_TIMEOUT_S - 0.001
        + FIT_TIMEOUT_S
        + (len(CATEGORIES) * len(REPLICATES) - 1) * FAST_SECONDS
    )


def test_build_timeout_uses_exact_cap_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "build.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    observed_timeouts: list[float] = []
    observed_numeric_threads: list[int] = []
    removed: list[str] = []

    monkeypatch.setattr(runner, "preflight_sandbox", lambda log=print: None)
    monkeypatch.setattr(runner, "resolve_sandbox", lambda: ("bwrap", "setpriv"))
    monkeypatch.setattr(runner, "_snapshot_submission", lambda _path: str(staged))
    monkeypatch.setattr(runner, "_prepare_build_tree", lambda _path: None)
    monkeypatch.setattr(runner, "_bwrap_prefix", lambda *_a, **_k: [])
    def fake_submission_env(numeric_threads):
        observed_numeric_threads.append(numeric_threads)
        return {}

    monkeypatch.setattr(runner, "_submission_env", fake_submission_env)
    monkeypatch.setattr(runner, "_rmtree_ro", lambda path: removed.append(path))

    def fake_run(_cmd, _cwd, timeout, _env):
        observed_timeouts.append(timeout)
        return 124, timeout, "TIMEOUT"

    monkeypatch.setattr(runner, "_run", fake_run)
    ok, built, note = runner.build_submission("/submission", log=lambda *_: None)
    assert not ok and built is None and "rc=124" in note
    assert observed_timeouts == [BUILD_TIMEOUT_S]
    assert observed_numeric_threads == [runner.WORKER_VCPUS]
    assert removed == [str(staged)]


def test_dataset_runner_uses_exact_nontransferable_phase_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission"
    public = tmp_path / "public"
    submission.mkdir()
    public.mkdir()
    for entrypoint in ("fit", "predict"):
        path = submission / entrypoint
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o700)
    observed_timeouts: list[float] = []
    observed_numeric_threads: list[int] = []

    monkeypatch.setattr(runner, "_open_private_work", lambda _path: None)
    monkeypatch.setattr(runner, "resolve_sandbox", lambda: ("bwrap", "setpriv"))
    monkeypatch.setattr(runner, "_bwrap_prefix", lambda *_a, **_k: [])
    def fake_submission_env(numeric_threads):
        observed_numeric_threads.append(numeric_threads)
        return {}

    monkeypatch.setattr(runner, "_submission_env", fake_submission_env)

    def fake_run(_cmd, cwd, timeout, _env):
        observed_timeouts.append(timeout)
        if len(observed_timeouts) == 1:
            Path(cwd, "model.out").write_text("model", encoding="utf-8")
        else:
            Path(cwd, "pred.csv").write_text("mean\n0.5\n", encoding="utf-8")
        return 0, 0.01, ""

    monkeypatch.setattr(runner, "_run", fake_run)
    result = runner.run_on_dataset(
        str(submission),
        str(public),
        n_test_expected=1,
        family="binomial-logit",
    )
    assert result["status"] == "ok"
    assert observed_timeouts == [FIT_TIMEOUT_S, PREDICT_TIMEOUT_S]
    assert observed_numeric_threads == [runner.WORKER_VCPUS]


def test_interrupted_build_cleans_staging_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "build.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    removed: list[str] = []

    monkeypatch.setattr(runner, "preflight_sandbox", lambda log=print: None)
    monkeypatch.setattr(runner, "resolve_sandbox", lambda: ("bwrap", "setpriv"))
    monkeypatch.setattr(runner, "_snapshot_submission", lambda _path: str(staged))
    monkeypatch.setattr(runner, "_prepare_build_tree", lambda _path: None)
    monkeypatch.setattr(runner, "_bwrap_prefix", lambda *_a, **_k: [])
    monkeypatch.setattr(runner, "_submission_env", lambda *_a, **_k: {})
    monkeypatch.setattr(runner, "_rmtree_ro", lambda path: removed.append(path))

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.build_submission("/submission", log=lambda *_: None)
    assert removed == [str(staged)]


def test_dataset_exception_never_finalizes_partial_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _entries(CATEGORIES)
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps({"datasets": entries, "meta": {"n_datasets": len(entries)}}),
        encoding="utf-8",
    )
    calls = 0

    def fail_second(dataset_dir, _submission, *, grade_calibration):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("trusted scoring interruption")
        dataset_id = os.path.basename(dataset_dir)
        category = os.path.basename(os.path.dirname(dataset_dir))
        return _ok_record(dataset_id, category)

    monkeypatch.setattr(grade, "load_corpus_key", lambda *_a, **_k: b"x" * 32)
    monkeypatch.setattr(
        grade,
        "_validate_corpus",
        lambda *_args: [str(entry["id"]) for entry in entries],
    )
    monkeypatch.setattr(grade, "calibration_seconds", lambda: 1.0)
    monkeypatch.setattr(
        grade,
        "build_submission",
        lambda *_a, **_k: (True, "/built", ""),
    )
    monkeypatch.setattr(grade, "grade_one", fail_second)
    monkeypatch.setattr(grade, "_rmtree_ro", lambda _path: None)
    monkeypatch.setenv("SVPGSBENCH_GRADER_NONCE", "c" * 32)

    with pytest.raises(RuntimeError, match="trusted scoring interruption"):
        grade.grade_corpus(
            str(corpus),
            "/submission",
            key_file="/private/test-corpus.key",
            out_dir=str(out),
            log=lambda *_args: None,
        )
    assert (out / "reward_detail.partial.json").is_file()
    assert not (out / "reward_detail.json").exists()
