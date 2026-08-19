"""Classification contract: solver failures hit the floor; trusted failures stay loud."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from grader import grade
from grader.skill import INVALID_REWARD, SKILL_LO
from grader.submission_runner import SandboxUnavailable


ENTRY = {
    "id": "category__r0",
    "path": "category/category__r0",
    "category": "category",
    "replicate": 0,
    "family": "binomial-logit",
    "weight": 1.0,
}


def _run(overrides):
    names = {
        "_validate_corpus", "calibration_seconds", "build_submission",
        "grade_one", "load_corpus_key", "_rmtree_ro",
    }
    original = {name: getattr(grade, name) for name in names}
    for name, value in overrides.items():
        setattr(grade, name, value)
    old_nonce = os.environ.get("SVPGSBENCH_GRADER_NONCE")
    os.environ["SVPGSBENCH_GRADER_NONCE"] = "a" * 32
    try:
        with tempfile.TemporaryDirectory() as corpus, tempfile.TemporaryDirectory() as out:
            with open(os.path.join(corpus, "manifest.json"), "w") as fh:
                json.dump({"datasets": [ENTRY], "meta": {"n_datasets": 1}}, fh)
            return grade.grade_corpus(
                corpus,
                "/submission",
                key_file="/private/test-corpus.key",
                out_dir=out,
            )
    finally:
        if old_nonce is None:
            os.environ.pop("SVPGSBENCH_GRADER_NONCE", None)
        else:
            os.environ["SVPGSBENCH_GRADER_NONCE"] = old_nonce
        for name, value in original.items():
            setattr(grade, name, value)


def _base_overrides():
    return {
        "_validate_corpus": lambda *_args: ["dummy"],
        "calibration_seconds": lambda: 1.0,
        "load_corpus_key": lambda *_args, **_kwargs: b"x" * 32,
        "_rmtree_ro": lambda _path: None,
    }


def test_build_rejection_is_complete_floor():
    overrides = _base_overrides()
    overrides["build_submission"] = lambda *_args, **_kwargs: (
        False, None, "special/oversized submission entry")
    reward, detail = _run(overrides)
    assert reward["reward"] == pytest.approx(SKILL_LO)
    assert detail["complete"] is True
    dataset = detail["datasets"][0]
    assert dataset["status"] == "build_failed"
    assert dataset["reward"] == dataset["raw_skill"] == SKILL_LO
    assert dataset["t_fit"] == 0.0 and dataset["t_predict"] == 0.0


def test_runtime_bad_output_is_complete_floor():
    overrides = _base_overrides()
    overrides["build_submission"] = lambda *_args, **_kwargs: (True, "/built", "")

    def failed_dataset(*_args, **_kwargs):
        aggregate = {"category": "category", "weight": 1.0,
                     "reward": INVALID_REWARD}
        detail = {"dataset": "dummy", "category": "category", "weight": 1.0,
                  "status": "bad_pred", "reward": INVALID_REWARD,
                  "accuracy": 0.0, "raw_skill": INVALID_REWARD,
                  "perf": 0.0, "note": "malformed pred.csv"}
        return aggregate, detail

    overrides["grade_one"] = failed_dataset
    reward, detail = _run(overrides)
    assert reward["reward"] == pytest.approx(SKILL_LO)
    assert detail["complete"] is True
    dataset = detail["datasets"][0]
    assert dataset["status"] == "bad_pred"
    assert dataset["reward"] == dataset["raw_skill"] == SKILL_LO


def test_grade_one_failure_records_satisfy_the_wrapper_contract():
    """Both early exits in grade_one must carry the signed field required by the
    TypeScript wrapper. Omitting it turns one bad dataset into a rejected report."""
    with tempfile.TemporaryDirectory() as dataset_dir:
        truth_dir = os.path.join(dataset_dir, "truth")
        os.mkdir(truth_dir)
        with open(os.path.join(truth_dir, "anchors.json"), "w") as fh:
            json.dump({
                "category": "category",
                "weight": 1.0,
                "family": "binomial-logit",
                "metrics_naive": {"auc": 0.5},
                "metrics_reference": {"auc": 0.7},
                "metrics_ref_naive_se": {"auc": 0.01},
                "metrics_naive_se": {"auc": 0.01},
            }, fh)
        with open(os.path.join(truth_dir, "y_test.csv"), "w") as fh:
            fh.write("sample_id,y\ns0,0\ns1,1\n")

        responses = [
            {"status": "bad_pred", "pred": None, "t_fit": 0.1,
             "t_predict": 0.2, "detail": "malformed pred.csv"},
            {"status": "ok",
             "pred": {"sample_id": ["s1", "s0"], "mean": [0.8, 0.2]},
             "t_fit": 0.1, "t_predict": 0.2, "detail": ""},
        ]
        original = grade.run_on_dataset
        try:
            for index, response in enumerate(responses):
                grade.run_on_dataset = lambda *_args, **_kwargs: response
                aggregate, detail = grade.grade_one(
                    dataset_dir,
                    "/submission",
                    grade_calibration=1.0,
                )
                assert aggregate["reward"] == SKILL_LO
                assert detail["reward"] == detail["raw_skill"] == SKILL_LO
                assert detail["accuracy"] == detail["perf"] == 0.0
                if index == 1:
                    assert detail["status"] == "bad_pred"
                    assert detail["note"] == "pred sample_id is not in test input order"
        finally:
            grade.run_on_dataset = original


def test_trusted_sandbox_failure_propagates():
    overrides = _base_overrides()

    def fail(*_args, **_kwargs):
        raise SandboxUnavailable("bwrap unavailable")

    overrides["build_submission"] = fail
    try:
        _run(overrides)
    except SandboxUnavailable:
        return
    raise AssertionError("trusted sandbox failure was converted to submission zero")


def main():
    for fn in (
        test_build_rejection_is_complete_floor,
        test_runtime_bad_output_is_complete_floor,
        test_grade_one_failure_records_satisfy_the_wrapper_contract,
        test_trusted_sandbox_failure_propagates,
    ):
        fn()
        print(f"  PASS  {fn.__name__}")
    print("\nFAILURE CLASSIFICATION TESTS PASS")


if __name__ == "__main__":
    main()
