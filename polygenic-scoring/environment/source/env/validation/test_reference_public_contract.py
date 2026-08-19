from __future__ import annotations

import csv
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import datagen.build_corpus as corpus_builder
from datagen.dgp import DGPConfig, generate
from datagen.materialize import materialize
from grader.truth import read_aligned_binary_truth_csv, read_binary_truth_csv
from reference import run_svpgs, baselines as strong_reference
from reference.protocol import (
    REFERENCE_CONVERGENCE_TOLERANCE,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
    REFERENCE_MAX_OUTER_ITERATIONS,
)
from reference.zoo_protocol import (
    REFERENCE_CANDIDATE_ORDER,
    REFERENCE_FULL_REFIT_RULE,
    REFERENCE_PROTOCOL_VERSION,
    validate_reference_diagnostics,
)
from reference.run_svpgs import load_prediction_dataset, load_training_dataset


def _materialized(tmp_path):
    generated = generate(DGPConfig(seed=31, n_samples=90, n_variants=28))
    nuisance = np.linspace(-3.0, 3.0, generated["G"].shape[0])
    info = materialize(
        generated,
        tmp_path / "dataset",
        family="binomial-logit",
        frac_train=generated["cfg"].frac_train_hint,
        prior_cols=("variant_class", "sv_log_length", "repeat_overlap"),
        cov_include=("age", "sex", "PC1"),
        extra_cov_cols={"nuisance_trap": nuisance},
        seed=31,
    )
    return generated, info


def test_shipping_compression_is_deterministic_and_removes_plain_genotypes(
    tmp_path,
) -> None:
    payloads = {
        "genotypes_train.tsv": b"sample_id\tv0\ns0\t1\n",
        "genotypes_test.tsv": b"sample_id\tv0\ns1\t2\n",
    }
    compressed = []
    for index in range(2):
        public = tmp_path / str(index)
        public.mkdir()
        for name, payload in payloads.items():
            (public / name).write_bytes(payload)

        corpus_builder._compress_public_genotypes(public)

        for name, payload in payloads.items():
            assert not (public / name).exists()
            target = public / f"{name}.gz"
            assert gzip.decompress(target.read_bytes()) == payload
        compressed.append((public / "genotypes_train.tsv.gz").read_bytes())

    assert compressed[0] == compressed[1]


def test_ensure_plain_directory_is_safe_for_parallel_builders(tmp_path) -> None:
    category_dir = tmp_path / "category"

    corpus_builder._ensure_plain_directory(category_dir, "category")
    corpus_builder._ensure_plain_directory(category_dir, "category")

    assert category_dir.is_dir()


def _run_public_reference_with_serialized_model(monkeypatch, tmp_path, serialized):
    public = tmp_path / "public"
    public.mkdir()
    for name in corpus_builder._PUBLIC_REFERENCE_INPUTS:
        (public / name).write_bytes(b"public input\n")

    training = SimpleNamespace(
        schema=SimpleNamespace(
            family="binomial-logit",
            covariate_names=("age",),
            variant_ids=("v0", "v1"),
        ),
        covariates=np.zeros((4, 1), dtype=np.float64),
        genotypes=np.zeros((4, 2), dtype=np.float64),
    )

    def fake_snapshot():
        return {
            "fit": b"#!/bin/sh\n",
            "predict": b"#!/bin/sh\n",
            "pgs_core.py": b"# exact public core\n",
        }, "3" * 64

    def fake_run(_command, *, cwd, **_kwargs):
        Path(cwd, "model.out").write_text(json.dumps(serialized), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        corpus_builder, "_public_reference_source_snapshot", fake_snapshot)
    monkeypatch.setattr(corpus_builder.subprocess, "run", fake_run)
    return corpus_builder._fit_best_of_family_reference(
        training, "d_calibration_contract", public)


def _valid_serialized_public_reference_model() -> dict:
    return {
        "runtime": {"numpy_version": corpus_builder.REFERENCE_NUMPY_VERSION},
        "family": "binomial-logit",
        "cov_cols": ["age"],
        "alpha": [0.0, 0.1],
        "beta": [0.2, -0.3],
        "gmean": [1.0, 1.0],
        "gsd": [0.5, 0.5],
        "vids": ["v0", "v1"],
        "calibration_intercept": -0.1,
        "calibration_slope": 0.8,
        "calibration_iterations": 4,
        "calibration_relative_gradient": 1e-10,
        "ann_cols": [],
        "classes": ["snv"],
        "recovered": {
            "iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "global_scale": 0.5,
        },
        "reference_method": "hierarchical_eb",
        "inner_selection_auc": {
            "hierarchical_eb": 0.70,
            "ridge_logistic": 0.65,
        },
        "inner_ridge_trials": [
            {"penalty": penalty, "auc": 0.65 - 0.001 * index}
            for index, penalty in enumerate((0.3, 0.1, 0.03, 0.01, 0.003, 0.001))
        ],
        "inner_hierarchical_recovered": {
            "iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "global_scale": 0.5,
        },
        "selection_seconds": 0.1,
        "full_refit_seconds": 0.2,
        "fit_seconds": 0.3,
    }


@pytest.mark.parametrize(
    "missing",
    [
        "calibration_intercept",
        "calibration_slope",
        "calibration_iterations",
        "calibration_relative_gradient",
    ],
)
def test_public_reference_model_schema_requires_calibration_coefficients(
    monkeypatch, tmp_path, missing,
) -> None:
    serialized = _valid_serialized_public_reference_model()
    serialized.pop(missing)

    with pytest.raises(
        corpus_builder.ReferenceFitError,
        match="public reference model schema is invalid",
    ):
        _run_public_reference_with_serialized_model(
            monkeypatch, tmp_path, serialized)


def test_public_reference_model_schema_rejects_legacy_calibration_fields(
    monkeypatch, tmp_path,
) -> None:
    serialized = _valid_serialized_public_reference_model()
    serialized.update(c=1.0, sd_cal=1.0, sigma_e2=1.0)

    with pytest.raises(
        corpus_builder.ReferenceFitError,
        match="public reference model schema is invalid",
    ):
        _run_public_reference_with_serialized_model(
            monkeypatch, tmp_path, serialized)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("calibration_intercept", float("nan")),
        ("calibration_intercept", float("inf")),
        ("calibration_intercept", True),
        ("calibration_slope", float("nan")),
        ("calibration_slope", float("inf")),
        ("calibration_slope", 0.0),
        ("calibration_slope", -0.1),
        ("calibration_slope", True),
        ("calibration_iterations", 0),
        ("calibration_iterations", 101),
        ("calibration_iterations", True),
        ("calibration_relative_gradient", float("nan")),
        ("calibration_relative_gradient", -1e-12),
        ("calibration_relative_gradient", 1.1e-8),
        ("calibration_relative_gradient", True),
    ],
)
def test_public_reference_model_rejects_invalid_calibration_coefficients(
    monkeypatch, tmp_path, field, invalid,
) -> None:
    serialized = _valid_serialized_public_reference_model()
    serialized[field] = invalid

    with pytest.raises(
        corpus_builder.ReferenceFitError,
        match="public reference coefficients are invalid",
    ):
        _run_public_reference_with_serialized_model(
            monkeypatch, tmp_path, serialized)


def _valid_fit_diagnostics() -> dict:
    validation_metrics = {"auc": 0.6, "brier": 0.2, "log_loss": 0.5}
    utility_components = {
        "auc": 0.2,
        "brier": 0.2,
        "log_loss": 2.0 / 7.0,
    }
    selection_utility = 38.0 / 175.0

    def optimizer() -> dict:
        return {
            "converged": True,
            "iterations": 3,
            "objective": 0.5,
            "lipschitz": 1.0,
        }

    def svpgs_fit(wall_time_s: float) -> dict:
        return {
            "converged": True,
            "max_outer_iterations": REFERENCE_MAX_OUTER_ITERATIONS,
            "convergence_tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
            "iterations_run": 25,
            "final_parameter_change": 5e-5,
            "final_predictor_change": 5e-5,
            "final_objective_change": 5e-5,
            "final_hyperparameter_change": 5e-5,
            "wall_time_s": wall_time_s,
        }

    def hyperparameters(name: str) -> dict:
        return {
            "sv_pgs": {
                "max_outer_iterations": REFERENCE_MAX_OUTER_ITERATIONS,
                "convergence_tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
            },
            "ridge_logistic": {"penalty": 0.1, "l1_ratio": 0.0},
            "dense_spike_logistic": {
                "dense_l2_penalty": 0.1,
                "sparse_l1_penalty": 0.05,
            },
            "elastic_net_logistic": {"penalty": 0.05, "l1_ratio": 0.5},
            "annotation_adaptive_en": {"penalty": 0.05, "l1_ratio": 0.5},
            "marginal_pt": {"keep_fraction": 0.1},
        }[name]

    def decomposition() -> dict:
        return {
            "transition": 0.5,
            "dense_l2_norm": 1.0,
            "sparse_l1_norm": 0.5,
            "sparse_nonzero_count": 1,
            "coefficient_count": 2,
            "reconstruction_max_abs_error": 0.0,
        }

    def pt_fit(*, full_refit: bool) -> dict:
        value = {
            "requested_variant_count": 2,
            "kept_variant_count": 1,
            "covariate_optimizer": optimizer(),
            "calibration_optimizer": optimizer(),
        }
        if full_refit:
            value["converged"] = True
        return value

    def tuning_trial(name: str) -> dict:
        trial = {
            "hyperparameters": hyperparameters(name),
            "validation_log_loss": 0.5,
        }
        if name == "sv_pgs":
            trial["fit"] = svpgs_fit(0.1)
        elif name in {"ridge_logistic", "elastic_net_logistic"}:
            trial["optimizer"] = optimizer()
        elif name == "annotation_adaptive_en":
            trial["optimizer"] = optimizer()
            trial["penalty_multiplier"] = {
                "minimum": 0.5,
                "median": 1.0,
                "maximum": 2.0,
            }
        elif name == "dense_spike_logistic":
            trial["optimizer"] = optimizer()
            trial["decomposition"] = decomposition()
        else:
            trial["fit"] = pt_fit(full_refit=False)
        return trial

    return {
        "protocol_version": REFERENCE_PROTOCOL_VERSION,
        "converged": True,
        "protocol": {
            "candidate_order": list(REFERENCE_CANDIDATE_ORDER),
            "tuning_labels": "training_dataset_inner_validation_only",
            "fit_input_type": "TrainingDataset",
            "fit_input_fields": [
                "schema",
                "sample_ids",
                "genotypes",
                "covariates",
                "targets",
                "variant_records",
            ],
            "full_refit_rule": REFERENCE_FULL_REFIT_RULE,
        },
        "split": {
            "seed": 1,
            "validation_fraction": 0.25,
            "inner_training_count": 4,
            "inner_validation_count": 2,
            "inner_validation_sample_ids": ["train-0", "train-1"],
            "inner_training_case_count": 2,
            "inner_validation_case_count": 1,
        },
        "candidates": {
            name: {
                "validation_log_loss": 0.5,
                "validation_metrics": validation_metrics,
                "selection_utility": selection_utility,
                "selection_utility_components": utility_components,
                "selected_hyperparameters": hyperparameters(name),
                "tuning_trials": [tuning_trial(name)],
                "cross_tuning": {
                    "fold_count": 2,
                    "fold_sizes": [1, 1],
                    "fold_log_losses": [0.5, 0.5],
                    "selected_hyperparameters_by_evaluation_fold": [
                        hyperparameters(name),
                        hyperparameters(name),
                    ],
                },
                "selected_for_full_refit": name in {"sv_pgs", "ridge_logistic"},
                "full_refit": (
                    svpgs_fit(0.2)
                    if name == "sv_pgs"
                    else (
                        {"converged": True, "optimizer": optimizer()}
                        if name == "ridge_logistic"
                        else None
                    )
                ),
            }
            for name in REFERENCE_CANDIDATE_ORDER
        },
        "selection": {
            "kind": "uncertainty_tied_top_two_ensemble",
            "best_single_candidate": "sv_pgs",
            "objective": {
                "name": "weighted_fraction_of_naive_to_perfect",
                "metric_weights": {"auc": 0.6, "brier": 0.2, "log_loss": 0.2},
                "perfect_metrics": {"auc": 1.0, "brier": 0.0, "log_loss": 0.0},
                "naive_validation_metrics": {
                    "auc": 0.5,
                    "brier": 0.25,
                    "log_loss": 0.7,
                },
                "naive_optimizer": optimizer(),
            },
            "refit_candidates": ["sv_pgs", "ridge_logistic"],
            "component_weights": {"sv_pgs": 0.5, "ridge_logistic": 0.5},
            "intercept": 0.0,
            "full_refit_distillation": {
                "sv_pgs": {"slope": 1.0, "intercept": 0.0, "correlation": 1.0},
                "ridge_logistic": {
                    "slope": 1.0,
                    "intercept": 0.0,
                    "correlation": 1.0,
                },
            },
            "low_dimensional_incumbent_guard": {
                "variant_count": 2_000,
                "eligible": False,
                "annotation_adaptive_in_top_two": False,
                "selected_candidate": "sv_pgs",
                "blend_clears_uncertainty": False,
                "applied": False,
            },
            "incumbent_svpgs_guard": {
                "variant_count": 2_000,
                "eligible": True,
                "sv_pgs_in_top_two": True,
                "sv_pgs_blend_weight": 0.0,
                "sv_pgs_supported_by_blend": False,
                "blend_clears_uncertainty": False,
                "full_refit_distillation_applied": False,
                "applied": False,
            },
            "single_candidate_gate": {
                "clear_winner": False,
                "reason": "top_two_indistinguishable",
                "best_single_candidate": "sv_pgs",
                "runner_up_candidate": "ridge_logistic",
                "validation_utility_gap": 0.0,
                "gap_standard_error": 0.0,
                "required_gap": 0.001,
                "bootstrap_replicates": 300,
            },
            "blend_gate": {
                "accepted": False,
                "reason": "validation_gate_not_met",
                "intercept": 0.0,
                "validation_log_loss": 0.5,
                "validation_metrics": validation_metrics,
                "selection_utility": selection_utility,
                "selection_utility_components": utility_components,
                "best_single_selection_utility": selection_utility,
                "improvement": 0.0,
                "improvement_standard_error": 0.0,
                "required_improvement": 0.001,
                "bootstrap_replicates": 300,
                "optimizer": {
                    "converged": True,
                    "iterations": 1,
                    "objective": 0.5,
                    "lipschitz": 1.0,
                },
                "candidate_weights": {
                    name: 0.0 for name in REFERENCE_CANDIDATE_ORDER
                },
            },
        },
        "training_output": {
            "finite": True,
            "minimum_logit": -1.0,
            "maximum_logit": 1.0,
        },
        "wall_time_s": 1.0,
    }


def test_strong_reference_diagnostics_accept_supported_incumbent_guard() -> None:
    diagnostics = _valid_fit_diagnostics()
    selection = diagnostics["selection"]
    selection["kind"] = "uncertainty_guarded_svpgs_incumbent"
    selection["refit_candidates"] = ["sv_pgs"]
    selection["component_weights"] = {"sv_pgs": 0.75}
    selection["intercept"] = -0.1
    selection["full_refit_distillation"] = {
        "sv_pgs": {"slope": 0.75, "intercept": -0.1, "correlation": 0.5}
    }
    selection["blend_gate"]["candidate_weights"]["sv_pgs"] = 0.02
    selection["incumbent_svpgs_guard"] = {
        "variant_count": 2_000,
        "eligible": True,
        "sv_pgs_in_top_two": True,
        "sv_pgs_blend_weight": 0.02,
        "sv_pgs_supported_by_blend": True,
        "blend_clears_uncertainty": False,
        "full_refit_distillation_applied": True,
        "applied": True,
    }
    selection["low_dimensional_incumbent_guard"] = {
        "variant_count": 2_000,
        "eligible": False,
        "annotation_adaptive_in_top_two": False,
        "selected_candidate": "ridge_logistic",
        "blend_clears_uncertainty": False,
        "applied": False,
    }
    diagnostics["candidates"]["ridge_logistic"]["selected_for_full_refit"] = False
    diagnostics["candidates"]["ridge_logistic"]["full_refit"] = None

    validate_reference_diagnostics(diagnostics, training_count=6)

    inconsistent = deepcopy(diagnostics)
    inconsistent["selection"]["incumbent_svpgs_guard"][
        "sv_pgs_supported_by_blend"
    ] = False
    with pytest.raises(ValueError, match="selection and blend gate"):
        validate_reference_diagnostics(inconsistent, training_count=6)


def test_reference_loader_uses_only_formula_selected_public_columns(tmp_path):
    _, info = _materialized(tmp_path)
    training = load_training_dataset(info["public"])
    prediction = load_prediction_dataset(
        info["public"],
        training=training,
    )

    assert training.schema.covariate_names == ("age", "sex", "PC1")
    assert training.covariates.shape[1] == 3
    assert prediction.covariates.shape[1] == 3
    assert training.schema.annotation_types == {
        "variant_class": "variant_class",
        "sv_log_length": "continuous",
        "repeat_overlap": "binary",
    }
    assert all(
        set(record.prior_continuous_features) == {"sv_log_length"}
        and set(record.prior_binary_features) == {"repeat_overlap"}
        for record in training.variant_records
    )
    observed_frequency = np.mean(training.genotypes, axis=0) / 2.0
    record_frequency = np.asarray(
        [record.allele_frequency for record in training.variant_records]
    )
    np.testing.assert_allclose(record_frequency, observed_frequency, rtol=0.0, atol=1e-7)


def test_training_loader_never_opens_prediction_files(monkeypatch, tmp_path):
    _, info = _materialized(tmp_path)
    opened: list[str] = []
    original = run_svpgs._open_public

    def recording_open(public_dir, name):
        opened.append(name)
        if "test" in name:
            raise AssertionError("training loader attempted to open prediction data")
        return original(public_dir, name)

    monkeypatch.setattr(run_svpgs, "_open_public", recording_open)
    training = load_training_dataset(info["public"])

    assert training.targets.size == len(training.sample_ids)
    assert all("test" not in name for name in opened)


def test_truth_reader_requires_exact_binary_rows_and_public_order(tmp_path):
    truth_path = tmp_path / "y_test.csv"
    truth_path.write_text("sample_id,y\ntest-0,0\ntest-1,1\n", encoding="utf-8")

    sample_ids, targets = read_binary_truth_csv(truth_path)
    assert sample_ids == ("test-0", "test-1")
    np.testing.assert_array_equal(
        read_aligned_binary_truth_csv(truth_path, sample_ids),
        targets,
    )
    with pytest.raises(ValueError, match="public test-row order"):
        read_aligned_binary_truth_csv(truth_path, tuple(reversed(sample_ids)))


@pytest.mark.parametrize(
    "payload",
    (
        "",
        "id,y\ntest-0,0\ntest-1,1\n",
        "sample_id,y\n\n",
        "sample_id,y\ntest-0\ntest-1,1\n",
        "sample_id,y\n,0\ntest-1,1\n",
        "sample_id,y\ntest-0,0\ntest-0,1\n",
        "sample_id,y\ntest-0,not-a-number\ntest-1,1\n",
        "sample_id,y\ntest-0,nan\ntest-1,1\n",
        "sample_id,y\ntest-0,inf\ntest-1,1\n",
        "sample_id,y\ntest-0,-inf\ntest-1,1\n",
        "sample_id,y\ntest-0,0\ntest-1,0\n",
        "sample_id,y\ntest-0,0\ntest-1,2\n",
    ),
)
def test_truth_reader_rejects_malformed_binary_contract(tmp_path, payload):
    truth_path = tmp_path / "y_test.csv"
    truth_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_binary_truth_csv(truth_path)


def test_truth_reader_rejects_symlinks(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("sample_id,y\ntest-0,0\ntest-1,1\n", encoding="utf-8")
    truth_path = tmp_path / "y_test.csv"
    truth_path.symlink_to(target)

    with pytest.raises(OSError):
        read_binary_truth_csv(truth_path)


def test_core_anchor_builder_records_real_fit_predict_truth_order(
    monkeypatch,
    tmp_path,
):
    events: list[str] = []
    training = SimpleNamespace(
        schema=SimpleNamespace(family="binomial-logit"),
        sample_ids=tuple(f"train-{index}" for index in range(6)),
        genotypes=np.zeros((6, 2), dtype=np.float32),
        covariates=np.zeros((6, 1), dtype=np.float32),
        targets=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        variant_records=(),
    )
    prediction = SimpleNamespace(
        sample_ids=tuple(f"test-{index}" for index in range(4)),
        genotypes=np.zeros((4, 2), dtype=np.float32),
        covariates=np.zeros((4, 1), dtype=np.float32),
    )

    class OrderedReferenceModel:
        model_bytes = b'{"locked":"model"}'

    class TruthReached(RuntimeError):
        pass

    def fake_generate(_config):
        events.append("generate")
        return object()

    def fake_materialize(*_args, **_kwargs):
        assert events == ["generate"]
        events.append("materialize")

    def fake_load_training(_path):
        assert events[-1] == "materialize"
        events.append("training_load")
        return training

    def fake_best_of_family(training_arg, ds_id, public_dir):
        # The reference is now the best of a principled family; the anchor builder
        # calls this ONE selection helper (which internally fits the candidates on an
        # inner split -- an implementation detail this order-contract test does not
        # probe). What it MUST guarantee is that the reference fit happens after the
        # naive fit and before any prediction table or truth file is opened.
        assert training_arg is training
        assert Path(public_dir).name == "public"
        assert events[-1] == "naive_fit"
        events.append("strong_fit")
        method_diag = {
            "fit_exit_code": 0,
            "fit_source_sha256": "3" * 64,
            "fit_python_executable": "/opt/svpgs-venv/bin/python",
            "fit_numpy_version": "2.1.3",
            "reference_total_fit_wall_s": 1.0,
            "reference_full_fit_wall_s": 0.9,
            "reference_program_reported_wall_s": 0.95,
            "selection_wall_s": 0.05,
            "calibration_intercept": -0.1,
            "calibration_slope": 0.8,
            "calibration_iterations": 4,
            "calibration_relative_gradient": 1e-10,
            "inner_ridge_trials": [
                {"penalty": penalty, "converged": True,
                 "auc": 0.65 - 0.001 * index}
                for index, penalty in enumerate(
                    (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
                )
            ],
            "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_global_scale": 0.5,
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.5,
        }
        return (
            "hierarchical_eb",
            OrderedReferenceModel(),
            method_diag,
            {"hierarchical_eb": 0.7, "ridge_logistic": 0.65},
            ["snv"],
        )

    expected_training = training

    def fake_load_prediction(_path, *, training: object):
        assert training is expected_training
        assert events[-2:] == ["strong_fit", "runtime_fit"]
        events.append("prediction_load")
        return prediction

    def fake_reference_predict(model, _public, _dataset_id, n_test, _family):
        assert model.model_bytes == OrderedReferenceModel.model_bytes
        assert n_test == 4
        assert events[-1] == "prediction_load"
        events.append("strong_prediction")
        pred_bytes = b"mean\n0.25\n0.75\n0.25\n0.75\n"
        return {
            "status": "ok", "detail": "", "predict_rc": 0,
            "t_predict": 0.1, "pred": {"mean": np.asarray([0.25, 0.75, 0.25, 0.75])},
            "pred_bytes": pred_bytes, "pred_sha256": hashlib.sha256(pred_bytes).hexdigest(),
            "model_sha256": hashlib.sha256(model.model_bytes).hexdigest(),
            "source_sha256": "3" * 64,
        }

    def fake_truth_reader(_path, expected_sample_ids):
        assert tuple(expected_sample_ids) == prediction.sample_ids
        assert events[-2:] == ["strong_prediction", "runtime_prediction"]
        events.append("truth_load")
        raise TruthReached

    monkeypatch.setattr(corpus_builder, "CORPUS", str(tmp_path / "corpus"))
    monkeypatch.setattr(corpus_builder, "generate", fake_generate)
    monkeypatch.setattr(corpus_builder, "materialize", fake_materialize)
    monkeypatch.setattr(run_svpgs, "load_training_dataset", fake_load_training)
    monkeypatch.setattr(run_svpgs, "load_prediction_dataset", fake_load_prediction)
    monkeypatch.setattr(
        corpus_builder, "_fit_best_of_family_reference", fake_best_of_family)
    monkeypatch.setattr(
        corpus_builder, "_predict_public_reference", fake_reference_predict)

    def fake_naive_fit(*_args, **_kwargs):
        events.append("naive_fit")
        return np.zeros(2)

    monkeypatch.setattr(
        corpus_builder,
        "_naive_logit_irls",
        fake_naive_fit,
    )

    def fake_runtime_fit(*_args, **_kwargs):
        events.append("runtime_fit")
        return object(), 0.1

    monkeypatch.setattr(
        corpus_builder,
        "_fit_runtime_anchor",
        fake_runtime_fit,
    )

    def fake_runtime_prediction(*_args, **_kwargs):
        assert events[-1] == "strong_prediction"
        events.append("runtime_prediction")
        return np.full(4, 0.5), 0.1

    monkeypatch.setattr(
        corpus_builder,
        "_predict_runtime_anchor",
        fake_runtime_prediction,
    )
    monkeypatch.setattr(
        corpus_builder,
        "read_aligned_binary_truth_csv",
        fake_truth_reader,
    )

    with pytest.raises(TruthReached):
        corpus_builder.build_one(
            next(iter(corpus_builder.CATEGORIES)),
            0,
            bytes(range(32)),
            pipeline_sha256="a" * 64,
        )
    assert events[-7:] == [
        "naive_fit",
        "strong_fit",
        "runtime_fit",
        "prediction_load",
        "strong_prediction",
        "runtime_prediction",
        "truth_load",
    ]


def test_combined_and_compatibility_dataset_apis_do_not_exist() -> None:
    for name in ("PublicDataset", "load_public_dataset", "fit_svpgs_public"):
        assert not hasattr(run_svpgs, name)
    assert not hasattr(strong_reference, "fit_strong_reference_public")


def test_loaded_dataset_values_are_deeply_read_only(tmp_path):
    _, info = _materialized(tmp_path)
    training = load_training_dataset(info["public"])
    prediction = load_prediction_dataset(
        info["public"],
        training=training,
    )

    for array in (
        training.genotypes,
        training.covariates,
        training.targets,
        prediction.genotypes,
        prediction.covariates,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        training.schema.annotation_types["new"] = "binary"
    with pytest.raises(TypeError):
        training.variant_records[0].prior_continuous_features["new"] = 1.0


def test_reference_records_use_serialized_annotation_values_not_generator_metadata(
    tmp_path,
):
    generated, info = _materialized(tmp_path)
    training = load_training_dataset(info["public"])
    with open(f"{info['public']}/variant_metadata.tsv", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    serialized = np.asarray([float(row["sv_log_length"]) for row in rows])
    fitted_values = np.asarray(
        [
            record.prior_continuous_features["sv_log_length"]
            for record in training.variant_records
        ]
    )
    raw_log_length = np.log10(np.maximum(generated["meta"]["length"], 1.0))

    np.testing.assert_array_equal(fitted_values, serialized)
    assert not np.allclose(fitted_values, raw_log_length)


def test_reference_loader_rejects_incomplete_annotation_type_contract(tmp_path):
    _, info = _materialized(tmp_path)
    dgp_path = tmp_path / "dataset" / "public" / "dgp.json"
    spec = json.loads(dgp_path.read_text())
    del spec["formula"]["annotation_types"]["repeat_overlap"]
    dgp_path.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="type every pgs annotation exactly"):
        load_training_dataset(info["public"])


def test_reference_loader_rejects_cross_table_sample_order_mismatch(tmp_path):
    _, info = _materialized(tmp_path)
    genotype_path = tmp_path / "dataset" / "public" / "genotypes_train.tsv"
    lines = genotype_path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    genotype_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="sample IDs/orders do not match"):
        load_training_dataset(info["public"])


def test_reference_loader_rejects_symlinked_public_files(tmp_path):
    _, info = _materialized(tmp_path)
    public = tmp_path / "dataset" / "public"
    family_path = public / "family.txt"
    target = tmp_path / "family-target.txt"
    target.write_bytes(family_path.read_bytes())
    family_path.unlink()
    family_path.symlink_to(target)

    with pytest.raises(ValueError, match="plain regular file"):
        load_training_dataset(info["public"])
