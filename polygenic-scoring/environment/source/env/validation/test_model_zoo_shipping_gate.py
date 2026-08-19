from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datagen.categories import CATEGORIES
from datagen.build_corpus import _generation_pipeline_sha256
from grader.contract import (
    CORPUS_PURPOSE,
    CORPUS_SCHEMA_VERSION,
    DEVELOPMENT_RESULT_HMAC_FIELD,
    MANIFEST_HMAC_FIELD,
    REPLICATES_PER_CATEGORY,
    SHIPPED_REQUESTED_N,
    SHIPPED_REQUESTED_P,
)
from grader.corpus_auth import (
    DEVELOPMENT_RESULT_HMAC_DOMAIN,
    MANIFEST_HMAC_DOMAIN,
    corpus_key_id,
    json_hmac_sha256,
    opaque_dataset_id,
)
from reference.protocol import (
    REFERENCE_CONVERGENCE_TOLERANCE,
    REFERENCE_FIT_PROTOCOL as SHIPPED_FIT_PROTOCOL,
    REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
    REFERENCE_MAX_OUTER_ITERATIONS,
    REFERENCE_RIDGE_PENALTIES,
    REFERENCE_PROTOCOL_VERSION as SHIPPED_PROTOCOL_VERSION,
)
from reference.zoo_protocol import REFERENCE_FULL_REFIT_RULE, REFERENCE_PROTOCOL_VERSION
import validation.model_zoo as model_zoo
from validation.model_zoo import (
    DEVELOPMENT_REPLICATES,
    METRIC_DIRECTIONS,
    MODEL_ORDER,
    REPORT_SCHEMA_VERSION,
    SCORING_SOURCE_RELATIVE_PATHS,
    ShippingGateError,
    ShippingThresholds,
    _apply_redundancy_threshold,
    _category_analysis,
    _development_dataset_id,
    _implementation_hashes,
    _model_result_from_probability,
    _read_json,
    _require_frozen_development_report_digest,
    _result_files,
    _scoring_source_sha256,
    _source_provenance_failures,
    _validated_final_result_payload,
    _validate_domain_separated_identities,
    _validate_manifest_grid,
    _validated_development_report,
    build_development_report,
)


AUTH_KEY = bytes(range(32))
_REAL_RECOMPUTE_DEVELOPMENT_METRIC_EVIDENCE = (
    model_zoo._recompute_development_metric_evidence
)


@pytest.fixture(autouse=True)
def _isolate_synthetic_report_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic profile tests isolate aggregation from vector recomputation.

    Their hand-authored skill matrices intentionally do not come from probabilities.
    Dedicated evidence tests below invoke the saved production function directly.
    """
    monkeypatch.setattr(
        model_zoo,
        "_recompute_development_metric_evidence",
        lambda _result, *, auth_key: None,
    )


def _inner_ridge_trials(best_auc: float = 0.65) -> list[dict]:
    return [
        {"penalty": penalty, "converged": True,
         "auc": best_auc - 0.001 * index}
        for index, penalty in enumerate(REFERENCE_RIDGE_PENALTIES)
    ]


def test_anchor_health_records_losses_for_reliability_filtering() -> None:
    health = model_zoo._anchor_health(
        "dataset",
        {"auc": 0.6, "brier": 0.2, "log_loss": 0.5},
        {"auc": 0.7, "brier": 0.21, "log_loss": 0.45},
        {"auc": 0.01, "brier": 0.005, "log_loss": 0.01},
        {"auc": 0.01, "brier": 0.005, "log_loss": 0.01},
    )

    assert health["brier"] == {
        "oriented_gap": pytest.approx(-0.01),
        "standard_error": 0.005,
        "z_score": pytest.approx(-2.0),
        "reliable_at_k_se": False,
        "naive_standard_error": 0.005,
        "resolution_ratio": pytest.approx(-2.0),
        "adequate_at_resolution_k": False,
    }


def test_anchor_health_separates_reliable_from_ADEQUATE() -> None:
    """A real gap and an adequately LARGE gap are different properties, and the
    health record must report both. `auc` here is overwhelmingly significant
    (z = 20) and still too small to be a denominator -- the exact signature of the
    five v6 categories that minted free credit.
    """
    health = model_zoo._anchor_health(
        "dataset",
        {"auc": 0.60, "brier": 0.2, "log_loss": 0.5},
        {"auc": 0.61, "brier": 0.15, "log_loss": 0.45},
        {"auc": 0.0005, "brier": 0.005, "log_loss": 0.01},
        {"auc": 0.01, "brier": 0.005, "log_loss": 0.01},
    )
    assert health["auc"]["z_score"] == pytest.approx(20.0)
    assert health["auc"]["reliable_at_k_se"] is True
    assert health["auc"]["resolution_ratio"] == pytest.approx(1.0)
    assert health["auc"]["adequate_at_resolution_k"] is False
    # brier clears both bars and is a usable denominator.
    assert health["brier"]["reliable_at_k_se"] is True
    assert health["brier"]["adequate_at_resolution_k"] is True


def test_anchor_health_rejects_negative_standard_error() -> None:
    with pytest.raises(ShippingGateError, match="invalid anchor standard error"):
        model_zoo._anchor_health(
            "dataset",
            {"auc": 0.6, "brier": 0.2, "log_loss": 0.5},
            {"auc": 0.7, "brier": 0.19, "log_loss": 0.45},
            {"auc": 0.01, "brier": -0.005, "log_loss": 0.01},
            {"auc": 0.01, "brier": 0.005, "log_loss": 0.01},
        )


def test_anchor_health_rejects_nonpositive_naive_standard_error() -> None:
    """se_naive is the adequacy denominator; zero would make any gap look
    infinitely adequate."""
    with pytest.raises(ShippingGateError, match="naive single-estimate standard error"):
        model_zoo._anchor_health(
            "dataset",
            {"auc": 0.6, "brier": 0.2, "log_loss": 0.5},
            {"auc": 0.7, "brier": 0.19, "log_loss": 0.45},
            {"auc": 0.01, "brier": 0.005, "log_loss": 0.01},
            {"auc": 0.01, "brier": 0.0, "log_loss": 0.01},
        )


def _optimizer_diagnostics() -> dict:
    return {
        "converged": True,
        "iterations": 3,
        "objective": 0.5,
        "lipschitz": 1.0,
    }


def _candidate_hyperparameters(name: str) -> dict:
    if name == "sv_pgs":
        return {
            "max_outer_iterations": REFERENCE_MAX_OUTER_ITERATIONS,
            "convergence_tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
        }
    if name == "ridge_logistic":
        return {"penalty": 0.1, "l1_ratio": 0.0}
    if name == "dense_spike_logistic":
        return {"dense_l2_penalty": 0.1, "sparse_l1_penalty": 0.05}
    if name in {"elastic_net_logistic", "annotation_adaptive_en"}:
        return {"penalty": 0.05, "l1_ratio": 0.5}
    if name == "marginal_pt":
        return {"keep_fraction": 0.1}
    raise AssertionError(f"unexpected candidate {name!r}")


def _decomposition_diagnostics() -> dict:
    return {
        "transition": 0.5,
        "dense_l2_norm": 1.0,
        "sparse_l1_norm": 0.5,
        "sparse_nonzero_count": 2,
        "coefficient_count": 8,
        "reconstruction_max_abs_error": 0.0,
    }


def _pt_fit_diagnostics(*, full_refit: bool) -> dict:
    diagnostics = {
        "requested_variant_count": 8,
        "kept_variant_count": 6,
        "covariate_optimizer": _optimizer_diagnostics(),
        "calibration_optimizer": _optimizer_diagnostics(),
    }
    if full_refit:
        diagnostics["converged"] = True
    return diagnostics


def _svpgs_fit_diagnostics(wall_time_s: float) -> dict:
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


def _candidate_tuning_trial(name: str) -> dict:
    trial = {
        "hyperparameters": _candidate_hyperparameters(name),
        "validation_log_loss": 0.5,
    }
    if name == "sv_pgs":
        trial["fit"] = _svpgs_fit_diagnostics(0.1)
    elif name in {"ridge_logistic", "elastic_net_logistic"}:
        trial["optimizer"] = _optimizer_diagnostics()
    elif name == "annotation_adaptive_en":
        trial["optimizer"] = _optimizer_diagnostics()
        trial["penalty_multiplier"] = {
            "minimum": 0.5,
            "median": 1.0,
            "maximum": 2.0,
        }
    elif name == "dense_spike_logistic":
        trial["optimizer"] = _optimizer_diagnostics()
        trial["decomposition"] = _decomposition_diagnostics()
    else:
        trial["fit"] = _pt_fit_diagnostics(full_refit=False)
    return trial


def _candidate_full_refit(name: str) -> dict:
    if name == "sv_pgs":
        return _svpgs_fit_diagnostics(0.2)
    if name in {"ridge_logistic", "elastic_net_logistic"}:
        return {"converged": True, "optimizer": _optimizer_diagnostics()}
    if name == "annotation_adaptive_en":
        return {
            "converged": True,
            "optimizer": _optimizer_diagnostics(),
            "penalty_multiplier": {
                "minimum": 0.5,
                "median": 1.0,
                "maximum": 2.0,
            },
        }
    if name == "dense_spike_logistic":
        return {
            "converged": True,
            "optimizer": _optimizer_diagnostics(),
            "decomposition": _decomposition_diagnostics(),
        }
    if name == "marginal_pt":
        return _pt_fit_diagnostics(full_refit=True)
    raise AssertionError(f"unexpected candidate {name!r}")


def _reference_diagnostics(training_count: int = 3_000) -> dict:
    candidate_names = tuple(model_zoo._strong._CANDIDATE_ORDER)
    validation_metrics = {"auc": 0.6, "brier": 0.2, "log_loss": 0.5}
    utility_components = {"auc": 0.2, "brier": 0.2, "log_loss": 2.0 / 7.0}
    selection_utility = 38.0 / 175.0
    validation_count = min(
        max(2, int(round(0.2 * training_count))),
        training_count - 2,
    )
    inner_training_count = training_count - validation_count
    validation_case_count = max(1, min(validation_count - 1, validation_count // 2))
    training_case_count = max(
        1,
        min(inner_training_count - 1, inner_training_count // 2),
    )
    return {
        "protocol_version": REFERENCE_PROTOCOL_VERSION,
        "converged": True,
        "protocol": {
            "candidate_order": list(candidate_names),
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
            "validation_fraction": 0.2,
            "inner_training_count": inner_training_count,
            "inner_validation_count": validation_count,
            "inner_validation_sample_ids": [
                f"train-{index:05d}" for index in range(validation_count)
            ],
            "inner_training_case_count": training_case_count,
            "inner_validation_case_count": validation_case_count,
        },
        "candidates": {
            name: {
                "validation_log_loss": 0.5,
                "validation_metrics": validation_metrics,
                "selection_utility": selection_utility,
                "selection_utility_components": utility_components,
                "selected_hyperparameters": _candidate_hyperparameters(name),
                "tuning_trials": [_candidate_tuning_trial(name)],
                "cross_tuning": {
                    "fold_count": 2,
                    "fold_sizes": [validation_count // 2, validation_count - validation_count // 2],
                    "fold_log_losses": [0.5, 0.5],
                    "selected_hyperparameters_by_evaluation_fold": [
                        _candidate_hyperparameters(name),
                        _candidate_hyperparameters(name),
                    ],
                },
                "selected_for_full_refit": name in {"sv_pgs", "ridge_logistic"},
                "full_refit": (
                    _candidate_full_refit(name)
                    if name in {"sv_pgs", "ridge_logistic"}
                    else None
                ),
            }
            for name in candidate_names
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
                "naive_optimizer": _optimizer_diagnostics(),
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
                "selected_candidate": "ridge_logistic",
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
                "candidate_weights": {name: 0.0 for name in candidate_names},
            },
        },
        "training_output": {
            "finite": True,
            "minimum_logit": -1.0,
            "maximum_logit": 1.0,
        },
        "wall_time_s": 1.0,
    }


def _development_fit_diagnostics(training_count: int = 3_000) -> dict:
    def adaptive(name: str) -> dict:
        l1_ratio = 0.0 if name == "class_adaptive_ridge" else 0.5
        return {
            "tuning_labels": "training_dataset_inner_validation_only",
            "selected_hyperparameters": {
                "penalty": 0.1,
                "l1_ratio": l1_ratio,
            },
            "validation_log_loss": 0.5,
            "tuning_trials": [
                {
                    "penalty": 0.1,
                    "l1_ratio": l1_ratio,
                    "validation_log_loss": 0.5,
                    "optimizer": _optimizer_diagnostics(),
                }
            ],
            "full_refit": _optimizer_diagnostics(),
            "penalty_multiplier": {
                "minimum": 0.5,
                "median": 1.0,
                "maximum": 2.0,
            },
        }

    return {
        "strong_reference": _reference_diagnostics(training_count=training_count),
        "shipped_reference": {
            "protocol_version": SHIPPED_PROTOCOL_VERSION,
            "reference_method": "hierarchical_eb",
            "converged": True,
            "fit_protocol": SHIPPED_FIT_PROTOCOL,
            "wall_time_s": 1.0,
            "fit_exit_code": 0,
            "fit_source_sha256": "3" * 64,
            "fit_python_executable": "/opt/svpgs-venv/bin/python",
            "fit_numpy_version": "2.1.3",
            "reference_total_fit_wall_s": 1.0,
            "reference_full_fit_wall_s": 0.5,
            "reference_program_reported_wall_s": 0.9,
            "selection_wall_s": 0.4,
            "calibration_intercept": -0.1,
            "calibration_slope": 0.8,
            "calibration_iterations": 4,
            "calibration_relative_gradient": 1e-10,
            "inner_selection_auc": {
                "hierarchical_eb": 0.71, "ridge_logistic": 0.65},
            "inner_ridge_trials": _inner_ridge_trials(),
            "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_global_scale": 0.5,
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.5,
            "prediction_exit_code": 0,
            "prediction_wall_s": 0.1,
            "prediction_output_sha256": "2" * 64,
            "prediction_output_byte_count": 64,
            "prediction_model_sha256": "1" * 64,
            "prediction_source_sha256": "3" * 64,
            "classes_present": ["snv"],
            "training_output": {
                "model_sha256": "1" * 64,
                "model_byte_count": 1024,
            },
        },
        "standalone_full_refits": {
            name: _candidate_full_refit(name)
            for name in model_zoo._strong._CANDIDATE_ORDER
        },
        "class_adaptive_ridge": adaptive("class_adaptive_ridge"),
        "fit_input_type": "TrainingDataset",
        "execution_order": [
            "all_model_fits",
            "prediction_data_load",
            "all_model_predictions",
            "truth_load",
        ],
    }


class _OrderedDecisionModel:
    def __init__(self, events: list[str], event: str, logits: np.ndarray):
        self._events = events
        self._event = event
        self._logits = np.asarray(logits, dtype=np.float64)

    def decision_function(self, genotypes, covariates):
        assert len(genotypes) == len(covariates) == self._logits.size
        self._events.append(self._event)
        return self._logits.copy()

    def predict_probability(self, genotypes, covariates):
        logits = self.decision_function(genotypes, covariates)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _sign_manifest(manifest: dict) -> None:
    manifest[MANIFEST_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        manifest,
        exclude_field=MANIFEST_HMAC_FIELD,
        domain=MANIFEST_HMAC_DOMAIN,
    )


def _sign_development_result(result: dict) -> None:
    result[DEVELOPMENT_RESULT_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        result,
        exclude_field=DEVELOPMENT_RESULT_HMAC_FIELD,
        domain=DEVELOPMENT_RESULT_HMAC_DOMAIN,
    )


def _final_manifest() -> dict:
    pipeline_sha256 = _generation_pipeline_sha256()
    entries = []
    for category in CATEGORIES:
        for replicate in range(REPLICATES_PER_CATEGORY):
            dataset_id = opaque_dataset_id(
                AUTH_KEY,
                purpose=CORPUS_PURPOSE,
                pipeline_sha256=pipeline_sha256,
                category=category,
                replicate=replicate,
            )
            entries.append(
                {
                    "id": dataset_id,
                    "path": f"{category}/{dataset_id}",
                    "category": category,
                    "replicate": replicate,
                    "family": "binomial-logit",
                    "weight": 1.0,
                    "sha256": {"public/dgp.json": "b" * 64},
                }
            )
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "datasets": entries,
        "meta": {
            "n_datasets": len(entries),
            "weight_rule": "equal_per_dataset",
            "categories": list(CATEGORIES),
            "replicates_per_category": REPLICATES_PER_CATEGORY,
            "requested_n": SHIPPED_REQUESTED_N,
            "requested_p": SHIPPED_REQUESTED_P,
            "development_report_sha256": "d" * 64,
            "purpose": CORPUS_PURPOSE,
            "key_id": corpus_key_id(AUTH_KEY),
            "generation_pipeline_sha256": pipeline_sha256,
        },
    }
    _sign_manifest(manifest)
    return manifest


def _profile(category_index: int, *, clone: bool) -> dict[str, float]:
    if clone:
        values = np.asarray([0.82, 0.54, 0.67, 0.64, 0.32, 0.73, 0.88])
    else:
        random = np.random.default_rng(91 + category_index)
        values = random.uniform(0.18, 0.72, size=len(MODEL_ORDER) - 1)
        values[category_index % values.size] = 0.90
    return {
        "strong_reference": 1.0,
        **{
            name: float(value)
            for name, value in zip(MODEL_ORDER[1:], values)
        },
    }


def _development_result(
    category: str,
    replicate: int,
    category_index: int,
    implementation_hashes: dict[str, str],
    *,
    clone: bool,
) -> dict:
    profile = _profile(category_index, clone=clone)
    replicate_offset = replicate - DEVELOPMENT_REPLICATES[
        len(DEVELOPMENT_REPLICATES) // 2
    ]
    models = {}
    for model_index, (name, value) in enumerate(profile.items()):
        jitter = (
            0.0
            if name == "strong_reference"
            else 0.001 * replicate_offset * (model_index + 1)
        )
        skill = value + jitter
        models[name] = {
            "metrics": {"auc": 0.7, "brier": 0.2, "log_loss": 0.6},
            "accuracy_skill": skill,
            "raw_skill": skill,
            "per_metric_skill": {"auc": skill},
        }
    health = {
        metric: {
            "oriented_gap": 0.1,
            "standard_error": 0.01,
            "z_score": 10.0,
            "reliable_at_k_se": True,
        }
        for metric in METRIC_DIRECTIONS
    }
    pipeline_sha256 = implementation_hashes["generation_pipeline_sha256"]
    dataset_id = _development_dataset_id(
        category,
        replicate,
        AUTH_KEY,
        pipeline_sha256,
    )
    _cfg_obj = CATEGORIES[category]["make_cfg"](
        replicate, SHIPPED_REQUESTED_N, SHIPPED_REQUESTED_P
    )
    dgp_config = asdict(_cfg_obj)
    del dgp_config["seed"]
    _n_train = int(_cfg_obj.n_samples * _cfg_obj.frac_train_hint)
    _n_test = _cfg_obj.n_samples - _n_train
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "purpose": "development",
        "key_id": corpus_key_id(AUTH_KEY),
        "generation_pipeline_sha256": pipeline_sha256,
        "implementation_sha256": implementation_hashes,
        "id": dataset_id,
        "path": f"{category}/{dataset_id}",
        "category": category,
        "replicate": replicate,
        "family": "binomial-logit",
        "requested_n": SHIPPED_REQUESTED_N,
        "requested_p": SHIPPED_REQUESTED_P,
        "dgp_config": dgp_config,
        "dataset_file_sha256": {"public/dgp.json": "c" * 64},
        "n_train": _n_train,
        "n_test": _n_test,
        "n_variants": _cfg_obj.n_variants,
        "anchor_health": health,
        "models": models,
        "metric_evidence": {},
        "fit_protocol": _development_fit_diagnostics(training_count=_n_train),
        "wall_time_s": 1.0 + category_index,
    }
    _sign_development_result(result)
    return result


def _development_results(*, clone: bool = False) -> list[dict]:
    implementation_hashes = _implementation_hashes()
    return [
        _development_result(
            category,
            replicate,
            category_index,
            implementation_hashes,
            clone=clone,
        )
        for category_index, category in enumerate(CATEGORIES)
        for replicate in DEVELOPMENT_REPLICATES
    ]


def test_final_manifest_requires_authenticated_canonical_replicate_grid() -> None:
    manifest = _final_manifest()
    _validate_manifest_grid(manifest, AUTH_KEY)
    missing = deepcopy(manifest)
    missing["datasets"].pop()
    _sign_manifest(missing)
    with pytest.raises(ShippingGateError, match="needs exactly"):
        _validate_manifest_grid(missing, AUTH_KEY)

    # A grid that is the right SIZE but the wrong ORDER must still be rejected:
    # the count check above cannot see this, and the grader pairs manifest entry i
    # with graded dataset i by position.
    reordered = deepcopy(manifest)
    entries = reordered["datasets"]
    entries[0], entries[1] = entries[1], entries[0]
    _sign_manifest(reordered)
    with pytest.raises(ShippingGateError, match="exact canonical"):
        _validate_manifest_grid(reordered, AUTH_KEY)

    wrong_identity = deepcopy(manifest)
    wrong_identity["datasets"][0]["id"] = "d_" + "0" * 32
    _sign_manifest(wrong_identity)
    with pytest.raises(ShippingGateError, match="identity mismatch"):
        _validate_manifest_grid(wrong_identity, AUTH_KEY)

    undersized = deepcopy(manifest)
    undersized["meta"]["requested_n"] = 4_000
    _sign_manifest(undersized)
    with pytest.raises(ShippingGateError, match="manifest dimensions"):
        _validate_manifest_grid(undersized, AUTH_KEY)

    tampered = deepcopy(manifest)
    tampered["meta"]["requested_n"] = 4_000
    with pytest.raises(ShippingGateError, match="manifest HMAC mismatch"):
        _validate_manifest_grid(tampered, AUTH_KEY)


def test_development_cli_uses_only_the_fixed_shipping_dimensions() -> None:
    arguments = model_zoo._parser().parse_args(
        [
            "--key-file",
            "/private/corpus.key",
            "develop",
            "--work-dir",
            "/tmp/development",
            "--out",
            "/tmp/report.json",
        ]
    )
    assert not hasattr(arguments, "n")
    assert not hasattr(arguments, "p")
    with pytest.raises(SystemExit):
        model_zoo._parser().parse_args(
            [
                "--key-file",
                "/private/corpus.key",
                "develop",
                "--n",
                "4000",
                "--work-dir",
                "/tmp/development",
                "--out",
                "/tmp/report.json",
            ]
        )


def test_development_report_requires_every_authenticated_replicate_once() -> None:
    results = _development_results()
    report = build_development_report(results, auth_key=AUTH_KEY)
    assert report["passed"] is True
    assert report["development_design"]["replicates"] == list(
        DEVELOPMENT_REPLICATES
    )
    _validated_development_report(report, AUTH_KEY)

    missing = results[:-1]
    with pytest.raises(ShippingGateError, match="expected .* received"):
        build_development_report(missing, auth_key=AUTH_KEY)

    wrong_replicate = deepcopy(results)
    wrong_replicate[0]["replicate"] = max(DEVELOPMENT_REPLICATES) + 1
    _sign_development_result(wrong_replicate[0])
    with pytest.raises(ShippingGateError, match="unexpected or duplicate"):
        build_development_report(wrong_replicate, auth_key=AUTH_KEY)

    undersized = deepcopy(results)
    undersized[0]["requested_n"] = 4_000
    _sign_development_result(undersized[0])
    with pytest.raises(ShippingGateError, match="generation controls mismatch"):
        build_development_report(undersized, auth_key=AUTH_KEY)


@pytest.mark.parametrize(
    "mutation",
    (
        "split_count",
        "candidate_winner",
        "standalone_extra_field",
        "adaptive_extra_field",
        "adaptive_optimizer",
    ),
)
def test_development_report_fails_closed_on_nested_fit_protocol_drift(
    mutation: str,
) -> None:
    results = _development_results()
    protocol = results[0]["fit_protocol"]
    if mutation == "split_count":
        protocol["strong_reference"]["split"]["inner_training_count"] -= 1
    elif mutation == "candidate_winner":
        protocol["strong_reference"]["candidates"]["ridge_logistic"][
            "selected_hyperparameters"
        ]["penalty"] = 0.03
    elif mutation == "standalone_extra_field":
        protocol["standalone_full_refits"]["ridge_logistic"]["legacy"] = True
    elif mutation == "adaptive_extra_field":
        protocol["class_adaptive_ridge"]["legacy"] = True
    else:
        protocol["class_adaptive_ridge"]["tuning_trials"][0]["optimizer"] = {}
    _sign_development_result(results[0])

    report = build_development_report(results, auth_key=AUTH_KEY)
    check = next(
        item for item in report["checks"] if item["name"] == "training_only_model_selection"
    )
    assert report["passed"] is False
    assert check["failures"] == [results[0]["id"]]


def test_development_fit_protocol_rejects_nonfinite_nested_loss() -> None:
    diagnostics = _development_fit_diagnostics()
    diagnostics["class_adaptive_ridge"]["tuning_trials"][0][
        "validation_log_loss"
    ] = float("nan")
    assert not model_zoo._valid_development_fit_protocol(
        diagnostics,
        training_count=3_000,
    )


def test_redundancy_analysis_detects_a_clone_development_matrix() -> None:
    category_report, distinctness = _category_analysis(
        _development_results(clone=True)
    )
    assert set(category_report) == set(CATEGORIES)
    _apply_redundancy_threshold(distinctness, ShippingThresholds())
    expected_pairs = len(CATEGORIES) * (len(CATEGORIES) - 1) // 2
    assert len(distinctness["redundant_pairs"]) == expected_pairs
    assert distinctness["redundant_components"] == [sorted(CATEGORIES)]


def test_development_report_fails_closed_on_unreliable_auc() -> None:
    results = _development_results()
    results[0]["anchor_health"]["auc"]["standard_error"] = 0.2
    _sign_development_result(results[0])
    report = build_development_report(results, auth_key=AUTH_KEY)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is False
    assert checks["reliable_reference_separation"]["failures"] == [
        {"id": results[0]["id"], "reliable_metrics": ["brier", "log_loss"]}
    ]


def test_development_report_fails_closed_on_uncalibrated_reference() -> None:
    results = _development_results()
    # For lower-is-better metrics, oriented_gap = naive - reference.  A gap of
    # -0.03 means the reference Brier is 0.03 worse than naive, beyond the shared
    # predeclared 0.02 absolute qualification bound.
    results[0]["anchor_health"]["brier"]["oriented_gap"] = -0.03
    _sign_development_result(results[0])
    report = build_development_report(results, auth_key=AUTH_KEY)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"] is False
    failures = checks["reference_calibration_qualified"]["failures"]
    assert failures[0]["id"] == results[0]["id"]
    assert failures[0]["violations"][0]["metric"] == "brier"
    assert failures[0]["violations"][0]["regret"] == pytest.approx(0.03)
    assert failures[0]["violations"][0]["limit"] == 0.02


def test_development_result_contract_rejects_stale_or_extra_models() -> None:
    results = _development_results()
    sorted_models = {
        name: results[0]["models"][name] for name in sorted(MODEL_ORDER)
    }
    results[0]["models"] = sorted_models
    assert (
        build_development_report(results, auth_key=AUTH_KEY)["schema_version"]
        == REPORT_SCHEMA_VERSION
    )

    stale = deepcopy(results)
    stale[0]["implementation_sha256"]["shipping_gate_sha256"] = "d" * 64
    _sign_development_result(stale[0])
    with pytest.raises(ShippingGateError, match="provenance mismatch"):
        build_development_report(stale, auth_key=AUTH_KEY)

    extra = deepcopy(results)
    extra[0]["models"]["undeclared"] = extra[0]["models"]["sv_pgs"]
    _sign_development_result(extra[0])
    with pytest.raises(ShippingGateError, match="provenance mismatch"):
        build_development_report(extra, auth_key=AUTH_KEY)


def test_development_and_shipping_identities_are_purpose_separated() -> None:
    development = {"dataset_results": _development_results()}
    development_id = development["dataset_results"][0]["id"]
    with pytest.raises(ShippingGateError, match="not purpose-separated"):
        _validate_domain_separated_identities(
            development,
            [{"id": development_id}],
        )

    pipeline_sha256 = _generation_pipeline_sha256()
    shipping_id = opaque_dataset_id(
        AUTH_KEY,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=pipeline_sha256,
        category=list(CATEGORIES)[0],
        replicate=DEVELOPMENT_REPLICATES[0],
    )
    _validate_domain_separated_identities(development, [{"id": shipping_id}])


def test_development_result_and_report_tampering_is_rejected() -> None:
    results = _development_results()
    tampered_result = deepcopy(results)
    tampered_result[0]["wall_time_s"] = 999.0
    with pytest.raises(ShippingGateError, match="authentication/provenance"):
        build_development_report(tampered_result, auth_key=AUTH_KEY)

    report = build_development_report(results, auth_key=AUTH_KEY)
    report["development_design"]["requested_n"] = 4_000
    with pytest.raises(ShippingGateError, match="report HMAC mismatch"):
        _validated_development_report(report, AUTH_KEY)


def _write_scoring_sources(root: Path) -> None:
    for relative in SCORING_SOURCE_RELATIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "grader/contract.py":
            path.write_text(
                "MASTERY_THRESHOLD = None\n"
                "MASTERY_TAIL_THRESHOLD = None\n"
                "SCIENTIFIC_VALUE = 1\n"
            )
        else:
            path.write_text("SCIENTIFIC_VALUE = 1\n")


@pytest.mark.parametrize("relative", SCORING_SOURCE_RELATIVE_PATHS)
def test_scoring_contract_digest_covers_every_active_source(
    tmp_path: Path, relative: str,
) -> None:
    _write_scoring_sources(tmp_path)
    initial = _scoring_source_sha256(tmp_path)
    path = tmp_path / relative
    path.write_text(path.read_text() + "SCIENTIFIC_CHANGE = 2\n")
    assert _scoring_source_sha256(tmp_path) != initial


def test_scoring_contract_digest_normalizes_only_mastery_values(tmp_path: Path) -> None:
    _write_scoring_sources(tmp_path)
    contract = tmp_path / "grader" / "contract.py"
    initial = _scoring_source_sha256(tmp_path)
    contract.write_text(
        "MASTERY_THRESHOLD = 0.456\n"
        "MASTERY_TAIL_THRESHOLD = 0.312\n"
        "SCIENTIFIC_VALUE = 1\n"
    )
    assert _scoring_source_sha256(tmp_path) == initial

    contract.write_text(
        "MASTERY_THRESHOLD = 0.456\n"
        "MASTERY_TAIL_THRESHOLD = 0.312\n"
        "SCIENTIFIC_VALUE = 2\n"
    )
    assert _scoring_source_sha256(tmp_path) != initial


def test_generation_pipeline_digest_is_current_not_merely_uniform() -> None:
    implementation_hashes = _implementation_hashes()
    assert (
        implementation_hashes["generation_pipeline_sha256"]
        == _generation_pipeline_sha256()
    )
    stale_results = [
        {
            "id": "first",
            "source_provenance": {
                "generation_pipeline_sha256": "a" * 64,
                "public_reference_source_sha256": implementation_hashes[
                    "public_reference_source_sha256"
                ],
            },
        },
        {
            "id": "second",
            "source_provenance": {
                "generation_pipeline_sha256": "a" * 64,
                "public_reference_source_sha256": implementation_hashes[
                    "public_reference_source_sha256"
                ],
            },
        },
    ]
    failures, observed = _source_provenance_failures(
        stale_results,
        implementation_hashes,
    )
    assert observed == {"a" * 64}
    assert failures == [
        "first",
        "second",
        "generation_pipeline_not_current_and_uniform",
    ]


def test_manifest_pins_the_exact_development_report_digest() -> None:
    manifest = _final_manifest()
    _require_frozen_development_report_digest(manifest, "d" * 64)
    with pytest.raises(ShippingGateError, match="frozen in the manifest"):
        _require_frozen_development_report_digest(manifest, "e" * 64)


def test_json_and_partial_result_readers_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"ok": True}), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ShippingGateError, match="plain file"):
        _read_json(link)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "one.json").symlink_to(target)
    with pytest.raises(ShippingGateError, match="non-JSON plain file"):
        _result_files(results_dir, {"one.json"})


def test_final_result_derived_fields_are_recomputed_from_probabilities_and_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    targets = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    probabilities = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
    schema = SimpleNamespace()
    training = SimpleNamespace(
        schema=schema,
        sample_ids=tuple(f"train-{index}" for index in range(6)),
        targets=np.zeros(6, dtype=np.float64),
        genotypes=np.zeros((6, 3), dtype=np.float64),
    )
    prediction = SimpleNamespace(
        sample_ids=("s0", "s1", "s2", "s3"),
    )
    naive = {"auc": 0.5, "brier": 0.25, "log_loss": 0.7}
    reference = {"auc": 0.9, "brier": 0.12, "log_loss": 0.4}
    standard_errors = {"auc": 0.01, "brier": 0.01, "log_loss": 0.01}
    naive_standard_errors = {"auc": 0.01, "brier": 0.01, "log_loss": 0.01}
    anchors = {
        "id": "d_" + "a" * 32,
        "path": "case/d_" + "a" * 32,
        "category": "case",
        "replicate": 0,
        "family": "binomial-logit",
        "metrics_naive": naive,
        "metrics_reference": reference,
        "metrics_ref_naive_se": standard_errors,
        "metrics_naive_se": naive_standard_errors,
        "provenance": {"generation_pipeline_sha256": "a" * 64},
        "reference_fit": {
            "public_source_sha256": "b" * 64,
        },
        "reference_protocol": {
            "reference_method": "hierarchical_eb",
            "inner_selection_auc": {
                "hierarchical_eb": 0.9, "ridge_logistic": 0.6},
            "prediction_exit_code": 0,
            "prediction_model_sha256": "1" * 64,
            "prediction_output_sha256": "2" * 64,
            "prediction_output_byte_count": 64,
            "prediction_source_sha256": "b" * 64,
        },
    }
    monkeypatch.setattr(model_zoo, "load_training_dataset", lambda _path: training)
    monkeypatch.setattr(
        model_zoo,
        "load_prediction_dataset",
        lambda _path, *, training: prediction,
    )
    monkeypatch.setattr(
        model_zoo,
        "_read_test_targets",
        lambda _path, _sample_ids: targets,
    )
    monkeypatch.setattr(model_zoo, "_read_json", lambda _path: anchors)
    entry = {
        "id": anchors["id"],
        "path": anchors["path"],
        "category": anchors["category"],
        "replicate": anchors["replicate"],
        "family": anchors["family"],
    }
    model_result = _model_result_from_probability(
        "strong_reference",
        targets,
        probabilities,
        naive,
        reference,
        standard_errors,
        naive_standard_errors,
    )
    result = {
        "n_train": 6,
        "n_test": 4,
        "n_variants": 3,
        "source_provenance": {
            "generation_pipeline_sha256": "a" * 64,
            "public_reference_source_sha256": "b" * 64,
        },
        "anchor_health": model_zoo._anchor_health(
            entry["id"], naive, reference, standard_errors,
            naive_standard_errors
        ),
        "strong_reference_reproduction_error": {
            metric: abs(model_result["metrics"][metric] - reference[metric])
            for metric in METRIC_DIRECTIONS
        },
        "strong_reference_probabilities": probabilities.tolist(),
        "models": {"strong_reference": model_result},
        "fit_protocol": {
            "reference_method": "hierarchical_eb",
            "inner_selection_auc": {
                "hierarchical_eb": 0.9, "ridge_logistic": 0.6},
            "prediction_exit_code": 0,
            "prediction_wall_s": 0.1,
            "prediction_model_sha256": "1" * 64,
            "prediction_output_sha256": "2" * 64,
            "prediction_output_byte_count": 64,
            "prediction_source_sha256": "b" * 64,
            "execution_order": [
                "strong_reference_fit",
                "prediction_data_load",
                "strong_reference_prediction",
                "truth_load",
            ],
        },
        "wall_time_s": 1.0,
        model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD: "d" * 64,
    }
    validated = _validated_final_result_payload(tmp_path, entry, result)
    assert "strong_reference_probabilities" not in validated
    assert set(validated["prediction_evidence"]) == (
        model_zoo.FINAL_AUDIT_PREDICTION_EVIDENCE_FIELDS
    )
    assert model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD not in validated

    tampered = deepcopy(result)
    tampered["anchor_health"]["auc"]["oriented_gap"] = 999.0
    with pytest.raises(ShippingGateError, match="does not reproduce trusted"):
        _validated_final_result_payload(tmp_path, entry, tampered)

    # The winning method's inner-selection AUC must be a finite number.
    protocol_tampered = deepcopy(result)
    protocol_tampered["fit_protocol"]["inner_selection_auc"]["hierarchical_eb"] = None
    with pytest.raises(ShippingGateError, match="final fit protocol is invalid"):
        _validated_final_result_payload(tmp_path, entry, protocol_tampered)

    selection_tampered = deepcopy(result)
    selection_tampered["fit_protocol"]["inner_selection_auc"]["hierarchical_eb"] = 0.91
    with pytest.raises(ShippingGateError, match="authenticated anchor"):
        _validated_final_result_payload(tmp_path, entry, selection_tampered)


def test_final_audit_persists_separately_authenticated_rebuildable_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = {
        "id": "d_" + "a" * 32,
        "path": "case/d_" + "a" * 32,
        "category": "case",
        "replicate": 0,
        "family": "binomial-logit",
    }
    manifest_sha256 = "b" * 64
    implementation_hashes = {"source": "c" * 64}
    partial = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "purpose": "final_audit",
        "manifest_sha256": manifest_sha256,
        "manifest_entry_sha256": model_zoo._canonical_json_sha256(entry),
        "implementation_sha256": implementation_hashes,
        **entry,
        "n_train": 6,
        "n_test": 4,
        "n_variants": 3,
        "source_provenance": {},
        "anchor_health": {},
        "strong_reference_reproduction_error": {},
        "strong_reference_probabilities": [0.2, 0.8, 0.3, 0.7],
        "models": {"strong_reference": {}},
        "fit_protocol": {
            "reference_method": "hierarchical_eb",
            "inner_selection_auc": {
                "hierarchical_eb": 0.9, "ridge_logistic": 0.6},
            "prediction_exit_code": 0,
            "prediction_wall_s": 0.1,
            "prediction_model_sha256": "1" * 64,
            "prediction_output_sha256": "2" * 64,
            "prediction_output_byte_count": 64,
            "prediction_source_sha256": "b" * 64,
            "execution_order": [
                "strong_reference_fit",
                "prediction_data_load",
                "strong_reference_prediction",
                "truth_load",
            ],
        },
        "wall_time_s": 1.0,
    }
    partial[model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        partial,
        exclude_field=model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD,
        domain=model_zoo.FINAL_AUDIT_RESULT_HMAC_DOMAIN,
    )
    sanitized = dict(partial)
    del sanitized["strong_reference_probabilities"]
    del sanitized[model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD]
    sanitized["prediction_evidence"] = model_zoo._encode_prediction_evidence(
        entry["id"], ("s0", "s1", "s2", "s3"),
        np.asarray(partial["strong_reference_probabilities"], dtype=np.float64),
    )

    monkeypatch.setattr(
        model_zoo,
        "_final_audit_context",
        lambda *_args: (
            ShippingThresholds(),
            object(),
            {},
            [entry],
            implementation_hashes,
        ),
    )
    monkeypatch.setattr(
        model_zoo,
        "_validated_final_result_payload",
        lambda _corpus, observed_entry, observed_partial: (
            dict(sanitized)
            if observed_entry == entry and observed_partial == partial
            else pytest.fail("unexpected raw partial ingestion")
        ),
    )
    monkeypatch.setattr(
        model_zoo,
        "_assemble_final_audit_report",
        lambda _manifest, _manifest_sha, rows, *_args, **_kwargs: {
            "dataset_results": list(rows)
        },
    )
    monkeypatch.setattr(
        model_zoo,
        "_recompute_persisted_final_evidence",
        lambda _corpus, observed_entry, _evidence: (
            None if observed_entry == entry else pytest.fail("unexpected evidence row")
        ),
    )

    report = model_zoo.build_final_audit_report_from_partials(
        tmp_path,
        {"datasets": [entry]},
        manifest_sha256,
        [partial],
        {},
        "d" * 64,
        auth_key=AUTH_KEY,
    )
    evidence = report["dataset_results"][0]
    assert set(evidence) == model_zoo.FINAL_AUDIT_EVIDENCE_FIELDS
    assert "strong_reference_probabilities" not in evidence
    assert set(evidence["prediction_evidence"]) == (
        model_zoo.FINAL_AUDIT_PREDICTION_EVIDENCE_FIELDS
    )
    assert model_zoo.FINAL_AUDIT_RESULT_HMAC_FIELD not in evidence
    assert model_zoo.verify_json_hmac(
        AUTH_KEY,
        evidence,
        field=model_zoo.FINAL_AUDIT_EVIDENCE_HMAC_FIELD,
        domain=model_zoo.FINAL_AUDIT_EVIDENCE_HMAC_DOMAIN,
    )

    regenerated = model_zoo.rebuild_final_audit_report_from_evidence(
        tmp_path,
        {"datasets": [entry]},
        manifest_sha256,
        report["dataset_results"],
        {},
        "d" * 64,
        auth_key=AUTH_KEY,
    )
    assert regenerated == report

    tampered = deepcopy(evidence)
    tampered["wall_time_s"] = 2.0
    with pytest.raises(ShippingGateError, match="evidence provenance mismatch"):
        model_zoo.rebuild_final_audit_report_from_evidence(
            tmp_path,
            {"datasets": [entry]},
            manifest_sha256,
            [tampered],
            {},
            "d" * 64,
            auth_key=AUTH_KEY,
        )

    raw_schema = deepcopy(evidence)
    raw_schema["strong_reference_probabilities"] = [0.5] * 4
    with pytest.raises(ShippingGateError, match="evidence provenance mismatch"):
        model_zoo.rebuild_final_audit_report_from_evidence(
            tmp_path,
            {"datasets": [entry]},
            manifest_sha256,
            [raw_schema],
            {},
            "d" * 64,
            auth_key=AUTH_KEY,
        )


def test_prediction_evidence_rejects_missing_tampered_and_reordered_content() -> None:
    dataset_id = "d_" + "e" * 32
    sample_ids = ("s0", "s1", "s2", "s3")
    probabilities = np.asarray([0.1, 0.8, 0.3, 0.9], dtype=np.float64)
    evidence = model_zoo._encode_prediction_evidence(
        dataset_id, sample_ids, probabilities
    )
    decoded = model_zoo._decode_prediction_evidence(
        evidence, dataset_id=dataset_id, sample_ids=sample_ids
    )
    assert np.array_equal(decoded, probabilities)

    missing = deepcopy(evidence)
    del missing["compressed_sha256"]
    with pytest.raises(ShippingGateError, match="identity, shape, or row order"):
        model_zoo._decode_prediction_evidence(
            missing, dataset_id=dataset_id, sample_ids=sample_ids
        )

    tampered = deepcopy(evidence)
    payload = tampered["compressed_base64"]
    tampered["compressed_base64"] = (
        ("A" if payload[0] != "A" else "B") + payload[1:]
    )
    with pytest.raises(ShippingGateError, match="prediction evidence"):
        model_zoo._decode_prediction_evidence(
            tampered, dataset_id=dataset_id, sample_ids=sample_ids
        )

    with pytest.raises(ShippingGateError, match="row order"):
        model_zoo._decode_prediction_evidence(
            evidence,
            dataset_id=dataset_id,
            sample_ids=tuple(reversed(sample_ids)),
        )


def test_development_evidence_recomputes_every_model_and_check_input() -> None:
    category = next(iter(CATEGORIES))
    replicate = DEVELOPMENT_REPLICATES[0]
    dataset_id = "d_" + "7" * 32
    # Enough held-out samples that the bootstrap naive-AUC SE is small relative to
    # the reference-naive gap: with only 24 rows the SE was so wide that 4*se_naive
    # exceeded the ~0.45 AUC gap and the (adequate, healthy) reference was excluded
    # as an inadequate denominator, leaving accuracy_skill with no active metric.
    sample_ids = tuple(f"s{index}" for index in range(200))
    targets = np.asarray(([0.0, 1.0] * 100), dtype=np.float64)
    naive = np.linspace(0.18, 0.62, targets.size, dtype=np.float64)
    reference = np.clip(0.17 + 0.66 * targets + np.linspace(
        -0.03, 0.03, targets.size), 0.02, 0.98)
    model_probabilities = {
        name: (
            reference
            if name == "strong_reference"
            else np.clip(
                naive + (index / (len(MODEL_ORDER) + 1)) * (reference - naive),
                0.01,
                0.99,
            )
        )
        for index, name in enumerate(MODEL_ORDER, start=1)
    }
    model_probabilities["strong_reference"] = reference
    pipeline_sha256 = "9" * 64
    bootstrap_seed = model_zoo.derive_stream_seed(
        AUTH_KEY,
        purpose="development",
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream="bootstrap",
    )
    naive_metrics = model_zoo._metric_values(targets, naive)
    reference_metrics = model_zoo._metric_values(targets, reference)
    standard_errors, naive_standard_errors = (
        model_zoo._bootstrap_anchor_standard_errors(
            targets, naive, reference, seed=bootstrap_seed
        )
    )
    result = {
        "id": dataset_id,
        "generation_pipeline_sha256": pipeline_sha256,
        "category": category,
        "replicate": replicate,
        "n_test": targets.size,
        "anchor_health": model_zoo._anchor_health(
            dataset_id,
            naive_metrics,
            reference_metrics,
            standard_errors,
            naive_standard_errors,
        ),
        "models": model_zoo._model_results(
            model_probabilities,
            targets,
            naive_metrics,
            reference_metrics,
            standard_errors,
            naive_standard_errors,
        ),
        "metric_evidence": {
            "sample_ids": list(sample_ids),
            "targets": model_zoo._encode_prediction_evidence(
                dataset_id, sample_ids, targets, model="truth"
            ),
            "naive": model_zoo._encode_prediction_evidence(
                dataset_id, sample_ids, naive, model="naive"
            ),
            "models": {
                name: model_zoo._encode_prediction_evidence(
                    dataset_id, sample_ids, values, model=name
                )
                for name, values in model_probabilities.items()
            },
        },
    }
    _REAL_RECOMPUTE_DEVELOPMENT_METRIC_EVIDENCE(result, auth_key=AUTH_KEY)

    metric_drift = deepcopy(result)
    metric_drift["models"]["ridge_logistic"]["raw_skill"] += 0.01
    with pytest.raises(ShippingGateError, match="do not reproduce retained vectors"):
        _REAL_RECOMPUTE_DEVELOPMENT_METRIC_EVIDENCE(
            metric_drift, auth_key=AUTH_KEY
        )

    reordered = deepcopy(result)
    reordered["metric_evidence"]["models"]["ridge_logistic"] = (
        model_zoo._encode_prediction_evidence(
            dataset_id,
            sample_ids,
            model_probabilities["ridge_logistic"][::-1],
            model="ridge_logistic",
        )
    )
    with pytest.raises(ShippingGateError, match="do not reproduce retained vectors"):
        _REAL_RECOMPUTE_DEVELOPMENT_METRIC_EVIDENCE(
            reordered, auth_key=AUTH_KEY
        )

    missing = deepcopy(result)
    del missing["metric_evidence"]["models"]["ridge_logistic"]
    with pytest.raises(ShippingGateError, match="evidence schema is invalid"):
        _REAL_RECOMPUTE_DEVELOPMENT_METRIC_EVIDENCE(missing, auth_key=AUTH_KEY)


def test_persisted_prediction_vectors_recompute_metrics_and_reject_reordering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = {
        "id": "d_" + "f" * 32,
        "path": "case/d_" + "f" * 32,
        "category": "case",
        "replicate": 0,
        "family": "binomial-logit",
    }
    sample_ids = ("s0", "s1", "s2", "s3")
    probabilities = np.asarray([0.1, 0.8, 0.3, 0.9], dtype=np.float64)
    training = SimpleNamespace()
    prediction = SimpleNamespace(sample_ids=sample_ids)
    monkeypatch.setattr(model_zoo, "load_training_dataset", lambda _path: training)
    monkeypatch.setattr(
        model_zoo,
        "load_prediction_dataset",
        lambda _path, *, training: prediction,
    )

    def order_sensitive_metric(values: np.ndarray) -> float:
        return float(np.dot(np.arange(1, values.size + 1), values))

    def recompute(_corpus, observed_entry, raw_result):
        assert observed_entry == entry
        values = np.asarray(
            raw_result["strong_reference_probabilities"], dtype=np.float64
        )
        return {
            "id": entry["id"],
            "models": {
                "strong_reference": {
                    "metrics": {"order_sensitive": order_sensitive_metric(values)}
                }
            },
            "prediction_evidence": model_zoo._encode_prediction_evidence(
                entry["id"], sample_ids, values
            ),
        }

    monkeypatch.setattr(model_zoo, "_validated_final_result_payload", recompute)
    evidence = {
        "id": entry["id"],
        "models": {
            "strong_reference": {
                "metrics": {"order_sensitive": order_sensitive_metric(probabilities)}
            }
        },
        "prediction_evidence": model_zoo._encode_prediction_evidence(
            entry["id"], sample_ids, probabilities
        ),
        model_zoo.FINAL_AUDIT_EVIDENCE_HMAC_FIELD: "authenticated-by-caller",
    }
    model_zoo._recompute_persisted_final_evidence(tmp_path, entry, evidence)

    metric_drift = deepcopy(evidence)
    metric_drift["models"]["strong_reference"]["metrics"]["order_sensitive"] += 0.01
    with pytest.raises(ShippingGateError, match="metrics or scientific fields drifted"):
        model_zoo._recompute_persisted_final_evidence(
            tmp_path, entry, metric_drift
        )

    reordered = deepcopy(evidence)
    reordered_values = probabilities[::-1]
    reordered["prediction_evidence"] = model_zoo._encode_prediction_evidence(
        entry["id"], sample_ids, reordered_values
    )
    with pytest.raises(ShippingGateError, match="metrics or scientific fields drifted"):
        model_zoo._recompute_persisted_final_evidence(tmp_path, entry, reordered)


def test_final_audit_loads_prediction_and_truth_only_after_fit_and_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    schema = SimpleNamespace(family="binomial-logit")
    training = SimpleNamespace(
        schema=schema,
        sample_ids=tuple(f"train-{index}" for index in range(6)),
        targets=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        genotypes=np.zeros((6, 3), dtype=np.float64),
        covariates=np.zeros((6, 2), dtype=np.float64),
        variant_records=(object(), object(), object()),
    )
    prediction = SimpleNamespace(
        sample_ids=("s0", "s1", "s2", "s3"),
        genotypes=np.zeros((4, 3), dtype=np.float64),
        covariates=np.zeros((4, 2), dtype=np.float64),
    )
    targets = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    model = _OrderedDecisionModel(
        events,
        "strong_prediction",
        np.asarray([-2.0, 2.0, -2.0, 2.0]),
    )
    model.model_bytes = b'{"locked":"model"}'
    model_sha256 = hashlib.sha256(model.model_bytes).hexdigest()
    pred_bytes = b"mean\n0.11920292202211755\n0.8807970779778823\n0.11920292202211755\n0.8807970779778823\n"
    pred_sha256 = hashlib.sha256(pred_bytes).hexdigest()
    diagnostics = _svpgs_fit_diagnostics(0.2)
    entry = {
        "id": "d_" + "a" * 32,
        "path": "case/d_" + "a" * 32,
        "category": "case",
        "replicate": 0,
        "family": "binomial-logit",
    }
    anchors = {
        **entry,
        "metrics_naive": {"auc": 0.5, "brier": 0.25, "log_loss": 0.7},
        "metrics_reference": {"auc": 1.0, "brier": 0.02, "log_loss": 0.13},
        "metrics_ref_naive_se": {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        "metrics_naive_se": {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        "provenance": {"generation_pipeline_sha256": "a" * 64},
        "reference_fit": {
            "public_source_sha256": "b" * 64,
        },
        "reference_protocol": {
            "protocol_version": SHIPPED_PROTOCOL_VERSION,
            "reference_method": "hierarchical_eb",
            "converged": True,
            "fit_protocol": SHIPPED_FIT_PROTOCOL,
            "wall_time_s": 1.0,
            "fit_exit_code": 0,
            "fit_source_sha256": "b" * 64,
            "fit_python_executable": "/opt/svpgs-venv/bin/python",
            "fit_numpy_version": "2.1.3",
            "reference_total_fit_wall_s": 1.0,
            "reference_full_fit_wall_s": 0.5,
            "reference_program_reported_wall_s": 0.9,
            "selection_wall_s": 0.4,
            "calibration_intercept": -0.1,
            "calibration_slope": 0.8,
            "calibration_iterations": 4,
            "calibration_relative_gradient": 1e-10,
            "inner_selection_auc": {
                "hierarchical_eb": 0.71, "ridge_logistic": 0.65},
            "inner_ridge_trials": _inner_ridge_trials(),
            "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_global_scale": 0.5,
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.5,
            "prediction_exit_code": 0,
            "prediction_wall_s": 0.1,
            "prediction_output_sha256": pred_sha256,
            "prediction_output_byte_count": len(pred_bytes),
            "prediction_model_sha256": model_sha256,
            "prediction_source_sha256": "b" * 64,
            "classes_present": ["snv"],
            "training_output": {
                "model_sha256": model_sha256,
                "model_byte_count": len(model.model_bytes),
            },
        },
    }

    def load_training(_path):
        events.append("training_load")
        return training

    def fit_reference(training_arg, dataset_id, public_dir):
        assert training_arg is training
        assert public_dir == tmp_path / entry["path"] / "public"
        events.append("fit")
        return ("hierarchical_eb", model, diagnostics,
                {"hierarchical_eb": 0.71, "ridge_logistic": 0.65}, ["snv"])

    expected_training = training

    def load_prediction(_path, *, training):
        assert training is expected_training
        assert events == ["training_load", "fit"]
        events.append("prediction_load")
        return prediction

    def run_reference_predict(model_arg, _public, _dataset_id, n_test, _family):
        assert model_arg.model_bytes == model.model_bytes
        assert n_test == 4
        assert events[-1] == "prediction_load"
        events.append("strong_prediction")
        return {
            "status": "ok", "detail": "", "predict_rc": 0,
            "t_predict": 0.1,
            "pred": {"mean": np.asarray([0.11920292202211755, 0.8807970779778823,
                                           0.11920292202211755, 0.8807970779778823])},
            "pred_bytes": pred_bytes,
            "pred_sha256": pred_sha256,
            "model_sha256": model_sha256,
            "source_sha256": "b" * 64,
        }

    def read_targets(_path, sample_ids):
        assert sample_ids == prediction.sample_ids
        assert events[-1] == "strong_prediction"
        events.append("truth_targets")
        return targets

    def read_anchors(_path):
        assert events[-1] == "truth_targets"
        events.append("truth_anchors")
        return anchors

    monkeypatch.setattr(model_zoo, "load_training_dataset", load_training)
    monkeypatch.setattr(model_zoo, "_fit_best_of_family_reference", fit_reference)
    monkeypatch.setattr(model_zoo, "load_prediction_dataset", load_prediction)
    monkeypatch.setattr(
        model_zoo, "_predict_public_reference", run_reference_predict)
    monkeypatch.setattr(
        model_zoo, "_public_reference_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(model_zoo, "_read_test_targets", read_targets)
    monkeypatch.setattr(model_zoo, "_read_json", read_anchors)

    result = model_zoo.evaluate_final_dataset(
        tmp_path,
        entry,
        manifest_sha256="d" * 64,
    )

    assert events == [
        "training_load",
        "fit",
        "prediction_load",
        "strong_prediction",
        "truth_targets",
        "truth_anchors",
    ]
    assert result["fit_protocol"]["execution_order"][-1] == "truth_load"


def test_development_model_zoo_fits_every_model_before_prediction_data_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    category = next(iter(CATEGORIES))
    schema = SimpleNamespace(family="binomial-logit")
    training = SimpleNamespace(
        schema=schema,
        sample_ids=tuple(f"train-{index}" for index in range(6)),
        targets=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        genotypes=np.zeros((6, 3), dtype=np.float64),
        covariates=np.zeros((6, 2), dtype=np.float64),
    )
    prediction = SimpleNamespace(
        sample_ids=("s0", "s1", "s2", "s3"),
        genotypes=np.zeros((4, 3), dtype=np.float64),
        covariates=np.zeros((4, 2), dtype=np.float64),
    )
    strong_logits = np.asarray([-2.0, 2.0, -2.0, 2.0])
    models = {
        name: _OrderedDecisionModel(
            events,
            f"model_prediction:{name}",
            strong_logits,
        )
        for name in MODEL_ORDER
    }
    naive_model = _OrderedDecisionModel(
        events,
        "naive_prediction",
        np.zeros(4, dtype=np.float64),
    )
    targets = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)

    monkeypatch.setattr(
        model_zoo,
        "_implementation_hashes",
        lambda: {"generation_pipeline_sha256": "a" * 64},
    )
    monkeypatch.setattr(model_zoo, "generate", lambda _config: object())

    def materialize_generated(*_args, **_kwargs):
        events.append("materialize")

    def load_training(_path):
        assert events == ["materialize"]
        events.append("training_load")
        return training

    def fit_zoo(value, config):
        assert value is training
        assert config is not None
        events.append("zoo_fit")
        return models, _development_fit_diagnostics(training_count=6)

    def fit_naive(value, config):
        assert value is training
        assert config is not None
        events.append("naive_fit")
        return naive_model

    expected_training = training

    def load_prediction(_path, *, training):
        assert training is expected_training
        assert events[-2:] == ["zoo_fit", "naive_fit"]
        events.append("prediction_load")
        return prediction

    def read_targets(_path, sample_ids):
        assert sample_ids == prediction.sample_ids
        assert events.count("naive_prediction") == 1
        assert sum(event.startswith("model_prediction:") for event in events) == len(
            MODEL_ORDER
        )
        events.append("truth_load")
        return targets

    def fit_best_of_family(train_arg, dataset_id, public_dir):
        # The dev reference is now the shipped best-of-family; return the SAME
        # strong_reference model so the override in evaluate_development_dataset is a
        # no-op here and the prediction-order events are unchanged.
        assert train_arg is training
        assert Path(public_dir).name == "public"

        class BestOfFamilyModel:
            model_bytes = b'{"locked":"development-model"}'

        diagnostics = {
            "fit_exit_code": 0,
            "fit_source_sha256": "b" * 64,
            "fit_python_executable": "/opt/svpgs-venv/bin/python",
            "fit_numpy_version": "2.1.3",
            "reference_total_fit_wall_s": 1.0,
            "reference_full_fit_wall_s": 0.5,
            "reference_program_reported_wall_s": 0.9,
            "selection_wall_s": 0.4,
            "calibration_intercept": -0.1,
            "calibration_slope": 0.8,
            "calibration_iterations": 4,
            "calibration_relative_gradient": 1e-10,
            "inner_ridge_trials": _inner_ridge_trials(),
            "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_global_scale": 0.5,
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.5,
        }
        return ("hierarchical_eb", BestOfFamilyModel(), diagnostics,
                {"hierarchical_eb": 0.7, "ridge_logistic": 0.65}, ["snv"])

    def predict_best_of_family(model_arg, _public, _dataset_id, _n_test, _family):
        assert model_arg.model_bytes == b'{"locked":"development-model"}'
        events.append("model_prediction:strong_reference")
        pred_bytes = b"mean\n0.11920292202211755\n0.8807970779778823\n0.11920292202211755\n0.8807970779778823\n"
        return {
            "status": "ok", "detail": "", "predict_rc": 0,
            "t_predict": 0.1,
            "pred": {"mean": 1.0 / (1.0 + np.exp(-strong_logits))},
            "pred_bytes": pred_bytes,
            "pred_sha256": hashlib.sha256(pred_bytes).hexdigest(),
            "model_sha256": hashlib.sha256(model_arg.model_bytes).hexdigest(),
            "source_sha256": "b" * 64,
        }

    monkeypatch.setattr(model_zoo, "materialize", materialize_generated)
    monkeypatch.setattr(model_zoo, "load_training_dataset", load_training)
    monkeypatch.setattr(model_zoo, "_fit_model_zoo", fit_zoo)
    monkeypatch.setattr(model_zoo, "_fit_covariate_model", fit_naive)
    monkeypatch.setattr(model_zoo, "_fit_best_of_family_reference", fit_best_of_family)
    monkeypatch.setattr(
        model_zoo, "_predict_public_reference", predict_best_of_family)
    monkeypatch.setattr(
        model_zoo, "_public_reference_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(model_zoo, "load_prediction_dataset", load_prediction)
    monkeypatch.setattr(model_zoo, "_read_test_targets", read_targets)
    monkeypatch.setattr(
        model_zoo,
        "_bootstrap_anchor_standard_errors",
        # Returns BOTH yardsticks from one resampling loop: (gap_se, naive_se).
        lambda *_args, **_kwargs: (
            {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
            {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        ),
    )
    monkeypatch.setattr(
        model_zoo,
        "_dataset_file_hashes",
        lambda _path: {"public/dgp.json": "b" * 64},
    )

    result = model_zoo.evaluate_development_dataset(
        category,
        0,
        tmp_path,
        config=model_zoo._declared_reference_config(),
        auth_key=AUTH_KEY,
    )

    assert events.index("zoo_fit") < events.index("prediction_load")
    assert events.index("naive_fit") < events.index("prediction_load")
    assert events[-1] == "truth_load"
    assert result["fit_protocol"]["execution_order"] == [
        "all_model_fits",
        "prediction_data_load",
        "all_model_predictions",
        "truth_load",
    ]


def test_standalone_refits_construct_cross_tuning_aware_selections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    strong_fit = SimpleNamespace(
        model=SimpleNamespace(component_names=(), component_models=()),
        diagnostics={
            "candidates": {
                name: {
                    "selected_hyperparameters": _candidate_hyperparameters(name),
                    "validation_log_loss": 0.5,
                }
                for name in model_zoo._strong._CANDIDATE_ORDER
            }
        },
    )

    def refit(selection, _dataset, _config, prepared):
        captured.append(selection)
        return SimpleNamespace(name=selection.name), {"converged": True}, prepared

    monkeypatch.setattr(model_zoo._strong, "_refit_candidate", refit)
    models, _ = model_zoo._standalone_reference_candidates(
        SimpleNamespace(),
        strong_fit,
        model_zoo._declared_reference_config(),
    )

    assert tuple(models) == model_zoo._strong._CANDIDATE_ORDER
    assert len(captured) == len(model_zoo._strong._CANDIDATE_ORDER)
    assert all(selection.cross_tuning == {} for selection in captured)
