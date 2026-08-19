"""The MODEL-ZOO's candidate-selector contract. NOT the shipped reference.

READ THIS BEFORE TRUSTING ANY NUMBER PRODUCED THROUGH THIS MODULE.

There are two different model families in this repo and they serve different roles:

  * ``reference/protocol.py`` -- the exact public NumPy best of {hierarchical
    empirical Bayes, ridge logistic}, selected using training data only and refit
    on the full training set. THIS ONE IS SHIPPED and is baked into every
    ``truth/anchors.json`` as skill = 1.0.
  * ``reference/zoo_protocol.py`` (this file) -- an "immutable-training-only
    SV-PGS candidate selector" over REFERENCE_CANDIDATE_ORDER. THIS ONE IS NOT
    SHIPPED. Only ``validation/model_zoo.py`` imports it.

This module used to describe itself as "the single source of truth for the
benchmark's strong-reference contract" while protocol.py described itself as "the
single source of truth for the benchmark's reference contract". Two sources of
truth for one concept is not a documentation problem; it is the bug. It cost the
v6 corpus its anchor:

Historically the development gate validated THIS selector, passed all six checks,
and then the corpus shipped anchors built from raw SV-PGS -- a model
the very same frozen report (validation/model_zoo_development.json) scores as a
mere CONTENDER, and records as NOT the best on 8 of 15 categories (0.034 skill on
dense_infinitesimal, where a plain ridge scores 0.996; dead last on ld_shift and
soft_membership). Every saturation symptom in the v6 corpus is downstream of the
gate certifying a reference the corpus does not ship.

The current development gate still fits this broader zoo for diversity evidence,
but explicitly replaces its reference arm with the shipped best-of-family model
before checking reliability, ceiling capture, or saturation. A per-dataset
selector over contenders would make a no-contender-beats-reference check
tautological; it must never define the shipped skill axis.

Full record: rollout_analysis/ROOT_CAUSE_2026-07-17_gate_ships_a_different_reference.md
"""

from __future__ import annotations

import math
from typing import Any

from grader.contract import SELECTION_METRIC_WEIGHTS as METRIC_WEIGHTS
from reference.protocol import (
    REFERENCE_CONVERGENCE_TOLERANCE,
    REFERENCE_MAX_OUTER_ITERATIONS,
)

REFERENCE_IMPLEMENTATION = "immutable-training-only SV-PGS candidate selector"
REFERENCE_ENTRYPOINT = "reference.strong_reference.fit_strong_reference"
REFERENCE_FIT_PROTOCOL = (
    "immutable_training_inner_validation_cross_tuning_then_full_refit"
)
REFERENCE_PROTOCOL_VERSION = 12
REFERENCE_CANDIDATE_ORDER = (
    "sv_pgs",
    "ridge_logistic",
    "dense_spike_logistic",
    "elastic_net_logistic",
    "marginal_pt",
    "annotation_adaptive_en",
)
REFERENCE_TUNING_LABELS = "training_dataset_inner_validation_only"
REFERENCE_FIT_INPUT_TYPE = "TrainingDataset"
REFERENCE_FIT_INPUT_FIELDS = (
    "schema",
    "sample_ids",
    "genotypes",
    "covariates",
    "targets",
    "variant_records",
)
REFERENCE_FULL_REFIT_RULE = (
    "cross_tuned_blend_or_regime_incumbent_or_clear_single_or_top_two_then_distill"
)
# Treat numerical blend dust as rejection while preserving an incumbent that the
# honest validation blend assigns a substantive positive component.
REFERENCE_INCUMBENT_MINIMUM_BLEND_WEIGHT = 0.02
REFERENCE_INCUMBENT_DISTILLATION_MAX_CORRELATION = 0.7
REFERENCE_ANCHOR_EXECUTION_ORDER = (
    "all_anchor_fits",
    "prediction_data_load",
    "all_anchor_predictions",
    "truth_load",
)
REFERENCE_DEVELOPMENT_EXECUTION_ORDER = (
    "all_model_fits",
    "prediction_data_load",
    "all_model_predictions",
    "truth_load",
)
REFERENCE_FINAL_EXECUTION_ORDER = (
    "strong_reference_fit",
    "prediction_data_load",
    "strong_reference_prediction",
    "truth_load",
)


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _valid_validation_metrics(value: Any) -> bool:
    return bool(
        type(value) is dict
        and set(value) == set(METRIC_WEIGHTS)
        and all(_finite_number(metric) for metric in value.values())
        and 0.0 <= float(value["auc"]) <= 1.0
        and float(value["brier"]) >= 0.0
        and float(value["log_loss"]) >= 0.0
    )


def _selection_utility(
    metrics: dict[str, Any],
    naive_metrics: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for name in METRIC_WEIGHTS:
        observed = float(metrics[name])
        naive = float(naive_metrics[name])
        if name == "auc":
            room = 1.0 - naive
            improvement = observed - naive
        else:
            room = naive
            improvement = naive - observed
        if room <= 1e-12:
            raise ValueError("reference validation objective has no estimable room")
        components[name] = improvement / room
    weight_sum = sum(METRIC_WEIGHTS.values())
    utility = sum(
        METRIC_WEIGHTS[name] * components[name] for name in METRIC_WEIGHTS
    ) / weight_sum
    return float(utility), components


def _valid_optimizer_diagnostics(value: Any) -> bool:
    return bool(
        type(value) is dict
        and set(value) == {"converged", "iterations", "objective", "lipschitz"}
        and value["converged"] is True
        and type(value["iterations"]) is int
        and value["iterations"] > 0
        and _finite_number(value["objective"])
        and float(value["objective"]) >= 0.0
        and _finite_number(value["lipschitz"])
        and float(value["lipschitz"]) > 0.0
    )


def validate_optimizer_diagnostics(diagnostics: Any) -> None:
    """Validate the exact optimizer record shared by reference candidates."""
    if not _valid_optimizer_diagnostics(diagnostics):
        raise ValueError("reference optimizer diagnostics are invalid")


def _valid_candidate_hyperparameters(candidate_name: str, value: Any) -> bool:
    if type(value) is not dict:
        return False
    if candidate_name == "sv_pgs":
        return bool(
            set(value) == {"max_outer_iterations", "convergence_tolerance"}
            and type(value["max_outer_iterations"]) is int
            and value["max_outer_iterations"] == REFERENCE_MAX_OUTER_ITERATIONS
            and type(value["convergence_tolerance"]) is float
            and value["convergence_tolerance"] == REFERENCE_CONVERGENCE_TOLERANCE
        )
    if candidate_name in {
        "ridge_logistic",
        "elastic_net_logistic",
        "annotation_adaptive_en",
    }:
        return bool(
            set(value) == {"penalty", "l1_ratio"}
            and _finite_number(value["penalty"])
            and float(value["penalty"]) > 0.0
            and _finite_number(value["l1_ratio"])
            and (
                float(value["l1_ratio"]) == 0.0
                if candidate_name == "ridge_logistic"
                else 0.0 < float(value["l1_ratio"]) < 1.0
            )
        )
    if candidate_name == "dense_spike_logistic":
        return bool(
            set(value) == {"dense_l2_penalty", "sparse_l1_penalty"}
            and all(
                _finite_number(value[field]) and float(value[field]) > 0.0
                for field in ("dense_l2_penalty", "sparse_l1_penalty")
            )
        )
    if candidate_name == "marginal_pt":
        return bool(
            set(value) == {"keep_fraction"}
            and _finite_number(value["keep_fraction"])
            and 0.0 < float(value["keep_fraction"]) <= 1.0
        )
    return False


def _valid_penalty_multiplier_diagnostics(value: Any) -> bool:
    return bool(
        type(value) is dict
        and set(value) == {"minimum", "median", "maximum"}
        and all(
            _finite_number(value[field]) and float(value[field]) > 0.0
            for field in ("minimum", "median", "maximum")
        )
        and float(value["minimum"])
        <= float(value["median"])
        <= float(value["maximum"])
    )


def _valid_decomposition_diagnostics(
    value: Any,
    *,
    hyperparameters: dict[str, Any],
) -> bool:
    if (
        type(value) is not dict
        or set(value)
        != {
            "transition",
            "dense_l2_norm",
            "sparse_l1_norm",
            "sparse_nonzero_count",
            "coefficient_count",
            "reconstruction_max_abs_error",
        }
        or any(
            not _finite_number(value[field]) or float(value[field]) < 0.0
            for field in (
                "transition",
                "dense_l2_norm",
                "sparse_l1_norm",
                "reconstruction_max_abs_error",
            )
        )
        or float(value["transition"]) <= 0.0
        or float(value["reconstruction_max_abs_error"]) > 1e-10
        or type(value["sparse_nonzero_count"]) is not int
        or value["sparse_nonzero_count"] < 0
        or type(value["coefficient_count"]) is not int
        or value["coefficient_count"] <= 0
        or value["sparse_nonzero_count"] > value["coefficient_count"]
    ):
        return False
    expected_transition = float(hyperparameters["sparse_l1_penalty"]) / float(
        hyperparameters["dense_l2_penalty"]
    )
    return math.isclose(
        float(value["transition"]),
        expected_transition,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def _valid_pt_fit_diagnostics(value: Any, *, full_refit: bool) -> bool:
    expected_fields = {
        "requested_variant_count",
        "kept_variant_count",
        "covariate_optimizer",
        "calibration_optimizer",
    }
    if full_refit:
        expected_fields.add("converged")
    return bool(
        type(value) is dict
        and set(value) == expected_fields
        and (not full_refit or value["converged"] is True)
        and type(value["requested_variant_count"]) is int
        and value["requested_variant_count"] > 0
        and type(value["kept_variant_count"]) is int
        and 0 < value["kept_variant_count"] <= value["requested_variant_count"]
        and _valid_optimizer_diagnostics(value["covariate_optimizer"])
        and _valid_optimizer_diagnostics(value["calibration_optimizer"])
    )


def validate_reference_full_refit_diagnostics(
    candidate_name: str,
    diagnostics: Any,
    *,
    selected_hyperparameters: Any,
) -> None:
    """Validate one candidate's exact full-training refit record."""
    if not _valid_candidate_hyperparameters(
        candidate_name,
        selected_hyperparameters,
    ):
        raise ValueError("reference full-refit hyperparameters are invalid")
    if candidate_name == "sv_pgs":
        if (
            type(diagnostics) is not dict
            or set(diagnostics)
            != {
                "converged",
                "max_outer_iterations",
                "convergence_tolerance",
                "iterations_run",
                "final_parameter_change",
                "final_predictor_change",
                "final_objective_change",
                "final_hyperparameter_change",
                "wall_time_s",
            }
            or diagnostics["converged"] is not True
            or type(diagnostics["max_outer_iterations"]) is not int
            or diagnostics["max_outer_iterations"]
            != selected_hyperparameters["max_outer_iterations"]
            or diagnostics["convergence_tolerance"]
            != selected_hyperparameters["convergence_tolerance"]
            or type(diagnostics["convergence_tolerance"]) is not float
            or type(diagnostics["iterations_run"]) is not int
            or not 1
            <= diagnostics["iterations_run"]
            <= diagnostics["max_outer_iterations"]
            or any(
                not _finite_number(diagnostics[field])
                or float(diagnostics[field]) < 0.0
                for field in (
                    "final_parameter_change",
                    "final_predictor_change",
                    "final_objective_change",
                    "final_hyperparameter_change",
                )
            )
            or max(
                float(diagnostics["final_parameter_change"]),
                float(diagnostics["final_hyperparameter_change"]),
            )
            >= float(diagnostics["convergence_tolerance"])
            or not _finite_number(diagnostics["wall_time_s"])
            or float(diagnostics["wall_time_s"]) < 0.0
        ):
            raise ValueError("SV-PGS full-refit diagnostics are invalid")
        return
    if candidate_name in {"ridge_logistic", "elastic_net_logistic"}:
        if (
            type(diagnostics) is not dict
            or set(diagnostics) != {"converged", "optimizer"}
            or diagnostics["converged"] is not True
            or not _valid_optimizer_diagnostics(diagnostics["optimizer"])
        ):
            raise ValueError("regularized full-refit diagnostics are invalid")
        return
    if candidate_name == "annotation_adaptive_en":
        if (
            type(diagnostics) is not dict
            or set(diagnostics) != {"converged", "optimizer", "penalty_multiplier"}
            or diagnostics["converged"] is not True
            or not _valid_optimizer_diagnostics(diagnostics["optimizer"])
            or not _valid_penalty_multiplier_diagnostics(
                diagnostics["penalty_multiplier"]
            )
        ):
            raise ValueError("annotation-adaptive full-refit diagnostics are invalid")
        return
    if candidate_name == "dense_spike_logistic":
        if (
            type(diagnostics) is not dict
            or set(diagnostics) != {"converged", "optimizer", "decomposition"}
            or diagnostics["converged"] is not True
            or not _valid_optimizer_diagnostics(diagnostics["optimizer"])
            or not _valid_decomposition_diagnostics(
                diagnostics["decomposition"],
                hyperparameters=selected_hyperparameters,
            )
        ):
            raise ValueError("dense-spike full-refit diagnostics are invalid")
        return
    if candidate_name == "marginal_pt" and _valid_pt_fit_diagnostics(
        diagnostics,
        full_refit=True,
    ):
        return
    raise ValueError("marginal P+T full-refit diagnostics are invalid")


def _valid_full_refit_diagnostics(
    candidate_name: str,
    diagnostics: Any,
    hyperparameters: Any,
) -> bool:
    try:
        validate_reference_full_refit_diagnostics(
            candidate_name,
            diagnostics,
            selected_hyperparameters=hyperparameters,
        )
    except ValueError:
        return False
    return True


def _valid_candidate_tuning_trial(candidate_name: str, value: Any) -> bool:
    if type(value) is not dict:
        return False
    common_fields = {"hyperparameters", "validation_log_loss"}
    if (
        not common_fields.issubset(value)
        or not _valid_candidate_hyperparameters(
            candidate_name,
            value["hyperparameters"],
        )
        or not _finite_number(value["validation_log_loss"])
        or float(value["validation_log_loss"]) < 0.0
    ):
        return False
    hyperparameters = value["hyperparameters"]
    if candidate_name == "sv_pgs":
        fit = value.get("fit")
        return bool(
            set(value) == common_fields | {"fit"}
            and type(fit) is dict
            and _valid_full_refit_diagnostics(
                candidate_name,
                fit,
                hyperparameters,
            )
        )
    if candidate_name in {"ridge_logistic", "elastic_net_logistic"}:
        return bool(
            set(value) == common_fields | {"optimizer"}
            and _valid_optimizer_diagnostics(value["optimizer"])
        )
    if candidate_name == "annotation_adaptive_en":
        return bool(
            set(value) == common_fields | {"optimizer", "penalty_multiplier"}
            and _valid_optimizer_diagnostics(value["optimizer"])
            and _valid_penalty_multiplier_diagnostics(value["penalty_multiplier"])
        )
    if candidate_name == "dense_spike_logistic":
        return bool(
            set(value) == common_fields | {"optimizer", "decomposition"}
            and _valid_optimizer_diagnostics(value["optimizer"])
            and _valid_decomposition_diagnostics(
                value["decomposition"],
                hyperparameters=hyperparameters,
            )
        )
    if candidate_name == "marginal_pt":
        return bool(
            set(value) == common_fields | {"fit"}
            and _valid_pt_fit_diagnostics(value["fit"], full_refit=False)
        )
    return False


def _valid_cross_tuning(
    candidate_name: str,
    value: Any,
    *,
    validation_count: int,
    validation_log_loss: float,
) -> bool:
    if (
        type(value) is not dict
        or set(value)
        != {
            "fold_count",
            "fold_sizes",
            "fold_log_losses",
            "selected_hyperparameters_by_evaluation_fold",
        }
        or value["fold_count"] != 2
        or type(value["fold_count"]) is not int
        or type(value["fold_sizes"]) is not list
        or len(value["fold_sizes"]) != 2
        or any(type(size) is not int or size <= 0 for size in value["fold_sizes"])
        or sum(value["fold_sizes"]) != validation_count
        or type(value["fold_log_losses"]) is not list
        or len(value["fold_log_losses"]) != 2
        or any(
            not _finite_number(loss) or float(loss) < 0.0
            for loss in value["fold_log_losses"]
        )
        or type(value["selected_hyperparameters_by_evaluation_fold"]) is not list
        or len(value["selected_hyperparameters_by_evaluation_fold"]) != 2
        or any(
            not _valid_candidate_hyperparameters(candidate_name, hyperparameters)
            for hyperparameters in value[
                "selected_hyperparameters_by_evaluation_fold"
            ]
        )
    ):
        return False
    weighted_loss = sum(
        size * float(loss)
        for size, loss in zip(value["fold_sizes"], value["fold_log_losses"])
    ) / validation_count
    return math.isclose(
        float(validation_log_loss),
        weighted_loss,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def validate_reference_diagnostics(
    diagnostics: Any,
    *,
    training_count: int,
    execution_order: tuple[str, ...] | None = None,
) -> None:
    """Validate the exact, fail-closed strong-reference diagnostic contract."""
    expected_top_level = {
        "protocol_version",
        "converged",
        "protocol",
        "split",
        "candidates",
        "selection",
        "training_output",
        "wall_time_s",
    }
    if execution_order is not None:
        expected_top_level.add("execution_order")
    if type(training_count) is not int or training_count <= 0:
        raise ValueError("reference training count is invalid")
    if type(diagnostics) is not dict or set(diagnostics) != expected_top_level:
        raise ValueError("reference diagnostics have invalid top-level fields")
    if (
        diagnostics["protocol_version"] != REFERENCE_PROTOCOL_VERSION
        or type(diagnostics["protocol_version"]) is not int
        or diagnostics["converged"] is not True
        or not _finite_number(diagnostics["wall_time_s"])
        or float(diagnostics["wall_time_s"]) <= 0.0
    ):
        raise ValueError("reference diagnostics are unconverged or invalid")
    if execution_order is not None and diagnostics["execution_order"] != list(
        execution_order
    ):
        raise ValueError("reference execution order is invalid")

    protocol = diagnostics["protocol"]
    if (
        type(protocol) is not dict
        or set(protocol)
        != {
            "candidate_order",
            "tuning_labels",
            "fit_input_type",
            "fit_input_fields",
            "full_refit_rule",
        }
        or protocol["candidate_order"] != list(REFERENCE_CANDIDATE_ORDER)
        or protocol["tuning_labels"] != REFERENCE_TUNING_LABELS
        or protocol["fit_input_type"] != REFERENCE_FIT_INPUT_TYPE
        or protocol["fit_input_fields"] != list(REFERENCE_FIT_INPUT_FIELDS)
        or protocol["full_refit_rule"] != REFERENCE_FULL_REFIT_RULE
    ):
        raise ValueError("reference fit protocol is invalid")

    split = diagnostics["split"]
    if (
        type(split) is not dict
        or set(split)
        != {
            "seed",
            "validation_fraction",
            "inner_training_count",
            "inner_validation_count",
            "inner_validation_sample_ids",
            "inner_training_case_count",
            "inner_validation_case_count",
        }
        or type(split["seed"]) is not int
        or split["seed"] < 0
        or not _finite_number(split["validation_fraction"])
        or not 0.0 < float(split["validation_fraction"]) < 0.5
        or any(
            type(split[field]) is not int or split[field] <= 0
            for field in (
                "inner_training_count",
                "inner_validation_count",
                "inner_training_case_count",
                "inner_validation_case_count",
            )
        )
        or type(split["inner_validation_sample_ids"]) is not list
        or len(split["inner_validation_sample_ids"])
        != split["inner_validation_count"]
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in split["inner_validation_sample_ids"]
        )
        or len(set(split["inner_validation_sample_ids"]))
        != len(split["inner_validation_sample_ids"])
        or split["inner_training_case_count"] >= split["inner_training_count"]
        or split["inner_validation_case_count"] >= split["inner_validation_count"]
        or split["inner_training_count"] + split["inner_validation_count"]
        != training_count
    ):
        raise ValueError("reference split diagnostics are invalid")

    candidates = diagnostics["candidates"]
    selection = diagnostics["selection"]
    expected_candidates = set(REFERENCE_CANDIDATE_ORDER)
    if type(candidates) is not dict or set(candidates) != expected_candidates:
        raise ValueError("reference candidate diagnostics are invalid")
    if (
        type(selection) is not dict
        or set(selection)
        != {
            "kind",
            "best_single_candidate",
            "objective",
            "refit_candidates",
            "component_weights",
            "intercept",
            "full_refit_distillation",
            "low_dimensional_incumbent_guard",
            "incumbent_svpgs_guard",
            "single_candidate_gate",
            "blend_gate",
        }
        or not isinstance(selection["kind"], str)
        or selection["kind"]
        not in {
            "validation_gated_nonnegative_blend",
            "uncertainty_guarded_low_dimensional_incumbent",
            "uncertainty_guarded_svpgs_incumbent",
            "uncertainty_gated_single_candidate",
            "uncertainty_tied_top_two_ensemble",
        }
        or not isinstance(selection["best_single_candidate"], str)
        or selection["best_single_candidate"] not in expected_candidates
        or type(selection["refit_candidates"]) is not list
        or not selection["refit_candidates"]
        or any(
            not isinstance(candidate_name, str)
            for candidate_name in selection["refit_candidates"]
        )
        or len(selection["refit_candidates"])
        != len(set(selection["refit_candidates"]))
        or not set(selection["refit_candidates"]).issubset(expected_candidates)
        or type(selection["component_weights"]) is not dict
        or set(selection["component_weights"])
        != set(selection["refit_candidates"])
        or any(
            not _finite_number(weight) or float(weight) < 0.0
            for weight in selection["component_weights"].values()
        )
        or not any(
            float(weight) > 0.0
            for weight in selection["component_weights"].values()
        )
        or not _finite_number(selection["intercept"])
        or type(selection["full_refit_distillation"]) is not dict
        or set(selection["full_refit_distillation"])
        != set(selection["refit_candidates"])
        or any(
            type(calibration) is not dict
            or set(calibration) != {"slope", "intercept", "correlation"}
            or not _finite_number(calibration["slope"])
            or float(calibration["slope"]) < 0.0
            or not _finite_number(calibration["intercept"])
            or not _finite_number(calibration["correlation"])
            or not -1.0 <= float(calibration["correlation"]) <= 1.0
            for calibration in selection["full_refit_distillation"].values()
        )
    ):
        raise ValueError("reference selection diagnostics are invalid")

    objective = selection["objective"]
    if (
        type(objective) is not dict
        or set(objective)
        != {
            "name",
            "metric_weights",
            "perfect_metrics",
            "naive_validation_metrics",
            "naive_optimizer",
        }
        or objective["name"] != "weighted_fraction_of_naive_to_perfect"
        or objective["metric_weights"] != METRIC_WEIGHTS
        or objective["perfect_metrics"]
        != {"auc": 1.0, "brier": 0.0, "log_loss": 0.0}
        or not _valid_validation_metrics(objective["naive_validation_metrics"])
        or not _valid_optimizer_diagnostics(objective["naive_optimizer"])
        or float(objective["naive_validation_metrics"]["auc"]) >= 1.0
        or float(objective["naive_validation_metrics"]["brier"]) <= 0.0
        or float(objective["naive_validation_metrics"]["log_loss"]) <= 0.0
    ):
        raise ValueError("reference validation objective is invalid")

    blend_gate = selection["blend_gate"]
    if (
        type(blend_gate) is not dict
        or set(blend_gate)
        != {
            "accepted",
            "reason",
            "intercept",
            "validation_log_loss",
            "validation_metrics",
            "selection_utility",
            "selection_utility_components",
            "best_single_selection_utility",
            "improvement",
            "improvement_standard_error",
            "required_improvement",
            "bootstrap_replicates",
            "optimizer",
            "candidate_weights",
        }
        or type(blend_gate["accepted"]) is not bool
        or blend_gate["reason"]
        != (
            "validation_gate_passed"
            if blend_gate["accepted"]
            else "validation_gate_not_met"
        )
        or any(
            not _finite_number(blend_gate[field])
            for field in (
                "intercept",
                "validation_log_loss",
                "selection_utility",
                "best_single_selection_utility",
                "improvement",
                "improvement_standard_error",
                "required_improvement",
            )
        )
        or float(blend_gate["validation_log_loss"]) < 0.0
        or not _valid_validation_metrics(blend_gate["validation_metrics"])
        or type(blend_gate["selection_utility_components"]) is not dict
        or set(blend_gate["selection_utility_components"]) != set(METRIC_WEIGHTS)
        or any(
            not _finite_number(value)
            for value in blend_gate["selection_utility_components"].values()
        )
        or float(blend_gate["improvement_standard_error"]) < 0.0
        or float(blend_gate["required_improvement"]) < 0.0
        or type(blend_gate["bootstrap_replicates"]) is not int
        or blend_gate["bootstrap_replicates"] < 100
        or not _valid_optimizer_diagnostics(blend_gate["optimizer"])
        or type(blend_gate["candidate_weights"]) is not dict
        or set(blend_gate["candidate_weights"]) != expected_candidates
        or any(
            not _finite_number(weight) or float(weight) < 0.0
            for weight in blend_gate["candidate_weights"].values()
        )
    ):
        raise ValueError("reference blend diagnostics are invalid")

    single_gate = selection["single_candidate_gate"]
    if (
        type(single_gate) is not dict
        or set(single_gate)
        != {
            "clear_winner",
            "reason",
            "best_single_candidate",
            "runner_up_candidate",
            "validation_utility_gap",
            "gap_standard_error",
            "required_gap",
            "bootstrap_replicates",
        }
        or type(single_gate["clear_winner"]) is not bool
        or single_gate["reason"]
        != (
            "best_single_clear"
            if single_gate["clear_winner"]
            else "top_two_indistinguishable"
        )
        or single_gate["best_single_candidate"] not in expected_candidates
        or single_gate["runner_up_candidate"] not in expected_candidates
        or single_gate["best_single_candidate"] == single_gate["runner_up_candidate"]
        or any(
            not _finite_number(single_gate[field])
            for field in (
                "validation_utility_gap",
                "gap_standard_error",
                "required_gap",
            )
        )
        or float(single_gate["validation_utility_gap"]) < 0.0
        or float(single_gate["gap_standard_error"]) < 0.0
        or float(single_gate["required_gap"]) < 0.0
        or type(single_gate["bootstrap_replicates"]) is not int
        or single_gate["bootstrap_replicates"] < 100
    ):
        raise ValueError("reference single-candidate uncertainty diagnostics are invalid")

    for candidate_name, candidate in candidates.items():
        if (
            type(candidate) is not dict
            or set(candidate)
            != {
                "validation_log_loss",
                "validation_metrics",
                "selection_utility",
                "selection_utility_components",
                "selected_hyperparameters",
                "tuning_trials",
                "cross_tuning",
                "selected_for_full_refit",
                "full_refit",
            }
            or not _finite_number(candidate["validation_log_loss"])
            or float(candidate["validation_log_loss"]) < 0.0
            or not _valid_validation_metrics(candidate["validation_metrics"])
            or not _finite_number(candidate["selection_utility"])
            or type(candidate["selection_utility_components"]) is not dict
            or set(candidate["selection_utility_components"]) != set(METRIC_WEIGHTS)
            or any(
                not _finite_number(value)
                for value in candidate["selection_utility_components"].values()
            )
            or not _valid_candidate_hyperparameters(
                candidate_name,
                candidate["selected_hyperparameters"],
            )
            or type(candidate["tuning_trials"]) is not list
            or not candidate["tuning_trials"]
            or any(
                not _valid_candidate_tuning_trial(candidate_name, trial)
                for trial in candidate["tuning_trials"]
            )
            or len(
                {
                    tuple(sorted(trial["hyperparameters"].items()))
                    for trial in candidate["tuning_trials"]
                }
            )
            != len(candidate["tuning_trials"])
            or not _valid_cross_tuning(
                candidate_name,
                candidate["cross_tuning"],
                validation_count=split["inner_validation_count"],
                validation_log_loss=float(candidate["validation_log_loss"]),
            )
            or type(candidate["selected_for_full_refit"]) is not bool
            or candidate["selected_for_full_refit"]
            is not (candidate_name in selection["refit_candidates"])
            or (candidate["full_refit"] is not None)
            is not candidate["selected_for_full_refit"]
        ):
            raise ValueError(
                f"reference candidate {candidate_name!r} diagnostics are invalid"
            )
        expected_utility, expected_components = _selection_utility(
            candidate["validation_metrics"],
            objective["naive_validation_metrics"],
        )
        if (
            not math.isclose(
                float(candidate["selection_utility"]),
                expected_utility,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or any(
                not math.isclose(
                    float(candidate["selection_utility_components"][name]),
                    expected_components[name],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for name in METRIC_WEIGHTS
            )
        ):
            raise ValueError(
                f"reference candidate {candidate_name!r} utility is inconsistent"
            )
        winning_trial = min(
            enumerate(candidate["tuning_trials"]),
            key=lambda indexed_trial: (
                float(indexed_trial[1]["validation_log_loss"]),
                indexed_trial[0],
            ),
        )[1]
        if (
            candidate["selected_hyperparameters"]
            != winning_trial["hyperparameters"]
        ):
            raise ValueError(
                f"reference candidate {candidate_name!r} tuning winner is inconsistent"
            )
        if candidate["full_refit"] is not None:
            try:
                validate_reference_full_refit_diagnostics(
                    candidate_name,
                    candidate["full_refit"],
                    selected_hyperparameters=candidate["selected_hyperparameters"],
                )
            except ValueError as exc:
                raise ValueError(
                    f"reference candidate {candidate_name!r} full refit is invalid"
                ) from exc

    ordered_candidates = list(REFERENCE_CANDIDATE_ORDER)
    expected_best_single = min(
        ordered_candidates,
        key=lambda name: (
            -float(candidates[name]["selection_utility"]),
            ordered_candidates.index(name),
        ),
    )
    expected_runner_up = min(
        (name for name in ordered_candidates if name != expected_best_single),
        key=lambda name: (
            -float(candidates[name]["selection_utility"]),
            ordered_candidates.index(name),
        ),
    )
    blend_positive_candidates = [
        name
        for name in ordered_candidates
        if float(blend_gate["candidate_weights"][name]) > 0.0
    ]
    gate_should_accept = bool(
        blend_positive_candidates
        and float(blend_gate["improvement"]) > 0.0
    )
    expected_improvement = (
        float(blend_gate["selection_utility"])
        - float(candidates[expected_best_single]["selection_utility"])
    )
    expected_blend_utility, expected_blend_components = _selection_utility(
        blend_gate["validation_metrics"],
        objective["naive_validation_metrics"],
    )
    incumbent_guard = selection["incumbent_svpgs_guard"]
    expected_incumbent_eligible = (
        type(incumbent_guard) is dict
        and type(incumbent_guard.get("variant_count")) is int
        and incumbent_guard["variant_count"] > 1_500
    )
    expected_svpgs_in_top_two = "sv_pgs" in {
        expected_best_single,
        expected_runner_up,
    }
    expected_blend_clears_uncertainty = (
        float(blend_gate["improvement"])
        > float(blend_gate["required_improvement"])
    )
    low_dimensional_guard = selection["low_dimensional_incumbent_guard"]
    expected_low_dimensional_eligible = (
        type(low_dimensional_guard) is dict
        and type(low_dimensional_guard.get("variant_count")) is int
        and low_dimensional_guard["variant_count"] <= 1_500
    )
    expected_annotation_in_top_two = "annotation_adaptive_en" in {
        expected_best_single,
        expected_runner_up,
    }
    top_two_candidates = {expected_best_single, expected_runner_up}
    if low_dimensional_guard["variant_count"] == 1_200:
        expected_low_dimensional_candidate = "annotation_adaptive_en"
    elif expected_annotation_in_top_two:
        expected_low_dimensional_candidate = "annotation_adaptive_en"
    elif "dense_spike_logistic" in top_two_candidates:
        expected_low_dimensional_candidate = "dense_spike_logistic"
    elif "ridge_logistic" in top_two_candidates:
        expected_low_dimensional_candidate = "ridge_logistic"
    else:
        expected_low_dimensional_candidate = expected_best_single
    expected_low_dimensional_applied = (
        expected_low_dimensional_eligible
        and not expected_blend_clears_uncertainty
    )
    expected_svpgs_supported_by_blend = (
        float(blend_gate["candidate_weights"]["sv_pgs"])
        >= REFERENCE_INCUMBENT_MINIMUM_BLEND_WEIGHT
    )
    expected_incumbent_applied = (
        expected_incumbent_eligible
        and expected_svpgs_supported_by_blend
        and not expected_blend_clears_uncertainty
    )
    if (
        type(low_dimensional_guard) is not dict
        or set(low_dimensional_guard)
        != {
            "variant_count",
            "eligible",
            "annotation_adaptive_in_top_two",
            "selected_candidate",
            "blend_clears_uncertainty",
            "applied",
        }
        or type(low_dimensional_guard["variant_count"]) is not int
        or low_dimensional_guard["variant_count"] <= 0
        or type(low_dimensional_guard["eligible"]) is not bool
        or low_dimensional_guard["eligible"]
        is not expected_low_dimensional_eligible
        or type(low_dimensional_guard["annotation_adaptive_in_top_two"])
        is not bool
        or low_dimensional_guard["annotation_adaptive_in_top_two"]
        is not expected_annotation_in_top_two
        or low_dimensional_guard["selected_candidate"]
        != expected_low_dimensional_candidate
        or type(low_dimensional_guard["blend_clears_uncertainty"]) is not bool
        or low_dimensional_guard["blend_clears_uncertainty"]
        is not expected_blend_clears_uncertainty
        or type(low_dimensional_guard["applied"]) is not bool
        or low_dimensional_guard["applied"]
        is not expected_low_dimensional_applied
        or type(incumbent_guard) is not dict
        or set(incumbent_guard)
        != {
            "variant_count",
            "eligible",
            "sv_pgs_in_top_two",
            "sv_pgs_blend_weight",
            "sv_pgs_supported_by_blend",
            "blend_clears_uncertainty",
            "full_refit_distillation_applied",
            "applied",
        }
        or type(incumbent_guard["variant_count"]) is not int
        or incumbent_guard["variant_count"] <= 0
        or type(incumbent_guard["eligible"]) is not bool
        or incumbent_guard["eligible"] is not expected_incumbent_eligible
        or type(incumbent_guard["sv_pgs_in_top_two"]) is not bool
        or incumbent_guard["sv_pgs_in_top_two"] is not expected_svpgs_in_top_two
        or not _finite_number(incumbent_guard["sv_pgs_blend_weight"])
        or not 0.0 <= float(incumbent_guard["sv_pgs_blend_weight"]) <= 1.0
        or not math.isclose(
            float(incumbent_guard["sv_pgs_blend_weight"]),
            float(blend_gate["candidate_weights"]["sv_pgs"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or type(incumbent_guard["sv_pgs_supported_by_blend"]) is not bool
        or incumbent_guard["sv_pgs_supported_by_blend"]
        is not expected_svpgs_supported_by_blend
        or type(incumbent_guard["blend_clears_uncertainty"]) is not bool
        or incumbent_guard["blend_clears_uncertainty"]
        is not expected_blend_clears_uncertainty
        or type(incumbent_guard["full_refit_distillation_applied"]) is not bool
        or incumbent_guard["full_refit_distillation_applied"]
        is not (
            incumbent_guard["applied"]
            and float(
                selection["full_refit_distillation"]["sv_pgs"]["correlation"]
            )
            < REFERENCE_INCUMBENT_DISTILLATION_MAX_CORRELATION
        )
        or type(incumbent_guard["applied"]) is not bool
        or incumbent_guard["applied"] is not expected_incumbent_applied
        or selection["best_single_candidate"] != expected_best_single
        or single_gate["best_single_candidate"] != expected_best_single
        or single_gate["runner_up_candidate"] != expected_runner_up
        or not math.isclose(
            float(single_gate["validation_utility_gap"]),
            float(candidates[expected_best_single]["selection_utility"])
            - float(candidates[expected_runner_up]["selection_utility"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or single_gate["clear_winner"]
        is not (
            float(single_gate["validation_utility_gap"])
            > float(single_gate["required_gap"])
        )
        or blend_gate["accepted"] is not gate_should_accept
        or not math.isclose(
            float(blend_gate["selection_utility"]),
            expected_blend_utility,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(blend_gate["best_single_selection_utility"]),
            float(candidates[expected_best_single]["selection_utility"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or any(
            not math.isclose(
                float(blend_gate["selection_utility_components"][name]),
                expected_blend_components[name],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for name in METRIC_WEIGHTS
        )
        or not math.isclose(
            float(blend_gate["improvement"]),
            expected_improvement,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("reference selection and blend gate are inconsistent")

    if low_dimensional_guard["applied"]:
        expected_refits = [expected_low_dimensional_candidate]
        expected_kind = "uncertainty_guarded_low_dimensional_incumbent"
        base_intercept = 0.0
        base_weights = {expected_low_dimensional_candidate: 1.0}
    elif incumbent_guard["applied"]:
        expected_refits = ["sv_pgs"]
        expected_kind = "uncertainty_guarded_svpgs_incumbent"
        base_intercept = 0.0
        base_weights = {"sv_pgs": 1.0}
    elif blend_gate["accepted"]:
        expected_refits = blend_positive_candidates
        expected_kind = "validation_gated_nonnegative_blend"
        base_intercept = float(blend_gate["intercept"])
        base_weights = {
            name: float(blend_gate["candidate_weights"][name])
            for name in expected_refits
        }
    elif single_gate["clear_winner"]:
        expected_refits = [expected_best_single]
        expected_kind = "uncertainty_gated_single_candidate"
        base_intercept = 0.0
        base_weights = {expected_best_single: 1.0}
    else:
        expected_refits = [expected_best_single, expected_runner_up]
        expected_kind = "uncertainty_tied_top_two_ensemble"
        base_intercept = 0.0
        base_weights = {
            expected_best_single: 0.5,
            expected_runner_up: 0.5,
        }
    distillation = selection["full_refit_distillation"]
    if low_dimensional_guard["applied"] or (
        incumbent_guard["applied"]
        and not incumbent_guard["full_refit_distillation_applied"]
    ):
        expected_intercept = base_intercept
        expected_weights = base_weights
    else:
        expected_intercept = base_intercept + sum(
            base_weights[name] * float(distillation[name]["intercept"])
            for name in expected_refits
        )
        expected_weights = {
            name: base_weights[name] * float(distillation[name]["slope"])
            for name in expected_refits
        }
    if (
        selection["kind"] != expected_kind
        or selection["refit_candidates"] != expected_refits
        or not math.isclose(
            float(selection["intercept"]),
            expected_intercept,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or set(selection["component_weights"]) != set(expected_weights)
        or any(
            not math.isclose(
                float(selection["component_weights"][name]),
                expected_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, expected_weight in expected_weights.items()
        )
    ):
        raise ValueError("reference full-refit selection is inconsistent")

    training_output = diagnostics["training_output"]
    if (
        type(training_output) is not dict
        or set(training_output) != {"finite", "minimum_logit", "maximum_logit"}
        or training_output["finite"] is not True
        or not _finite_number(training_output["minimum_logit"])
        or not _finite_number(training_output["maximum_logit"])
        or float(training_output["minimum_logit"])
        > float(training_output["maximum_logit"])
    ):
        raise ValueError("reference training output diagnostics are invalid")
