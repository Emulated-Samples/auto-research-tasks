"""The SHIPPED reference contract -- what every anchor and every score means.

The reference standard is the exact, self-contained public NumPy program in
``gold/{fit,predict,pgs_core.py}``. Per dataset it selects between its joint
hierarchical empirical-Bayes estimator and fixed-grid ridge logistic using a
deterministic public-training inner split, then refits the winner on all training
rows. A positive-slope Platt map learned from the selected candidate's inner-validation
logits calibrates probabilities without changing AUC. ``datagen.build_corpus`` executes
that exact program in a clean work directory
containing only public training files, retains its exact model.out, and later executes
the locked gold/predict in the mandatory public sandbox after test-feature staging.
Anchors score the actual pred.csv round-trip bytes and bind model/source/output hashes,
exit status, and timing. There is no second trusted estimator and no private engine to
imitate.

WHY BEST-OF-FAMILY: hierarchical shrinkage is strong on annotation/heavy-tailed
regimes but genuinely loses to a plain ridge on dense near-Gaussian ones
(dense_infinitesimal, low_prevalence), so a single fixed method cannot anchor every
well-posed category. Picking the best PRINCIPLED method per regime is what a demigod
does, so skill=1 stays reachable and no category needs dropping. The family EXCLUDES
cheats (marginal_pt) and weak baselines, so the reference can never BE a cheat.

The hidden JAX SV-PGS adapter remains only as one actively measured model-zoo arm for
difficulty/diversity analysis. It does not define anchors, provenance, score 1, or the
gold proof.

This module pins the provenance strings the grader checks, the ``REFERENCE_METHODS``
family, the tie-margin, and the exact shape of the method-aware, fail-closed reference
diagnostics recorded in each anchor (validated by ``validate_reference_diagnostics``).
"""
from __future__ import annotations

import math
from typing import Any

REFERENCE_IMPLEMENTATION = "public NumPy best-of-family (hierarchical_eb | ridge_logistic)"
REFERENCE_ENTRYPOINT = "gold/fit"
REFERENCE_FIT_PROTOCOL = "public_numpy_family_inner_validation_platt_full_refit"
REFERENCE_MAX_OUTER_ITERATIONS = 400
REFERENCE_CONVERGENCE_TOLERANCE = 1e-4
REFERENCE_PROTOCOL_VERSION = 9
# The reference method chosen per dataset is one of these principled estimators. A
# cheat / weak baseline can never be the reference -- the family excludes them.
REFERENCE_METHODS = ("hierarchical_eb", "ridge_logistic")
REFERENCE_RIDGE_PENALTIES = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
# Ridge must beat hierarchical EB on inner-validation AUC by at least this margin;
# otherwise selection defaults to hierarchical EB. The validator re-checks the
# declared winner against this rule so a recorded method that did NOT win is rejected.
REFERENCE_FAMILY_WIN_MARGIN = 0.03
REFERENCE_HIERARCHICAL_OUTER_ITERATIONS = 6
REFERENCE_RIDGE_MAX_ITERATIONS = 500
REFERENCE_RIDGE_TOLERANCE = 1e-5
REFERENCE_CALIBRATION_MAX_ITERATIONS = 100
REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE = 1e-8
REFERENCE_PUBLIC_FIT_BUDGET_S = 165.0
REFERENCE_PROCESS_TIMEOUT_S = 170.0
REFERENCE_PUBLIC_PREDICT_BUDGET_S = 30.0
REFERENCE_SOURCE_RELATIVE_PATHS = (
    "gold/fit",
    "gold/predict",
    "gold/pgs_core.py",
)
REFERENCE_PYTHON_EXECUTABLE = "/opt/svpgs-venv/bin/python"
REFERENCE_NUMPY_VERSION = "2.1.3"
# Anchor build executes every fit before any prediction table is opened, then
# predicts, then loads truth -- so a reference can never see held-out rows.
REFERENCE_ANCHOR_EXECUTION_ORDER = (
    "all_anchor_fits",
    "prediction_data_load",
    "all_anchor_predictions",
    "truth_load",
)


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def validate_reference_diagnostics(
    diagnostics: Any,
    *,
    training_count: int,
    execution_order: tuple[str, ...] | None = None,
) -> None:
    """Validate the exact fail-closed public NumPy reference diagnostics."""
    if type(training_count) is not int or training_count <= 0:
        raise ValueError("reference training count is invalid")
    if type(diagnostics) is not dict:
        raise ValueError("reference diagnostics have invalid top-level fields")
    method = diagnostics.get("reference_method")
    if method not in REFERENCE_METHODS:
        raise ValueError("reference diagnostics have an unknown reference_method")

    common_fields = {
        "protocol_version", "reference_method", "converged", "fit_protocol",
        "wall_time_s", "reference_total_fit_wall_s", "reference_full_fit_wall_s",
        "reference_program_reported_wall_s", "selection_wall_s",
        "calibration_intercept", "calibration_slope", "calibration_iterations",
        "calibration_relative_gradient",
        "fit_exit_code", "fit_source_sha256", "fit_python_executable",
        "fit_numpy_version",
        "inner_selection_auc", "inner_ridge_trials",
        "hierarchical_outer_iterations", "hierarchical_inner_iterations_run",
        "hierarchical_inner_global_scale",
        "prediction_exit_code", "prediction_wall_s",
        "prediction_output_sha256", "prediction_output_byte_count",
        "prediction_model_sha256", "prediction_source_sha256",
        "classes_present", "training_output",
    }
    hierarchical_fields = {
        "hierarchical_iterations_run", "hierarchical_global_scale",
    }
    ridge_fields = {
        "ridge_penalty", "ridge_penalty_grid", "ridge_max_iterations",
        "ridge_iterations_run", "ridge_tolerance", "ridge_final_relative_gradient",
    }
    expected_top_level = common_fields | (
        hierarchical_fields if method == "hierarchical_eb" else ridge_fields)
    if execution_order is not None:
        expected_top_level.add("execution_order")
    if set(diagnostics) != expected_top_level:
        raise ValueError("reference diagnostics have invalid top-level fields")

    if (
        diagnostics["protocol_version"] != REFERENCE_PROTOCOL_VERSION
        or type(diagnostics["protocol_version"]) is not int
        or diagnostics["converged"] is not True
        or diagnostics["fit_protocol"] != REFERENCE_FIT_PROTOCOL
        or not _finite_number(diagnostics["wall_time_s"])
        or float(diagnostics["wall_time_s"]) <= 0.0
    ):
        raise ValueError("reference diagnostics are unconverged or invalid")

    # The externally measured wall covers interpreter startup, selection, refit and
    # serialization. The program-reported wall is bounded by the exact public guard.
    if (
        not _finite_number(diagnostics["reference_total_fit_wall_s"])
        or float(diagnostics["reference_total_fit_wall_s"]) <= 0.0
        or abs(float(diagnostics["reference_total_fit_wall_s"])
               - float(diagnostics["wall_time_s"])) > 1e-6
        or not _finite_number(diagnostics["reference_full_fit_wall_s"])
        or float(diagnostics["reference_full_fit_wall_s"]) <= 0.0
        or float(diagnostics["reference_full_fit_wall_s"])
        > float(diagnostics["wall_time_s"]) + 1e-6
        or not _finite_number(diagnostics["reference_program_reported_wall_s"])
        or not 0.0 < float(diagnostics["reference_program_reported_wall_s"])
        < REFERENCE_PUBLIC_FIT_BUDGET_S
        or float(diagnostics["reference_program_reported_wall_s"])
        > float(diagnostics["wall_time_s"]) + 1e-6
        or not _finite_number(diagnostics["selection_wall_s"])
        or not 0.0 < float(diagnostics["selection_wall_s"])
        < float(diagnostics["reference_program_reported_wall_s"])
        or float(diagnostics["reference_total_fit_wall_s"])
        > REFERENCE_PROCESS_TIMEOUT_S + 1e-6
    ):
        raise ValueError("public reference timing diagnostics are invalid")
    if (
        not _finite_number(diagnostics["calibration_intercept"])
        or not _finite_number(diagnostics["calibration_slope"])
        or float(diagnostics["calibration_slope"]) <= 0.0
        or type(diagnostics["calibration_iterations"]) is not int
        or not 1 <= diagnostics["calibration_iterations"] <= REFERENCE_CALIBRATION_MAX_ITERATIONS
        or not _finite_number(diagnostics["calibration_relative_gradient"])
        or not 0.0 <= float(diagnostics["calibration_relative_gradient"])
        <= REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE
    ):
        raise ValueError("public reference calibration diagnostics are invalid")
    if (
        diagnostics["fit_exit_code"] != 0
        or type(diagnostics["fit_exit_code"]) is not int
        or diagnostics["fit_python_executable"] != REFERENCE_PYTHON_EXECUTABLE
        or diagnostics["fit_numpy_version"] != REFERENCE_NUMPY_VERSION
        or not isinstance(diagnostics["fit_source_sha256"], str)
        or len(diagnostics["fit_source_sha256"]) != 64
    ):
        raise ValueError("public reference fit execution identity is invalid")

    # Both declared family members must participate successfully. Letting one
    # disappear would turn best-of-family into survivor-of-family.
    selection = diagnostics["inner_selection_auc"]
    if (
        type(selection) is not dict
        or set(selection) != {"hierarchical_eb", "ridge_logistic"}
        or any(
            not _finite_number(v) or not 0.0 <= float(v) <= 1.0
            for v in selection.values()
        )
    ):
        raise ValueError("reference inner-selection AUCs are invalid")

    trials = diagnostics["inner_ridge_trials"]
    if type(trials) is not list or len(trials) != len(REFERENCE_RIDGE_PENALTIES):
        raise ValueError("reference ridge inner-trial diagnostics are incomplete")
    observed_penalties = []
    observed_aucs = []
    for trial in trials:
        if (
            type(trial) is not dict
            or set(trial) != {"penalty", "converged", "auc"}
            or trial["converged"] is not True
            or not _finite_number(trial["penalty"])
            or not _finite_number(trial["auc"])
            or not 0.0 <= float(trial["auc"]) <= 1.0
        ):
            raise ValueError("reference ridge inner-trial diagnostics are invalid")
        observed_penalties.append(float(trial["penalty"]))
        observed_aucs.append(float(trial["auc"]))
    if tuple(observed_penalties) != REFERENCE_RIDGE_PENALTIES:
        raise ValueError("reference ridge penalty grid does not match the contract")
    if abs(max(observed_aucs) - float(selection["ridge_logistic"])) > 1e-12:
        raise ValueError("reference ridge selection AUC is not the best declared trial")
    best_ridge_index = max(range(len(observed_aucs)), key=observed_aucs.__getitem__)
    best_ridge_penalty = observed_penalties[best_ridge_index]
    if (
        method == "ridge_logistic"
        and float(diagnostics["ridge_penalty"]) != best_ridge_penalty
    ):
        raise ValueError(
            "ridge reference was not refit at the winning inner-trial penalty"
        )

    # The DECLARED method must be the one the tie-margin rule actually selects, so a
    # recorded winner that did not win (or a razor-thin ridge win) is rejected as a
    # provenance failure rather than trusted.
    hierarchical = selection["hierarchical_eb"]
    ridge = selection["ridge_logistic"]
    hierarchical_wins = ridge <= hierarchical + REFERENCE_FAMILY_WIN_MARGIN
    expected_method = "hierarchical_eb" if hierarchical_wins else "ridge_logistic"
    if method != expected_method:
        raise ValueError(
            f"reference_method {method!r} disagrees with the inner-selection winner "
            f"{expected_method!r} (hierarchical_eb={hierarchical}, "
            f"ridge_logistic={ridge})")

    if (
        diagnostics["hierarchical_outer_iterations"]
        != REFERENCE_HIERARCHICAL_OUTER_ITERATIONS
        or type(diagnostics["hierarchical_outer_iterations"]) is not int
        or diagnostics["hierarchical_inner_iterations_run"]
        != REFERENCE_HIERARCHICAL_OUTER_ITERATIONS
        or type(diagnostics["hierarchical_inner_iterations_run"]) is not int
        or not _finite_number(diagnostics["hierarchical_inner_global_scale"])
        or float(diagnostics["hierarchical_inner_global_scale"]) <= 0.0
    ):
        raise ValueError("hierarchical inner-fit diagnostics are invalid")

    if method == "hierarchical_eb":
        if (
            type(diagnostics["hierarchical_iterations_run"]) is not int
            or diagnostics["hierarchical_iterations_run"]
            != REFERENCE_HIERARCHICAL_OUTER_ITERATIONS
            or not _finite_number(diagnostics["hierarchical_global_scale"])
            or float(diagnostics["hierarchical_global_scale"]) <= 0.0
        ):
            raise ValueError("hierarchical reference diagnostics are invalid")
    else:  # ridge_logistic
        grid = diagnostics["ridge_penalty_grid"]
        if (
            not _finite_number(diagnostics["ridge_penalty"])
            or float(diagnostics["ridge_penalty"]) <= 0.0
            or type(grid) is not list
            or not grid
            or any(not _finite_number(p) or float(p) <= 0.0 for p in grid)
            or tuple(float(p) for p in grid) != REFERENCE_RIDGE_PENALTIES
            or float(diagnostics["ridge_penalty"]) not in {float(p) for p in grid}
            or diagnostics["ridge_max_iterations"] != REFERENCE_RIDGE_MAX_ITERATIONS
            or type(diagnostics["ridge_max_iterations"]) is not int
            or type(diagnostics["ridge_iterations_run"]) is not int
            or not 1 <= diagnostics["ridge_iterations_run"] <= diagnostics["ridge_max_iterations"]
            or not _finite_number(diagnostics["ridge_tolerance"])
            or float(diagnostics["ridge_tolerance"]) != REFERENCE_RIDGE_TOLERANCE
            or not _finite_number(diagnostics["ridge_final_relative_gradient"])
            or float(diagnostics["ridge_final_relative_gradient"]) < 0.0
            or float(diagnostics["ridge_final_relative_gradient"])
            > float(diagnostics["ridge_tolerance"])
        ):
            raise ValueError("ridge reference diagnostics are invalid")

    if execution_order is not None and diagnostics["execution_order"] != list(
        execution_order
    ):
        raise ValueError("reference execution order is invalid")

    classes_present = diagnostics["classes_present"]
    if (
        type(classes_present) is not list
        or not classes_present
        or any(not isinstance(name, str) or not name for name in classes_present)
        or len(set(classes_present)) != len(classes_present)
    ):
        raise ValueError("reference classes_present is invalid")

    training_output = diagnostics["training_output"]
    if (
        type(training_output) is not dict
        or set(training_output) != {"model_sha256", "model_byte_count"}
        or not isinstance(training_output["model_sha256"], str)
        or len(training_output["model_sha256"]) != 64
        or type(training_output["model_byte_count"]) is not int
        or training_output["model_byte_count"] <= 0
        or diagnostics["prediction_exit_code"] != 0
        or type(diagnostics["prediction_exit_code"]) is not int
        or not _finite_number(diagnostics["prediction_wall_s"])
        or not 0.0 < float(diagnostics["prediction_wall_s"])
        <= REFERENCE_PUBLIC_PREDICT_BUDGET_S
        or type(diagnostics["prediction_output_byte_count"]) is not int
        or diagnostics["prediction_output_byte_count"] <= 0
        or diagnostics["prediction_model_sha256"] != training_output["model_sha256"]
        or diagnostics["prediction_source_sha256"] != diagnostics["fit_source_sha256"]
        or any(
            not isinstance(diagnostics[field], str)
            or len(diagnostics[field]) != 64
            for field in (
                "prediction_output_sha256", "prediction_model_sha256",
                "prediction_source_sha256",
            )
        )
    ):
        raise ValueError("reference serialized-model/prediction diagnostics are invalid")
