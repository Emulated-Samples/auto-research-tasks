"""Fail-closed contract tests for the exact public NumPy reference."""
from __future__ import annotations

import copy

import pytest

from reference.protocol import (
    REFERENCE_ANCHOR_EXECUTION_ORDER,
    REFERENCE_CALIBRATION_MAX_ITERATIONS,
    REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE,
    REFERENCE_ENTRYPOINT,
    REFERENCE_FAMILY_WIN_MARGIN,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
    REFERENCE_IMPLEMENTATION,
    REFERENCE_METHODS,
    REFERENCE_PROTOCOL_VERSION,
    REFERENCE_PUBLIC_FIT_BUDGET_S,
    REFERENCE_RIDGE_MAX_ITERATIONS,
    REFERENCE_RIDGE_PENALTIES,
    REFERENCE_RIDGE_TOLERANCE,
    REFERENCE_SOURCE_RELATIVE_PATHS,
    validate_reference_diagnostics,
)


def _trials(best: float) -> list[dict]:
    return [
        {"penalty": penalty, "converged": True, "auc": best - index * 0.001}
        for index, penalty in enumerate(REFERENCE_RIDGE_PENALTIES)
    ]


def _common(method: str, hierarchical_auc: float, ridge_auc: float) -> dict:
    value = {
        "protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_method": method,
        "converged": True,
        "fit_protocol": REFERENCE_FIT_PROTOCOL,
        "wall_time_s": 100.0,
        "fit_exit_code": 0,
        "fit_source_sha256": "3" * 64,
        "fit_python_executable": "/opt/svpgs-venv/bin/python",
        "fit_numpy_version": "2.1.3",
        "reference_total_fit_wall_s": 100.0,
        "reference_full_fit_wall_s": 35.0,
        "reference_program_reported_wall_s": 99.0,
        "selection_wall_s": 60.0,
        "calibration_intercept": -0.1,
        "calibration_slope": 0.8,
        "calibration_iterations": 4,
        "calibration_relative_gradient": 1e-10,
        "inner_selection_auc": {
            "hierarchical_eb": hierarchical_auc,
            "ridge_logistic": ridge_auc,
        },
        "inner_ridge_trials": _trials(ridge_auc),
        "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
        "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
        "hierarchical_inner_global_scale": 0.03,
        "prediction_exit_code": 0,
        "prediction_wall_s": 0.25,
        "prediction_output_sha256": "2" * 64,
        "prediction_output_byte_count": 128,
        "prediction_model_sha256": "1" * 64,
        "prediction_source_sha256": "3" * 64,
        "classes_present": ["snv", "deletion_long"],
        "training_output": {
            "model_sha256": "1" * 64,
            "model_byte_count": 4096,
        },
    }
    if method == "hierarchical_eb":
        value.update({
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.025,
        })
    else:
        value.update({
            "ridge_penalty": REFERENCE_RIDGE_PENALTIES[0],
            "ridge_penalty_grid": list(REFERENCE_RIDGE_PENALTIES),
            "ridge_max_iterations": REFERENCE_RIDGE_MAX_ITERATIONS,
            "ridge_iterations_run": 80,
            "ridge_tolerance": REFERENCE_RIDGE_TOLERANCE,
            "ridge_final_relative_gradient": 5e-6,
        })
    return value


def _hierarchical() -> dict:
    return _common("hierarchical_eb", 0.72, 0.70)


def _ridge() -> dict:
    return _common("ridge_logistic", 0.68, 0.72)


def test_public_reference_identity_is_exact_and_packageable() -> None:
    assert REFERENCE_IMPLEMENTATION == (
        "public NumPy best-of-family (hierarchical_eb | ridge_logistic)"
    )
    assert REFERENCE_ENTRYPOINT == "gold/fit"
    assert REFERENCE_PROTOCOL_VERSION == 9
    assert REFERENCE_FIT_PROTOCOL == (
        "public_numpy_family_inner_validation_platt_full_refit"
    )
    assert REFERENCE_CALIBRATION_MAX_ITERATIONS == 100
    assert REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE == 1e-8
    assert REFERENCE_METHODS == ("hierarchical_eb", "ridge_logistic")
    assert REFERENCE_SOURCE_RELATIVE_PATHS == (
        "gold/fit", "gold/predict", "gold/pgs_core.py"
    )


@pytest.mark.parametrize("fixture", [_hierarchical, _ridge])
def test_both_public_reference_methods_validate(fixture) -> None:
    value = fixture()
    validate_reference_diagnostics(value, training_count=1800)
    value["execution_order"] = list(REFERENCE_ANCHOR_EXECUTION_ORDER)
    validate_reference_diagnostics(
        value,
        training_count=1800,
        execution_order=REFERENCE_ANCHOR_EXECUTION_ORDER,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(reference_method="ridge_logistic"),
        lambda d: d["inner_selection_auc"].update(hierarchical_eb=float("nan")),
        lambda d: d["inner_ridge_trials"].pop(),
        lambda d: d.update(reference_program_reported_wall_s=REFERENCE_PUBLIC_FIT_BUDGET_S),
        lambda d: d.update(selection_wall_s=d["reference_program_reported_wall_s"]),
        lambda d: d.pop("calibration_intercept"),
        lambda d: d.pop("calibration_slope"),
        lambda d: d.pop("calibration_iterations"),
        lambda d: d.pop("calibration_relative_gradient"),
        lambda d: d.update(calibration_intercept=float("nan")),
        lambda d: d.update(calibration_intercept=float("inf")),
        lambda d: d.update(calibration_intercept=True),
        lambda d: d.update(calibration_slope=float("nan")),
        lambda d: d.update(calibration_slope=float("inf")),
        lambda d: d.update(calibration_slope=0.0),
        lambda d: d.update(calibration_slope=-0.1),
        lambda d: d.update(calibration_slope=True),
        lambda d: d.update(calibration_iterations=0),
        lambda d: d.update(calibration_iterations=101),
        lambda d: d.update(calibration_iterations=True),
        lambda d: d.update(calibration_relative_gradient=float("nan")),
        lambda d: d.update(calibration_relative_gradient=-1e-12),
        lambda d: d.update(calibration_relative_gradient=1.1e-8),
        lambda d: d.update(calibration_relative_gradient=True),
        lambda d: d.update(hierarchical_inner_iterations_run=9),
        lambda d: d.update(hierarchical_global_scale=0.0),
        lambda d: d["training_output"].update(model_sha256="x"),
        lambda d: d.update(prediction_exit_code=1),
        lambda d: d.update(prediction_model_sha256="4" * 64),
    ],
)
def test_reference_diagnostics_fail_closed(mutation) -> None:
    value = copy.deepcopy(_hierarchical())
    mutation(value)
    with pytest.raises(ValueError):
        validate_reference_diagnostics(value, training_count=1800)


def test_ridge_must_win_by_the_declared_margin() -> None:
    value = _ridge()
    value["inner_selection_auc"] = {
        "hierarchical_eb": 0.70,
        "ridge_logistic": 0.70 + REFERENCE_FAMILY_WIN_MARGIN,
    }
    value["inner_ridge_trials"] = _trials(0.70 + REFERENCE_FAMILY_WIN_MARGIN)
    with pytest.raises(ValueError, match="disagrees"):
        validate_reference_diagnostics(value, training_count=1800)
