"""Public-contract tests for the standalone gold family selector.

These tests deliberately import only the submission's own plumbing module.  A
gold arm that reaches into trusted benchmark packages is not a solvability
witness, even if its predictions happen to match.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "gold"


def _load_core():
    spec = importlib.util.spec_from_file_location("gold_public_core", GOLD / "pgs_core.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gold_contains_only_public_standalone_imports() -> None:
    forbidden_roots = {"datagen", "grader", "reference", "validation"}
    for path in (GOLD / "fit", GOLD / "predict", GOLD / "pgs_core.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert not imported_roots & forbidden_roots


def test_gold_selector_constants_match_the_public_protocol() -> None:
    core = _load_core()
    assert core.INNER_VALIDATION_FRACTION == 0.4
    assert core.INNER_SPLIT_SEED == 20260718
    assert core.RIDGE_PENALTIES == (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    assert core.FAMILY_WIN_MARGIN == 0.03


def test_gold_split_is_stable_stratified_and_row_order_independent() -> None:
    core = _load_core()
    ids = tuple(f"sample-{index}" for index in range(20))
    targets = np.asarray([0.0, 1.0] * 10)
    training, validation = core.deterministic_stratified_inner_split(ids, targets)

    expected = []
    for target in (0.0, 1.0):
        members = np.flatnonzero(targets == target)
        ranked = sorted(
            (int(index) for index in members),
            key=lambda index: (
                hashlib.sha256(
                    f"20260718\0{ids[index]}".encode("utf-8")
                ).digest(),
                ids[index],
            ),
        )
        validation_count = int(round(core.INNER_VALIDATION_FRACTION * len(ranked)))
        expected.extend(ranked[:validation_count])
    assert validation.tolist() == sorted(expected)
    assert set(training).isdisjoint(validation)
    assert sorted(np.concatenate([training, validation]).tolist()) == list(range(20))
    assert set(targets[training]) == {0.0, 1.0}
    assert set(targets[validation]) == {0.0, 1.0}


def test_gold_auc_uses_average_ranks_for_ties() -> None:
    core = _load_core()
    targets = np.asarray([0.0, 1.0, 0.0, 1.0])
    assert core.auc(targets, np.asarray([0.1, 0.9, 0.2, 0.8])) == 1.0
    assert core.auc(targets, np.ones(4)) == 0.5


def test_gold_platt_calibrator_is_deterministic_positive_and_improves_probabilities() -> None:
    core = _load_core()
    assert core.PLATT_L2 == 1e-3
    logits = np.linspace(-5.0, 5.0, 200)
    true_probability = 1.0 / (1.0 + np.exp(-(-0.7 + 0.28 * logits)))
    # Deterministic quantile construction avoids making optimizer coverage depend on RNG.
    targets = (
        ((np.arange(logits.size, dtype=np.float64) * 0.6180339887498949) % 1.0)
        < true_probability
    ).astype(np.float64)
    first = core.fit_platt_calibrator(logits, targets)
    second = core.fit_platt_calibrator(logits, targets)
    assert first == second
    assert first["slope"] > 0.0
    assert np.isfinite(first["intercept"])
    assert first["relative_gradient"] <= core.PLATT_RELATIVE_GRADIENT_TOLERANCE

    raw_probability = 1.0 / (1.0 + np.exp(-logits))
    calibrated_probability = 1.0 / (
        1.0 + np.exp(-(first["intercept"] + first["slope"] * logits))
    )
    raw_log_loss = -np.mean(
        targets * np.log(raw_probability)
        + (1.0 - targets) * np.log1p(-raw_probability)
    )
    calibrated_log_loss = -np.mean(
        targets * np.log(calibrated_probability)
        + (1.0 - targets) * np.log1p(-calibrated_probability)
    )
    assert calibrated_log_loss < raw_log_loss
    assert np.mean((calibrated_probability - targets) ** 2) < np.mean(
        (raw_probability - targets) ** 2
    )
    assert core.auc(targets, calibrated_probability) == core.auc(targets, logits)


def test_gold_platt_calibrator_rejects_invalid_or_non_predictive_inputs() -> None:
    import pytest

    core = _load_core()
    invalid = (
        (np.asarray([0.0, 1.0, np.nan, 2.0]), np.asarray([0.0, 1.0, 0.0, 1.0])),
        (np.asarray([0.0, 1.0, 2.0, 3.0]), np.asarray([0.0, 0.0, 0.0, 0.0])),
        (np.ones(4), np.asarray([0.0, 1.0, 0.0, 1.0])),
        (np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 0.0, 1.0])),
    )
    for logits, targets in invalid:
        with pytest.raises(ValueError):
            core.fit_platt_calibrator(logits, targets)


def test_gold_platt_calibrator_fails_closed_when_ranking_is_reversed() -> None:
    import pytest

    core = _load_core()
    logits = np.linspace(-3.0, 3.0, 80)
    targets = (logits < 0.0).astype(np.float64)
    with pytest.raises(ValueError):
        core.fit_platt_calibrator(logits, targets)


def test_gold_ridge_is_deterministic_and_serializes_public_prediction_axes() -> None:
    core = _load_core()
    rng = np.random.default_rng(20260718)
    genotypes = rng.integers(0, 3, size=(80, 12)).astype(np.float64)
    covariates = rng.normal(size=(80, 2))
    signal = 0.8 * (genotypes[:, 0] - 1.0) + 0.4 * covariates[:, 0]
    targets = (signal + rng.logistic(size=80) > 0.0).astype(np.float64)
    first = core.fit_ridge_logistic(genotypes, covariates, targets, 0.1)
    second = core.fit_ridge_logistic(genotypes, covariates, targets, 0.1)
    assert np.array_equal(first["alpha"], second["alpha"])
    assert np.array_equal(first["beta"], second["beta"])
    logits = core.linear_predict(first, genotypes, covariates)
    assert logits.shape == (80,)
    assert np.all(np.isfinite(logits))


def test_gold_ridge_matches_the_shipped_ridge_candidate() -> None:
    from reference.runtime_anchor import fit_fixed_ridge_logistic

    core = _load_core()
    rng = np.random.default_rng(41)
    genotypes = rng.integers(0, 3, size=(96, 15)).astype(np.float64)
    covariates = rng.normal(size=(96, 3))
    signal = genotypes[:, 1] - 0.5 * genotypes[:, 4] + covariates[:, 0]
    targets = (signal + rng.logistic(size=96) > np.median(signal)).astype(np.float64)
    public = core.fit_ridge_logistic(genotypes, covariates, targets, 0.03)
    shipped, diagnostics = fit_fixed_ridge_logistic(
        genotypes, covariates, targets, penalty=0.03
    )
    assert diagnostics["converged"] is True
    public_logits = core.linear_predict(public, genotypes, covariates)
    shipped_logits = shipped.decision_function(genotypes, covariates)
    assert np.allclose(public_logits, shipped_logits, rtol=1e-12, atol=1e-12)


def test_gold_fit_pays_for_complete_selection_before_full_refit() -> None:
    source = (GOLD / "fit").read_text(encoding="utf-8")
    assert "for penalty in RIDGE_PENALTIES" in source
    assert "ridge_auc > hierarchical_auc + FAMILY_WIN_MARGIN" in source
    assert "hierarchical inner candidate missed its selection reserve" in source
    assert "best-of-family selection exhausted its full-refit reserve" in source
    assert "FIT_BUDGET_S = 165.0" in source
    assert "INNER_HIERARCHICAL_DEADLINE_S = 60.0" in source
    assert "SELECTION_DEADLINE_S = 100.0" in source
    assert "HIERARCHICAL_OUTER_ITERATIONS = 6" in source


def test_gold_probability_postprocessing_is_shared_and_extreme_safe() -> None:
    core = _load_core()
    model = {
        "alpha": np.asarray([0.0]),
        "beta": np.asarray([1000.0]),
        "gmean": np.asarray([0.0]),
        "gsd": np.asarray([1.0]),
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
    }
    genotypes = np.asarray([[-1.0], [1.0]])
    covariates = np.empty((2, 0))
    probability = core.predict_probability(model, genotypes, covariates)
    expected = 1.0 / (1.0 + np.exp(-np.asarray([-30.0, 30.0])))
    assert np.array_equal(probability, expected)
    predict_source = (GOLD / "predict").read_text(encoding="utf-8")
    assert "predict_probability(model, G, Xcov)" in predict_source
    assert "nan_to_num" not in predict_source
    assert "pos.get" not in predict_source


def test_gold_probability_postprocessing_requires_valid_positive_calibration() -> None:
    import pytest

    core = _load_core()
    base = {
        "alpha": np.asarray([0.0]),
        "beta": np.asarray([1.0]),
        "gmean": np.asarray([0.0]),
        "gsd": np.asarray([1.0]),
    }
    genotypes = np.asarray([[-1.0], [1.0]])
    covariates = np.empty((2, 0))
    for calibration in (
        {},
        {"calibration_intercept": 0.0, "calibration_slope": 0.0},
        {"calibration_intercept": np.nan, "calibration_slope": 1.0},
        {"calibration_intercept": 0.0, "calibration_slope": np.inf},
    ):
        with pytest.raises(ValueError):
            core.predict_probability({**base, **calibration}, genotypes, covariates)

    calibrated = core.predict_probability(
        {**base, "calibration_intercept": -0.5, "calibration_slope": 0.25},
        genotypes,
        covariates,
    )
    expected = 1.0 / (1.0 + np.exp(-np.asarray([-0.75, -0.25])))
    assert np.array_equal(calibrated, expected)


def test_corpus_builder_executes_exact_public_gold_without_hidden_engine() -> None:
    source = (ROOT / "datagen" / "build_corpus.py").read_text(encoding="utf-8")
    assert '[REFERENCE_PYTHON_EXECUTABLE, os.path.join(submission_dir, "fit")]' in source
    assert "_public_reference_source_snapshot()" in source
    assert "_fit_svpgs" not in source
    assert 'os.path.join(os.path.dirname(_ROOT), "SV-PGS")' not in source
    assert "run_prefit_predict(" in source
    assert "reference_model.predict_probability" not in source
    assert "from gold.pgs_core" not in source
