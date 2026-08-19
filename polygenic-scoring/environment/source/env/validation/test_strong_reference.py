from __future__ import annotations

from dataclasses import fields, replace
import inspect
import json
from typing import Any

import numpy as np
import pytest
import sv_pgs.model as engine_model

from reference import run_svpgs, baselines as strong_reference
from reference.zoo_protocol import REFERENCE_PROTOCOL_VERSION
from reference.run_svpgs import (
    PredictionDataset,
    PublicPredictorSchema,
    ReferenceVariantRecord,
    TrainingDataset,
    VariantClass,
)
from reference.baselines import (
    StrongReferenceConfig,
    StrongReferenceProtocolError,
    deterministic_stratified_inner_split,
    fit_fixed_ridge_logistic,
    fit_strong_reference,
)
from sv_pgs.data import VariantRecord


class _FakeSvPgsModel:
    def __init__(self, genotypes: np.ndarray, covariates: np.ndarray, targets: np.ndarray):
        centered_targets = np.asarray(targets, dtype=np.float64) - float(np.mean(targets))
        centered_genotypes = np.asarray(genotypes, dtype=np.float64) - np.mean(genotypes, axis=0)
        self.genotype_coefficients = centered_genotypes.T @ centered_targets
        self.genotype_coefficients /= max(genotypes.shape[0] * genotypes.shape[1], 1)
        self.covariate_coefficients = np.zeros(covariates.shape[1], dtype=np.float64)
        prevalence = float(np.clip(np.mean(targets), 1e-4, 1.0 - 1e-4))
        self.intercept = float(np.log(prevalence / (1.0 - prevalence)))

    def decision_function(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.intercept
            + np.asarray(genotypes) @ self.genotype_coefficients
            + np.asarray(covariates) @ self.covariate_coefficients,
            dtype=np.float64,
        )


def _fake_svpgs(
    genotypes,
    covariates,
    targets,
    records,
    *,
    max_outer_iterations,
    convergence_tolerance,
):
    del records, max_outer_iterations
    return {
        "model": _FakeSvPgsModel(genotypes, covariates, targets),
        "converged": True,
        "convergence_tolerance": convergence_tolerance,
        "iterations_run": 1,
        "final_parameter_change": convergence_tolerance / 2.0,
        "final_predictor_change": convergence_tolerance / 2.0,
        "final_objective_change": convergence_tolerance / 2.0,
        "final_hyperparameter_change": convergence_tolerance / 2.0,
        "wall_time_s": 0.0,
    }


def _synthetic_datasets(seed: int = 9) -> tuple[TrainingDataset, PredictionDataset]:
    random = np.random.default_rng(seed)
    sample_count, test_count, variant_count = 96, 24, 18
    genotypes = random.binomial(2, 0.3, size=(sample_count, variant_count)).astype(np.float32)
    covariates = random.normal(size=(sample_count, 2)).astype(np.float32)
    linear_predictor = (
        -0.25
        + 0.55 * covariates[:, 0]
        + 0.45 * (genotypes[:, 0] - 0.6)
        - 0.35 * (genotypes[:, 3] - 0.6)
    )
    probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    targets = (random.random(sample_count) < probability).astype(np.float32)
    if set(np.unique(targets)) != {0.0, 1.0}:
        raise AssertionError("synthetic targets lost a class")
    records = [
        ReferenceVariantRecord(
            variant_id=f"v{index}",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=index + 1,
            allele_frequency=0.3,
            training_support=int(np.count_nonzero(genotypes[:, index])),
        )
        for index in range(variant_count)
    ]
    schema = PublicPredictorSchema(
        family="binomial-logit",
        covariate_names=("age", "pc1"),
        variant_ids=tuple(f"v{index}" for index in range(variant_count)),
        annotation_types={"variant_class": "variant_class"},
    )
    training = TrainingDataset(
        schema=schema,
        sample_ids=tuple(f"train-{index:03d}" for index in range(sample_count)),
        genotypes=genotypes,
        covariates=covariates,
        targets=targets,
        variant_records=tuple(records),
    )
    prediction = PredictionDataset(
        schema=schema,
        sample_ids=tuple(f"test-{index:03d}" for index in range(test_count)),
        genotypes=random.binomial(
            2,
            0.3,
            size=(test_count, variant_count),
        ).astype(np.float32),
        covariates=random.normal(size=(test_count, 2)).astype(np.float32),
    )
    return training, prediction


def _small_config() -> StrongReferenceConfig:
    return StrongReferenceConfig(
        inner_validation_fraction=0.25,
        ridge_penalties=(0.1, 0.03),
        dense_spike_dense_penalties=(0.1, 0.03),
        dense_spike_sparse_penalties=(0.05, 0.01),
        elastic_net_penalties=(0.05, 0.01),
        elastic_net_l1_ratios=(0.5,),
        pt_keep_fractions=(0.1, 0.25),
        optimizer_max_iterations=800,
        optimizer_tolerance=2e-5,
        selection_min_absolute_gap=10.0,
    )


def test_deterministic_stratified_split_is_id_keyed_and_preserves_classes():
    dataset, _ = _synthetic_datasets()
    train_a, validation_a = deterministic_stratified_inner_split(
        dataset.sample_ids,
        dataset.targets,
        validation_fraction=0.25,
        seed=41,
    )
    train_b, validation_b = deterministic_stratified_inner_split(
        dataset.sample_ids,
        dataset.targets,
        validation_fraction=0.25,
        seed=41,
    )
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert set(np.unique(dataset.targets[train_a])) == {0.0, 1.0}
    assert set(np.unique(dataset.targets[validation_a])) == {0.0, 1.0}

    permutation = np.random.default_rng(3).permutation(len(dataset.sample_ids))
    permuted_ids = tuple(dataset.sample_ids[int(index)] for index in permutation)
    _, validation_permuted = deterministic_stratified_inner_split(
        permuted_ids,
        dataset.targets[permutation],
        validation_fraction=0.25,
        seed=41,
    )
    selected_ids = {dataset.sample_ids[int(index)] for index in validation_a}
    permuted_selected_ids = {permuted_ids[int(index)] for index in validation_permuted}
    assert selected_ids == permuted_selected_ids


def test_fit_accepts_only_the_structurally_training_only_type(monkeypatch):
    monkeypatch.setattr(strong_reference, "_fit_svpgs", _fake_svpgs)
    training, prediction = _synthetic_datasets()

    assert "test_targets" not in inspect.signature(fit_strong_reference).parameters
    assert [field.name for field in fields(TrainingDataset)] == [
        "schema",
        "sample_ids",
        "genotypes",
        "covariates",
        "targets",
        "variant_records",
    ]
    with pytest.raises(TypeError, match="exactly TrainingDataset"):
        fit_strong_reference(prediction, config=_small_config())  # type: ignore[arg-type]
    result = fit_strong_reference(training, config=_small_config())
    assert result.diagnostics["protocol_version"] == REFERENCE_PROTOCOL_VERSION
    assert result.diagnostics["protocol"]["fit_input_type"] == "TrainingDataset"
    assert result.diagnostics["protocol"]["fit_input_fields"] == [
        field.name for field in fields(TrainingDataset)
    ]


def test_mutable_engine_records_are_created_only_at_the_fit_boundary(monkeypatch):
    training, _ = _synthetic_datasets()

    class BoundaryReached(RuntimeError):
        pass

    class CapturingEngine:
        def __init__(self, _config):
            pass

        def fit(self, _genotypes, _covariates, _targets, records):
            assert all(type(record) is VariantRecord for record in records)
            assert all(
                type(source) is ReferenceVariantRecord
                for source in training.variant_records
            )
            raise BoundaryReached

    # The hidden engine is imported lazily at the active model-zoo fit boundary;
    # patch its defining module rather than resurrecting a dead module-global alias.
    monkeypatch.setattr(engine_model, "BayesianPGS", CapturingEngine)
    with pytest.raises(BoundaryReached):
        run_svpgs._fit_svpgs(
            training.genotypes,
            training.covariates,
            training.targets,
            training.variant_records,
            max_outer_iterations=1,
            convergence_tolerance=1e-4,
        )


def test_training_dto_rejects_schema_record_order_drift_and_invalid_dosage():
    training, _ = _synthetic_datasets()
    with pytest.raises(ValueError, match="does not align"):
        TrainingDataset(
            schema=training.schema,
            sample_ids=training.sample_ids,
            genotypes=training.genotypes,
            covariates=training.covariates,
            targets=training.targets,
            variant_records=tuple(reversed(training.variant_records)),
        )

    invalid_genotypes = np.asarray(training.genotypes).copy()
    invalid_genotypes[0, 0] = 1.5
    with pytest.raises(ValueError, match="does not align"):
        TrainingDataset(
            schema=training.schema,
            sample_ids=training.sample_ids,
            genotypes=invalid_genotypes,
            covariates=training.covariates,
            targets=training.targets,
            variant_records=training.variant_records,
        )


def test_public_schema_and_reference_records_reject_malformed_identifiers():
    with pytest.raises(ValueError, match="schema is invalid"):
        PublicPredictorSchema(
            family="binomial-logit",
            covariate_names=("",),
            variant_ids=("v0",),
            annotation_types={"variant_class": "variant_class"},
        )
    with pytest.raises(ValueError, match="record is invalid"):
        ReferenceVariantRecord(
            variant_id="",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=1,
        )


def test_strong_reference_outputs_are_finite_and_protocol_diagnostics_are_complete(
    monkeypatch,
):
    monkeypatch.setattr(strong_reference, "_fit_svpgs", _fake_svpgs)
    dataset, prediction = _synthetic_datasets()
    result = fit_strong_reference(dataset, config=_small_config())
    logits = result.model.decision_function(
        prediction.genotypes,
        prediction.covariates,
    )

    assert logits.shape == (len(prediction.sample_ids),)
    assert np.all(np.isfinite(logits))
    diagnostics = result.diagnostics
    json.dumps(diagnostics, allow_nan=False)
    assert diagnostics["converged"] is True
    assert diagnostics["training_output"]["finite"] is True
    assert diagnostics["protocol"]["candidate_order"] == [
        "sv_pgs",
        "ridge_logistic",
        "dense_spike_logistic",
        "elastic_net_logistic",
        "marginal_pt",
        "annotation_adaptive_en",
    ]
    assert set(diagnostics["candidates"]) == set(diagnostics["protocol"]["candidate_order"])
    assert diagnostics["selection"]["objective"]["metric_weights"] == {
        "auc": 0.6,
        "brier": 0.2,
        "log_loss": 0.2,
    }
    expected_best = max(
        diagnostics["protocol"]["candidate_order"],
        key=lambda name: diagnostics["candidates"][name]["selection_utility"],
    )
    assert diagnostics["selection"]["best_single_candidate"] == expected_best
    assert diagnostics["selection"]["blend_gate"]["bootstrap_replicates"] == 300
    blend_gate = diagnostics["selection"]["blend_gate"]
    assert blend_gate["accepted"] is bool(
        any(weight > 0.0 for weight in blend_gate["candidate_weights"].values())
        and blend_gate["improvement"] > 0.0
    )
    assert (
        diagnostics["selection"]["single_candidate_gate"]["bootstrap_replicates"]
        == 300
    )
    refit_candidates = set(diagnostics["selection"]["refit_candidates"])
    assert refit_candidates
    for name, candidate in diagnostics["candidates"].items():
        assert np.isfinite(candidate["validation_log_loss"])
        assert set(candidate["validation_metrics"]) == {"auc", "brier", "log_loss"}
        assert np.isfinite(candidate["selection_utility"])
        assert candidate["tuning_trials"]
        assert candidate["selected_for_full_refit"] is (name in refit_candidates)
        assert (candidate["full_refit"] is not None) is (name in refit_candidates)


def test_annotation_design_contains_nonlinear_and_class_interaction_terms():
    continuous = np.asarray([-2.0, -1.0, 1.0, 2.0])
    classes = (
        VariantClass.SNV,
        VariantClass.SNV,
        VariantClass.DELETION_SHORT,
        VariantClass.DELETION_SHORT,
    )
    records = tuple(
        ReferenceVariantRecord(
            variant_id=f"v{index}",
            variant_class=variant_class,
            chromosome="1",
            position=index + 1,
            prior_continuous_features={"length": float(value)},
        )
        for index, (variant_class, value) in enumerate(zip(classes, continuous))
    )

    design = strong_reference._annotation_design(records)
    assert design.shape == (4, 5)
    assert any(
        np.array_equal(design[:, index], continuous * continuous)
        for index in range(design.shape[1])
    )
    class_indicator = np.asarray([1.0, 1.0, 0.0, 0.0])
    assert any(
        np.array_equal(design[:, index], class_indicator * continuous)
        for index in range(design.shape[1])
    )


def test_full_refit_distillation_recovers_scale_without_outcomes():
    honest = np.asarray([-2.0, -1.0, 1.0, 2.0])
    full_refit = 2.0 * honest + 3.0

    calibration = strong_reference._full_refit_distillation(honest, full_refit)
    reconstructed = (
        calibration["intercept"] + calibration["slope"] * full_refit
    )
    np.testing.assert_allclose(reconstructed, honest, rtol=0.0, atol=1e-12)
    assert calibration["correlation"] == pytest.approx(1.0)

    reversed_calibration = strong_reference._full_refit_distillation(
        honest,
        -honest,
    )
    assert reversed_calibration["slope"] == 0.0
    assert reversed_calibration["correlation"] == pytest.approx(-1.0)


def test_single_candidate_gate_distinguishes_clear_wins_from_sampling_ties():
    targets = np.tile(np.asarray([0.0, 1.0]), 100)
    naive_metrics = strong_reference._validation_metrics(
        targets,
        np.zeros(targets.size, dtype=np.float64),
    )

    def selection(name, logits):
        logits = np.asarray(logits, dtype=np.float64)
        metrics = strong_reference._validation_metrics(targets, logits)
        utility, components = strong_reference._validation_utility(
            metrics,
            naive_metrics,
        )
        return (
            strong_reference._CandidateSelection(
                name=name,
                hyperparameters={},
                validation_logits=logits,
                validation_log_loss=float(metrics["log_loss"]),
                tuning_trials=(),
                cross_tuning={},
            ),
            {
                "validation_metrics": metrics,
                "selection_utility": utility,
                "selection_utility_components": components,
            },
        )

    clear_best = np.where(targets == 1.0, 3.0, -3.0)
    weak_runner = np.zeros(targets.size, dtype=np.float64)
    best_selection, best_validation = selection("best", clear_best)
    runner_selection, runner_validation = selection("runner", weak_runner)
    config = replace(
        _small_config(),
        selection_min_absolute_gap=0.002,
        selection_bootstrap_replicates=100,
    )
    clear = strong_reference._single_candidate_uncertainty_gate(
        (best_selection, runner_selection),
        targets,
        (0, 1),
        (best_validation, runner_validation),
        naive_metrics,
        config,
    )
    assert clear["clear_winner"] is True
    assert clear["validation_utility_gap"] > clear["required_gap"]

    tied_selection, tied_validation = selection("runner", clear_best.copy())
    tied = strong_reference._single_candidate_uncertainty_gate(
        (best_selection, tied_selection),
        targets,
        (0, 1),
        (best_validation, tied_validation),
        naive_metrics,
        config,
    )
    assert tied["clear_winner"] is False
    assert tied["validation_utility_gap"] == 0.0
    assert tied["reason"] == "top_two_indistinguishable"


def test_cross_tuning_chooses_each_evaluation_folds_trial_on_other_fold():
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    folds = np.asarray([0, 1, 0, 1], dtype=np.int8)
    trial_a = np.asarray([-3.0, 3.0, 3.0, -3.0])
    trial_b = -trial_a
    tuning_trials = (
        {"hyperparameters": {"penalty": 0.1}, "validation_log_loss": 1.0},
        {"hyperparameters": {"penalty": 0.2}, "validation_log_loss": 1.0},
    )

    selection = strong_reference._cross_tuned_candidate(
        "test",
        (({"penalty": 0.1}, trial_a), ({"penalty": 0.2}, trial_b)),
        targets,
        folds,
        tuning_trials,
    )

    assert selection.cross_tuning[
        "selected_hyperparameters_by_evaluation_fold"
    ] == [{"penalty": 0.2}, {"penalty": 0.1}]
    np.testing.assert_array_equal(
        selection.validation_logits,
        np.asarray([3.0, 3.0, -3.0, -3.0]),
    )


def test_cross_tuning_fold_assignment_is_deterministic_and_stratified():
    sample_ids = tuple(f"sample-{index}" for index in range(20))
    targets = np.tile(np.asarray([0.0, 1.0]), 10)
    first = strong_reference._cross_tuning_folds(sample_ids, targets, seed=41)
    second = strong_reference._cross_tuning_folds(sample_ids, targets, seed=41)

    np.testing.assert_array_equal(first, second)
    for outcome in (0.0, 1.0):
        assert set(first[targets == outcome]) == {0, 1}


def test_dense_spike_penalty_is_exact_dense_sparse_infimal_convolution():
    coefficients = np.asarray([-3.0, -0.4, 0.0, 0.7, 4.0], dtype=np.float64)
    dense_l2_penalty = 2.0
    sparse_l1_penalty = 1.0
    transition = sparse_l1_penalty / dense_l2_penalty
    dense = np.clip(coefficients, -transition, transition)
    sparse = coefficients - dense
    explicit = (
        0.5 * dense_l2_penalty * float(np.dot(dense, dense))
        + sparse_l1_penalty * float(np.sum(np.abs(sparse)))
    )
    reduced = strong_reference._dense_spike_penalty_value(
        coefficients,
        dense_l2_penalty=dense_l2_penalty,
        sparse_l1_penalty=sparse_l1_penalty,
    )
    assert reduced == pytest.approx(explicit)

    step_size = 0.3
    proximal = strong_reference._dense_spike_prox(
        coefficients,
        step_size=step_size,
        dense_l2_penalty=dense_l2_penalty,
        sparse_l1_penalty=sparse_l1_penalty,
    )
    proximal_boundary = transition + step_size * sparse_l1_penalty
    expected = np.where(
        np.abs(coefficients) <= proximal_boundary,
        coefficients / (1.0 + step_size * dense_l2_penalty),
        np.sign(coefficients)
        * (np.abs(coefficients) - step_size * sparse_l1_penalty),
    )
    np.testing.assert_allclose(proximal, expected, rtol=0.0, atol=1e-15)


def test_dense_spike_candidate_tunes_two_penalties_and_refits_convergently(monkeypatch):
    monkeypatch.setattr(strong_reference, "_fit_svpgs", _fake_svpgs)
    dataset, prediction = _synthetic_datasets()
    config = _small_config()
    result = fit_strong_reference(dataset, config=config)
    candidate = result.diagnostics["candidates"]["dense_spike_logistic"]

    assert len(candidate["tuning_trials"]) == (
        len(config.dense_spike_dense_penalties)
        * len(config.dense_spike_sparse_penalties)
    )
    assert set(candidate["selected_hyperparameters"]) == {
        "dense_l2_penalty",
        "sparse_l1_penalty",
    }
    for trial in candidate["tuning_trials"]:
        assert trial["optimizer"]["converged"] is True
        assert trial["decomposition"]["reconstruction_max_abs_error"] == 0.0
    selection = strong_reference._CandidateSelection(
        name="dense_spike_logistic",
        hyperparameters=candidate["selected_hyperparameters"],
        validation_logits=np.zeros(0, dtype=np.float64),
        validation_log_loss=float(candidate["validation_log_loss"]),
        tuning_trials=(),
        cross_tuning={},
    )
    model, refit, _prepared = strong_reference._refit_candidate(
        selection,
        dataset,
        config,
        None,
    )
    assert refit["optimizer"]["converged"] is True
    assert refit["decomposition"]["reconstruction_max_abs_error"] == 0.0
    logits = model.decision_function(prediction.genotypes, prediction.covariates)
    assert np.all(np.isfinite(logits))


def test_dense_spike_candidate_nonconvergence_fails_closed(monkeypatch):
    dataset, _ = _synthetic_datasets()
    config = _small_config()
    prepared = strong_reference._prepare_training_design(
        dataset.genotypes,
        dataset.covariates,
        dataset.targets,
    )
    failed = strong_reference._OptimizerResult(
        coefficients=np.zeros(prepared.matrix.shape[1], dtype=np.float64),
        converged=False,
        iterations=config.optimizer_max_iterations,
        objective=0.7,
        lipschitz=1.0,
    )
    monkeypatch.setattr(
        strong_reference,
        "_fit_dense_spike_logistic",
        lambda *_args, **_kwargs: failed,
    )
    with pytest.raises(StrongReferenceProtocolError, match="did not converge"):
        strong_reference._tune_dense_spike_candidate(
            prepared,
            dataset.genotypes[:24],
            dataset.covariates[:24],
            dataset.targets[:24],
            config,
            np.tile(np.asarray([0, 1], dtype=np.int8), 12),
        )


def test_svpgs_candidate_nonconvergence_fails_closed(monkeypatch):
    def nonconverged(*args: Any, **kwargs: Any):
        del args, kwargs
        return {"model": object(), "converged": False, "wall_time_s": 0.0}

    monkeypatch.setattr(strong_reference, "_fit_svpgs", nonconverged)
    with pytest.raises(StrongReferenceProtocolError, match="did not converge"):
        training, _ = _synthetic_datasets()
        fit_strong_reference(training, config=_small_config())


def test_fixed_ridge_runtime_anchor_is_finite_and_converged():
    dataset, prediction = _synthetic_datasets()
    model, diagnostics = fit_fixed_ridge_logistic(
        dataset.genotypes,
        dataset.covariates,
        dataset.targets,
        config=_small_config(),
    )
    logits = model.decision_function(
        prediction.genotypes,
        prediction.covariates,
    )
    assert diagnostics["converged"] is True
    assert logits.shape == (len(prediction.sample_ids),)
    assert np.all(np.isfinite(logits))
