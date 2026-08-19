"""Fast ridge-logistic runtime anchor for report-only efficiency diagnostics.

This is the corpus's *timing* baseline only -- a practical production-style
penalized logistic PGS that a competent engineer would ship. It is deliberately
NOT the accuracy reference (that is the exact public NumPy best of hierarchical
empirical Bayes and ridge in ``gold/fit``). Runtime is compared against this anchor
only in the diagnostics produced by ``grader.perf``.

Covariates are unpenalized fixed effects; the standardized genotype block carries
a single ridge penalty. The estimator is a matrix-free FISTA (accelerated
proximal-gradient) solve that streams matvecs against the genotype matrix, so it
costs O(iterations * N * P) and stays fast in the disclosed ``P >= N`` regime
without a cubic ``P x P`` Newton factorization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _FixedRidgeModel:
    """A linear logit model on raw dosage: eta = intercept + C@theta + G@weights."""

    intercept: float
    covariate_coefficients: np.ndarray
    genotype_weights: np.ndarray
    genotype_mean: np.ndarray

    def decision_function(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        genotype_array = np.asarray(genotypes, dtype=np.float64)
        covariate_array = np.asarray(covariates, dtype=np.float64)
        if genotype_array.ndim != 2 or genotype_array.shape[1] != self.genotype_weights.size:
            raise ValueError("genotype columns do not match the fitted runtime anchor")
        if covariate_array.ndim != 2 or covariate_array.shape != (
            genotype_array.shape[0],
            self.covariate_coefficients.size,
        ):
            raise ValueError("covariate columns do not match the fitted runtime anchor")
        score = (
            float(self.intercept)
            + covariate_array @ self.covariate_coefficients
            + (genotype_array - self.genotype_mean) @ self.genotype_weights
        )
        if not np.all(np.isfinite(score)):
            raise ValueError("runtime anchor produced non-finite scores")
        return np.asarray(score, dtype=np.float64)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    centered = values - mean
    scale = np.sqrt(np.einsum("ij,ij->j", centered, centered) / max(values.shape[0], 1))
    scale = np.where(scale < 1e-8, 1.0, scale)
    return centered / scale, mean, scale


def fit_fixed_ridge_logistic(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    *,
    penalty: float = 1.0,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    nuisance_penalty: float = 1e-8,
) -> tuple[_FixedRidgeModel, dict]:
    """Fit one deterministic ridge-logistic model (no model selection, no CV).

    ``penalty`` is the ridge strength on the standardized genotype block. Returns
    (model, diagnostics); ``diagnostics['converged']`` reports first-order stopping.
    """
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(nuisance_penalty) or nuisance_penalty < 0.0:
        raise ValueError("nuisance_penalty must be finite and nonnegative")
    genotype_array = np.asarray(genotypes, dtype=np.float64)
    covariate_array = np.asarray(covariates, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1)
    sample_count, variant_count = genotype_array.shape
    if (
        covariate_array.ndim != 2
        or covariate_array.shape[0] != sample_count
        or target_array.size != sample_count
        or set(np.unique(target_array)) - {0.0, 1.0}
        or not np.all(np.isfinite(genotype_array))
        or not np.all(np.isfinite(covariate_array))
    ):
        raise ValueError("runtime anchor received invalid training data")

    standardized_genotypes, genotype_mean, genotype_scale = _standardize(genotype_array)
    standardized_covariates, covariate_mean, covariate_scale = _standardize(covariate_array)
    covariate_count = standardized_covariates.shape[1]
    design = np.empty((sample_count, 1 + covariate_count + variant_count), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1 : 1 + covariate_count] = standardized_covariates
    design[:, 1 + covariate_count :] = standardized_genotypes
    genetic_start = 1 + covariate_count

    l2 = np.zeros(design.shape[1], dtype=np.float64)
    l2[1:genetic_start] = nuisance_penalty
    l2[genetic_start:] = penalty

    # matrix-free FISTA with backtracking line search on the smooth, strongly
    # convex (l2 > 0 on the genotype block) penalized logistic objective.
    def objective_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        clipped = np.clip(logits, -30, 30)
        value = float(
            np.mean(np.logaddexp(0.0, clipped) - target_array * clipped)
            + 0.5 * float(np.dot(l2, beta * beta))
        )
        probabilities = 1.0 / (1.0 + np.exp(-clipped))
        gradient = design.T @ (probabilities - target_array) / sample_count + l2 * beta
        return value, gradient

    def objective(beta: np.ndarray) -> float:
        clipped = np.clip(design @ beta, -30, 30)
        return float(
            np.mean(np.logaddexp(0.0, clipped) - target_array * clipped)
            + 0.5 * float(np.dot(l2, beta * beta))
        )

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    extrapolated = coefficients.copy()
    acceleration = 1.0
    lipschitz = 1.0
    # gradient-norm stopping scale from the covariate-only baseline gradient
    grad0_norm = np.linalg.norm(objective_and_gradient(coefficients)[1]) + 1e-30

    converged = False
    relative_gradient = float("inf")
    for _iteration in range(max_iterations):
        smooth_at_extrapolated, gradient = objective_and_gradient(extrapolated)
        trial = max(lipschitz * 0.5, 1e-8)
        while True:
            proposal = extrapolated - gradient / trial
            difference = proposal - extrapolated
            bound = (
                smooth_at_extrapolated
                + float(np.dot(gradient, difference))
                + 0.5 * trial * float(np.dot(difference, difference))
            )
            if objective(proposal) <= bound + 1e-12:
                break
            trial *= 2.0
            if not np.isfinite(trial):
                raise ValueError("runtime anchor backtracking diverged")
        lipschitz = trial
        # adaptive restart if the step is not a descent direction
        if float(np.dot(extrapolated - proposal, proposal - coefficients)) > 0.0:
            acceleration = 1.0
        next_acceleration = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration * acceleration))
        extrapolated = proposal + ((acceleration - 1.0) / next_acceleration) * (
            proposal - coefficients
        )
        coefficients = proposal
        acceleration = next_acceleration
        relative_gradient = float(
            np.linalg.norm(objective_and_gradient(coefficients)[1]) / grad0_norm
        )
        if relative_gradient <= tolerance:
            converged = True
            break

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("runtime anchor optimizer diverged")

    # de-standardize back onto raw covariate / raw dosage axes
    covariate_coefficients = coefficients[1:genetic_start] / covariate_scale
    genotype_weights = coefficients[genetic_start:] / genotype_scale
    intercept = float(
        coefficients[0]
        - np.dot(covariate_mean, covariate_coefficients)
    )
    model = _FixedRidgeModel(
        intercept=intercept,
        covariate_coefficients=np.asarray(covariate_coefficients, dtype=np.float64),
        genotype_weights=np.asarray(genotype_weights, dtype=np.float64),
        genotype_mean=np.asarray(genotype_mean, dtype=np.float64),
    )
    return model, {
        "converged": bool(converged),
        "penalty": float(penalty),
        "max_iterations": int(max_iterations),
        "iterations_run": int(_iteration + 1),
        "tolerance": float(tolerance),
        "final_relative_gradient": relative_gradient,
    }
