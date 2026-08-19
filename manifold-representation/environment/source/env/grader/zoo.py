"""Additive-manifold zoo used by manifold-bench.

This is a benchmark-focused adaptation of the local manifold-zoo generators in
``Manifold-SAE/experiments/amm_zoo`` and ``gam/bench/bsf_manifold_zoo.py``.
The important invariants are retained and tested here: analytic objects plus
compact neural-activation coordinate clouds, per-factor centering and RMS
calibration, isometric ambient frames, sparse additive superposition, and
separately seeded train/match/score splits.

Unlike those research scripts, this module deliberately exposes only ``x`` to a
submission.  Active masks, coordinates, frames, and additive contributions are
private grading truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .neural_bank import load_neural_bank
from .specs import SuiteSpec


CORE_KINDS: tuple[str, ...] = (
    "segment",
    "circle",
    "disk",
    "sphere",
    "torus",
    "mobius",
    "swiss",
    "helix",
)
EXTENDED_KINDS: tuple[str, ...] = CORE_KINDS + (
    "ellipse",
    "lemniscate",
    "lissajous",
    "annulus",
    "cylinder",
    "cone",
    "saddle",
    "paraboloid",
    "clifford_torus",
    "double_circle",
)
SPAN_DIM: dict[str, int] = {
    "segment": 1,
    "circle": 2,
    "disk": 2,
    "sphere": 3,
    "torus": 3,
    "mobius": 3,
    "swiss": 3,
    "helix": 3,
    "ellipse": 2,
    "lemniscate": 2,
    "lissajous": 2,
    "annulus": 2,
    "cylinder": 3,
    "cone": 3,
    "saddle": 3,
    "paraboloid": 3,
    "clifford_torus": 4,
    "double_circle": 3,
}
INTRINSIC_DIM: dict[str, int] = {
    "segment": 1,
    "circle": 1,
    "disk": 2,
    "sphere": 2,
    "torus": 2,
    "mobius": 2,
    "swiss": 2,
    "helix": 1,
    "ellipse": 1,
    "lemniscate": 1,
    "lissajous": 1,
    "annulus": 2,
    "cylinder": 2,
    "cone": 2,
    "saddle": 2,
    "paraboloid": 2,
    "clifford_torus": 2,
    "double_circle": 1,
}
NEURAL_PREFIX = "neural_"


@dataclass(frozen=True)
class Factor:
    fid: int
    kind: str
    intrinsic_dim: int
    base_span_dim: int
    span_dim: int
    parameters: dict[str, float]
    lift_weights: np.ndarray
    lift_phases: np.ndarray
    lift_scale: float
    center: np.ndarray
    rms: float
    frame: np.ndarray


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    active: np.ndarray
    contributions: np.ndarray
    coordinates: np.ndarray
    pair_id: np.ndarray
    changed_factor: np.ndarray


@dataclass(frozen=True)
class Dataset:
    spec: SuiteSpec
    factors: tuple[Factor, ...]
    train: Split
    match: Split
    score: Split


def _raw_coordinates(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.full((n, 2), np.nan, dtype=np.float64)
    if kind == "segment":
        values[:, 0] = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    elif kind in {"circle", "ellipse", "lemniscate", "lissajous", "double_circle"}:
        values[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
        if kind == "double_circle":
            values[:, 1] = rng.integers(0, 2, n)
    elif kind in {"disk", "annulus"}:
        lower = 0.35 if kind == "annulus" else 0.0
        values[:, 0] = np.sqrt(rng.uniform(lower * lower, 1.0, n))
        values[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n)
    elif kind == "sphere":
        values[:, 0] = np.arccos(rng.uniform(-1.0, 1.0, n))
        values[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n)
    elif kind in {"torus", "mobius", "clifford_torus"}:
        values[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
        values[:, 1] = (
            rng.uniform(-1.0, 1.0, n) if kind == "mobius" else rng.uniform(0.0, 2.0 * np.pi, n)
        )
    elif kind == "swiss":
        values[:, 0] = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
        values[:, 1] = rng.uniform(-1.0, 1.0, n)
    elif kind == "helix":
        values[:, 0] = rng.uniform(0.0, 4.0 * np.pi, n)
    elif kind in {"cylinder", "cone"}:
        values[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
        values[:, 1] = rng.uniform(-1.0, 1.0, n) if kind == "cylinder" else rng.uniform(0.2, 1.0, n)
    elif kind in {"saddle", "paraboloid"}:
        values[:, 0] = rng.uniform(-1.0, 1.0, n)
        values[:, 1] = rng.uniform(-1.0, 1.0, n)
    else:
        raise ValueError(f"unknown manifold kind: {kind}")
    return values


def _neural_index(kind: str) -> int | None:
    if not kind.startswith(NEURAL_PREFIX):
        return None
    suffix = kind.removeprefix(NEURAL_PREFIX)
    if not suffix.isdigit():
        raise ValueError(f"invalid neural factor kind: {kind}")
    return int(suffix)


def _base_span_dim(kind: str) -> int:
    index = _neural_index(kind)
    return load_neural_bank().chart_dim if index is not None else SPAN_DIM[kind]


def _intrinsic_dim(kind: str) -> int:
    index = _neural_index(kind)
    return load_neural_bank().chart_dim if index is not None else INTRINSIC_DIM[kind]


def _apply_lift(
    base: np.ndarray,
    weights: np.ndarray,
    phases: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Injectively bend a local factor into a higher-dimensional ambient span.

    Retaining the base coordinates makes the map injective.  The sinusoidal
    coordinates ensure that recovering a factor requires its nonlinear image,
    not merely the flat span containing it.
    """
    values = np.asarray(base, dtype=np.float64)
    if weights.shape[1] == 0:
        return values
    nonlinear = np.sin(values @ weights + phases)
    return np.concatenate([values, scale * nonlinear], axis=1)


def _lift_embedding(base: np.ndarray, factor: Factor) -> np.ndarray:
    return _apply_lift(
        base,
        factor.lift_weights,
        factor.lift_phases,
        factor.lift_scale,
    )


def _raw_embedding(kind: str, coordinates: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    u = coordinates[:, 0]
    if kind == "segment":
        embedded = u[:, None]
    elif kind == "circle":
        embedded = np.stack([np.cos(u), np.sin(u)], axis=1)
    elif kind == "ellipse":
        aspect = parameters["aspect"]
        embedded = np.stack([aspect * np.cos(u), np.sin(u) / aspect], axis=1)
    elif kind == "lemniscate":
        embedded = np.stack([np.cos(u), np.sin(u) * np.cos(u)], axis=1)
    elif kind == "lissajous":
        embedded = np.stack(
            [
                np.sin(parameters["frequency_a"] * u + parameters["phase"]),
                np.sin(parameters["frequency_b"] * u),
            ],
            axis=1,
        )
    elif kind == "double_circle":
        component = 2.0 * coordinates[:, 1] - 1.0
        embedded = np.stack([np.cos(u), np.sin(u), parameters["separation"] * component], axis=1)
    elif kind in {"disk", "annulus"}:
        r, theta = u, coordinates[:, 1]
        embedded = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    elif kind == "sphere":
        phi, theta = u, coordinates[:, 1]
        embedded = np.stack(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
            axis=1,
        )
    elif kind == "torus":
        theta, phi = u, coordinates[:, 1]
        major = parameters["major"]
        minor = parameters["minor"]
        radial = major + minor * np.cos(phi)
        embedded = np.stack(
            [radial * np.cos(theta), radial * np.sin(theta), minor * np.sin(phi)], axis=1
        )
    elif kind == "clifford_torus":
        theta, phi = u, coordinates[:, 1]
        embedded = np.stack(
            [np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)], axis=1
        ) / np.sqrt(2.0)
    elif kind == "mobius":
        theta, width = u, coordinates[:, 1] * parameters["width"]
        radial = 1.0 + width * np.cos(theta / 2.0)
        embedded = np.stack(
            [radial * np.cos(theta), radial * np.sin(theta), width * np.sin(theta / 2.0)],
            axis=1,
        )
    elif kind == "swiss":
        theta, height = u, coordinates[:, 1] * parameters["height"]
        embedded = np.stack([theta * np.cos(theta), height, theta * np.sin(theta)], axis=1)
    elif kind == "helix":
        pitch = parameters["pitch"]
        embedded = np.stack([np.cos(u), np.sin(u), pitch * (u - 2.0 * np.pi)], axis=1)
    elif kind == "cylinder":
        embedded = np.stack(
            [np.cos(u), np.sin(u), parameters["height"] * coordinates[:, 1]], axis=1
        )
    elif kind == "cone":
        radius = coordinates[:, 1]
        embedded = np.stack(
            [radius * np.cos(u), radius * np.sin(u), parameters["slope"] * radius], axis=1
        )
    elif kind == "saddle":
        v = coordinates[:, 1]
        embedded = np.stack([u, v, parameters["curvature"] * u * v], axis=1)
    elif kind == "paraboloid":
        v = coordinates[:, 1]
        embedded = np.stack([u, v, parameters["curvature"] * (u * u + v * v)], axis=1)
    else:
        raise ValueError(f"unknown manifold kind: {kind}")
    if embedded.shape[1] != SPAN_DIM[kind] or not np.isfinite(embedded).all():
        raise RuntimeError(f"invalid {kind} embedding")
    return embedded


def _parameters(kind: str, rng: np.random.Generator) -> dict[str, float]:
    if kind == "torus":
        return {"major": float(rng.uniform(1.4, 2.4)), "minor": float(rng.uniform(0.35, 0.8))}
    if kind == "mobius":
        return {"width": float(rng.uniform(0.25, 0.65))}
    if kind == "swiss":
        return {"height": float(rng.uniform(1.0, 3.0))}
    if kind == "helix":
        return {"pitch": float(rng.uniform(0.08, 0.28))}
    if kind == "ellipse":
        return {"aspect": float(rng.uniform(1.3, 2.4))}
    if kind == "lissajous":
        return {
            "frequency_a": float(rng.choice(np.array([2, 3, 4]))),
            "frequency_b": float(rng.choice(np.array([3, 5]))),
            "phase": float(rng.uniform(0.15, 1.0)),
        }
    if kind == "double_circle":
        return {"separation": float(rng.uniform(0.45, 1.1))}
    if kind == "cylinder":
        return {"height": float(rng.uniform(0.5, 1.5))}
    if kind == "cone":
        return {"slope": float(rng.uniform(0.6, 1.4))}
    if kind in {"saddle", "paraboloid"}:
        return {"curvature": float(rng.uniform(0.6, 1.8))}
    return {}


def _factor_kinds(spec: SuiteSpec, rng: np.random.Generator) -> list[str]:
    bank = load_neural_bank()
    families = spec.empirical_families
    if (
        len(set(families)) != len(families)
        or len(families) > spec.n_factors
        or any(not 0 <= family < bank.n_families for family in families)
    ):
        raise ValueError(f"invalid empirical families: {families}")
    analytic_factors = spec.n_factors - len(families)
    n_curved = round(spec.curved_fraction * analytic_factors)
    source = CORE_KINDS if spec.zoo_profile == "core" else EXTENDED_KINDS
    if spec.zoo_profile not in {"core", "extended"}:
        raise ValueError(f"unknown zoo profile: {spec.zoo_profile}")
    curved = [kind for kind in source if kind != "segment"]
    kinds = ["segment"] * (analytic_factors - n_curved)
    kinds.extend(curved[index % len(curved)] for index in range(n_curved))
    kinds.extend(f"{NEURAL_PREFIX}{index}" for index in families)
    rng.shuffle(kinds)
    return kinds


def _make_frames(
    spec: SuiteSpec, span_dims: list[int], rng: np.random.Generator
) -> list[np.ndarray]:
    max_span = max(span_dims)
    if max_span > spec.ambient_dim:
        raise ValueError("factor span exceeds ambient dimension")
    shared_raw = rng.standard_normal((spec.ambient_dim, max_span))
    shared, _ = np.linalg.qr(shared_raw)
    frames: list[np.ndarray] = []
    for span in span_dims:
        raw = rng.standard_normal((spec.ambient_dim, span))
        if spec.coherence > 0.0:
            # Both terms must be on the same scale for the mixture weight to mean
            # anything. A raw Gaussian column has expected norm sqrt(ambient_dim)
            # -- about 6.6 at d=44 -- while a column of `shared` is orthonormal
            # with norm 1. Mixing them directly let the independent term dominate
            # by ~6.6x, so `coherence=0.58` moved the mean squared canonical
            # overlap by 0.003 (0.0666 -> 0.0695) against a random-frame baseline:
            # the suite named coherent_subspaces was generating essentially random
            # subspaces, and only c>=0.99 did anything. Orthonormalize first, then
            # mix, then re-orthonormalize.
            independent, _ = np.linalg.qr(raw)
            raw = (
                np.sqrt(1.0 - spec.coherence) * independent[:, :span]
                + np.sqrt(spec.coherence) * shared[:, :span]
            )
        q, _ = np.linalg.qr(raw)
        frame = q[:, :span].T
        if not np.allclose(frame @ frame.T, np.eye(span), atol=1e-10):
            raise RuntimeError("ambient factor frame is not isometric")
        frames.append(frame)
    return frames


def make_factors(spec: SuiteSpec) -> tuple[Factor, ...]:
    rng = np.random.default_rng(spec.seed)
    kinds = _factor_kinds(spec, rng)
    base_dims = [_base_span_dim(kind) for kind in kinds]
    lift_dims = [base + spec.lift_extra_dims for base in base_dims]
    lift_parameters: list[tuple[np.ndarray, np.ndarray, float]] = []
    for base in base_dims:
        weights = rng.standard_normal((base, spec.lift_extra_dims)) / np.sqrt(base)
        phases = rng.uniform(-np.pi, np.pi, spec.lift_extra_dims)
        lift_parameters.append((weights, phases, 0.55))
    frames = _make_frames(spec, lift_dims, rng)
    bank = load_neural_bank()
    factors: list[Factor] = []
    for fid, (kind, base_dim, frame, lift) in enumerate(
        zip(kinds, base_dims, frames, lift_parameters, strict=True)
    ):
        lift_weights, lift_phases, lift_scale = lift
        neural_index = _neural_index(kind)
        if neural_index is None:
            parameters = _parameters(kind, rng)
            calibration_rng = np.random.default_rng(spec.seed * 10_000 + fid)
            coords = _raw_coordinates(kind, 4_096, calibration_rng)
            base = _raw_embedding(kind, coords, parameters)
        else:
            parameters = {"bank_index": float(neural_index)}
            base = np.asarray(bank.points(neural_index, "train"), dtype=np.float64)
        raw = _apply_lift(base, lift_weights, lift_phases, lift_scale)
        center = raw.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((raw - center) ** 2, axis=1))))
        if not np.isfinite(rms) or rms <= 1e-8:
            raise RuntimeError(f"degenerate calibration for factor {fid}")
        factors.append(
            Factor(
                fid=fid,
                kind=kind,
                intrinsic_dim=_intrinsic_dim(kind),
                base_span_dim=base_dim,
                span_dim=raw.shape[1],
                parameters=parameters,
                lift_weights=lift_weights,
                lift_phases=lift_phases,
                lift_scale=lift_scale,
                center=center,
                rms=rms,
                frame=frame,
            )
        )
    return tuple(factors)


def _sampling_weights(spec: SuiteSpec, exponent: float) -> np.ndarray:
    if exponent <= 0.0:
        return np.full(spec.n_factors, 1.0 / spec.n_factors)
    ranks = np.arange(1, spec.n_factors + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    return weights / weights.sum()


def _active_mask(
    spec: SuiteSpec,
    n: int,
    *,
    evaluation: bool,
    exponent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = (
        spec.test_support_mean
        if evaluation and spec.test_support_mean is not None
        else spec.support_mean
    )
    maximum = (
        spec.test_support_max
        if evaluation and spec.test_support_max is not None
        else spec.support_max
    )
    if not 0.0 < mean < spec.n_factors or not 2 <= maximum < spec.n_factors:
        raise ValueError("invalid support distribution")

    # Ordinary rows have a broad, unknown support count.  Explicit null and
    # singleton rows make the noise floor and every factor individually
    # observable, so exact labels are identifiable even under correlated
    # co-occurrence.  Their positions are shuffled and never exposed.
    counts = np.clip(rng.poisson(mean, size=n), 2, maximum)
    null_count = min(n, round(spec.null_fraction * n))
    anchor_count = min(n - null_count, max(spec.n_factors, round(spec.anchor_fraction * n)))
    counts[:null_count] = 0
    counts[null_count : null_count + anchor_count] = 1

    weights = _sampling_weights(spec, exponent)
    clusters = np.arange(spec.n_factors) // max(3, round(np.sqrt(spec.n_factors)))
    result = np.zeros((n, spec.n_factors), dtype=bool)
    anchor_rows = np.flatnonzero(counts == 1)
    anchor_factors = np.resize(rng.permutation(spec.n_factors), len(anchor_rows))
    result[anchor_rows, anchor_factors] = True
    for row in np.flatnonzero(counts >= 2):
        chosen: list[int] = []
        first = int(rng.choice(spec.n_factors, p=weights))
        chosen.append(first)
        while len(chosen) < counts[row]:
            candidates = np.array([index for index in range(spec.n_factors) if index not in chosen])
            local = candidates[clusters[candidates] == clusters[first]]
            if local.size and rng.random() < spec.correlation:
                local_weights = weights[local]
                pick = int(rng.choice(local, p=local_weights / local_weights.sum()))
            else:
                candidate_weights = weights[candidates]
                pick = int(rng.choice(candidates, p=candidate_weights / candidate_weights.sum()))
            chosen.append(pick)
        result[row, chosen] = True
    order = rng.permutation(n)
    return result[order]


def sample_split(
    spec: SuiteSpec,
    factors: tuple[Factor, ...],
    n: int,
    seed: int,
    *,
    evaluation: bool,
    split_name: str,
) -> Split:
    rng = np.random.default_rng(seed)
    exponent = (
        spec.test_zipf_exponent
        if evaluation and spec.test_zipf_exponent is not None
        else spec.zipf_exponent
    )
    active = _active_mask(spec, n, evaluation=evaluation, exponent=exponent, rng=rng)
    contributions = np.zeros((n, spec.n_factors, spec.ambient_dim), dtype=np.float64)
    latent_contributions = np.zeros_like(contributions)
    bank = load_neural_bank()
    coordinate_dim = max(2, bank.chart_dim)
    latent_coordinates = np.full((n, spec.n_factors, coordinate_dim), np.nan, dtype=np.float64)
    for factor in factors:
        neural_index = _neural_index(factor.kind)
        if neural_index is None:
            coords = _raw_coordinates(factor.kind, n, rng)
            base = _raw_embedding(factor.kind, coords, factor.parameters)
        else:
            coords = bank.sample(neural_index, split_name, n, rng)
            base = coords
        raw = _lift_embedding(base, factor)
        local = (raw - factor.center) / factor.rms
        amplitude = np.ones(n, dtype=np.float64)
        if spec.amplitude_sigma > 0.0:
            sigma = spec.amplitude_sigma
            amplitude *= np.exp(sigma * rng.standard_normal(n) - 0.5 * sigma * sigma)
        if spec.signed_amplitude:
            amplitude *= rng.choice(np.array([-1.0, 1.0]), size=n)
        # An active object must have an observable causal effect. Several
        # centered manifolds pass arbitrarily close to the origin, where an
        # active label would otherwise be information-theoretically
        # indistinguishable from absence under observation noise. Preserve the
        # sampled position and sign, but censor only the vanishing lower tail
        # of its ambient contribution norm. Upper amplitude tails are untouched.
        local_norm = np.linalg.norm(local, axis=1)
        effect_norm = np.abs(amplitude) * local_norm
        adjustment = np.maximum(
            1.0,
            spec.min_active_norm / np.maximum(effect_norm, 1e-12),
        )
        amplitude *= adjustment
        ambient = (local @ factor.frame) * amplitude[:, None]
        latent_contributions[:, factor.fid, :] = ambient
        contributions[active[:, factor.fid], factor.fid, :] = ambient[active[:, factor.fid]]
        latent_coordinates[:, factor.fid, : coords.shape[1]] = coords

    pair_id = np.full(n, -1, dtype=np.int32)
    changed_factor = np.full(n, -1, dtype=np.int32)
    if evaluation and spec.counterfactual_fraction > 0.0:
        eligible = np.flatnonzero(active.any(axis=1))
        n_pairs = min(len(eligible) // 2, round(spec.counterfactual_fraction * n / 2.0))
        selected = rng.choice(eligible, size=2 * n_pairs, replace=False)
        for identifier, (left, right) in enumerate(selected.reshape(-1, 2)):
            factor_id = int(rng.choice(np.flatnonzero(active[left])))
            replacement_coordinate = latent_coordinates[right, factor_id].copy()
            active[right] = active[left]
            contributions[right] = contributions[left]
            contributions[right, factor_id] = latent_contributions[right, factor_id]
            latent_coordinates[right] = latent_coordinates[left]
            # Keep the independently sampled coordinate for the changed factor.
            latent_coordinates[right, factor_id] = replacement_coordinate
            pair_id[[left, right]] = identifier
            changed_factor[[left, right]] = factor_id

    coordinates = np.where(active[:, :, None], latent_coordinates, np.nan)
    signal = contributions.sum(axis=1)
    noise = spec.noise_std * rng.standard_normal(signal.shape)
    x = signal + noise
    if not np.isfinite(x).all():
        raise RuntimeError("zoo generated non-finite observations")
    order = rng.permutation(n)
    return Split(
        x=x[order].astype(np.float32),
        active=active[order],
        contributions=contributions[order].astype(np.float32),
        coordinates=coordinates[order].astype(np.float32),
        pair_id=pair_id[order],
        changed_factor=changed_factor[order],
    )


def build_dataset(spec: SuiteSpec) -> Dataset:
    factors = make_factors(spec)
    return Dataset(
        spec=spec,
        factors=factors,
        train=sample_split(
            spec, factors, spec.n_train, spec.seed + 11, evaluation=False, split_name="train"
        ),
        match=sample_split(
            spec, factors, spec.n_match, spec.seed + 23, evaluation=True, split_name="match"
        ),
        score=sample_split(
            spec, factors, spec.n_score, spec.seed + 37, evaluation=True, split_name="score"
        ),
    )


def public_config(spec: SuiteSpec) -> dict[str, Any]:
    """Information legitimately available to a solver at fit and transform time."""
    return {
        "schema_version": "1.0",
        "ambient_dim": spec.ambient_dim,
        "max_features": spec.max_features,
        "train_rows": spec.n_train,
        "predictions_file": "predictions.npz",
    }
