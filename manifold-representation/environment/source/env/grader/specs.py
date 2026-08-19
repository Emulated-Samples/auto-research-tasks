"""Versioned production and calibration suite definitions.

Production suites are never used to choose reference hyperparameters.  Each has
one structurally matched calibration counterpart with a disjoint seed and, for
the neural suite, disjoint source families.  The public task describes the
behavioral problem while these private declarations vary geometry, support,
noise, prevalence, coherence, and scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


SCORING_VERSION = "manifold-bench-v14"


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    category: str
    seed: int
    ambient_dim: int
    n_factors: int
    max_features: int
    support_mean: float
    support_max: int
    n_train: int
    n_match: int
    n_score: int
    curved_fraction: float
    noise_std: float = 0.0
    correlation: float = 0.0
    coherence: float = 0.0
    zipf_exponent: float = 0.0
    test_zipf_exponent: float | None = None
    amplitude_sigma: float = 0.0
    signed_amplitude: bool = False
    test_support_mean: float | None = None
    test_support_max: int | None = None
    null_fraction: float = 0.04
    anchor_fraction: float = 0.14
    counterfactual_fraction: float = 0.18
    min_active_norm: float = 0.25
    lift_extra_dims: int = 1
    fit_timeout_seconds: int = 75
    transform_timeout_seconds: int = 15
    weight: float = 1.0
    zoo_profile: str = "core"
    empirical_families: tuple[int, ...] = ()

    @property
    def full_credit_seconds(self) -> float:
        """Generous public runtime region; efficiency remains a minor score."""
        return 0.85 * (self.fit_timeout_seconds + self.transform_timeout_seconds)


_BASE = SuiteSpec(
    name="core_balanced",
    category="decomposition",
    # v13 production holdout. The reference -- and with it every witnessed
    # target and floor gap -- was replaced after v12's production outcomes were
    # observed (four Opus cohorts ran against the v12 seeds and their traces
    # informed the new reference's design), so per the same policy that retired
    # v11's seeds, the v12 production draws are retired rather than rescored
    # under a ruler they helped select. Calibration seeds are unchanged: they
    # are the declared tuning set.
    seed=9_531_101,
    ambient_dim=36,
    n_factors=12,
    max_features=22,
    support_mean=3.0,
    support_max=6,
    n_train=1_900,
    n_match=420,
    n_score=720,
    curved_fraction=0.67,
    noise_std=0.01,
)


SUITES: dict[str, SuiteSpec] = {
    _BASE.name: _BASE,
    "neural_hybrid": replace(
        _BASE,
        name="neural_hybrid",
        category="geometry",
        seed=9_532_101,
        ambient_dim=48,
        n_factors=16,
        max_features=27,
        support_mean=3.8,
        support_max=7,
        n_train=2_300,
        n_match=480,
        n_score=820,
        curved_fraction=0.75,
        noise_std=0.02,
        # Entire activation families are held out from reference-development
        # calibration, not merely different rows from the same families.
        empirical_families=(2, 3, 6, 7),
        lift_extra_dims=2,
    ),
    "extended_topologies": replace(
        _BASE,
        name="extended_topologies",
        category="geometry",
        seed=9_532_551,
        ambient_dim=64,
        n_factors=24,
        max_features=35,
        support_mean=4.2,
        support_max=8,
        n_train=2_900,
        n_match=620,
        n_score=980,
        curved_fraction=0.88,
        noise_std=0.025,
        amplitude_sigma=0.20,
        weight=1.25,
        zoo_profile="extended",
        lift_extra_dims=2,
    ),
    "correlated_supports": replace(
        _BASE,
        name="correlated_supports",
        category="identifiability",
        seed=9_533_101,
        ambient_dim=48,
        n_factors=16,
        max_features=29,
        support_mean=3.7,
        support_max=7,
        n_train=2_500,
        n_match=520,
        n_score=880,
        curved_fraction=0.65,
        correlation=0.78,
        anchor_fraction=0.20,
        noise_std=0.02,
        weight=1.2,
    ),
    "coherent_subspaces": replace(
        _BASE,
        name="coherent_subspaces",
        category="identifiability",
        seed=9_534_101,
        ambient_dim=44,
        n_factors=15,
        max_features=24,
        support_mean=3.2,
        support_max=6,
        n_train=2_500,
        n_match=520,
        n_score=880,
        curved_fraction=0.70,
        # The regime's real coherence, restored. Until _make_frames was fixed
        # this 0.58 was decorative -- it mixed a norm-1 orthonormal column with a
        # norm-6.6 Gaussian one, so the realized mean squared canonical overlap
        # was 0.0695 against a 0.0666 random baseline and the suite was a
        # disguised duplicate of core_balanced. Fixing the mixture made 0.58 real
        # (overlap 0.400) and the rank-6 reference collapsed on it: the worst of
        # twelve draws landed 0.0425 above the non-solution floor, under the 0.05
        # the discrimination guard requires. It was softened to 0.45 on the
        # explicit record that "the binding constraint is the reference's skill,
        # not the regime. A stronger reference would let this go back to 0.58."
        #
        # That prediction was tested and held. With `_block_rank` at 3, at the
        # full 12 replicates and re-measuring the arms rather than reusing a
        # floor (tools/probe_hard_regimes.py):
        #
        #   c=0.58: all SIX metrics discriminate; recovery gap 0.4058 against a
        #   0.05 bar (8x), reference recovery 0.499 vs floor 0.002.
        #
        # So the suite is back at its real setting, and the earlier softening was
        # an artifact of the anchor rather than a property of the regime -- which
        # is exactly what a benchmark should never let a weak reference decide.
        coherence=0.58,
        anchor_fraction=0.18,
        noise_std=0.025,
        weight=1.2,
    ),
    "long_tail": replace(
        _BASE,
        name="long_tail",
        category="generalization",
        seed=9_535_101,
        ambient_dim=48,
        n_factors=20,
        max_features=32,
        support_mean=4.0,
        support_max=7,
        n_train=2_700,
        n_match=570,
        n_score=920,
        curved_fraction=0.70,
        # The regime's real skew, restored. The shift under test is train-skewed
        # -> test-uniform prevalence, and 1.25 is the setting the suite was
        # designed around.
        #
        # It was softened twice, and BOTH softenings were the rank-6 anchor
        # failing rather than the regime being unmeasurable. That reading was
        # recorded at the time ("the binding constraint is the reference's skill
        # rather than the regime's realism") and the numbers were real -- measured
        # at 12 replicates, watching the floor-to-target GAP rather than the
        # anchor's score:
        #
        #   zipf   ref_aap  arm_aap    gap   scored metrics   [rank-6 anchor]
        #   0.90     0.395    0.311  0.057   4/6
        #   1.05     0.423    0.340  0.072   4/6
        #   1.25     0.388    0.372  0.016   2/6   <- four metrics blind
        #
        # At 1.25 the rank-6 reference landed 0.016 above a non-solution and four
        # of six metrics went blind: the conclusion drawn was that "at 1.25 rare
        # factors are close to absent in training and nothing recovers them".
        # The first clause was true of that anchor. The second was not true of the
        # regime. With `_block_rank` at 3, same test, same 12 replicates,
        # re-measuring the arms (tools/probe_hard_regimes.py):
        #
        #   zipf 1.25: SIX of six metrics discriminate; recovery gap 0.4709
        #   against a 0.05 bar (9.4x), reference recovery 0.680 vs floor 0.081.
        #
        # The one metric that stays marginal is assignment_coherence at 0.058 --
        # live, but only just, and it is the number to watch if this regime is
        # pushed further.
        zipf_exponent=1.25,
        test_zipf_exponent=0.35,
        noise_std=0.025,
        weight=1.2,
    ),
    "noisy_amplitudes": replace(
        _BASE,
        name="noisy_amplitudes",
        category="robustness",
        seed=9_536_101,
        ambient_dim=56,
        n_factors=18,
        max_features=31,
        support_mean=3.8,
        support_max=7,
        n_train=2_800,
        n_match=570,
        n_score=920,
        curved_fraction=0.72,
        noise_std=0.11,
        amplitude_sigma=0.55,
        signed_amplitude=True,
        weight=1.25,
    ),
    "support_shift": replace(
        _BASE,
        name="support_shift",
        category="generalization",
        seed=9_537_101,
        ambient_dim=50,
        n_factors=18,
        max_features=28,
        support_mean=2.4,
        support_max=5,
        test_support_mean=4.8,
        test_support_max=8,
        n_train=2_700,
        n_match=570,
        n_score=920,
        curved_fraction=0.70,
        noise_std=0.035,
        amplitude_sigma=0.30,
        weight=1.25,
    ),
    "scaled_mixture": replace(
        _BASE,
        name="scaled_mixture",
        category="scale",
        seed=9_538_101,
        ambient_dim=72,
        n_factors=24,
        max_features=39,
        support_mean=4.8,
        support_max=9,
        n_train=3_500,
        n_match=680,
        n_score=1_020,
        curved_fraction=0.72,
        noise_std=0.04,
        amplitude_sigma=0.25,
        fit_timeout_seconds=100,
        transform_timeout_seconds=20,
        weight=1.35,
        lift_extra_dims=2,
    ),
}


# Reference-development counterparts.  These are the only suites on which
# reference choices may be iterated.  Production seeds above remain an audit,
# and the neural counterpart uses disjoint LLM/vision families.
CALIBRATION_SEEDS = {
    "core_balanced": 431_101,
    "neural_hybrid": 432_101,
    "extended_topologies": 432_551,
    "correlated_supports": 433_101,
    "coherent_subspaces": 434_101,
    "long_tail": 435_101,
    "noisy_amplitudes": 436_101,
    "support_shift": 437_101,
    "scaled_mixture": 438_101,
}

CALIBRATION_SUITES: dict[str, SuiteSpec] = {
    name: replace(
        spec,
        seed=CALIBRATION_SEEDS[name],
        empirical_families=(0, 1, 4, 5) if name == "neural_hybrid" else spec.empirical_families,
    )
    for name, spec in SUITES.items()
}

CALIBRATION_REPLICATES = 24


def calibration_specs(name: str) -> tuple[SuiteSpec, ...]:
    """Independent reference-development draws for a production regime.

    The count is not cosmetic. Targets are a dispersion-aware lower envelope of
    these witnesses, and with only three draws neither the minimum nor the
    standard deviation is worth anything, which is why v10 needed a single blunt
    0.65 multiplicative margin to stay safe. Twelve sufficed while the
    reference's own optimizer noise dominated; v13's deterministic reference
    left the data draw as the only noise, and a 36-draw probe of long_tail
    found a merge-collapse left tail (~8% of draws) that twelve draws had
    never sampled -- the v13 sealed audit failed on exactly that under-sampled
    tail. Twenty-four halves the chance a production draw lands below the
    witnessed envelope, and at ~2 seconds per deterministic fit the doubling
    is nearly free.
    """
    base = CALIBRATION_SUITES[name]
    return tuple(
        replace(base, seed=base.seed + 90_001 * replicate)
        for replicate in range(CALIBRATION_REPLICATES)
    )


PROBLEM_SUITES: dict[str, tuple[str, ...]] = {
    "M6_full": tuple(SUITES),
}


def suites_for_problem(problem_id: str) -> tuple[SuiteSpec, ...]:
    names = PROBLEM_SUITES.get(problem_id)
    if names is None:
        raise ValueError(f"unknown manifold-bench problem: {problem_id}")
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate suite in problem {problem_id}")
    return tuple(SUITES[name] for name in names)
