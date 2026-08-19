"""
Behavioral tests for varseq.variant.score_variant_effects on the seeded oracle model.

These assert CONTRACT + INVARIANTS + DETERMINISM only -- deliberately no frozen numeric
values (those live grader-side). They pin the improved gold state: the {effect_size, mean,
sd, pvalue, q-value} return contract, the FDR q-value, the 1e-8 degenerate-background guard,
the allele-length-homogeneous-batch contract, and the check_ref rename.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from varseq.variant import score_variant_effects, predict_variant_effects
from tests.vep_fixtures import (
    GENOME_PATH,
    N_SHUFFLES,
    SEED,
    SEQ_LEN,
    _COMP,
    _ref_match,
    build_oracle_model,
    variant_sets,
)

EXPECTED_KEYS = {"effect_size", "mean", "sd", "pvalue", "q-value"}
KINDS = ["snv", "ins", "del", "mnv"]


@pytest.fixture(scope="module")
def model():
    return build_oracle_model()


@pytest.fixture(scope="module")
def sets():
    return variant_sets()


def _run(model, variants, seed=SEED, **kw):
    return score_variant_effects(
        model=model,
        variants=variants.copy(),
        genome=GENOME_PATH,
        seq_len=SEQ_LEN,
        devices="cpu",
        n_shuffles=N_SHUFFLES,
        seed=seed,
        compare_func="log2FC",
        **kw,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_return_contract(model, sets, kind):
    v = sets[kind]
    out = _run(model, v)
    # exact key set -- no extras, no omissions
    assert set(out.keys()) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert len(out[key]) == len(v), f"{key} length != #variants"
        # JSON-serializable python floats, row-aligned to the input
        assert all(isinstance(x, float) for x in out[key]), f"{key} not list[float]"


@pytest.mark.parametrize("kind", KINDS)
def test_statistical_invariants(model, sets, kind):
    out = _run(model, sets[kind])
    sd = np.array(out["sd"])
    p = np.array(out["pvalue"])
    q = np.array(out["q-value"])
    # non-degenerate background on the RF-9 model (a k=1 model would give sd==0)
    assert np.all(sd > 1e-9), f"{kind}: degenerate background sd={sd}"
    # probabilities in range, finite
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    assert np.all(np.isfinite(q)) and np.all((q >= 0) & (q <= 1))
    # FDR never reports a q below its p
    assert np.all(q >= p - 1e-12), f"{kind}: q<p violates FDR"


@pytest.mark.parametrize("kind", KINDS)
def test_determinism(model, sets, kind):
    v = sets[kind]
    a = _run(model, v, seed=SEED)
    b = _run(model, v, seed=SEED)
    # same seed -> byte-identical output
    for key in EXPECTED_KEYS:
        assert a[key] == b[key], f"{kind}/{key} not reproducible at fixed seed"
    # different seed -> effect_size is unchanged (it does not depend on the background)
    c = _run(model, v, seed=SEED + 1)
    assert np.allclose(a["effect_size"], c["effect_size"], rtol=1e-9), (
        f"{kind}: effect_size must not depend on the background seed"
    )


def test_background_moves_with_seed(model, sets):
    # on a non-degenerate set, changing the seed reshuffles the background stats
    a = _run(model, sets["snv"], seed=SEED)
    c = _run(model, sets["snv"], seed=SEED + 1)
    assert not np.allclose(a["mean"], c["mean"], rtol=1e-6), "background did not move"


def test_single_variant_q_equals_p(model, sets):
    # BH on a single hypothesis leaves it unchanged: q == p
    v = sets["snv"].iloc[[0]].reset_index(drop=True)
    out = _run(model, v)
    assert len(out["pvalue"]) == 1
    assert np.isclose(out["q-value"][0], out["pvalue"][0], rtol=1e-9)


def test_degenerate_background_finite_p(model):
    # ref == alt -> the effect is identically 0 across every shuffle -> sd == 0.
    # The 1e-8 z-denominator guard must yield a finite p (~1.0), never nan/inf.
    pos = 60
    ref = _ref_match(pos, 1)
    v = pd.DataFrame([{"chrom": "chr1", "pos": pos, "ref": ref, "alt": ref}])
    out = _run(model, v)
    p = out["pvalue"][0]
    assert np.isfinite(p), "degenerate background produced a non-finite p"
    assert np.isfinite(out["q-value"][0])
    assert 0.0 <= p <= 1.0


def test_mixed_allele_lengths_raise(model, sets):
    # A single call whose variants do not share |ref|/|alt| must raise loudly
    # (strings_to_indices stacks alleles), not silently mis-align.
    mixed = pd.concat([sets["snv"], sets["ins"]], ignore_index=True)
    with pytest.raises(AssertionError):
        _run(model, mixed)


def test_check_ref_flag_runs(model, sets):
    # The renamed check_ref flag must exercise the reference-check path without the
    # old TypeError (the bool no longer shadows the check_reference function).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = predict_variant_effects(
            variants=sets["snv"].copy(),
            model=model,
            genome=GENOME_PATH,
            seq_len=SEQ_LEN,
            devices="cpu",
            compare_func="divide",
            return_ad=False,
            check_ref=True,
        )
    assert effects.shape[0] == len(sets["snv"])


def test_old_check_reference_kwarg_removed(model, sets):
    # Regression guard for the rename: the shadowing param name is gone.
    with pytest.raises(TypeError):
        predict_variant_effects(
            variants=sets["snv"].copy(),
            model=model,
            genome=GENOME_PATH,
            seq_len=SEQ_LEN,
            devices="cpu",
            return_ad=False,
            check_reference=True,
        )
