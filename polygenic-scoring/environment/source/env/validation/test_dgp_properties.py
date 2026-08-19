"""Property tests for the DGP mechanisms each category NAMES.

Every private category promises a statistical regime that these tests verify.
Before the 2026-07-13 fix the generator applied ONE heavy-tailed TPB slab (plus an
optional spike-at-zero) to everything, so several of those promises were false --
`dense_infinitesimal` had excess kurtosis ~29 and its top 1% of variants carried 41%
of the squared effect mass, and `suppressor_ld`'s pair geometry did not exist at
all (its marginal/joint sign-flip rate was INDISTINGUISHABLE from a positive-LD
control). These tests MEASURE the promised property with predeclared bounds, so a
regression to "one slab for everything" fails CI.

The bounds are deliberately loose relative to the measured values (recorded in each
test) -- they catch a missing mechanism, not seed noise. Small N/P replicas keep the
file to a few seconds; the shipped corpus uses the same code path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datagen.categories import CATEGORIES
from datagen.dgp import DGPConfig, generate
from grader.contract import SHIPPED_FRAC_TRAIN

N, P = 900, 500
SEEDS = (1, 2, 3)


def _gen(category, seed, **override):
    cfg = CATEGORIES[category]["make_cfg"](seed, N, P)
    for key, value in override.items():
        setattr(cfg, key, value)
    return generate(cfg)


def _gen_dense_gaussian(seed):
    # Exercise the ACTUAL shipped recipe, not a hand-maintained mirror that can
    # stay green while CATEGORIES drifts.
    return _gen("dense_infinitesimal", seed)


def _excess_kurtosis(beta):
    nz = beta[beta != 0.0]
    z = (nz - nz.mean()) / nz.std()
    return float(np.mean(z ** 4) - 3.0)


def _top_squared_mass(beta, frac=0.01):
    b2 = np.sort(beta ** 2)[::-1]
    k = max(1, int(round(frac * len(b2))))
    return float(b2[:k].sum() / b2.sum())


def _marginal_cov(dataset):
    """Cov(G_j, genetic liability) -- what a marginal (P+T) screen sees."""
    Gstd = dataset["Gstd"]
    beta = dataset["truth"]["beta_effective"]
    return Gstd.T @ (Gstd @ beta) / Gstd.shape[0]


# ---- DGP-001: the dense_gaussian family is actually dense + near-Gaussian ----
@pytest.mark.parametrize("seed", SEEDS)
def test_dense_gaussian_is_light_tailed_and_has_no_spikes(seed):
    beta = _gen_dense_gaussian(seed)["truth"]["beta_effective"]

    assert np.all(beta != 0.0), "a dense architecture has no null set"
    # measured after the fix: |excess kurtosis| ~ 0.11 (was 29 / 19.7 on the slab)
    assert abs(_excess_kurtosis(beta)) < 1.0
    # measured after the fix: top 1% carry ~0.09 of squared mass (was 0.41 / 0.39)
    assert _top_squared_mass(beta) < 0.15
    # effect scale must be near-homoskedastic: no annotation/class-driven spikes
    scale_ratio = np.exp(np.ptp(np.log(np.sqrt(
        _gen_dense_gaussian(seed)["truth"]["tau2_effective"]))))
    assert scale_ratio < 3.0, "dense arm must not re-introduce a wide prior-scale spread"


def test_dense_gaussian_truth_contains_only_the_scale_it_used():
    dataset = _gen_dense_gaussian(5)
    truth = dataset["truth"]
    cfg = dataset["cfg"]
    np.testing.assert_allclose(
        truth["tau2_raw"], (truth["global_scale_raw"] * truth["s"]) ** 2
    )
    assert len(set(truth["class_log_baseline"].values())) == 1
    assert truth["gamma_len"] == pytest.approx(
        cfg.dense_scale_spread * cfg.length_coef_mean
    )
    assert truth["gamma_repeat"] == pytest.approx(
        cfg.dense_scale_spread * cfg.repeat_coef_mean
    )
    for unused in ("lam", "delta", "shape_a", "shape_b", "tail_temper"):
        assert unused not in truth


def test_dense_gaussian_archive_omits_unused_tpb_latents(tmp_path):
    from datagen.materialize import materialize

    output = materialize(
        _gen_dense_gaussian(5),
        tmp_path / "dense",
        family="binomial-logit",
        frac_train=SHIPPED_FRAC_TRAIN,
        prior_cols=("variant_class", "sv_log_length", "repeat_overlap"),
        seed=5,
    )
    with np.load(Path(output["truth"]) / "truth.npz") as archive:
        for unused in ("lam", "shape_a", "shape_b"):
            assert unused not in archive.files


# ---- DGP-002: sparse_dense_mix is a controlled two-component mixture --------
@pytest.mark.parametrize("seed", SEEDS)
def test_sparse_dense_mix_has_both_components_with_controlled_variance_shares(seed):
    dataset = _gen("sparse_dense_mix", seed)
    truth = dataset["truth"]
    dense = truth["beta_dense_effective"]
    spike = truth["beta_spike_effective"]
    Gstd = dataset["Gstd"]

    # a dense FLOOR: every variant carries an effect (the old build zeroed 65%)
    assert np.all(dense != 0.0)
    # and a genuinely sparse spike layer on top
    cfg = CATEGORIES["sparse_dense_mix"]["make_cfg"](seed, N, P)
    assert int(np.count_nonzero(spike)) == int(round(len(spike) * cfg.spike_frac))

    var_dense = float((Gstd @ dense).var())
    var_spike = float((Gstd @ spike).var())
    share = var_dense / (var_dense + var_spike)
    target = cfg.dense_var_share
    # the shares are CONTROLLED, not an accident of the slab (measured: 0.500)
    assert abs(share - target) < 0.05
    # The sparse component has much larger per-active-variant RMS than the dense
    # floor. RMS matches the declared variance-component fact; a median is unstable
    # for only ~10 heavy-tailed spikes.
    spike_rms = float(np.sqrt(np.mean(spike[spike != 0.0] ** 2)))
    floor_rms = float(np.sqrt(np.mean(dense ** 2)))
    assert spike_rms / floor_rms > 5.0


# ---- DGP-003: suppressor_ld is sparse opposing-effect pair geometry ---------
def _reversed_effect_mass(dataset):
    """Share of the causal squared-effect mass whose MARGINAL association is nulled
    (< 30% of the joint effect) or sign-reversed.

    An unweighted sign-flip RATE is the wrong statistic and is how the absent
    mechanism hid for so long: under strong LD with random-signed neighbours, tiny
    effects flip sign by chance, so even a positive-LD control shows a ~0.35 flip rate
    (the audit measured 0.3137 signed vs 0.3353 control). Weighting by beta^2 asks the
    question that matters -- how much of the SIGNAL is invisible to a marginal screen."""
    truth = dataset["truth"]
    beta = truth["beta_effective"]
    marg = _marginal_cov(dataset)
    causal = beta != 0.0
    b, m = beta[causal], marg[causal]
    hidden = (np.sign(m) != np.sign(b)) | (np.abs(m) < 0.3 * np.abs(b))
    return float((b ** 2 * hidden).sum() / (b ** 2).sum())


@pytest.mark.parametrize("seed", SEEDS)
def test_suppressor_ld_reverses_the_sign_of_the_marginal_association(seed):
    """ATTENUATION IS NOT REVERSAL.

    An equal-and-opposite pair (b, -b) gives the target a marginal covariance of
    (1-r)*b: shrunk, but with the CORRECT SIGN, so a marginal screen still reads the
    sign right and the category only half-exists (its measured reversal rate was 0.20,
    which is finite-sample noise around a construction that can never reverse). The
    partner effect must OVERSHOOT: with b_partner = -(1+m)*b/r the target's marginal
    covariance is exactly -m*b, so the sign flip is SET by construction."""
    dataset = _gen("suppressor_ld", seed)
    control = _gen("suppressor_ld", seed, suppressor_block_frac=0.0)

    truth = dataset["truth"]
    target = truth["suppressor_target"]
    partner = truth["suppressor_partner"]
    assert target.sum() >= 5, "no suppressor pairs were constructed"
    assert target.sum() == partner.sum()

    beta = truth["beta_effective"]
    Gstd = dataset["Gstd"]
    block_id = dataset["meta"]["block_id"]
    cfg = dataset["cfg"]
    for block in np.unique(block_id[target]):
        cols = np.where(block_id == block)[0]
        left = cols[target[cols]]
        right = cols[partner[cols]]
        assert len(left) == len(right) == 1
        pair = np.array([left[0], right[0]])
        correlation = float(np.mean(Gstd[:, pair[0]] * Gstd[:, pair[1]]))
        assert correlation >= cfg.suppressor_min_corr
        pair_beta = beta[pair]
        pair_r = np.array([[1.0, correlation], [correlation, 1.0]])
        pair_marginal = pair_r @ pair_beta
        # the PAIR-ONLY identity: the target's marginal is -m * beta_target, with the
        # overshoot m inside the configured band, so it is REVERSED with real margin.
        m = -pair_marginal[0] / pair_beta[0]
        assert cfg.suppressor_overshoot_lo - 1e-9 <= m <= cfg.suppressor_overshoot_hi + 1e-9
        assert np.sign(pair_marginal[0]) == -np.sign(pair_beta[0])

    # The realized (whole-genome) marginal association carries finite-sample cross-block
    # covariance on top of the pair identity, so the reversal must survive THAT.
    marg = _marginal_cov(dataset)
    kappa = marg[target] / beta[target]
    # kappa = -m by construction: negative for essentially every target (measured ~1.0),
    # far above the >0.35 the mechanism needs to be more than half-present.
    assert float(np.mean(kappa < 0.0)) > 0.90
    assert float(np.median(kappa)) < -0.2
    assert float(np.mean(np.abs(kappa) < 1.0)) > 0.8

    # and the matched positive-LD control must NOT show it -- otherwise the "mechanism"
    # is just an artifact of strong LD plus random-signed neighbours.
    control_kappa = (_marginal_cov(control)[control["truth"]["causal_mask"]]
                     / control["truth"]["beta_effective"][control["truth"]["causal_mask"]])
    assert float(np.mean(control_kappa < 0.0)) < 0.45

    hidden_signed = _reversed_effect_mass(dataset)
    hidden_control = _reversed_effect_mass(control)
    assert hidden_signed > 0.04
    assert hidden_signed > 2.0 * hidden_control

    sup = target | partner
    v_pair = float((Gstd[:, sup] @ beta[sup]).var())
    v_rest = float((Gstd[:, ~sup] @ beta[~sup]).var())
    share = v_pair / (v_pair + v_rest)
    assert share == pytest.approx(cfg.suppressor_var_share, abs=1e-10)
    assert float(np.mean(beta != 0.0)) < 0.25


def test_suppressor_ld_generation_is_seed_reproducible():
    first = _gen("suppressor_ld", 7)
    second = _gen("suppressor_ld", 7)
    for key in ("G", "Gstd", "y"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(
        first["truth"]["beta_effective"], second["truth"]["beta_effective"]
    )


# ---- DGP-004: the LD-shift cohort really changes the correlation structure ---
def _adjacent_correlation(G, rows, block_id):
    X = G[rows].astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    same_block = block_id[:-1] == block_id[1:]
    corr = (X[:, :-1] * X[:, 1:]).mean(0)
    return np.abs(corr[same_block])


@pytest.mark.parametrize("seed", SEEDS)
def test_ld_shift_moves_the_correlation_structure_but_not_the_effects_or_maf(seed):
    dataset = _gen("ld_shift", seed)
    cohort = dataset["cohort"]
    train = np.where(cohort == 0)[0]
    test = np.where(cohort == 1)[0]
    assert len(train) > 0 and len(test) > 0

    block_id = dataset["meta"]["block_id"]
    ld_train = _adjacent_correlation(dataset["G"], train, block_id)
    ld_test = _adjacent_correlation(dataset["G"], test, block_id)
    # measured after the fix: |adjacent within-block corr| ~0.37 train vs ~0.05 test
    assert ld_train.mean() > 0.25
    assert ld_test.mean() < 0.15
    assert ld_train.mean() - ld_test.mean() > 0.15

    # the effects are FIXED across the shift (only the tagging structure moved), and
    # so is the allele-frequency distribution -- otherwise the shift is confounded.
    G = dataset["G"].astype(float)
    freq_train = G[train].mean(0) / 2.0
    freq_test = G[test].mean(0) / 2.0
    assert float(np.mean(np.abs(freq_train - freq_test))) < 0.02
    assert np.corrcoef(freq_train, freq_test)[0, 1] > 0.95


def test_only_the_ld_shift_category_splits_by_cohort():
    for category, spec in CATEGORIES.items():
        cfg = spec["make_cfg"](1, N, P)
        expected = category == "ld_shift"
        assert bool(cfg.ld_shift) is expected


# ---- DGP-006: causal-inclusion provenance ----------------------------------
@pytest.mark.parametrize("category", ["sparse_heavy_tail", "rare_variant_maf",
                                      "sparse_dense_mix"])
def test_causal_mask_records_where_the_slab_conditional_actually_holds(category):
    truth = _gen(category, 1)["truth"]
    beta = truth["beta_effective"]
    mask = truth["causal_mask"]

    np.testing.assert_array_equal(mask, beta != 0.0)
    # tau2 is documented as the slab variance CONDITIONAL ON INCLUSION: it stays
    # positive off the mask, which is exactly why the mask has to be stored.
    assert np.all(truth["tau2_effective"] > 0.0)
    if category == "sparse_dense_mix":
        assert mask.all(), "dense architectures have no excluded variants"
    else:
        assert not mask.all(), "masked architectures must exclude some variants"


def test_dense_gaussian_causal_mask_covers_every_variant():
    # The shipped dense_infinitesimal category must obey the causal-mask contract:
    # a dense architecture excludes nothing.
    truth = _gen_dense_gaussian(1)["truth"]
    np.testing.assert_array_equal(truth["causal_mask"], truth["beta_effective"] != 0.0)
    assert truth["causal_mask"].all(), "dense architectures have no excluded variants"
    assert np.all(truth["tau2_effective"] > 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_low_prevalence_category_realizes_its_declared_rare_outcome(seed):
    dataset = _gen("low_prevalence", seed)
    cfg = dataset["cfg"]
    realized = float(np.mean(dataset["y"]))
    assert cfg.prevalence == pytest.approx(0.15)
    assert cfg.effect_family == "dense_gaussian"
    # Loose, preregistered realization band: catches a missing intercept/prevalence
    # mechanism without tuning to one seed (N=900 gives sd about 0.012).
    assert 0.10 <= realized <= 0.20


def test_dense_effect_families_reject_a_zeroing_mask():
    # A "dense" label plus a null_frac spike is precisely the bug this fixes.
    with pytest.raises(ValueError, match="dense effect families forbid"):
        generate(DGPConfig(n_samples=200, n_variants=60, seed=1,
                           effect_family="dense_gaussian", null_frac=0.5))
    with pytest.raises(ValueError, match="unknown effect_family"):
        generate(DGPConfig(n_samples=200, n_variants=60, seed=1,
                           effect_family="not_a_family"))


# ---- CONTRACT-002: public bytes never depend on held-out labels ----------------
def _tree_bytes(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_public_materialization_is_independent_of_hidden_test_labels(tmp_path):
    from datagen.materialize import materialize

    spec = CATEGORIES["decoy_annotations"]
    dataset = _gen("decoy_annotations", 3)
    first = materialize(dataset, tmp_path / "first", family=spec["family"],
                        frac_train=dataset["cfg"].frac_train_hint,
                        prior_cols=spec["prior_cols"], seed=3)

    with open(Path(first["truth"]) / "y_test.csv") as handle:
        handle.readline()
        test_indices = [int(line.split(",", 1)[0][1:]) for line in handle if line.strip()]
    changed = dict(dataset)
    changed["y"] = dataset["y"].copy()
    changed["y"][test_indices] = 1 - changed["y"][test_indices]
    second = materialize(changed, tmp_path / "second", family=spec["family"],
                         frac_train=changed["cfg"].frac_train_hint,
                         prior_cols=spec["prior_cols"], seed=3)

    assert _tree_bytes(first["public"]) == _tree_bytes(second["public"])


def _read_csv(path):
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split(",")
        rows = [line.rstrip("\n").split(",") for line in handle if line.strip()]
    return header, rows


def _columns(header, rows, names):
    index = {name: i for i, name in enumerate(header)}
    return np.array([[float(row[index[name]]) for name in names] for row in rows])


def test_unlisted_covariate_columns_punish_an_all_columns_parser(tmp_path):
    """The formula-scoping contract must COST something to violate.

    The batch artifact is drawn independently and the phenotype is generated FROM it
    with a cohort-dependent sign, so the public column is never a function of the
    hidden labels (the test above pins that) -- but it predicts y in train and
    reverses out of sample, so a submission that regresses on every non-ID column
    instead of the formula-declared ones loses real held-out score."""
    from datagen.materialize import materialize

    spec = CATEGORIES["decoy_annotations"]
    dataset = _gen("decoy_annotations", 3)
    out = materialize(dataset, tmp_path / "ds", family=spec["family"],
                      frac_train=dataset["cfg"].frac_train_hint,
                      prior_cols=spec["prior_cols"], seed=3)
    public = out["public"]

    head_tr, rows_tr = _read_csv(f"{public}/covariates_train.csv")
    head_te, rows_te = _read_csv(f"{public}/covariates_test.csv")
    with open(f"{public}/formula.txt") as handle:
        formula = handle.read()
    declared = formula.split("covariate(")[1].split(")")[0].split(", ")

    # the machinery is actually DEPLOYED: columns on disk, absent from the formula
    unlisted = [c for c in head_te if c not in declared and c != "sample_id"]
    assert unlisted, "no unlisted covariate columns were written"
    assert all(name not in formula for name in unlisted)

    y_train = _columns(head_tr, rows_tr, ["y"]).ravel()
    _, y_rows = _read_csv(f"{out['truth']}/y_test.csv")
    y_test = np.array([float(row[1]) for row in y_rows])

    def fit_predict(names):
        Xtr = np.column_stack([np.ones(len(rows_tr)), _columns(head_tr, rows_tr, names)])
        Xte = np.column_stack([np.ones(len(rows_te)), _columns(head_te, rows_te, names)])
        beta, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)
        return _auc(y_test, Xte @ beta)

    honest = fit_predict(declared)                   # formula-scoped
    naive = fit_predict(declared + unlisted)         # "use every non-ID column"
    assert honest - naive > 0.05, (honest, naive)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"nuisance_cov_cols": -1}, "nonnegative integer"),
        ({"nuisance_cov_cols": True}, "nonnegative integer"),
        (
            {"nuisance_cov_cols": 1, "nuisance_cov_strength": 0.0},
            "positive nuisance_cov_strength",
        ),
        (
            {"nuisance_cov_cols": 1, "frac_train_hint": 1.0},
            "nonempty train and test cohorts",
        ),
    ],
)
def test_nuisance_covariate_configuration_fails_closed(updates, message):
    config = DGPConfig(seed=7, n_samples=40, n_variants=20, **updates)
    with pytest.raises(ValueError, match=message):
        generate(config)


def _auc(y, score):
    y = np.asarray(y).astype(int)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
