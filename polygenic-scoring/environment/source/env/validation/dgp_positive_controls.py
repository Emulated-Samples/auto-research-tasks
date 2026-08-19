"""Positive controls for the datagen mechanisms (DGP-001..004).

A category is only worth having if the RIGHT inductive bias WINS in it. These are
deliberately minimal, self-contained numpy estimators (they are NOT the graded
reference and they do not touch validation/model_zoo.py) used to check the
model-RANKING each mechanism is supposed to produce:

    dense_infinitesimal : ridge            >  lasso / adaptive lasso
    sparse_dense_mix    : dense+sparse     >  pure ridge AND pure lasso
    suppressor_ld       : joint estimator  >  marginal screening (P+T)
    ld_shift            : joint estimator  >  tag-only marginal predictor (on the
                                              LD-shifted test cohort)

Everything is a linear-probability fit on covariate-residualized y, which is enough
to rank models by held-out AUC and keeps each replicate to a second or two.

Run as a script for the full paired-seed table:
    .venv/bin/python -m validation.dgp_positive_controls
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen.categories import CATEGORIES          # noqa: E402
from datagen.dgp import generate                   # noqa: E402


# ---- metric ---------------------------------------------------------------
def auc(y, score):
    y = np.asarray(y).astype(int)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# ---- data prep ------------------------------------------------------------
def _prep(cat, seed, N, P):
    """Train/test split matching materialize(): cohort split under ld_shift, else a
    plain random split. Returns covariate-residualized train target + standardized
    genotypes (train statistics only, exactly what a submission can compute).

    The pseudo-category "ld_shift_no_shift_control" is the ld_shift recipe with the
    test cohort's LD drawn from the SAME range as train -- the matched control that
    isolates the shift from every other property of the recipe."""
    control = cat == "ld_shift_no_shift_control"
    if control:
        cat = "ld_shift"
    cfg = CATEGORIES[cat]["make_cfg"](seed, N, P)
    if control:
        cfg.ld_shift_rho_lo, cfg.ld_shift_rho_hi = cfg.ld_rho_lo, cfg.ld_rho_hi
    d = generate(cfg)
    G, y, C = d["G"].astype(float), d["y"].astype(float), d["cov"]
    n = len(y)
    if cfg.ld_shift:
        cohort = d["cohort"]
        tr, te = np.where(cohort == 0)[0], np.where(cohort == 1)[0]
    else:
        rng = np.random.default_rng(seed + 999)
        idx = rng.permutation(n)
        ntr = int(cfg.frac_train_hint * n)
        tr, te = idx[:ntr], idx[ntr:]
    mu, sd = G[tr].mean(0), G[tr].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xtr, Xte = (G[tr] - mu) / sd, (G[te] - mu) / sd
    # covariate adjustment (unpenalized), fit on train only
    beta_c, *_ = np.linalg.lstsq(C[tr], y[tr], rcond=None)
    rtr = y[tr] - C[tr] @ beta_c
    cov_te = C[te] @ beta_c
    return dict(Xtr=Xtr, Xte=Xte, rtr=rtr, cov_te=cov_te, ytr=y[tr], yte=y[te],
                block=d["meta"]["block_id"], truth=d["truth"],
                variant_class=d["meta"]["variant_class"])


def _folds(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]


# ---- estimators (all return test-set genetic scores) -----------------------
# Everything is fit from the Gram matrix (X'X/n, X'r/n), computed ONCE per training
# slice and reused across the whole penalty grid -- otherwise the paired-seed sweep
# is dominated by redundant matrix products.
class _Fit:
    def __init__(self, X, r):
        self.n, self.p = X.shape
        self.X, self.r = X, r
        self.G = X.T @ X / self.n
        self.c = X.T @ r / self.n
        self.L = float(np.max(np.linalg.eigvalsh(self.G))) + 1e-9

    def ridge(self, lam):
        return np.linalg.solve(self.G + (lam / self.n) * np.eye(self.p), self.c)

    def lasso(self, lam, w=None, iters=250):
        w = np.ones(self.p) if w is None else w
        b = np.zeros(self.p)
        z = b.copy()
        t = 1.0
        for _ in range(iters):
            u = z - (self.G @ z - self.c) / self.L
            thr = lam * w / self.L
            b_new = np.sign(u) * np.maximum(np.abs(u) - thr, 0.0)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = b_new + ((t - 1) / t_new) * (b_new - b)
            b, t = b_new, t_new
        return b

    def enet(self, lam_alpha, iters=250):
        """Elastic net: lam * (alpha*|b|_1 + (1-alpha)/2*|b|^2). alpha=0 is ridge and
        alpha=1 is lasso, so the mixed-penalty fit and its two pure corners are the
        SAME estimator family -- which is what makes 'the mixture needs both' testable
        rather than a comparison between differently-tuned models."""
        lam, alpha = lam_alpha
        ridge_part = (1.0 - alpha) * lam
        L = self.L + ridge_part
        b = np.zeros(self.p)
        z = b.copy()
        t = 1.0
        for _ in range(iters):
            grad = self.G @ z - self.c + ridge_part * z
            u = z - grad / L
            thr = alpha * lam / L
            b_new = np.sign(u) * np.maximum(np.abs(u) - thr, 0.0)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = b_new + ((t - 1) / t_new) * (b_new - b)
            b, t = b_new, t_new
        return b


_RIDGE_GRID = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
_LASSO_GRID = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064]


def _cv_select(X, r, method, grid, seed, **kw):
    """3-fold CV over `grid`, then refit the winner on the full training slice."""
    fits = [(_Fit(X[tr], r[tr]), tr, va) for tr, va in _folds(len(r), 3, seed)]
    best, best_lam = -np.inf, grid[0]
    for lam in grid:
        s = 0.0
        for f, _tr, va in fits:
            b = getattr(f, method)(lam, **kw)
            s += -np.mean((r[va] - X[va] @ b) ** 2)
        if s > best:
            best, best_lam = s, lam
    full = _Fit(X, r)
    return best_lam, getattr(full, method)(best_lam, **kw)


def ridge(d, seed):
    _, b = _cv_select(d["Xtr"], d["rtr"], "ridge", _RIDGE_GRID, seed)
    return d["Xte"] @ b, b


def lasso(d, seed):
    _, b = _cv_select(d["Xtr"], d["rtr"], "lasso", _LASSO_GRID, seed)
    return d["Xte"] @ b, b


def adaptive_lasso(d, seed):
    _, b0 = ridge(d, seed)
    w = 1.0 / (np.abs(b0) + 1e-4)
    w = w / w.mean()
    _, b = _cv_select(d["Xtr"], d["rtr"], "lasso", _LASSO_GRID, seed, w=w)
    return d["Xte"] @ b, b


_ENET_GRID = [(lam, alpha)
              for alpha in (0.15, 0.35, 0.6, 0.85)
              for lam in (0.02, 0.05, 0.15, 0.4, 1.0, 3.0)]
# the two PURE corners of the same estimator family (alpha = 0 / 1)
_ENET_RIDGE_GRID = [(lam, 0.0) for lam in (0.05, 0.15, 0.4, 1.0, 3.0, 10.0, 30.0)]
_ENET_LASSO_GRID = [(lam, 1.0) for lam in (0.002, 0.004, 0.008, 0.016, 0.032, 0.064)]


def dense_plus_sparse(d, seed):
    """The mixed-penalty (elastic-net) member of the family: a dense L2 floor AND a
    sparse L1 spike layer at once, with both penalties chosen by CV."""
    _, b = _cv_select(d["Xtr"], d["rtr"], "enet", _ENET_GRID, seed)
    return d["Xte"] @ b, b


def enet_ridge_corner(d, seed):
    _, b = _cv_select(d["Xtr"], d["rtr"], "enet", _ENET_RIDGE_GRID, seed)
    return d["Xte"] @ b, b


def enet_lasso_corner(d, seed):
    _, b = _cv_select(d["Xtr"], d["rtr"], "enet", _ENET_LASSO_GRID, seed)
    return d["Xte"] @ b, b


def annotation_ridge(d, seed):
    """Ridge with a per-variant penalty learned from the ANNOTATION (variant_class).

    Uses only public information: it fits one prior effect-scale per class from the
    training data (the mean squared marginal effect within each class, an annotation-
    level quantity), then penalizes each variant inversely to its class scale. This is
    the minimal annotation-aware joint model -- exactly the capability the benchmark
    exists to reward -- and under an LD shift it is the ONLY way to tell a causal
    variant from the variants that merely tag it in the training cohort."""
    X, r = d["Xtr"], d["rtr"]
    n = X.shape[0]
    vclass = np.asarray(d["variant_class"])
    marg = X.T @ r / n
    noise = float(np.mean(r ** 2)) / n              # per-variant sampling variance
    scale = np.ones(X.shape[1])
    for cls in np.unique(vclass):
        mask = vclass == cls
        # method-of-moments prior scale for this class: E[marg^2] - sampling noise
        excess = max(float(np.mean(marg[mask] ** 2)) - noise, 1e-12)
        scale[mask] = np.sqrt(excess)
    scale = scale / scale.mean()
    w = 1.0 / (scale ** 2)                          # penalty weight ~ 1 / prior variance

    def fit(_Fit_obj, lam):
        return np.linalg.solve(
            _Fit_obj.G + (lam / _Fit_obj.n) * np.diag(w), _Fit_obj.c)

    fits = [(_Fit(X[tr], r[tr]), va) for tr, va in _folds(len(r), 3, seed)]
    best, best_lam = -np.inf, _RIDGE_GRID[0]
    for lam in _RIDGE_GRID:
        s = sum(-np.mean((r[va] - X[va] @ fit(f, lam)) ** 2) for f, va in fits)
        if s > best:
            best, best_lam = s, lam
    b = fit(_Fit(X, r), best_lam)
    return d["Xte"] @ b, b


def pt_marginal(d, seed):
    """Pruning + thresholding: marginal effects, LD-pruned (keep the top-|z| variant
    per block), thresholded at a CV-chosen number of variants. The canonical
    tag-friendly / marginal-screening predictor."""
    X, r = d["Xtr"], d["rtr"]
    n = X.shape[0]
    block = d["block"]
    marg = X.T @ r / n
    se = np.sqrt(np.maximum(np.mean(r ** 2) / n, 1e-12))
    zscore = np.abs(marg) / se
    # LD pruning: one (best) variant per block
    keep = []
    for blk in np.unique(block):
        cols = np.where(block == blk)[0]
        keep.append(cols[np.argmax(zscore[cols])])
    keep = np.array(sorted(keep))
    order = keep[np.argsort(-zscore[keep])]
    best, best_b = -np.inf, np.zeros(X.shape[1])
    for k in (5, 10, 25, 50, 100, len(order)):
        k = min(k, len(order))
        sel = order[:k]
        b = np.zeros(X.shape[1])
        b[sel] = marg[sel]
        s = 0.0
        for tr, va in _folds(n, 3, seed):
            mtr = X[tr].T @ r[tr] / len(tr)
            bb = np.zeros(X.shape[1])
            bb[sel] = mtr[sel]
            s += -np.mean((r[va] - X[va] @ bb) ** 2)
        if s > best:
            best, best_b = s, b
    return d["Xte"] @ best_b, best_b


MODELS = {
    "ridge": ridge,
    "lasso": lasso,
    "adaptive_lasso": adaptive_lasso,
    "dense_plus_sparse": dense_plus_sparse,
    "enet_ridge_corner": enet_ridge_corner,
    "enet_lasso_corner": enet_lasso_corner,
    "annotation_ridge": annotation_ridge,
    "pt_marginal": pt_marginal,
}


def oracle_auc(d):
    """Held-out AUC of the TRUE effects. The LD-shifted cohort is intrinsically easier
    than the matched no-shift cohort (measured oracle AUC 0.662 vs 0.646), so raw AUC
    cannot be compared across the two arms -- every shift claim must be made on the
    GAP TO ORACLE."""
    return auc(d["yte"], d["Xte"] @ d["truth"]["beta_effective"])


def score(cat, seed, models, N=1200, P=500):
    d = _prep(cat, seed, N, P)
    out = {}
    for name in models:
        gscore, _ = MODELS[name](d, seed)
        out[name] = auc(d["yte"], d["cov_te"] + gscore)
    return out


def paired(cat, models, seeds, N=1200, P=500):
    rows = [score(cat, s, models, N, P) for s in seeds]
    return {m: np.array([r[m] for r in rows]) for m in models}


def win_rate(a, b):
    return float(np.mean(a > b)), float(np.mean(a - b))


if __name__ == "__main__":
    seeds = list(range(1, 11))
    print("paired seeds:", seeds, "(N=1200, P=500)\n")

    r = paired("dense_infinitesimal", ["ridge", "lasso", "adaptive_lasso"], seeds)
    print("dense_infinitesimal  mean AUC:",
          {k: round(float(v.mean()), 4) for k, v in r.items()})
    for opp in ("lasso", "adaptive_lasso"):
        w, dlt = win_rate(r["ridge"], r[opp])
        print(f"  ridge > {opp:15s} win rate {w:.2f}  mean delta {dlt:+.4f}")

    r = paired("sparse_dense_mix",
               ["enet_ridge_corner", "enet_lasso_corner", "dense_plus_sparse"], seeds)
    print("\nsparse_dense_mix     mean AUC:",
          {k: round(float(v.mean()), 4) for k, v in r.items()})
    for opp in ("enet_ridge_corner", "enet_lasso_corner"):
        w, dlt = win_rate(r["dense_plus_sparse"], r[opp])
        print(f"  dense_plus_sparse > {opp:18s} win rate {w:.2f}  mean delta {dlt:+.4f}")

    r = paired("suppressor_ld", ["ridge", "pt_marginal"], seeds)
    print("\nsuppressor_ld        mean AUC:",
          {k: round(float(v.mean()), 4) for k, v in r.items()})
    w, dlt = win_rate(r["ridge"], r["pt_marginal"])
    print(f"  joint(ridge) > P+T    win rate {w:.2f}  mean delta {dlt:+.4f}")

    r = paired("ld_shift", ["ridge", "pt_marginal"], seeds)
    print("\nld_shift             mean AUC:",
          {k: round(float(v.mean()), 4) for k, v in r.items()})
    w, dlt = win_rate(r["ridge"], r["pt_marginal"])
    print(f"  joint(ridge) > tag-only P+T  win rate {w:.2f}  mean delta {dlt:+.4f}")

    # FALSIFICATION arm: the claim is not "P+T is worse here", it is "P+T DEGRADES
    # BECAUSE OF THE SHIFT". Without the matched no-shift control (same category, LD
    # held fixed across the cohorts) a constant P+T handicap would look like a shift
    # effect -- which is exactly how the absent signed-LD mechanism hid.
    control = paired("ld_shift_no_shift_control", ["ridge", "pt_marginal"], seeds)
    print("\nld_shift CONTROL (same category, no shift) mean AUC:",
          {k: round(float(v.mean()), 4) for k, v in control.items()})
    print("  P+T lost to the shift:   "
          f"{float(control['pt_marginal'].mean() - r['pt_marginal'].mean()):+.4f}")
    print("  joint lost to the shift: "
          f"{float(control['ridge'].mean() - r['ridge'].mean()):+.4f}")
