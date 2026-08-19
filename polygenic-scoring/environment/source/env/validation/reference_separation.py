"""Reference separation: does the REAL SV-PGS engine have reward signal?

Generates one moderate BINARY dataset, splits train/test by sha256(sample_id)
(mirroring the AoU 20% held-out policy), and compares methods on HELD-OUT rank-AUC:

  naive         cov-only logistic
  oracle        true beta (genetic ceiling)
  uniform-ridge single-penalty (class-blind) ridge-logistic, penalty grid via IRLS
  marginal P+T  univariate z + LD-block clump + threshold grid
  SV-PGS        the real engine (reference/run_svpgs.fit_svpgs_generated)

Every method emits a genetic score on train & test; we then STACK it with the
covariates via a common logistic re-fit (y_train ~ [cov | genetic_score]) and
score the held-out set. This gives all methods identical covariate adjustment so
the comparison isolates the genetic component. SV-PGS's own genetic component
(decision_components) is used, then re-stacked the same way as the others.

Reports: raw AUC, incremental-normalized AUC (auc-naive)/(oracle-naive),
prior-recovery Spearman(SV-PGS recovered per-class baseline vs TRUE per-class
log-baseline), and SV-PGS fit wall-time.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import numpy as np
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_SVPGS = os.environ.get("SVPGS_HOME", os.path.join(os.path.dirname(_ROOT), "SV-PGS"))
if os.path.isdir(_SVPGS):
    sys.path.insert(0, _SVPGS)

from datagen.dgp import generate, DGPConfig  # noqa: E402
from reference.run_svpgs import fit_svpgs_generated  # noqa: E402


# ----------------------------------------------------------------------------
def sha256_test_mask(n, salt="svpgsbench", frac=0.2):
    m = np.zeros(n, dtype=bool)
    for i in range(n):
        h = hashlib.sha256(f"{salt}|sample_{i}".encode()).hexdigest()
        bucket = int(h[:16], 16) / float(1 << 64)
        m[i] = bucket < frac
    return m


def standardize_train(G_tr, G_te):
    mu = G_tr.mean(axis=0)
    css = ((G_tr - mu) ** 2).sum(axis=0)
    sd = np.sqrt(css / G_tr.shape[0])
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (G_tr - mu) / sd, (G_te - mu) / sd, mu, sd


def auc(y, s):
    # tie-aware Mann-Whitney AUC
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        ranks_sorted[i:j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ranks[order] = ranks_sorted
    n_pos = (y == 1).sum()
    sum_r_pos = ranks[y == 1].sum()
    return (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * (len(s) - n_pos))


# ---- logistic helpers -------------------------------------------------------
def _logit_irls(X, y, pen_vec, max_iter=50, tol=1e-8):
    """Newton/IRLS for penalized logistic. pen_vec: per-column L2 penalty (ridge)."""
    n, p = X.shape
    beta = np.zeros(p)
    P = np.diag(pen_vec)
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-6)
        grad = X.T @ (y - mu) - pen_vec * beta
        H = (X * w[:, None]).T @ X + P
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def stack_and_score(cov_tr, cov_te, g_tr, g_te, y_tr):
    """Common covariate+genetic stacking: y ~ [1, cov, g]. Small ridge on all
    non-intercept columns for numerical stability; returns test linear predictor."""
    def design(cov, g):
        return np.column_stack([np.ones(len(cov)), cov, g])
    Xtr = design(cov_tr, g_tr)
    Xte = design(cov_te, g_te)
    pen = np.full(Xtr.shape[1], 1e-3)
    pen[0] = 0.0
    beta = _logit_irls(Xtr, y_tr.astype(float), pen)
    return Xte @ beta


# ---- uniform ridge (class-blind) -------------------------------------------
def uniform_ridge_genetic(Z_tr, Z_te, cov_tr, y_tr, lam_grid, valid_frac=0.25, seed=0):
    """Single-penalty ridge-logistic on [1,cov,Z]; cov unpenalized, all genetic
    variants share ONE penalty lam. Pick lam by internal validation AUC."""
    rng = np.random.default_rng(seed)
    n = Z_tr.shape[0]
    vmask = rng.random(n) < valid_frac
    Xfull = np.column_stack([np.ones(n), cov_tr, Z_tr])
    k = 1 + cov_tr.shape[1]
    best = (None, -1, None)
    for lam in lam_grid:
        pen = np.concatenate([np.zeros(k), np.full(Z_tr.shape[1], lam)])
        beta = _logit_irls(Xfull[~vmask], y_tr[~vmask].astype(float), pen)
        g_val = Z_tr[vmask] @ beta[k:]
        a = auc(y_tr[vmask], g_val)
        if a > best[1]:
            best = (lam, a, beta)
    lam, _, beta = best
    # refit on full train at chosen lam
    pen = np.concatenate([np.zeros(k), np.full(Z_tr.shape[1], lam)])
    beta = _logit_irls(Xfull, y_tr.astype(float), pen)
    coef = beta[k:]
    return Z_tr @ coef, Z_te @ coef, lam


# ---- marginal P+T -----------------------------------------------------------
def _marginal_effects(Z, cov, y, block_id):
    """Covariate-residualized marginal effect, z, and top-|z|-per-block clump reps,
    computed on the given rows ONLY (no leakage of other rows into the effects)."""
    m = Z.shape[0]
    C = np.column_stack([np.ones(m), cov])
    beta_c, *_ = np.linalg.lstsq(C, y.astype(float), rcond=None)
    yr = y.astype(float) - C @ beta_c
    s2 = (yr @ yr) / m
    num = Z.T @ yr                          # X'yr
    denom = (Z ** 2).sum(axis=0)            # ~ m
    beta_marg = num / np.maximum(denom, 1e-8)
    zscore = num / np.sqrt(np.maximum(s2 * denom, 1e-12))
    # LD-block clump: keep the top-|z| variant per block
    seen_block, clump_reps = set(), []
    for j in np.argsort(-np.abs(zscore)):
        b = int(block_id[j])
        if b in seen_block:
            continue
        seen_block.add(b)
        clump_reps.append(j)
    return beta_marg, zscore, np.array(clump_reps)


def marginal_pt_genetic(Z_tr, Z_te, cov_tr, y_tr, block_id, z_grid, valid_frac=0.25, seed=0):
    n, P = Z_tr.shape
    # Draw the inner-validation split FIRST, then fit marginal effects on the
    # inner-TRAIN rows only (mirrors uniform_ridge_genetic). Fitting on all train
    # rows before validating would leak the validation labels into the effects the
    # threshold is selected against, inflating P+T's apparent AUC.
    rng = np.random.default_rng(seed)
    vmask = rng.random(n) < valid_frac
    beta_in, z_in, reps_in = _marginal_effects(Z_tr[~vmask], cov_tr[~vmask],
                                               y_tr[~vmask], block_id)
    absz_in = np.abs(z_in[reps_in])
    best = (None, -1)
    for zt in z_grid:
        sel = reps_in[absz_in >= zt]
        if len(sel) == 0:
            continue
        g_val = Z_tr[vmask][:, sel] @ beta_in[sel]
        a = auc(y_tr[vmask], g_val)
        if a > best[1]:
            best = (zt, a)
    zt = best[0] if best[0] is not None else 0.0
    # Refit marginal effects + clump on the FULL train at the chosen threshold.
    beta_marg, zscore, clump_reps = _marginal_effects(Z_tr, cov_tr, y_tr, block_id)
    absz_rep = np.abs(zscore[clump_reps])
    sel = clump_reps[absz_rep >= zt]
    if len(sel) == 0:
        sel = clump_reps[np.argsort(-absz_rep)[:1]]
    g_tr = Z_tr[:, sel] @ beta_marg[sel]
    g_te = Z_te[:, sel] @ beta_marg[sel]
    return g_tr, g_te, zt, len(sel)


# ----------------------------------------------------------------------------
def run(N=3000, P=1000, seed=7, max_iter_svpgs=15,
        heritability=0.7, tail_temper=2.5, ancestry_confounding=0.3,
        ld_rho_lo=0.85, ld_rho_hi=0.98, log=print):
    t_start = time.perf_counter()
    log(f"[sep] generating binary dataset N={N} P={P} seed={seed} "
        f"h2={heritability} tail_temper={tail_temper} ld_rho=[{ld_rho_lo},{ld_rho_hi}]", flush=True)
    d = generate(DGPConfig(seed=seed, n_samples=N, n_variants=P,
                           heritability=heritability, prevalence=0.3, tail_temper=tail_temper,
                           ancestry_confounding=ancestry_confounding,
                           ld_rho_lo=ld_rho_lo, ld_rho_hi=ld_rho_hi))
    meta, truth = d["meta"], d["truth"]
    G = np.asarray(d["G"], np.float32)
    cov_full = np.asarray(d["cov"], np.float32)
    y = np.asarray(d["y"]).astype(int)
    P = G.shape[1]
    log(f"[sep] realized P={P}  case_rate={y.mean():.3f}", flush=True)

    # strip leading intercept column from cov (fit prepends its own)
    cov = cov_full[:, 1:] if np.allclose(cov_full[:, 0], 1.0) else cov_full

    te = sha256_test_mask(len(y))
    tr = ~te
    log(f"[sep] train={tr.sum()} test={te.sum()}", flush=True)

    G_tr, G_te = G[tr], G[te]
    cov_tr, cov_te = cov[tr], cov[te]
    y_tr, y_te = y[tr], y[te]
    Z_tr, Z_te, mu, sd = standardize_train(G_tr, G_te)

    results = {}

    # --- naive (cov only) ---
    s = stack_and_score(cov_tr, cov_te, np.zeros(len(y_tr)), np.zeros(len(y_te)), y_tr)
    results["naive"] = auc(y_te, s)
    log(f"[sep] naive        AUC={results['naive']:.4f}", flush=True)

    # --- oracle (true beta) ---
    g_tr = Z_tr @ truth["beta_effective"]
    g_te = Z_te @ truth["beta_effective"]
    s = stack_and_score(cov_tr, cov_te, g_tr, g_te, y_tr)
    results["oracle"] = auc(y_te, s)
    log(f"[sep] oracle       AUC={results['oracle']:.4f}", flush=True)

    # --- uniform ridge ---
    lam_grid = [0.5, 2.0, 8.0, 32.0, 128.0, 512.0]
    g_tr, g_te, lam = uniform_ridge_genetic(Z_tr, Z_te, cov_tr, y_tr, lam_grid)
    s = stack_and_score(cov_tr, cov_te, g_tr, g_te, y_tr)
    results["ridge"] = auc(y_te, s)
    log(f"[sep] uniform-ridge AUC={results['ridge']:.4f}  (lam*={lam})", flush=True)

    # --- marginal P+T ---
    z_grid = [0.0, 1.0, 1.64, 2.0, 2.58, 3.0]
    g_tr, g_te, zt, nsel = marginal_pt_genetic(Z_tr, Z_te, cov_tr, y_tr, meta["block_id"], z_grid)
    s = stack_and_score(cov_tr, cov_te, g_tr, g_te, y_tr)
    results["pt"] = auc(y_te, s)
    log(f"[sep] marginal-P+T AUC={results['pt']:.4f}  (z*={zt} nsel={nsel})", flush=True)

    # --- SV-PGS (real engine) on TRAIN, score TEST ---
    d_train = dict(G=G_tr, Gstd=Z_tr, y=y_tr, cov=cov_full[tr], meta=meta,
                   truth=truth, cfg=d["cfg"], classnames=d["classnames"])
    log("[sep] fitting SV-PGS on train ...", flush=True)
    fit = fit_svpgs_generated(d_train, max_outer_iterations=max_iter_svpgs)
    log(f"[sep] SV-PGS fit done in {fit['wall_time_s']:.1f}s converged={fit['converged']}", flush=True)
    # genetic component on test (re-stack same as others for fair cov adjustment)
    model = fit["model"]
    g_tr_comp, _ = model.decision_components(G_tr, cov_tr)
    g_te_comp, _ = model.decision_components(G_te, cov_te)
    s = stack_and_score(cov_tr, cov_te, np.asarray(g_tr_comp, float),
                        np.asarray(g_te_comp, float), y_tr)
    results["svpgs"] = auc(y_te, s)
    # also the model's OWN full linear predictor (its native covariate handling)
    lp_te = model.decision_function(G_te, cov_te)
    results["svpgs_native"] = auc(y_te, np.asarray(lp_te, float))
    log(f"[sep] SV-PGS       AUC={results['svpgs']:.4f}  (native LP AUC={results['svpgs_native']:.4f})", flush=True)

    # --- normalized AUC ---
    base, ceil = results["naive"], results["oracle"]
    def norm(a):
        return (a - base) / (ceil - base) if ceil > base else float("nan")
    log("\n[sep] ===== HELD-OUT AUC SUMMARY =====", flush=True)
    for k in ["naive", "ridge", "pt", "svpgs", "svpgs_native", "oracle"]:
        log(f"[sep]   {k:14s} raw={results[k]:.4f}  norm={norm(results[k]):+.3f}", flush=True)

    # --- prior recovery ---
    cs = fit["classes_present"]
    true_lb = truth["class_log_baseline"]
    ytrue = [true_lb[c] for c in cs]
    x_off = [fit["type_offset_coef"][c] for c in cs]
    x_pv = [fit["per_class_mean_prior_var"][c] for c in cs]
    x_mb = [fit["per_class_mean_abs_beta"][c] for c in cs]
    sp_off = spearmanr(x_off, ytrue).statistic
    sp_pv = spearmanr(x_pv, ytrue).statistic
    sp_mb = spearmanr(x_mb, ytrue).statistic
    log("\n[sep] ===== PRIOR RECOVERY (Spearman vs true per-class log-baseline) =====", flush=True)
    log(f"[sep]   type_offset_coef      rho={sp_off:+.3f}", flush=True)
    log(f"[sep]   per-class prior_var   rho={sp_pv:+.3f}", flush=True)
    log(f"[sep]   per-class mean|beta|  rho={sp_mb:+.3f}", flush=True)
    log(f"[sep]   recovered global_scale={fit['global_scale']:.4g}", flush=True)
    # annotation-coefficient recovery (length +, repeat -)
    cb = fit["scale_model_coef_by_name"]
    len_lin = cb.get("continuous_spline::sv_log_length::basis_0", float("nan"))
    rep_true = cb.get("factor_level::repeat_overlap::true", float("nan"))
    log(f"[sep]   sv_log_length linear coef = {len_lin:+.3f} (true gamma_len={truth['gamma_len']:+.2f})", flush=True)
    log(f"[sep]   repeat_overlap::true coef = {rep_true:+.3f} (true gamma_repeat={truth['gamma_repeat']:+.2f})", flush=True)

    # SV-vs-SNV effect scale check
    sv_classes = [c for c in cs if c not in ("snv", "small_indel")]
    snv_off = fit["type_offset_coef"].get("snv", float("nan"))
    sv_off_mean = np.nanmean([fit["type_offset_coef"][c] for c in sv_classes]) if sv_classes else float("nan")
    log(f"[sep]   SNV type_offset={snv_off:+.3f}  mean SV type_offset={sv_off_mean:+.3f}  "
        f"(SV>SNV: {'YES' if sv_off_mean > snv_off else 'NO'})", flush=True)

    log(f"\n[sep] total wall={time.perf_counter()-t_start:.1f}s  (SV-PGS fit={fit['wall_time_s']:.1f}s)", flush=True)
    return results, fit


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(N=N, P=P)
