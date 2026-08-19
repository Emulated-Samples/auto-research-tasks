"""Category robustness harness: over >=5 seeds per category, compare the REAL
SV-PGS engine against the class-blind / marginal cheats and the oracle on
HELD-OUT prediction, and report the normalized accuracy separation.

Every method emits a genetic score that is stacked with covariates through the
same logistic refit, then evaluated by held-out rank-AUC normalized between the
naive and oracle anchors.

Run ONE heavy category at a time (memory). CLI:
  python validation/robustness.py <category> [n_seeds] [N] [P]
  python validation/robustness.py ALL 5 3000 800
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_SVPGS = os.environ.get("SVPGS_HOME", os.path.join(os.path.dirname(_ROOT), "SV-PGS"))
if os.path.isdir(_SVPGS):
    sys.path.insert(0, _SVPGS)

from datagen.dgp import generate, DGPConfig            # noqa: E402
from datagen.categories import CATEGORIES              # noqa: E402
from reference.run_svpgs import fit_svpgs_generated    # noqa: E402
from validation.reference_separation import (          # noqa: E402
    sha256_test_mask, standardize_train, auc,
    stack_and_score, uniform_ridge_genetic, marginal_pt_genetic,
)


# ---------------------------------------------------------------------------
def eval_one(cfg: DGPConfig, family: str, *, max_iter_svpgs=25, log=print):
    """One dataset/seed: returns dict of normalized scores + raw + timings."""
    if family != "binomial-logit":
        raise ValueError(f"unsupported family {family!r}")
    d = generate(cfg)
    meta, truth = d["meta"], d["truth"]
    G = np.asarray(d["G"], np.float32)
    cov_full = np.asarray(d["cov"], np.float32)
    y = np.asarray(d["y"])
    P = G.shape[1]

    cov = cov_full[:, 1:] if np.allclose(cov_full[:, 0], 1.0) else cov_full
    te = sha256_test_mask(len(y))
    tr = ~te
    G_tr, G_te = G[tr], G[te]
    cov_tr, cov_te = cov[tr], cov[te]
    y_tr, y_te = y[tr], y[te]
    Z_tr, Z_te, mu, sd = standardize_train(G_tr, G_te)

    res = {}
    def score(gt, ge):
        return auc(y_te, stack_and_score(cov_tr, cov_te, gt, ge, y_tr))

    zt = np.zeros(len(y_tr))
    ze = np.zeros(len(y_te))
    res["naive"] = score(zt, ze)
    res["oracle"] = score(
        Z_tr @ truth["beta_effective"], Z_te @ truth["beta_effective"])

    lam_grid = [0.5, 2.0, 8.0, 32.0, 128.0, 512.0]
    z_grid = [0.0, 1.0, 1.64, 2.0, 2.58, 3.0]
    g_tr, g_te, lam = uniform_ridge_genetic(Z_tr, Z_te, cov_tr, y_tr, lam_grid)
    res["ridge"] = score(g_tr, g_te)
    g_tr, g_te, zsel, nsel = marginal_pt_genetic(
        Z_tr, Z_te, cov_tr, y_tr, meta["block_id"], z_grid)
    res["pt"] = score(g_tr, g_te)

    d_train = dict(G=G_tr, Gstd=Z_tr, y=y_tr, cov=cov_full[tr], meta=meta,
                   truth=truth, cfg=cfg, classnames=d["classnames"])
    t0 = time.perf_counter()
    fit = fit_svpgs_generated(d_train, max_outer_iterations=max_iter_svpgs)
    fit_wall = time.perf_counter() - t0
    model = fit["model"]
    g_tr_c, _ = model.decision_components(G_tr, cov_tr)
    g_te_c, _ = model.decision_components(G_te, cov_te)
    res["svpgs"] = score(np.asarray(g_tr_c, float), np.asarray(g_te_c, float))

    base, ceil = res["naive"], res["oracle"]
    span = ceil - base
    norm = {k: (res[k] - base) / span if span > 1e-9 else float("nan")
            for k in ["naive", "ridge", "pt", "svpgs", "oracle"]}
    margin = norm["svpgs"] - max(norm["ridge"], norm["pt"])
    out = dict(seed=cfg.seed, P=P, raw=res, norm=norm, margin=margin,
               fit_wall=fit_wall, gap=span, case_rate=float(np.mean(y)))
    log(f"[{cfg.seed}] P={P} raw(naive/ridge/pt/svpgs/oracle)="
        f"{res['naive']:.4f}/{res['ridge']:.4f}/{res['pt']:.4f}/{res['svpgs']:.4f}/{res['oracle']:.4f}"
        f"  norm svpgs={norm['svpgs']:+.3f} ridge={norm['ridge']:+.3f} pt={norm['pt']:+.3f}"
        f"  MARGIN={margin:+.3f}  gap={span:.4f}  fit={fit_wall:.1f}s", flush=True)
    return out


def run_category(name, n_seeds=5, N=3000, P=800, seed0=100, log=print):
    cat = CATEGORIES[name]
    log(f"\n===== CATEGORY {name} ({cat['family']}) =====", flush=True)
    log(f"      {cat['desc']}", flush=True)
    runs = []
    for s in range(seed0, seed0 + n_seeds):
        cfg = cat["make_cfg"](s, N, P)
        runs.append(eval_one(cfg, cat["family"], log=log))
    ms = np.array([r["margin"] for r in runs])
    sv = np.array([r["norm"]["svpgs"] for r in runs])
    rd = np.array([r["norm"]["ridge"] for r in runs])
    pt = np.array([r["norm"]["pt"] for r in runs])
    summary = dict(
        category=name, family=cat["family"],
        mean_margin=float(ms.mean()), min_margin=float(ms.min()),
        mean_svpgs=float(sv.mean()), mean_ridge=float(rd.mean()), mean_pt=float(pt.mean()),
        mean_gap=float(np.mean([r["gap"] for r in runs])),
        mean_fit_s=float(np.mean([r["fit_wall"] for r in runs])),
        seeds=[r["seed"] for r in runs],
        seed_margins=[round(m, 3) for m in ms],
        seed_svpgs=[round(x, 3) for x in sv],
        passes=bool(ms.mean() >= 0.10 and sv.mean() > 0 and ms.min() > 0),
    )
    log(f"[SUMMARY {name}] mean margin={ms.mean():+.3f} (min {ms.min():+.3f})  "
        f"mean svpgs={sv.mean():+.3f} ridge={rd.mean():+.3f} pt={pt.mean():+.3f}  "
        f"PASS={summary['passes']}", flush=True)
    return summary


def _run_one_cli(category, seed, N, P, outfile):
    """Run a SINGLE (category, seed) in a short-lived process and append one JSON
    line to outfile. Resilient to OOM: if killed, only this seed is lost and the
    driver retries it; completed seeds persist on disk."""
    cat = CATEGORIES[category]
    cfg = cat["make_cfg"](seed, N, P)
    r = eval_one(cfg, cat["family"], log=print)
    rec = dict(category=category, seed=seed, family=cat["family"],
               margin=r["margin"], gap=r["gap"], fit_wall=r["fit_wall"],
               norm=r["norm"], raw=r["raw"])
    with open(outfile, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[ONE-DONE] {category} seed={seed} margin={r['margin']:+.3f}", flush=True)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    if len(sys.argv) > 1 and sys.argv[1] == "ONE":
        _run_one_cli(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                     int(sys.argv[5]), sys.argv[6])
        sys.exit(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    P = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    names = list(CATEGORIES) if which == "ALL" else [which]
    summaries = []
    for nm in names:
        summaries.append(run_category(nm, n_seeds=n_seeds, N=N, P=P))
    print("\n\n########## OVERALL SEPARATION TABLE ##########", flush=True)
    print(f"{'category':22s} {'svpgs':>7s} {'ridge':>7s} {'pt':>7s} "
          f"{'margin':>7s} {'min':>7s} {'gap':>6s} {'fit_s':>6s}  PASS", flush=True)
    for s in summaries:
        print(f"{s['category']:22s} {s['mean_svpgs']:+7.3f} "
              f"{s['mean_ridge']:+7.3f} {s['mean_pt']:+7.3f} {s['mean_margin']:+7.3f} "
              f"{s['min_margin']:+7.3f} {s['mean_gap']:6.3f} {s['mean_fit_s']:6.1f}  "
              f"{'YES' if s['passes'] else 'no'}", flush=True)
    with open(os.path.join(_ROOT, "validation", "robustness_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print("\nwrote validation/robustness_summary.json", flush=True)
