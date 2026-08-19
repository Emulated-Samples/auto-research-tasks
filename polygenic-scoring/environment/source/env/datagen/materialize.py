"""Materialize a generate() dataset into the agent-facing on-disk task format,
plus a hidden truth bundle for the grader.

Agent-visible files (written under <out>/public/):
  family.txt            binomial-logit
  formula.txt           y ~ pgs(<annotation cols>) + covariate(<cov cols>)
  dgp.json              exact machine-readable family and column contract
  variant_metadata.tsv  keyed by variant_id: variant_class + annotation columns
                        (+ soft-membership / nested columns for those categories)
  covariates_train.csv  sample_id, <covariate cols>, y
  genotypes_train.tsv   sample_id + one column per variant_id (dosage 0/1/2)
  covariates_test.csv   sample_id, <covariate cols>            (NO y)
  genotypes_test.tsv    sample_id + one column per variant_id

Hidden files (written under <out>/truth/, never given to the agent):
  y_test.csv            held-out phenotype (for accuracy scoring)
  truth.npz             generative latents (for analysis/provenance), stored with a
                        CLEAN raw-vs-effective split:
                          raw:       beta_raw, tau2_raw, s, global_scale_raw
                                     (NON-identifiable -- see below), class
                                     log-baselines, gamma_len, gamma_repeat;
                                     TPB latents only for families that use them
                          effective: beta_effective, alpha_effective,
                                     tau2_effective, intercept_effective,
                                     heritability, class labels

Program contract the agent implements (documented in the task instruction):
  fit     reads public/*  (train split) -> writes model.out
  predict reads model.out + public test files -> writes pred.csv
Grading is OUTCOME-ONLY: held-out predictive accuracy (AUC / Brier / log-loss).
There is NO prior-recovery reward; the truth bundle is for corpus validation and
analysis, not for grading model.out.
"""
from __future__ import annotations

import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.dgp import generate, DGPConfig

# covariate column names in the same order as dgp.generate builds `cov`:
# [intercept, age_s, age_s^2, sex, eth1, eth2, PC1..PC{n_pcs}]. The PC block width
# tracks cfg.n_pcs (configurable), so derive the names from the actual count rather
# than hardcoding 10 (a non-default n_pcs would otherwise mismatch the row width).
COV_FIXED_NAMES = ["age", "age2", "sex", "eth1", "eth2"]


def _cov_names(n_pcs: int):
    return COV_FIXED_NAMES + [f"PC{i + 1}" for i in range(n_pcs)]


def _split(n, frac_train, seed, pc1=None, ancestry_shift=0.0):
    """Train/test index split. Default: random permutation. When `ancestry_shift`
    > 0 and PC1 is given, the split is TILTED by ancestry (PC1): the test cohort is
    enriched for the high-PC1 tail while train leans low-PC1, so train and test have
    DIFFERENT ancestry distributions. `ancestry_shift` blends the PC1 rank against
    noise (larger = stronger, more deterministic shift), so covariate adjustment
    must generalize across the shift rather than overfit the train ancestry."""
    rng = np.random.default_rng(seed + 999)
    ntr = int(n * frac_train)
    if ancestry_shift > 0.0 and pc1 is not None:
        z = (np.asarray(pc1, float) - np.mean(pc1)) / (np.std(pc1) + 1e-12)
        score = ancestry_shift * z + rng.standard_normal(n)
        order = np.argsort(score)          # low PC1 -> high PC1 (by tilted score)
        return order[:ntr], order[ntr:]    # train = low tail, test = high tail
    idx = rng.permutation(n)
    return idx[:ntr], idx[ntr:]


def _write_dgp_json(
    path,
    *,
    family,
    prior_cols,
    annotation_types,
    cov_cols,
):
    """Write the exact machine-readable public column and link contract."""
    annotation_types = dict(annotation_types)
    if set(annotation_types) != set(prior_cols):
        raise ValueError("annotation_types must describe every pgs annotation exactly")
    allowed_annotation_types = {
        "variant_class",
        "class_members",
        "class_membership",
        "binary",
        "continuous",
        "categorical",
    }
    invalid_types = sorted(set(annotation_types.values()) - allowed_annotation_types)
    if invalid_types:
        raise ValueError(f"unsupported annotation types: {invalid_types}")
    if annotation_types.get("variant_class") != "variant_class":
        raise ValueError("variant_class must use annotation type variant_class")
    class_members = annotation_types.get("prior_class_members") == "class_members"
    class_membership = (
        annotation_types.get("prior_class_membership") == "class_membership"
    )
    if class_members != class_membership:
        raise ValueError("soft class members and membership annotations must be paired")
    spec = {
        "family": family,
        "formula": {"response": "y", "pgs_annotations": list(prior_cols),
                    "annotation_types": annotation_types,
                    "covariates": list(cov_cols)},
    }
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)


def _declared_annotation_types(prior_cols, metadata_columns):
    """Return the explicit public type contract for every formula annotation."""
    special_types = {
        "variant_class": "variant_class",
        "prior_class_members": "class_members",
        "prior_class_membership": "class_membership",
        "repeat_overlap": "binary",
    }
    types = {}
    for name in prior_cols:
        if name not in metadata_columns:
            raise ValueError(f"declared annotation {name!r} is absent from metadata")
        if name in special_types:
            types[name] = special_types[name]
            continue
        values = np.asarray(metadata_columns[name])
        if values.dtype.kind in "iufb":
            if not np.all(np.isfinite(values.astype(float))):
                raise ValueError(f"declared annotation {name!r} contains non-finite values")
            types[name] = "continuous"
        else:
            types[name] = "categorical"
    return types


def materialize(dataset: dict, out_dir: str, *, family: str,
                frac_train: float,
                prior_cols=("variant_class", "sv_log_length", "repeat_overlap"),
                seed: int = 0,
                cov_include=None, extra_cov_cols=None,
                ancestry_shift=0.0):
    public_dir = os.path.join(out_dir, "public")
    truth_dir = os.path.join(out_dir, "truth")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)

    def pub(filename):
        return os.path.join(public_dir, filename)

    def tru(filename):
        return os.path.join(truth_dir, filename)

    G = dataset["G"]                 # raw dosages N x P
    cov = dataset["cov"]            # N x (1+len(COV_NAMES)); col0 = intercept
    y = dataset["y"]
    meta = dataset["meta"]
    cfg = dataset["cfg"]
    truth = dataset["truth"]
    N, P = G.shape
    vid = meta["variant_id"]

    # PC1 (= ancestry axis) is cov column 6 (intercept, age, age^2, sex, eth1,
    # eth2, PC1, ...); used for an optional ancestry-shifted train/test split.
    pc1 = cov[:, 6] if cov.shape[1] > 6 else None
    cohort = dataset["cohort"]
    if cfg.ld_shift or cfg.nuisance_cov_cols > 0:
        # The generator built these cohorts to DIFFER: different block correlation
        # matrices (ld_shift) and/or an opposite-signed batch artifact
        # (nuisance_cov_cols). The split must follow the cohort labels -- a random
        # split would blend the two regimes and erase the mechanism.
        if ancestry_shift:
            raise ValueError(
                "cohort-generated datasets and ancestry_shift splits are exclusive")
        tr = np.where(np.asarray(cohort) == 0)[0]
        te = np.where(np.asarray(cohort) == 1)[0]
        if abs(len(tr) - int(N * frac_train)) > 1:
            raise ValueError(
                f"cohort split (n_train={len(tr)}) disagrees with "
                f"frac_train={frac_train} (cfg.frac_train_hint={cfg.frac_train_hint})")
    else:
        tr, te = _split(N, frac_train, seed, pc1=pc1, ancestry_shift=ancestry_shift)
    sid = np.array([f"s{i}" for i in range(N)])

    # ---- variant_metadata.tsv (annotation columns) ----
    # emit length & repeat as CUSTOM columns (reserved fields don't enter the
    # SV-PGS scale model; custom ones do -- ref doc 03 sec 2C).
    log_len = np.log10(np.maximum(meta["length"], 1.0))
    log_len_z = (log_len - log_len.mean()) / (log_len.std() + 1e-12)
    cols = {
        "variant_id": vid,
        "variant_class": meta["variant_class"],
        "sv_log_length": np.round(log_len_z, 5),
        "repeat_overlap": meta["is_repeat"].astype(int),
        "allele_frequency": np.round(meta["allele_frequency"], 5),
        "is_copy_number": meta["is_copy_number"].astype(int),
    }
    # optional soft-membership / nested columns added by category-specific callers
    for extra in ("prior_class_members", "prior_class_membership", "gene_context"):
        if extra in meta:
            cols[extra] = meta[extra]
    # Any other annotation column a category exposes via prior_cols but not covered
    # above is copied through without interpreting its name or intended role.
    for k in prior_cols:
        if k not in cols and k in meta:
            cols[k] = meta[k]
    header = list(cols)
    with open(pub("variant_metadata.tsv"), "w") as fh:
        fh.write("\t".join(header) + "\n")
        for j in range(P):
            fh.write("\t".join(str(cols[k][j]) for k in header) + "\n")

    # ---- covariate tables (drop intercept col0; agent adds its own) ----
    covM = cov[:, 1:]
    cov_names = _cov_names(cfg.n_pcs)   # PC block width tracks the config
    assert covM.shape[1] == len(cov_names), (
        f"covariate width {covM.shape[1]} != {len(cov_names)} names (n_pcs={cfg.n_pcs})")
    # The FORMULA lists only `cov_include` (default: every generative covariate).
    # A conformance/decoy category passes a strict subset here, so a submission
    # that respects the formula uses only these while extra columns are written but
    # not listed. Validate the subset names.
    cov_include = list(cov_names if cov_include is None else cov_include)
    unknown = [c for c in cov_include if c not in cov_names]
    if unknown:
        raise ValueError(f"cov_include names not among covariates: {unknown}")
    # `extra_cov_cols`: length-N arrays written to the covariate table but NOT listed
    # in the formula (CONTRACT-002). They come from the GENERATOR (which draws the
    # batch artifact independently and then generates y from it) or from an explicit
    # caller override. Materialization never derives a public column from the held-out
    # labels: changing y_test must leave every public byte unchanged.
    if extra_cov_cols is None:
        extra_cov_cols = dict(dataset.get("extra_cov_cols") or {})
    else:
        extra_cov_cols = dict(extra_cov_cols)
    for name, arr in extra_cov_cols.items():
        if name in cov_names:
            raise ValueError(f"extra_cov_col {name!r} collides with a real covariate")
        if len(arr) != N:
            raise ValueError(f"extra_cov_col {name!r} length {len(arr)} != N={N}")
    written_names = cov_names + list(extra_cov_cols)   # real cols then decoys
    def write_cov(path, rows, with_y):
        with open(path, "w") as fh:
            head = ["sample_id"] + written_names + (["y"] if with_y else [])
            fh.write(",".join(head) + "\n")
            for i in rows:
                vals = [sid[i]] + [f"{v:.6g}" for v in covM[i]]
                vals += [f"{float(extra_cov_cols[name][i]):.6g}" for name in extra_cov_cols]
                if with_y:
                    vals.append(f"{y[i]:.6g}")
                fh.write(",".join(vals) + "\n")
    write_cov(pub("covariates_train.csv"), tr, with_y=True)
    write_cov(pub("covariates_test.csv"), te, with_y=False)

    # ---- genotype tables (samples x variants, dosage) ----
    def write_geno(path, rows):
        with open(path, "w") as fh:
            fh.write("sample_id\t" + "\t".join(vid) + "\n")
            for i in rows:
                fh.write(sid[i] + "\t" + "\t".join(str(int(g)) for g in G[i]) + "\n")
    write_geno(pub("genotypes_train.tsv"), tr)
    write_geno(pub("genotypes_test.tsv"), te)

    # ---- family / formula contract ----
    with open(pub("family.txt"), "w") as fh:
        fh.write(family + "\n")
    with open(pub("formula.txt"), "w") as fh:
        fh.write(f"y ~ pgs({' + '.join(prior_cols)}) + "
                 f"covariate({', '.join(cov_include)})\n")
    # Machine-readable model spec lists only observable, load-bearing columns.
    _write_dgp_json(
        pub("dgp.json"),
        family=family,
        prior_cols=prior_cols,
        annotation_types=_declared_annotation_types(prior_cols, cols),
        cov_cols=cov_include,
    )

    # ---- hidden truth for the grader ----
    with open(tru("y_test.csv"), "w") as fh:
        fh.write("sample_id,y\n")
        for i in te:
            fh.write(f"{sid[i]},{y[i]:.6g}\n")
    classnames = sorted(set(meta["variant_class"]))
    # truth.npz stores CLEARLY SEPARATED raw- vs effective-scale fields (no longer a
    # single "phenotype-scale" claim mixing the two axes):
    #   * RAW generative constants -- beta_raw, tau2_raw, s, global_scale_raw
    #     (NON-identifiable: it cancels under the beta/sqrt(gvar) renormalization, so
    #     it is provenance only, never a recovery target), the spread-adjusted
    #     per-class log baselines the DGP used, gamma_len, gamma_repeat.
    #   * EFFECTIVE (phenotype/liability axis) fields that ACTUALLY generated y --
    #     beta_effective, alpha_effective, tau2_effective (= k^2*tau2_raw, so
    #     beta_effective ~ N(0, tau2_effective) holds) and intercept_effective (with
    #     the covariate-centering constant folded in, so intercept_effective +
    #     C@alpha_effective + Gstd@beta_effective reconstructs the linear predictor).
    # Grading is outcome-only; these are for validation/analysis, not model.out.
    adj_baseline = truth["class_log_baseline"]
    archive = {
        "variant_class": meta["variant_class"],
        # ---- raw-scale generative constants ----
        "beta_raw": truth["beta_raw"],
        "tau2_raw": truth["tau2_raw"],
        "s": truth["s"],
        "global_scale_raw": truth["global_scale_raw"],
        "class_log_baseline": json.dumps({
            c: float(adj_baseline[c]) for c in classnames
        }),
        "gamma_len": truth["gamma_len"],
        "gamma_repeat": truth["gamma_repeat"],
        # ---- effective (phenotype/liability axis) fields ----
        "beta_effective": truth["beta_effective"],
        "alpha_effective": truth["alpha_effective"],
        "tau2_effective": truth["tau2_effective"],
        "intercept_effective": truth["intercept_effective"],
        "heritability": truth["heritability"],
        # ---- effect provenance ----
        "causal_mask": truth["causal_mask"],
        "effect_family": truth["effect_family"],
        "beta_dense_effective": truth["beta_dense_effective"],
        "beta_spike_effective": truth["beta_spike_effective"],
        "spike_mask": truth["spike_mask"],
        "suppressor_target": truth["suppressor_target"],
        "suppressor_partner": truth["suppressor_partner"],
        "ld_shift": truth["ld_shift"],
        "train_idx": tr,
        "test_idx": te,
        "family": family,
    }
    for name in ("lam", "shape_a", "shape_b"):
        if name in truth:
            archive[name] = truth[name]
    np.savez(tru("truth.npz"), **archive)
    return {"n_train": len(tr), "n_test": len(te), "n_variants": P,
            "public": public_dir, "truth": truth_dir}
