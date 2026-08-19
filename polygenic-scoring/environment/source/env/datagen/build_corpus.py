"""Build the categorized SV-PGS benchmark corpus + grader anchors.

Per dataset it generates + materializes the task (public/ + truth/), fits every
anchor from immutable training data, loads prediction features only after all
fits, and opens truth/y_test.csv only after every prediction is complete:
  metrics_naive     = covariates-only GLM
  metrics_reference = exact public NumPy gold/fit + gold/predict execution over
                      {hierarchical_eb, ridge_logistic}, selected on a deterministic
                      training-inner-validation split and refit on full training.
and writes a schema-v8 truth/anchors.json with an opaque dataset identity, method-aware
reference fit provenance/convergence (reference_method + inner_selection_auc + the
complete fit/predict source, model, output, exit, and timing identity), and
naive/reference metrics.  The trusted builder never calls the prediction math in
process: it scores the actual pred.csv bytes emitted by gold/predict under the same
networkless 30-second sandbox contract as a submission.
Metrics: {auc, brier, log_loss} (AUC is the only positive-credit scored metric).

Each dataset is built in its own short-lived subprocess (CLI `ONE`), driven by
build_corpus.sh; `FINALIZE` assigns every dataset the same weight and writes a
schema-v8 corpus/manifest.json containing SHA-256 digests for every public/truth file.

CLI:
  python datagen/build_corpus.py --key-file <absolute-path> ONE <category> <replicate>
  python datagen/build_corpus.py --key-file <absolute-path> CHECK <category> <replicate>
  python datagen/build_corpus.py --key-file <absolute-path> CHECK_FINALIZED <category> <replicate>
  python datagen/build_corpus.py --key-file <absolute-path> FINALIZE
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from datagen.dgp import generate                     # noqa: E402
from datagen.materialize import materialize          # noqa: E402
from datagen.categories import CATEGORIES            # noqa: E402
from grader.perf import calibration_seconds          # noqa: E402
# Score anchors through the SAME metric functions the grader uses at grade time
# (grader/metrics.py) so naive/reference anchors are directly comparable to
# submissions -- there is no private, divergently-clipped metric path (fix 12).
from grader.metrics import compute_metrics           # noqa: E402
from grader.truth import read_aligned_binary_truth_csv  # noqa: E402
from grader.submission_runner import run_prefit_predict  # noqa: E402
from grader.contract import (                        # noqa: E402
    ANCHOR_HMAC_FIELD,
    CORPUS_SCHEMA_VERSION,
    CORPUS_PURPOSE,
    CORPUS_BUILDER_CODE_VERSION,
    DATASET_WEIGHT,
    FIT_TIMEOUT_S,
    REFERENCE_FIT_FEASIBILITY_FRACTION,
    REFERENCE_FIT_HEADROOM_FRACTION,
    DEVELOPMENT_REPORT_HMAC_FIELD,
    DEVELOPMENT_REPORT_RELATIVE_PATH,
    GENERATION_CONTRACT_ITEMS,
    GENERATION_PIPELINE_RELATIVE_PATHS,
    MANIFEST_HMAC_FIELD,
    METRIC_WEIGHTS,
    MODEL_ZOO_REPORT_SCHEMA_VERSION,
    REPLICATES_PER_CATEGORY,
    SHIPPED_CATEGORY_COUNT,
    SHIPPED_DATASET_COUNT,
    SHIPPED_REQUESTED_N,
    SHIPPED_REQUESTED_P,
    WEIGHT_RULE,
)
from grader.corpus_auth import (                      # noqa: E402
    ANCHOR_HMAC_DOMAIN,
    DEVELOPMENT_REPORT_HMAC_DOMAIN,
    MANIFEST_HMAC_DOMAIN,
    corpus_key_id,
    derive_stream_seed,
    json_hmac_sha256,
    load_corpus_key,
    opaque_dataset_id,
    verify_json_hmac,
)
from grader.skill import (  # noqa: E402
    K_SE,
    RESOLUTION_K,
    reference_calibration_qualification,
)
from reference.protocol import (                    # noqa: E402
    REFERENCE_ANCHOR_EXECUTION_ORDER,
    REFERENCE_CALIBRATION_MAX_ITERATIONS,
    REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE,
    REFERENCE_ENTRYPOINT,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
    REFERENCE_IMPLEMENTATION,
    REFERENCE_PROTOCOL_VERSION,
    REFERENCE_NUMPY_VERSION,
    REFERENCE_PYTHON_EXECUTABLE,
    REFERENCE_SOURCE_RELATIVE_PATHS,
    validate_reference_diagnostics,
)

CORPUS = os.path.join(_ROOT, "corpus")
METRIC_DIRECTIONS = {"auc": True, "brier": False, "log_loss": False}


class ReferenceFitError(RuntimeError):
    """The declared strong-reference protocol was invalid or unconverged.

    A reference failure invalidates the corpus build. It must never select a more
    favorable early-stopped fit or silently omit the failed dataset.
    """


# ---- metric helpers --------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _score_metrics(family, y, mean):
    """Score an anchor prediction through grader/metrics.py -- the EXACT path used
    at grade time (fix 12). ``mean`` is the predicted probability. Returns
    {name: scalar} (the anchor schema), dropping the higher_is_better flags
    compute_metrics attaches."""
    pred = {"mean": np.asarray(mean, float)}
    return {k: float(v[0]) for k, v in compute_metrics(family, np.asarray(y), pred).items()}


def _validate_anchor_quality(dataset_id, naive, reference, ref_naive_se, naive_se):
    """Require at least one metric whose anchor gap is BOTH real AND large enough
    to be a skill denominator -- applying the grader's exact activation rule.

    This gate previously asked only "does the reference reliably beat naive?"
    (``gap >= K_SE * SE_paired``). That is a test of EXISTENCE, and it shipped a
    corpus in which five of fifteen categories carried a naive->reference AUC gap
    of 0.0047-0.0208 -- real, precisely measured, and far too small to divide by.
    Because ``skill = (sub - naive)/gap``, those categories amplified ordinary
    held-out noise until a 1.5-second marginal cheat, a uniform ridge and the
    from-scratch gold arm ALL pinned at +1.500 AUC skill and became
    indistinguishable. Half of the cheat's headline score was minted there. See
    rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md.

    The reliability test could not have caught it and raising K_SE could not
    either: the paired SE shrinks WITH the gap (corr = +0.925 over the 45 shipped
    datasets), so the corpus's smallest gap carried its HIGHEST z. Adequacy needs
    an absolute yardstick, and ``naive_se`` -- the SE of the single naive estimate
    -- is one, because it does not collapse when the reference does.

    A dataset with no adequate metric is a BROKEN ANCHOR: nothing about a
    submission can be measured on it. It fails the build LOUDLY rather than
    shipping as a category that silently mints free credit.
    """
    expected = set(METRIC_DIRECTIONS)
    if set(naive) != expected or set(reference) != expected:
        raise ReferenceFitError(
            f"{dataset_id}: anchor metric keys do not match {sorted(expected)}"
        )
    if set(ref_naive_se) != expected:
        raise ReferenceFitError(
            f"{dataset_id}: reference-gap SE keys do not match {sorted(expected)}"
        )
    if set(naive_se) != expected:
        raise ReferenceFitError(
            f"{dataset_id}: naive single-estimate SE keys do not match "
            f"{sorted(expected)}"
        )
    active_metrics, inadequate = [], []
    for name, higher_is_better in METRIC_DIRECTIONS.items():
        naive_value = float(naive[name])
        reference_value = float(reference[name])
        standard_error = float(ref_naive_se[name])
        single_se = float(naive_se[name])
        oriented_gap = (
            reference_value - naive_value
            if higher_is_better
            else naive_value - reference_value
        )
        if not (
            np.isfinite(naive_value)
            and np.isfinite(reference_value)
            and np.isfinite(standard_error)
            and standard_error > 0.0
            and np.isfinite(single_se)
            and single_se > 0.0
        ):
            raise ReferenceFitError(
                f"{dataset_id}: reference anchor contains non-finite metric data"
            )
        reliability_threshold = K_SE * standard_error
        adequacy_threshold = RESOLUTION_K * single_se
        if oriented_gap <= -reliability_threshold:
            print(
                f"[warn] {dataset_id}: reference reliably worse than naive on {name} "
                f"({naive_value:.6f} -> {reference_value:.6f}); metric excluded, as "
                f"the grader would",
                flush=True,
            )
        elif oriented_gap <= 1e-9 or oriented_gap < reliability_threshold:
            pass  # not a real gap; the grader excludes it too
        elif oriented_gap < adequacy_threshold:
            # Real, but too small to be a denominator. Name it explicitly: this is
            # the case that used to pass silently and mint clamped free credit.
            inadequate.append(
                f"{name} (gap {oriented_gap:.6f} < {RESOLUTION_K:g} x "
                f"se_naive {single_se:.6f} = {adequacy_threshold:.6f})"
            )
        else:
            active_metrics.append(name)
    if not active_metrics:
        detail = f"; inadequate: {', '.join(inadequate)}" if inadequate else ""
        raise ReferenceFitError(
            f"{dataset_id}: BROKEN ANCHOR -- no reference metric is both reliably "
            f"better than naive and large enough to be a skill denominator{detail}"
        )
    # Match the SCORING predicate: accuracy_skill weights only metrics in
    # METRIC_WEIGHTS (AUC). An anchor whose AUC gap is inadequate but whose brier gap
    # is adequate would pass the check above yet score 0 for every submission and the
    # reference alike. Require at least one *weighted* metric to be adequate.
    weighted_active = [m for m in active_metrics if METRIC_WEIGHTS.get(m, 0.0) > 0.0]
    if not weighted_active:
        raise ReferenceFitError(
            f"{dataset_id}: BROKEN ANCHOR -- no SCORED reference metric (weight>0) is "
            f"large enough to be a skill denominator; active but unweighted: "
            f"{', '.join(active_metrics)}"
        )


def _validate_reference_feasibility(dataset_id, time_reference):
    """The reference's own fit must fit in the budget a SUBMISSION is given.

    ``grader/contract.py`` states this invariant in prose -- the exact public
    reference must finish well inside the same cap given to a submission, otherwise
    skill=1.0 is unreachable by construction. The v6 corpus shipped two datasets
    whose hidden SV-PGS reference fit
    exceeds FIT_TIMEOUT_S outright (198.2 s and 188.0 s against a 170 s cap), where
    a submission that exactly reproduced the reference is SIGKILLed and scores the
    invalid floor. An anchor nobody can reach is not an anchor.

    Fails the build LOUDLY at the cap; warns in the headroom band, because the cap
    is measured on the BUILD host and grading runs on a smaller one.
    """
    if type(time_reference) not in (int, float) or not np.isfinite(float(time_reference)):
        raise ReferenceFitError(f"{dataset_id}: reference fit time is not a finite number")
    seconds = float(time_reference)
    if seconds <= 0.0:
        raise ReferenceFitError(f"{dataset_id}: reference fit time must be positive")
    hard_limit = REFERENCE_FIT_FEASIBILITY_FRACTION * FIT_TIMEOUT_S
    if seconds >= hard_limit:
        raise ReferenceFitError(
            f"{dataset_id}: UNREACHABLE ANCHOR -- the reference fit took "
            f"{seconds:.1f}s against a {FIT_TIMEOUT_S}s submission fit cap, so a "
            f"submission reproducing it is SIGKILLed and skill=1.0 cannot be "
            f"reached on this dataset"
        )
    warn_limit = REFERENCE_FIT_HEADROOM_FRACTION * FIT_TIMEOUT_S
    if seconds >= warn_limit:
        print(
            f"[warn] {dataset_id}: reference fit took {seconds:.1f}s = "
            f"{seconds / FIT_TIMEOUT_S:.0%} of the {FIT_TIMEOUT_S}s submission cap. "
            f"Timed on the BUILD host; grading runs on t3.xlarge/4 vCPU, so the "
            f"exact public program must also be witnessed in the target sandbox.",
            flush=True,
        )


def _validate_reference_protocol(dataset_id, diagnostics, training_count):
    """Require auditable training-only selection diagnostics for the reference."""
    try:
        validate_reference_diagnostics(
            diagnostics,
            training_count=training_count,
            execution_order=REFERENCE_ANCHOR_EXECUTION_ORDER,
        )
    except ValueError as exc:
        raise ReferenceFitError(
            f"{dataset_id}: invalid training-only reference diagnostics"
        ) from exc


def _bootstrap_anchor_ses(family, y, naive_prob, ref_prob, *, n_boot=300, seed=0):
    """Bootstrap BOTH anchor standard errors from one resampling loop.

    Returns ``(ref_naive_se, naive_se)``:

    * ``ref_naive_se`` -- SD of the oriented (reference - naive) GAP across
      resamples. Answers "is the gap REAL?" and feeds skill.metric_skill's
      reliability gate (K_SE).
    * ``naive_se`` -- SD of the SINGLE naive estimate across resamples. Answers
      "is the gap BIG ENOUGH TO DIVIDE BY?" and feeds the adequacy gate
      (skill.RESOLUTION_K).

    THESE ARE DIFFERENT QUESTIONS AND ONLY THE SECOND ONE PROTECTS THE SCORE.
    The paired gap SE is not a usable yardstick for size: as a reference collapses
    ONTO naive, the gap and its PAIRED SE shrink together, so ``gap/SE_paired`` is
    near scale-invariant. Measured over the 45 datasets of the shipped v6 corpus
    (rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md):
        corr(gap, SE_paired) = +0.925,  corr(gap, gap/SE_paired) = +0.121
    and the corpus's SMALLEST AUC gap (+0.0019) carried a HIGHER reliability z
    (17.5) than its LARGEST (+0.1035, z=16.2). The single-estimate ``naive_se`` is
    a property of the data and the naive predictor alone and does NOT shrink when
    the reference collapses, so it is the yardstick that works.

    The naive metrics are computed on every resample regardless -- the previous
    version of this function discarded them after differencing -- so the second SE
    costs nothing.

    Oriented by the metric's higher_is_better sign so the gap is a genuine
    'improvement'. ``naive_se`` needs no orientation: it is a dispersion.
    """
    y = np.asarray(y, float)
    naive_prob = np.asarray(naive_prob, float)
    ref_prob = np.asarray(ref_prob, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in METRIC_DIRECTIONS}
    naive_values = {k: [] for k in METRIC_DIRECTIONS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # A degenerate resample (single class) makes AUC undefined; skip it.
        if family == "binomial-logit" and len(np.unique(yb)) < 2:
            continue
        mn = _score_metrics(family, yb, naive_prob[idx])
        mr = _score_metrics(family, yb, ref_prob[idx])
        for k, higher in METRIC_DIRECTIONS.items():
            if k in mn:
                naive_values[k].append(mn[k])
            if k in mn and k in mr:
                d = (mr[k] - mn[k]) if higher else (mn[k] - mr[k])
                diffs[k].append(d)
    return (
        {k: float(np.std(v, ddof=1)) for k, v in diffs.items() if len(v) > 2},
        {k: float(np.std(v, ddof=1)) for k, v in naive_values.items() if len(v) > 2},
    )


def _naive_logit_irls(X, y, pen, *, n_iter=40, tol=1e-7):
    """Fit the small covariates-only baseline with fail-closed Newton steps."""
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-6, None)
        H = (X * w[:, None]).T @ X
        H[np.diag_indices_from(H)] += pen
        grad = X.T @ (y - mu) - pen * b
        step = np.linalg.solve(H, grad)
        mx = np.max(np.abs(step))
        if mx > 2.0:
            step = step * (2.0 / mx)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b


def _fit_runtime_anchor(covariates, genotypes, targets, *, ridge=1.0):
    """Fit the practical runtime anchor without access to prediction features."""
    from reference.runtime_anchor import fit_fixed_ridge_logistic

    t0 = time.perf_counter()
    model, diagnostics = fit_fixed_ridge_logistic(
        genotypes,
        covariates,
        targets,
        penalty=ridge,
    )
    if diagnostics.get("converged") is not True:
        raise ReferenceFitError("runtime ridge anchor did not converge")
    return model, float(time.perf_counter() - t0)


def _predict_runtime_anchor(model, covariates, genotypes):
    """Predict only after the separate prediction dataset has been loaded."""
    t0 = time.perf_counter()
    probability = _sigmoid(model.decision_function(genotypes, covariates))
    return probability, float(time.perf_counter() - t0)


# The reference is not reimplemented in trusted code. The corpus builder executes the
# exact three-file public gold program that can be packaged as a submission, against
# only the immutable public training files, and consumes its serialized model.
_PUBLIC_REFERENCE_INPUTS = (
    "family.txt",
    "formula.txt",
    "covariates_train.csv",
    "genotypes_train.tsv",
    "variant_metadata.tsv",
)


class _PublicNumpyReferenceArtifact:
    def __init__(self, serialized, model_bytes, source_files, source_sha256):
        self.serialized = serialized
        self.model_bytes = model_bytes
        self.source_files = source_files
        self.source_sha256 = source_sha256


def _public_reference_source_snapshot():
    """Read and digest the exact locked three-file submission once."""
    source_files = {}
    digest = hashlib.sha256()
    digest.update(b"svpgsbench|public-reference-source-v1\0")
    for relative in REFERENCE_SOURCE_RELATIVE_PATHS:
        data = Path(_ROOT, *relative.split("/")).read_bytes()
        source_files[Path(relative).name] = data
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return source_files, digest.hexdigest()


def _fit_best_of_family_reference(training, ds_id, public_dir):
    """Run the exact packageable public NumPy reference executable.

    No trusted estimator, private dependency, category label, prediction row, or test
    truth enters the subprocess. Failure of either family arm or any ridge-grid trial
    fails the executable and therefore fails the corpus build; there is no fallback.
    """
    source_files, source_sha256 = _public_reference_source_snapshot()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="svpgs-public-reference-") as work:
            submission_dir = os.path.join(work, "submission")
            os.mkdir(submission_dir)
            for name, data in source_files.items():
                destination = os.path.join(submission_dir, name)
                with open(destination, "wb") as handle:
                    handle.write(data)
                os.chmod(destination, 0o755 if name in {"fit", "predict"} else 0o644)
            for name in _PUBLIC_REFERENCE_INPUTS:
                source = os.path.join(os.fspath(public_dir), name)
                if os.path.isfile(source):
                    # Build-time public dirs are plain files; symlink them.
                    os.symlink(os.path.abspath(source), os.path.join(work, name))
                elif os.path.isfile(source + ".gz"):
                    # SHIPPED corpora compress the genotype TSVs, but the public
                    # reference reads the plain file. The shipping audit re-runs this
                    # reference against the compressed corpus, so decompress the .gz
                    # into the work dir exactly as the grader's public staging does
                    # (grader/submission_runner.py). Build-time (uncompressed) dirs
                    # never reach this branch, so anchor bytes are unchanged.
                    with gzip.open(source + ".gz", "rb") as fin, \
                            open(os.path.join(work, name), "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                else:
                    raise ReferenceFitError(
                        f"{ds_id}: public reference input {name!r} is missing"
                    )
            # Match the public runner's sanitized four-vCPU process environment.
            # Inheriting PYTHONPATH/VIRTUAL_ENV/private configuration would destroy
            # the claim that these exact packageable bytes caused the anchor.
            reference_env = {
                "PATH": "/opt/svpgs-venv/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": work,
                "TMPDIR": "/tmp",
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
                "VECLIB_MAXIMUM_THREADS": "4",
            }
            completed = subprocess.run(
                [REFERENCE_PYTHON_EXECUTABLE, os.path.join(submission_dir, "fit")],
                cwd=work,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=FIT_TIMEOUT_S,
                check=False,
                text=True,
                env=reference_env,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise ReferenceFitError(
                    f"{ds_id}: public NumPy reference exited {completed.returncode}: "
                    f"{detail[-2000:]}"
                )
            if completed.stdout.strip() or completed.stderr.strip():
                raise ReferenceFitError(
                    f"{ds_id}: public NumPy reference emitted unexpected console output"
                )
            model_path = os.path.join(work, "model.out")
            if not os.path.isfile(model_path):
                raise ReferenceFitError(
                    f"{ds_id}: public NumPy reference wrote no model.out"
                )
            with open(model_path, "rb") as handle:
                model_bytes = handle.read()
            try:
                serialized = json.loads(model_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReferenceFitError(
                    f"{ds_id}: public reference model.out is invalid JSON") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReferenceFitError(
            f"{ds_id}: public NumPy reference exceeded the {FIT_TIMEOUT_S}s fit cap"
        ) from exc
    total_wall = float(time.perf_counter() - started)

    required = {
        "runtime", "family", "cov_cols", "alpha", "beta", "gmean", "gsd", "vids",
        "calibration_intercept", "calibration_slope", "calibration_iterations",
        "calibration_relative_gradient", "ann_cols", "classes", "recovered",
        "reference_method", "inner_selection_auc", "inner_ridge_trials",
        "inner_hierarchical_recovered", "selection_seconds",
        "full_refit_seconds", "fit_seconds",
    }
    if type(serialized) is not dict or set(serialized) != required:
        raise ReferenceFitError(f"{ds_id}: public reference model schema is invalid")
    if (
        serialized["runtime"] != {"numpy_version": REFERENCE_NUMPY_VERSION}
        or serialized["family"] != training.schema.family
        or serialized["cov_cols"] != list(training.schema.covariate_names)
        or serialized["vids"] != list(training.schema.variant_ids)
        or serialized["reference_method"] not in {"hierarchical_eb", "ridge_logistic"}
    ):
        raise ReferenceFitError(f"{ds_id}: public reference model identity drifted")
    alpha = np.asarray(serialized["alpha"], dtype=float)
    beta = np.asarray(serialized["beta"], dtype=float)
    gmean = np.asarray(serialized["gmean"], dtype=float)
    gsd = np.asarray(serialized["gsd"], dtype=float)
    calibration_intercept = serialized["calibration_intercept"]
    calibration_slope = serialized["calibration_slope"]
    calibration_iterations = serialized["calibration_iterations"]
    calibration_relative_gradient = serialized["calibration_relative_gradient"]
    if (
        alpha.shape != (training.covariates.shape[1] + 1,)
        or beta.shape != (training.genotypes.shape[1],)
        or gmean.shape != beta.shape
        or gsd.shape != beta.shape
        or not all(np.all(np.isfinite(value)) for value in (alpha, beta, gmean, gsd))
        or np.any(gsd <= 0.0)
        or type(calibration_intercept) not in (int, float)
        or not np.isfinite(float(calibration_intercept))
        or type(calibration_slope) not in (int, float)
        or not np.isfinite(float(calibration_slope))
        or float(calibration_slope) <= 0.0
        or type(calibration_iterations) is not int
        or not 1 <= calibration_iterations <= REFERENCE_CALIBRATION_MAX_ITERATIONS
        or type(calibration_relative_gradient) not in (int, float)
        or not np.isfinite(float(calibration_relative_gradient))
        or not 0.0 <= float(calibration_relative_gradient)
        <= REFERENCE_CALIBRATION_RELATIVE_GRADIENT_TOLERANCE
    ):
        raise ReferenceFitError(f"{ds_id}: public reference coefficients are invalid")

    method = serialized["reference_method"]
    selection = serialized["inner_selection_auc"]
    trials = serialized["inner_ridge_trials"]
    inner_recovered = serialized["inner_hierarchical_recovered"]
    recovered = serialized["recovered"]
    method_diag = {
        "fit_exit_code": 0,
        "fit_source_sha256": source_sha256,
        "fit_python_executable": REFERENCE_PYTHON_EXECUTABLE,
        "fit_numpy_version": REFERENCE_NUMPY_VERSION,
        "reference_total_fit_wall_s": total_wall,
        "reference_full_fit_wall_s": float(serialized["full_refit_seconds"]),
        "reference_program_reported_wall_s": float(serialized["fit_seconds"]),
        "selection_wall_s": float(serialized["selection_seconds"]),
        "calibration_intercept": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
        "calibration_iterations": int(calibration_iterations),
        "calibration_relative_gradient": float(calibration_relative_gradient),
        "inner_ridge_trials": [dict(trial, converged=True) for trial in trials],
        "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
        "hierarchical_inner_iterations_run": int(inner_recovered["iterations_run"]),
        "hierarchical_inner_global_scale": float(inner_recovered["global_scale"]),
    }
    if method == "hierarchical_eb":
        method_diag.update({
            "hierarchical_iterations_run": int(recovered["iterations_run"]),
            "hierarchical_global_scale": float(recovered["global_scale"]),
        })
    else:
        method_diag.update({
            "ridge_penalty": float(recovered["ridge_penalty"]),
            "ridge_penalty_grid": [
                float(trial["penalty"]) for trial in trials
            ],
            "ridge_iterations_run": int(recovered["ridge_iterations"]),
            "ridge_max_iterations": 500,
            "ridge_tolerance": 1e-5,
            "ridge_final_relative_gradient": float(
                recovered["ridge_relative_gradient"]
            ),
        })
    return (
        method,
        _PublicNumpyReferenceArtifact(
            serialized, model_bytes, source_files, source_sha256),
        method_diag,
        selection,
        list(serialized["classes"]),
    )


def _predict_public_reference(reference_artifact, public_dir, ds_id, n_test, family):
    """Run the retained fit artifact through its exact snapshotted predictor."""
    if _public_reference_source_sha256() != reference_artifact.source_sha256:
        raise ReferenceFitError(
            f"{ds_id}: authoritative gold source changed between fit and predict")
    with tempfile.TemporaryDirectory(prefix="svpgs-public-predict-source-") as source:
        os.chmod(source, 0o755)
        for name, data in reference_artifact.source_files.items():
            destination = os.path.join(source, name)
            with open(destination, "wb") as handle:
                handle.write(data)
            os.chmod(destination, 0o755 if name in {"fit", "predict"} else 0o644)
        result = run_prefit_predict(
            source,
            public_dir,
            reference_artifact.model_bytes,
            n_test_expected=n_test,
            family=family,
        )
    if result.get("status") != "ok":
        raise ReferenceFitError(
            f"{ds_id}: exact public reference predict failed: "
            f"{result.get('status')}: {result.get('detail')}")
    result["source_sha256"] = reference_artifact.source_sha256
    return result


# ---- schema-v8 identity, digests, and provenance ---------------------------
def _require_shipping_cell(category, replicate):
    if category not in CATEGORIES:
        raise ValueError(f"unknown shipping category {category!r}")
    if type(replicate) is not int or not 0 <= replicate < REPLICATES_PER_CATEGORY:
        raise ValueError(
            "shipping replicate must be an integer in "
            f"[0, {REPLICATES_PER_CATEGORY}), got {replicate!r}"
        )


def _dataset_id(category, replicate, auth_key, pipeline_sha256):
    _require_shipping_cell(category, replicate)
    return opaque_dataset_id(
        auth_key,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
    )


def _dataset_path(category, replicate, auth_key, pipeline_sha256):
    return f"{category}/{_dataset_id(category, replicate, auth_key, pipeline_sha256)}"


def _require_plain_directory(path, label):
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise FileNotFoundError(f"{label} is absent: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is not a plain directory: {path}")


def _ensure_plain_directory(path, label):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    _require_plain_directory(path, label)


def _compress_public_genotypes(public_dir):
    for name in ("genotypes_train.tsv", "genotypes_test.tsv"):
        source = os.path.join(public_dir, name)
        target = source + ".gz"
        _require_plain_file(source, f"public {name}")
        if os.path.lexists(target):
            raise FileExistsError(f"compressed genotype output already exists: {target}")
        with open(source, "rb") as input_file, open(target, "xb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_output,
                mtime=0,
            ) as output_file:
                shutil.copyfileobj(input_file, output_file, length=1 << 20)
        os.unlink(source)


def _require_plain_file(path, label):
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise FileNotFoundError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a plain regular file: {path}")


def _hash_file(path):
    _require_plain_file(path, "hashed file")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_hashes(dataset_dir, *, exclude=()):
    """Return exact SHA-256s for every regular file below public/ and truth/.

    Paths are POSIX-style and relative to the dataset directory so the manifest
    is platform-independent. Non-regular files fail the build rather than being
    omitted from the integrity contract.
    """
    excluded = set(exclude)
    hashes = {}
    for top_level in ("public", "truth"):
        root = os.path.join(dataset_dir, top_level)
        _require_plain_directory(root, f"dataset {top_level}")
        for current, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            for dirname in dirnames:
                directory_path = os.path.join(current, dirname)
                if stat.S_ISLNK(os.lstat(directory_path).st_mode):
                    raise ValueError(f"dataset contains symlinked directory: {directory_path}")
            for filename in filenames:
                path = os.path.join(current, filename)
                relative = os.path.relpath(path, dataset_dir).replace(os.sep, "/")
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    raise ValueError(f"dataset contains non-regular file: {path}")
                if relative in excluded:
                    continue
                hashes[relative] = _hash_file(path)
    return dict(sorted(hashes.items()))


def _public_reference_source_sha256():
    """Digest exactly the three packageable files that define score 1."""
    digest = hashlib.sha256()
    digest.update(b"svpgsbench|public-reference-source-v1\0")
    for relative in REFERENCE_SOURCE_RELATIVE_PATHS:
        path = os.path.join(_ROOT, *relative.split("/"))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _generation_pipeline_sha256():
    """Digest only the immutable controls and sources that determine data bytes."""
    digest = hashlib.sha256()
    digest.update(b"svpgsbench|data-generation-contract-v1\0")
    digest.update(
        json.dumps(
            dict(GENERATION_CONTRACT_ITEMS),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    for relative in GENERATION_PIPELINE_RELATIVE_PATHS:
        path = os.path.join(_ROOT, *relative.split("/"))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _development_report_sha256(auth_key):
    path = os.path.join(_ROOT, *DEVELOPMENT_REPORT_RELATIVE_PATH.split("/"))
    _require_plain_file(path, "frozen development report")
    with open(path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    pipeline_sha256 = _generation_pipeline_sha256()
    expected_design = {
        "purpose": "development",
        "key_id": corpus_key_id(auth_key),
        "generation_pipeline_sha256": pipeline_sha256,
        "categories": list(CATEGORIES),
        "category_count": len(CATEGORIES),
        "replicates": list(range(REPLICATES_PER_CATEGORY)),
        "replicates_per_category": REPLICATES_PER_CATEGORY,
        "requested_n": SHIPPED_REQUESTED_N,
        "requested_p": SHIPPED_REQUESTED_P,
        "dataset_count": SHIPPED_DATASET_COUNT,
    }
    required_report_fields = {
        "schema_version",
        "report_kind",
        "passed",
        "protocol",
        "development_design",
        "checks",
        "categories",
        "distinctness",
        "dataset_results",
        DEVELOPMENT_REPORT_HMAC_FIELD,
    }
    if (
        not isinstance(report, dict)
        or set(report) != required_report_fields
        or report.get("schema_version") != MODEL_ZOO_REPORT_SCHEMA_VERSION
        or report.get("report_kind") != "development_model_zoo"
        or report.get("passed") is not True
        or report.get("development_design") != expected_design
        or not verify_json_hmac(
            auth_key,
            report,
            field=DEVELOPMENT_REPORT_HMAC_FIELD,
            domain=DEVELOPMENT_REPORT_HMAC_DOMAIN,
        )
    ):
        raise ValueError(
            f"{DEVELOPMENT_REPORT_RELATIVE_PATH} is not the exact passing "
            "full-size development report"
        )
    return _hash_file(path)


def _reference_source_provenance():
    return {
        "implementation": REFERENCE_IMPLEMENTATION,
        "entrypoint": REFERENCE_ENTRYPOINT,
        "fit_protocol": REFERENCE_FIT_PROTOCOL,
        "python_executable": REFERENCE_PYTHON_EXECUTABLE,
        "numpy_version": REFERENCE_NUMPY_VERSION,
        "public_source_sha256": _public_reference_source_sha256(),
    }


def _builder_sha256():
    return _hash_file(os.path.abspath(__file__))


def _config_provenance(
    category,
    replicate,
    auth_key,
    pipeline_sha256,
):
    """The generation config for a category/replicate, without secret seeds.

    This records everything that determines the dataset except the keyed stream
    values and produced file bytes. The HMAC-authenticated anchor proves those
    omitted stream values came from the declared key/pipeline context.
    """
    import dataclasses

    spec = CATEGORIES[category]
    cfg = spec["make_cfg"](
        replicate,
        SHIPPED_REQUESTED_N,
        SHIPPED_REQUESTED_P,
    )
    cfg_fields = json.loads(json.dumps(dataclasses.asdict(cfg),
                                       default=str, sort_keys=True))
    del cfg_fields["seed"]
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "code_version": CORPUS_BUILDER_CODE_VERSION,
        "builder_sha256": _builder_sha256(),
        "purpose": CORPUS_PURPOSE,
        "generation_pipeline_sha256": pipeline_sha256,
        "id": _dataset_id(category, replicate, auth_key, pipeline_sha256),
        "path": _dataset_path(category, replicate, auth_key, pipeline_sha256),
        "category": category,
        "replicate": int(replicate),
        "N": SHIPPED_REQUESTED_N,
        "P": SHIPPED_REQUESTED_P,
        "family": spec["family"],
        "prior_cols": list(spec["prior_cols"]),
        "cfg": cfg_fields,
    }


def _manifest_authenticates_dataset(
    out_dir,
    anchor,
    auth_key,
    pipeline_sha256,
):
    """Require a keyed, finalized schema-v8 manifest to pin the full dataset."""
    manifest_path = os.path.join(CORPUS, "manifest.json")
    try:
        _require_plain_file(manifest_path, "corpus manifest")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != CORPUS_SCHEMA_VERSION
            or not isinstance(manifest.get("datasets"), list)
            or manifest.get("meta", {}).get("key_id") != corpus_key_id(auth_key)
            or manifest.get("meta", {}).get("generation_pipeline_sha256")
            != pipeline_sha256
            or not verify_json_hmac(
                auth_key,
                manifest,
                field=MANIFEST_HMAC_FIELD,
                domain=MANIFEST_HMAC_DOMAIN,
            )
        ):
            return False
        matches = [
            entry
            for entry in manifest["datasets"]
            if isinstance(entry, dict) and entry.get("id") == anchor["id"]
        ]
        if len(matches) != 1:
            return False
        entry = matches[0]
        expected_identity = {
            "id": anchor["id"],
            "path": anchor["path"],
            "category": anchor["category"],
            "replicate": anchor["replicate"],
            "family": anchor["family"],
            "weight": DATASET_WEIGHT,
        }
        if any(entry.get(key) != value for key, value in expected_identity.items()):
            return False
        return entry.get("sha256") == _dataset_hashes(out_dir)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _provenance_matches(
    out_dir,
    category,
    replicate,
    auth_key,
    pipeline_sha256,
    *,
    require_manifest=False,
):
    """Return true only for a complete, authenticated schema-v8 dataset."""
    expected_out_dir = os.path.join(
        CORPUS,
        category,
        _dataset_id(category, replicate, auth_key, pipeline_sha256),
    )
    if os.path.abspath(out_dir) != os.path.abspath(expected_out_dir):
        return False, "dataset path is not the canonical category/id path"
    try:
        _require_plain_directory(CORPUS, "corpus root")
        _require_plain_directory(
            os.path.join(CORPUS, category),
            f"{category} category",
        )
        _require_plain_directory(out_dir, "dataset directory")
    except (OSError, ValueError) as exc:
        return False, f"dataset ancestry is invalid ({type(exc).__name__})"
    anch = os.path.join(out_dir, "truth", "anchors.json")
    try:
        _require_plain_file(anch, "anchor file")
        with open(anch, encoding="utf-8") as anchor_file:
            a = json.load(anchor_file)
    except Exception as e:
        return False, f"unreadable anchors ({type(e).__name__})"
    if (not isinstance(a, dict)
            or type(a.get("schema_version")) is not int
            or a["schema_version"] != CORPUS_SCHEMA_VERSION):
        return False, f"anchor schema is not v{CORPUS_SCHEMA_VERSION}"
    identity = {
        "id": _dataset_id(category, replicate, auth_key, pipeline_sha256),
        "path": _dataset_path(category, replicate, auth_key, pipeline_sha256),
        "category": category,
        "replicate": int(replicate),
    }
    for key, value in identity.items():
        if a.get(key) != value:
            return False, f"anchor identity mismatch on {key!r}"
    reference_fit = a.get("reference_fit")
    if not isinstance(reference_fit, dict) or reference_fit.get("converged") is not True:
        return False, "reference fit is missing or unconverged"
    try:
        expected_reference = dict(_reference_source_provenance(), converged=True)
    except (OSError, ValueError) as exc:
        return False, f"reference provenance unavailable ({type(exc).__name__})"
    if reference_fit != expected_reference:
        return False, "reference source/protocol provenance changed"
    if not verify_json_hmac(
        auth_key,
        a,
        field=ANCHOR_HMAC_FIELD,
        domain=ANCHOR_HMAC_DOMAIN,
    ):
        return False, "anchor HMAC mismatch"
    prov = a.get("provenance")
    if not isinstance(prov, dict):
        return False, "missing schema-v8 provenance"
    exp = _config_provenance(
        category,
        replicate,
        auth_key,
        pipeline_sha256,
    )
    for k, v in exp.items():
        if prov.get(k) != v:
            return False, f"config mismatch on {k!r}"
    try:
        actual_hashes = _dataset_hashes(out_dir, exclude={"truth/anchors.json"})
    except (OSError, ValueError) as exc:
        return False, f"dataset files unreadable ({type(exc).__name__})"
    if actual_hashes != prov.get("file_sha256"):
        return False, "public/truth files changed since anchors were built"
    try:
        weight = a["weight"]
        if type(weight) not in (int, float) or float(weight) != DATASET_WEIGHT:
            return False, "anchor weight is invalid"
        _validate_anchor_quality(
            a["id"],
            a["metrics_naive"],
            a["metrics_reference"],
            a["metrics_ref_naive_se"],
            a["metrics_naive_se"],
        )
        _validate_reference_protocol(a["id"], a["reference_protocol"], a["_n_train"])
        _validate_reference_feasibility(a["id"], a["time_reference"])
        for timing_name in (
            "time_reference",
            "time_calibration",
            "time_runtime_anchor",
            "time_runtime_calibration",
        ):
            timing = a[timing_name]
            if (type(timing) not in (int, float)
                    or not np.isfinite(float(timing))
                    or float(timing) <= 0.0):
                return False, f"anchor {timing_name} is invalid"
    except (KeyError, TypeError, ValueError, ReferenceFitError) as exc:
        return False, f"anchor semantics are invalid ({type(exc).__name__})"
    if require_manifest and not _manifest_authenticates_dataset(
        out_dir,
        a,
        auth_key,
        pipeline_sha256,
    ):
        return False, "dataset is not pinned by the finalized schema-v8 manifest"
    return True, "match"


# ---- build one dataset -----------------------------------------------------
@functools.lru_cache(maxsize=1)
def _host_calibration():
    """One speed calibration for the whole build: (estimate, raw repetitions).

    The calibration workload measures the BUILD HOST, not the dataset, so every
    anchor in a build shares a single estimate. The raw repetitions are retained
    in the anchor so a postmortem can see the spread the estimate came from.
    """
    reps = tuple(float(calibration_seconds(reps=1)) for _ in range(5))
    return float(min(reps)), reps


def build_one(category, replicate, auth_key, pipeline_sha256=None):
    # Public data loading is independent of the optional hidden model-zoo engine.
    from reference.run_svpgs import (
        load_prediction_dataset,
        load_training_dataset,
    )

    if pipeline_sha256 is None:
        pipeline_sha256 = _generation_pipeline_sha256()
    spec = CATEGORIES[category]
    family = spec["family"]
    if family != "binomial-logit":
        raise ValueError(f"unsupported category family {family!r}")
    dgp_seed = derive_stream_seed(
        auth_key,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream="dgp",
    )
    materialize_seed = derive_stream_seed(
        auth_key,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream="materialize",
    )
    bootstrap_seed = derive_stream_seed(
        auth_key,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream="bootstrap",
    )
    cfg = spec["make_cfg"](
        replicate,
        SHIPPED_REQUESTED_N,
        SHIPPED_REQUESTED_P,
    )
    cfg.seed = dgp_seed
    ds_id = _dataset_id(category, replicate, auth_key, pipeline_sha256)
    ds_path = _dataset_path(category, replicate, auth_key, pipeline_sha256)
    _ensure_plain_directory(CORPUS, "corpus root")
    category_dir = os.path.join(CORPUS, category)
    _ensure_plain_directory(category_dir, f"{category} category")
    out_dir = os.path.join(category_dir, ds_id)
    public_dir = os.path.join(out_dir, "public")

    # A rebuild is a fresh dataset transaction. Keeping files from an older
    # schema/config would make them part of the new digest set despite no longer
    # being produced, and a stale anchors.json could mask a failed rebuild.
    if os.path.lexists(out_dir):
        if not stat.S_ISDIR(os.lstat(out_dir).st_mode):
            raise ValueError(f"dataset output path is not a directory: {out_dir}")
        shutil.rmtree(out_dir)

    d = generate(cfg)
    materialize(
        d,
        out_dir,
        family=family,
        frac_train=cfg.frac_train_hint,
        prior_cols=spec["prior_cols"],
        seed=materialize_seed,
        ancestry_shift=spec["ancestry_shift"],
    )
    del d

    # The returned value contains only training rows. This trusted call site does
    # not invoke the separate prediction loader until every fit is complete.
    training = load_training_dataset(public_dir)
    tr_sids = training.sample_ids
    cov_tr = training.covariates
    G_tr = training.genotypes
    y_tr = training.targets
    gfamily = training.schema.family
    execution_events: list[str] = []

    # --- naive fit: covariates only ---
    Xtr = np.column_stack([np.ones(len(tr_sids)), cov_tr])
    pen = np.full(Xtr.shape[1], 1e-3)
    pen[0] = 0.0
    b = _naive_logit_irls(Xtr, y_tr.astype(float), pen)

    # --- exact packageable public NumPy reference (gold/fit) ---
    # Selected by deterministic training-inner-validation AUC, then refit on the FULL
    # training bytes. hierarchical EB wins where structured shrinkage matters; a
    # ridge wins on dense near-Gaussian regimes. Picking the best principled method per
    # regime is what a demigod does, so skill=1 stays reachable and every well-posed
    # category is adequately anchored (no co-selection). No test truth is touched.
    try:
        (
            reference_method,
            reference_model,
            method_diagnostics,
            selection_aucs,
            reference_classes_present,
        ) = _fit_best_of_family_reference(training, ds_id, public_dir)
    except ReferenceFitError:
        raise
    except Exception as exc:
        raise ReferenceFitError(
            f"{ds_id}: reference fit failed: {type(exc).__name__}: {exc}"
        ) from exc
    # A submission is not told which family member wins on a hidden cohort. Its exact
    # reference witness must therefore perform the training-only selection AND refit
    # the winner. Feasibility/time_reference record that complete procedure, not only
    # the final refit.
    reference_fit_time = float(method_diagnostics["reference_total_fit_wall_s"])
    if reference_model is None or not reference_model.model_bytes:
        raise ReferenceFitError(f"{ds_id}: reference returned no serialized model")
    model_sha256 = hashlib.sha256(reference_model.model_bytes).hexdigest()
    reference_diagnostics = {
        "protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_method": reference_method,
        "converged": True,
        "fit_protocol": REFERENCE_FIT_PROTOCOL,
        "wall_time_s": reference_fit_time,
        "inner_selection_auc": selection_aucs,
        "classes_present": [str(c) for c in reference_classes_present],
        "training_output": {
            "model_sha256": model_sha256,
            "model_byte_count": len(reference_model.model_bytes),
        },
        **method_diagnostics,
    }

    # --- runtime-anchor fit ---
    # This is independent of the slow accuracy reference and is also fitted before
    # prediction files are opened.
    runtime_model, runtime_fit_time = _fit_runtime_anchor(cov_tr, G_tr, y_tr)
    execution_events.append("all_anchor_fits")

    # Only fitted models survive across this boundary. Prediction features contain
    # no labels and are loaded only after every model fit above has completed.
    prediction = load_prediction_dataset(
        public_dir,
        training=training,
    )
    execution_events.append("prediction_data_load")
    te_sids = prediction.sample_ids
    cov_te = prediction.covariates
    G_te = prediction.genotypes

    Xte = np.column_stack([np.ones(len(te_sids)), cov_te])
    naive_prob = _sigmoid(Xte @ b)
    if naive_prob.shape != (len(te_sids),) or not np.all(np.isfinite(naive_prob)):
        raise ValueError(f"{ds_id}: naive anchor produced invalid probabilities")

    # The anchor is caused by the exact locked public predictor, not an in-process
    # helper that merely implements similar math.  Only after every fit and the
    # trusted prediction-data load above do we stage the two public test files in
    # the mandatory networkless submission sandbox.  The exact model.out bytes
    # from gold/fit are consumed by gold/predict under the normal 30-second cap;
    # metrics below parse the actual pred.csv round-trip bytes it wrote.
    prediction_result = _predict_public_reference(
        reference_model, public_dir, ds_id, len(te_sids), gfamily)
    ref_prob = np.asarray(prediction_result["pred"]["mean"], dtype=float)
    if ref_prob.shape != (len(te_sids),) or not np.all(np.isfinite(ref_prob)):
        raise ReferenceFitError(f"{ds_id}: reference produced invalid probabilities")
    reference_diagnostics.update({
        "prediction_exit_code": int(prediction_result["predict_rc"]),
        "prediction_wall_s": float(prediction_result["t_predict"]),
        "prediction_output_sha256": prediction_result["pred_sha256"],
        "prediction_output_byte_count": len(prediction_result["pred_bytes"]),
        "prediction_model_sha256": prediction_result["model_sha256"],
        "prediction_source_sha256": prediction_result["source_sha256"],
    })
    time_ref = reference_fit_time + float(prediction_result["t_predict"])

    runtime_prob, runtime_prediction_time = _predict_runtime_anchor(
        runtime_model,
        cov_te,
        G_te,
    )
    time_runtime = runtime_fit_time + runtime_prediction_time
    if runtime_prob.shape != (len(te_sids),) or not np.all(np.isfinite(runtime_prob)):
        raise ValueError(f"{ds_id}: runtime anchor produced invalid probabilities")
    execution_events.append("all_anchor_predictions")

    # Truth is opened only after all fits and all predictions are complete.
    try:
        y_te = read_aligned_binary_truth_csv(
            os.path.join(out_dir, "truth", "y_test.csv"),
            te_sids,
        )
    except ValueError as exc:
        raise ValueError(f"{ds_id}: {exc}") from exc
    execution_events.append("truth_load")

    reference_diagnostics["execution_order"] = list(execution_events)
    _validate_reference_protocol(ds_id, reference_diagnostics, len(tr_sids))
    m_naive = _score_metrics(gfamily, y_te, naive_prob)
    m_ref = _score_metrics(gfamily, y_te, ref_prob)
    # The submission-side calibration factor is reference-relative, so the
    # reference itself is exactly one by construction.  Refuse to let a materially
    # miscalibrated reference define that scale: otherwise a high-AUC but wildly
    # overconfident anchor would certify itself as perfect and make the prompt's
    # calibration requirement vacuous.
    reference_calibration = reference_calibration_qualification(m_ref, m_naive)
    if not reference_calibration["passed"]:
        raise ReferenceFitError(
            f"{ds_id}: reference fails the predeclared absolute calibration "
            f"qualification: {reference_calibration['violations']!r}"
        )
    # Anchor SEs: the grader excludes a metric when its reference-minus-naive gap
    # is either (a) not distinguishable from zero -- RELIABILITY, via the paired
    # gap SE -- or (b) too small to be a skill denominator -- ADEQUACY, via the
    # single-estimate naive SE. Both are bootstrapped from one resampling loop.
    # Neither ever changes the anchor denominator, preserving exact reference=1
    # semantics; an excluded metric drops out and the rest renormalize.
    ref_naive_se, naive_se = _bootstrap_anchor_ses(
        gfamily,
        y_te,
        naive_prob,
        ref_prob,
        seed=bootstrap_seed,
    )
    _validate_anchor_quality(ds_id, m_naive, m_ref, ref_naive_se, naive_se)
    _compress_public_genotypes(public_dir)

    time_calibration, calibration_reps = _host_calibration()
    # The accuracy reference and the runtime anchor are timed on the SAME host in
    # the SAME process, so they share ONE calibration estimate. Measuring it twice
    # (as this used to) sampled the same quantity twice and injected the
    # difference straight into the reported efficiency ratio as avoidable noise.
    time_runtime_calibration = time_calibration
    for timing_name, timing_value in {
        "time_reference": time_ref,
        "time_runtime_anchor": time_runtime,
        "time_calibration": time_calibration,
        "time_runtime_calibration": time_runtime_calibration,
    }.items():
        if not np.isfinite(timing_value) or timing_value <= 0.0:
            raise ValueError(f"{ds_id}: invalid {timing_name}={timing_value!r}")

    gap = m_ref["auc"] - m_naive["auc"]
    # Pin the anchor to the generation contract, all materialized data bytes, and
    # the exact reference adapter/engine source. The anchor file itself is excluded
    # here because the manifest records its digest after FINALIZE serializes it.
    provenance = _config_provenance(
        category,
        replicate,
        auth_key,
        pipeline_sha256,
    )
    provenance["file_sha256"] = _dataset_hashes(
        out_dir,
        exclude={"truth/anchors.json"},
    )
    reference_fit = dict(_reference_source_provenance(), converged=True)
    anchors = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "id": ds_id,
        "path": ds_path,
        "family": family,
        "category": category,
        "replicate": int(replicate),
        "weight": DATASET_WEIGHT,
        "metrics_naive": m_naive,
        "metrics_reference": m_ref,
        # Per-metric bootstrap SE of the (reference - naive) GAP: the RELIABILITY
        # gate in grader/skill.metric_skill (K_SE). Tests whether the gap is real.
        "metrics_ref_naive_se": ref_naive_se,
        # Per-metric bootstrap SE of the SINGLE naive estimate: the ADEQUACY gate
        # in grader/skill.metric_skill (RESOLUTION_K). Tests whether the gap is
        # large enough to divide by -- a different question, and the one that
        # protects the score. Unlike the paired SE above, this does not shrink
        # when the reference collapses onto naive.
        "metrics_naive_se": naive_se,
        "time_reference": time_ref,
        # Separate fast practical runtime anchor for report-only efficiency;
        # schema v8 requires it independently of time_reference.
        "time_runtime_anchor": time_runtime,
        "reference_fit": reference_fit,
        "reference_protocol": reference_diagnostics,
        "provenance": provenance,
        # reference-host speed calibration: the grader measures the same workload
        # at grade time and rescales the anchor time by the ratio, so the perf
        # factor is portable across hardware (see grader/perf.py). The runtime
        # anchor is timed on the same host, so it shares this calibration.
        "time_calibration": time_calibration,
        "time_runtime_calibration": time_runtime_calibration,
        "time_calibration_repetitions": list(calibration_reps),
        "_n_train": int(len(tr_sids)), "_n_test": int(len(te_sids)),
        "_n_variants": int(G_tr.shape[1]), "_gap": float(gap),
        "_prior_cols": list(spec["prior_cols"]),
    }
    anchors[ANCHOR_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        anchors,
        exclude_field=ANCHOR_HMAC_FIELD,
        domain=ANCHOR_HMAC_DOMAIN,
    )
    with open(os.path.join(out_dir, "truth", "anchors.json"), "w") as fh:
        json.dump(anchors, fh, indent=2, allow_nan=False)
    print(f"[ONE-DONE] {ds_id} gap={gap:+.4f} naive={m_naive} ref={m_ref} "
          f"time_ref={time_ref:.1f}s", flush=True)


def _expected_datasets():
    """Return the exact canonical category/replicate grid for a shippable build."""
    cats = list(CATEGORIES)
    if len(cats) != SHIPPED_CATEGORY_COUNT:
        raise ValueError(
            f"shipping requires exactly {SHIPPED_CATEGORY_COUNT} categories"
        )
    expected = [
        (cat, replicate)
        for cat in cats
        for replicate in range(REPLICATES_PER_CATEGORY)
    ]
    if len(expected) != SHIPPED_DATASET_COUNT:
        raise ValueError(
            f"shipping requires exactly {SHIPPED_DATASET_COUNT} datasets"
        )
    return expected


# ---- finalize: equal weights + immutable manifest --------------------------
def finalize(auth_key, pipeline_sha256=None):
    """Write a keyed schema-v8 manifest over exactly the expected datasets.

    Every dataset receives the same weight. The manifest pins every regular file
    below public/ and truth/ by SHA-256, including anchors.json itself.

    Operates ONLY over the immutable shipping category/replicate grid. Aborts
    (nonzero) on any missing, duplicate,
    unexpected/orphaned, or stale (provenance-mismatched) dataset rather than
    globbing whatever anchors.json happen to exist -- so a partial or stale corpus
    can never be finalized into a shippable manifest (fix 13c).

    The expected category/replicate grid and dimensions come only from the
    shipped contract; callers cannot widen or shrink them."""
    if pipeline_sha256 is None:
        pipeline_sha256 = _generation_pipeline_sha256()
    try:
        development_report_sha256 = _development_report_sha256(auth_key)
    except (OSError, TypeError, ValueError) as exc:
        sys.exit(f"[FINALIZE-ABORT] invalid frozen development report: {exc}")
    expected_list = _expected_datasets()
    expected_set = set(expected_list)
    if len(expected_list) != len(expected_set):
        dups = sorted({k for k in expected_list if expected_list.count(k) > 1})
        sys.exit(f"[FINALIZE-ABORT] duplicate entries in expected list: {dups}")

    categories = list(dict.fromkeys(category for category, _replicate in expected_list))
    if not categories:
        sys.exit("[FINALIZE-ABORT] expected dataset grid is empty")
    canonical_categories = list(CATEGORIES)
    if categories != canonical_categories:
        sys.exit(
            "[FINALIZE-ABORT] categories must exactly match the canonical eval "
            f"order: expected={canonical_categories}, got={categories}"
        )
    category_counts = {
        category: sum(1 for item_category, _replicate in expected_list
                      if item_category == category)
        for category in categories
    }
    if len(set(category_counts.values())) != 1:
        sys.exit(
            "[FINALIZE-ABORT] category replicate counts are not rectangular: "
            f"{category_counts}"
        )
    replicates_per_category = next(iter(category_counts.values()))
    if replicates_per_category != REPLICATES_PER_CATEGORY:
        sys.exit(
            "[FINALIZE-ABORT] replicates_per_category must equal "
            f"{REPLICATES_PER_CATEGORY}, got {replicates_per_category}"
        )
    declared_grid = [
        (category, replicate)
        for category in categories
        for replicate in range(replicates_per_category)
    ]
    if expected_list != declared_grid:
        sys.exit(
            "[FINALIZE-ABORT] expected datasets are not the exact declared "
            f"category/replicate grid: expected={declared_grid}, got={expected_list}"
        )

    try:
        _require_plain_directory(CORPUS, "corpus root")
    except (OSError, ValueError) as exc:
        sys.exit(f"[FINALIZE-ABORT] invalid corpus root: {exc}")

    expected_paths = {
        _dataset_path(category, replicate, auth_key, pipeline_sha256): (
            category,
            replicate,
        )
        for category, replicate in expected_list
    }
    expected_categories = set(categories)
    actual_paths = set()
    for category_entry in sorted(os.scandir(CORPUS), key=lambda entry: entry.name):
        if category_entry.name == "manifest.json":
            if not category_entry.is_file(follow_symlinks=False):
                sys.exit("[FINALIZE-ABORT] manifest.json is not a plain file")
            continue
        if not category_entry.is_dir(follow_symlinks=False):
            sys.exit(
                f"[FINALIZE-ABORT] unexpected corpus-root entry: {category_entry.name}"
            )
        if category_entry.name not in expected_categories:
            sys.exit(
                f"[FINALIZE-ABORT] unexpected category directory: {category_entry.name}"
            )
        for dataset_entry in sorted(
            os.scandir(category_entry.path),
            key=lambda entry: entry.name,
        ):
            if not dataset_entry.is_dir(follow_symlinks=False):
                sys.exit(
                    "[FINALIZE-ABORT] non-directory entry in category "
                    f"{category_entry.name}: {dataset_entry.name}"
                )
            actual_paths.add(f"{category_entry.name}/{dataset_entry.name}")

    missing_paths = sorted(set(expected_paths) - actual_paths)
    unexpected_paths = sorted(actual_paths - set(expected_paths))
    if missing_paths:
        sys.exit(
            f"[FINALIZE-ABORT] expected dataset directories missing: {missing_paths}"
        )
    if unexpected_paths:
        sys.exit(
            f"[FINALIZE-ABORT] unexpected/orphaned dataset directories: {unexpected_paths}"
        )

    # Load the anchor at every exact grid path; anchorless directories are invalid.
    found = {}
    for rel, key in expected_paths.items():
        ds_dir = os.path.join(CORPUS, *rel.split("/"))
        ap = os.path.join(ds_dir, "truth", "anchors.json")
        try:
            _require_plain_file(ap, "anchor file")
            with open(ap, encoding="utf-8") as anchor_file:
                a = json.load(anchor_file)
        except (OSError, ValueError) as exc:
            sys.exit(f"[FINALIZE-ABORT] invalid anchor for {rel}: {exc}")
        key = (a.get("category"), a.get("replicate"))
        if key in found:
            sys.exit(f"[FINALIZE-ABORT] duplicate dataset on disk for {key}: "
                     f"{rel} and {found[key][2]}")
        found[key] = (ap, a, rel, ds_dir)

    missing = sorted(expected_set - set(found))
    unexpected = sorted(set(found) - expected_set)
    if missing:
        sys.exit(f"[FINALIZE-ABORT] {len(missing)} expected dataset(s) missing: {missing}")
    if unexpected:
        sys.exit(f"[FINALIZE-ABORT] {len(unexpected)} unexpected/orphaned dataset(s) "
                 f"present (not in expected list): {unexpected}")

    # Stale check: each dataset's stored provenance must match the requested config
    # and its own public bytes.
    for cat, replicate in expected_list:
        _, _, _, ds_dir = found[(cat, replicate)]
        ok, why = _provenance_matches(
            ds_dir,
            cat,
            replicate,
            auth_key,
            pipeline_sha256,
        )
        if not ok:
            sys.exit(
                f"[FINALIZE-ABORT] stale dataset {cat} "
                f"replicate={replicate}: {why}"
            )

    recs = [found[k] for k in expected_list]  # (ap, anchor, rel, dataset_dir)
    datasets = []
    for _ap, a, rel, ds_dir in recs:
        expected_id = _dataset_id(
            a["category"],
            a["replicate"],
            auth_key,
            pipeline_sha256,
        )
        expected_path = _dataset_path(
            a["category"],
            a["replicate"],
            auth_key,
            pipeline_sha256,
        )
        if a.get("id") != expected_id or a.get("path") != expected_path or rel != expected_path:
            sys.exit(
                f"[FINALIZE-ABORT] dataset identity/path mismatch for {rel}: "
                f"id={a.get('id')!r}, path={a.get('path')!r}"
            )
        datasets.append({
            "id": expected_id,
            "path": expected_path,
            "category": a["category"],
            "family": a["family"],
            "replicate": int(a["replicate"]),
            "weight": DATASET_WEIGHT,
            "sha256": _dataset_hashes(ds_dir),
        })
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "datasets": datasets,
        "meta": {
            "n_datasets": len(datasets),
            "weight_rule": WEIGHT_RULE,
            "categories": categories,
            "replicates_per_category": replicates_per_category,
            "requested_n": SHIPPED_REQUESTED_N,
            "requested_p": SHIPPED_REQUESTED_P,
            "development_report_sha256": development_report_sha256,
            "purpose": CORPUS_PURPOSE,
            "key_id": corpus_key_id(auth_key),
            "generation_pipeline_sha256": pipeline_sha256,
        },
    }
    manifest[MANIFEST_HMAC_FIELD] = json_hmac_sha256(
        auth_key,
        manifest,
        exclude_field=MANIFEST_HMAC_FIELD,
        domain=MANIFEST_HMAC_DOMAIN,
    )
    from grader.grade import CorpusValidationError, _validate_corpus
    manifest_path = os.path.join(CORPUS, "manifest.json")
    manifest_tmp = os.path.join(os.path.dirname(CORPUS), ".svpgs-manifest.tmp")
    manifest_backup = os.path.join(
        os.path.dirname(CORPUS), ".svpgs-manifest.previous"
    )
    if os.path.exists(manifest_backup):
        # Recover an interrupted transaction before starting a new one. If both
        # files exist, their relationship is ambiguous and requires owner review.
        if os.path.exists(manifest_path):
            sys.exit(
                "[FINALIZE-ABORT] both current and recovery manifests exist"
            )
        os.replace(manifest_backup, manifest_path)

    had_previous_manifest = os.path.exists(manifest_path)
    if had_previous_manifest:
        # Pre-manifest validation intentionally requires an otherwise exact tree.
        # Move the old manifest outside the corpus root while retaining an atomic
        # rollback copy; never validate or publish against stale manifest bytes.
        os.replace(manifest_path, manifest_backup)

    def rollback_manifest_transaction():
        """Restore the exact pre-FINALIZE manifest state after any exception."""
        # Restoration is attempted even if removal of the just-written manifest
        # fails: os.replace can atomically put the recovery copy back over it. Temp
        # cleanup runs last so its own failure cannot strand the published manifest.
        try:
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)
        finally:
            try:
                if had_previous_manifest and os.path.exists(manifest_backup):
                    os.replace(manifest_backup, manifest_path)
            finally:
                if os.path.exists(manifest_tmp):
                    os.unlink(manifest_tmp)

    try:
        # Validate the exact dataset tree before publishing the manifest. A clean
        # corpus intentionally has no manifest yet; accepting that one missing file
        # here avoids relying on a stale previous manifest while still rejecting any
        # undeclared category/dataset material.
        _validate_corpus(
            CORPUS,
            manifest,
            datasets,
            auth_key,
            require_manifest_file=False,
        )
    except BaseException as exc:
        # The old manifest must never remain stranded in the recovery path merely
        # because validation raised an unexpected I/O/value error, KeyboardInterrupt,
        # or another exception outside CorpusValidationError.
        rollback_manifest_transaction()
        if isinstance(exc, (CorpusValidationError, OSError, ValueError)):
            sys.exit(
                f"[FINALIZE-ABORT] schema-v8 pre-manifest validation failed: {exc}"
            )
        raise
    try:
        with open(manifest_tmp, "w") as fh:
            json.dump(manifest, fh, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(manifest_tmp, manifest_path)
        _validate_corpus(CORPUS, manifest, datasets, auth_key)
    except BaseException as exc:
        rollback_manifest_transaction()
        if isinstance(exc, (CorpusValidationError, OSError, ValueError)):
            sys.exit(f"[FINALIZE-ABORT] schema-v8 finalized validation failed: {exc}")
        raise
    if had_previous_manifest:
        try:
            os.unlink(manifest_backup)
        except BaseException:
            # Cleanup is part of the transaction. If the recovery copy cannot be
            # retired, roll back rather than leave both current and recovery manifests
            # and force an ambiguous next FINALIZE.
            rollback_manifest_transaction()
            raise
    print(f"[FINALIZE] {len(datasets)} datasets -> {CORPUS}/manifest.json", flush=True)
    print(f"{'path':34s} {'family':16s} {'gap':>7s} {'weight':>8s}")
    for _ap, a, rel, _ds_dir in recs:
        print(f"{rel:34s} {a['family']:16s} {a['_gap']:+7.4f} {a['weight']:8.4f}")


def build_all(auth_key):
    """Build the whole corpus sequentially (ONE heavy fit at a time -> memory
    safe), resuming only from datasets whose keyed anchors authenticate the exact
    source/config and every produced non-anchor byte. Anything stale or mismatched
    is rebuilt. Finalization independently authenticates the complete grid and
    writes the immutable manifest. This is the implementation of `ALL`."""
    pipeline_sha256 = _generation_pipeline_sha256()
    for cat, replicate in _expected_datasets():
        dataset_id = _dataset_id(cat, replicate, auth_key, pipeline_sha256)
        out_dir = os.path.join(CORPUS, cat, dataset_id)
        ok, why = _provenance_matches(
            out_dir,
            cat,
            replicate,
            auth_key,
            pipeline_sha256,
        )
        if ok:
            print(
                f"[skip] {cat} replicate={replicate} (provenance match)",
                flush=True,
            )
            continue
        if os.path.exists(os.path.join(out_dir, "truth", "anchors.json")):
            print(f"[rebuild] {cat} replicate={replicate}: {why}", flush=True)
        print(
            f"[build] {cat} replicate={replicate} "
            f"N={SHIPPED_REQUESTED_N} P={SHIPPED_REQUESTED_P}",
            flush=True,
        )
        build_one(cat, replicate, auth_key, pipeline_sha256)
    finalize(auth_key, pipeline_sha256)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", required=True)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_dataset_mode(name):
        command = subparsers.add_parser(name)
        command.add_argument("category", choices=list(CATEGORIES))
        command.add_argument(
            "replicate",
            type=int,
            choices=range(REPLICATES_PER_CATEGORY),
        )

    for name in ("ONE", "CHECK", "CHECK_FINALIZED"):
        add_dataset_mode(name)
    subparsers.add_parser("FINALIZE")
    subparsers.add_parser("ALL")
    return parser.parse_args()


def _main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = _parse_args()
    auth_key = load_corpus_key(args.key_file, repository_root=_ROOT)
    pipeline_sha256 = _generation_pipeline_sha256()
    if args.mode == "ONE":
        build_one(
            args.category,
            args.replicate,
            auth_key,
            pipeline_sha256,
        )
    elif args.mode == "CHECK":
        # A keyed anchor authenticates its own contents, exact source/config, and
        # every produced non-anchor byte. This permits safe partial-build resume;
        # FINALIZE separately authenticates the complete corpus manifest.
        dataset_id = _dataset_id(
            args.category,
            args.replicate,
            auth_key,
            pipeline_sha256,
        )
        out_dir = os.path.join(CORPUS, args.category, dataset_id)
        ok, why = _provenance_matches(
            out_dir,
            args.category,
            args.replicate,
            auth_key,
            pipeline_sha256,
        )
        print(
            f"[check] {args.category} replicate={args.replicate}: "
            f"{'match' if ok else why}",
            flush=True,
        )
        sys.exit(0 if ok else 1)
    elif args.mode == "CHECK_FINALIZED":
        # Reuse is permitted only when the finalized schema-v8 manifest authenticates
        # every dataset byte, including anchors.json.
        dataset_id = _dataset_id(
            args.category,
            args.replicate,
            auth_key,
            pipeline_sha256,
        )
        out_dir = os.path.join(CORPUS, args.category, dataset_id)
        ok, why = _provenance_matches(
            out_dir,
            args.category,
            args.replicate,
            auth_key,
            pipeline_sha256,
            require_manifest=True,
        )
        print(
            f"[check-finalized] {args.category} replicate={args.replicate}: "
            f"{'match' if ok else why}",
            flush=True,
        )
        sys.exit(0 if ok else 1)
    elif args.mode == "FINALIZE":
        finalize(auth_key, pipeline_sha256)
    else:
        build_all(auth_key)


if __name__ == "__main__":
    _main()
