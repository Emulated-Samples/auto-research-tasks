from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from datagen import build_corpus
from grader.contract import (
    ANCHOR_HMAC_FIELD,
    CORPUS_BUILDER_CODE_VERSION,
    CORPUS_SCHEMA_VERSION,
    CORPUS_PURPOSE,
    DATASET_WEIGHT,
    GENERATION_PIPELINE_RELATIVE_PATHS,
    MANIFEST_HMAC_FIELD,
    REPLICATES_PER_CATEGORY,
    SHIPPED_CATEGORIES,
    SHIPPED_DATASET_COUNT,
    SHIPPED_REQUESTED_N,
    SHIPPED_REQUESTED_P,
    WEIGHT_RULE,
)
from grader.corpus_auth import (
    ANCHOR_HMAC_DOMAIN,
    MANIFEST_HMAC_DOMAIN,
    corpus_key_id,
    derive_stream_seed,
    json_hmac_sha256,
    load_corpus_key,
    opaque_dataset_id,
)
from grader.grade import (
    CorpusValidationError,
    REFERENCE_ENTRYPOINT,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_IMPLEMENTATION,
    _validate_corpus,
)
from grader.submission_runner import PUBLIC_FILES
from reference.protocol import (
    REFERENCE_CONVERGENCE_TOLERANCE,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
    REFERENCE_MAX_OUTER_ITERATIONS,
    REFERENCE_PROTOCOL_VERSION,
)
from reference.zoo_protocol import REFERENCE_CANDIDATE_ORDER


CATEGORIES = SHIPPED_CATEGORIES
CATEGORY = CATEGORIES[0]
REPLICATE = 0
AUTH_KEY = bytes(range(32))
PIPELINE_SHA256 = "c" * 64
FAMILY = "binomial-logit"


def _optimizer_diagnostics() -> dict:
    return {
        "converged": True,
        "iterations": 3,
        "objective": 0.5,
        "lipschitz": 1.0,
    }


def _candidate_hyperparameters(name: str) -> dict:
    return {
        "sv_pgs": {
            "max_outer_iterations": REFERENCE_MAX_OUTER_ITERATIONS,
            "convergence_tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
        },
        "ridge_logistic": {"penalty": 0.1, "l1_ratio": 0.0},
        "dense_spike_logistic": {
            "dense_l2_penalty": 0.1,
            "sparse_l1_penalty": 0.05,
        },
        "elastic_net_logistic": {"penalty": 0.05, "l1_ratio": 0.5},
        "annotation_adaptive_en": {"penalty": 0.05, "l1_ratio": 0.5},
        "marginal_pt": {"keep_fraction": 0.1},
    }[name]


def _decomposition_diagnostics() -> dict:
    return {
        "transition": 0.5,
        "dense_l2_norm": 1.0,
        "sparse_l1_norm": 0.5,
        "sparse_nonzero_count": 1,
        "coefficient_count": 4,
        "reconstruction_max_abs_error": 0.0,
    }


def _pt_fit_diagnostics(*, full_refit: bool) -> dict:
    diagnostics = {
        "requested_variant_count": 4,
        "kept_variant_count": 2,
        "covariate_optimizer": _optimizer_diagnostics(),
        "calibration_optimizer": _optimizer_diagnostics(),
    }
    if full_refit:
        diagnostics["converged"] = True
    return diagnostics


def _svpgs_fit_diagnostics(wall_time_s: float) -> dict:
    return {
        "converged": True,
        "max_outer_iterations": REFERENCE_MAX_OUTER_ITERATIONS,
        "convergence_tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
        "iterations_run": 25,
        "final_parameter_change": 5e-5,
        "final_predictor_change": 5e-5,
        "final_objective_change": 5e-5,
        "final_hyperparameter_change": 5e-5,
        "wall_time_s": wall_time_s,
    }


def _candidate_tuning_trial(name: str) -> dict:
    trial = {
        "hyperparameters": _candidate_hyperparameters(name),
        "validation_log_loss": 0.5,
    }
    if name == "sv_pgs":
        trial["fit"] = _svpgs_fit_diagnostics(0.1)
    elif name in {"ridge_logistic", "elastic_net_logistic"}:
        trial["optimizer"] = _optimizer_diagnostics()
    elif name == "annotation_adaptive_en":
        trial["optimizer"] = _optimizer_diagnostics()
        trial["penalty_multiplier"] = {
            "minimum": 0.5,
            "median": 1.0,
            "maximum": 2.0,
        }
    elif name == "dense_spike_logistic":
        trial["optimizer"] = _optimizer_diagnostics()
        trial["decomposition"] = _decomposition_diagnostics()
    else:
        trial["fit"] = _pt_fit_diagnostics(full_refit=False)
    return trial


def test_generation_digest_excludes_reference_and_scoring_sources() -> None:
    assert GENERATION_PIPELINE_RELATIVE_PATHS == (
        "datagen/categories.py",
        "datagen/dgp.py",
        "datagen/materialize.py",
        "grader/corpus_auth.py",
    )
    assert all(
        not relative.startswith(("reference/", "validation/"))
        and relative not in {"grader/metrics.py", "grader/truth.py"}
        for relative in GENERATION_PIPELINE_RELATIVE_PATHS
    )


def test_generation_digest_commits_to_explicit_shape_controls(monkeypatch) -> None:
    original = build_corpus._generation_pipeline_sha256()
    monkeypatch.setattr(
        build_corpus,
        "GENERATION_CONTRACT_ITEMS",
        (("categories", SHIPPED_CATEGORIES), ("requested_n", 123)),
    )
    assert build_corpus._generation_pipeline_sha256() != original


def _identity(category: str, replicate: int) -> tuple[str, str]:
    dataset_id = opaque_dataset_id(
        AUTH_KEY,
        purpose=CORPUS_PURPOSE,
        pipeline_sha256=PIPELINE_SHA256,
        category=category,
        replicate=replicate,
    )
    return dataset_id, f"{category}/{dataset_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hashes(dataset_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(dataset_dir).as_posix(): _sha256(path)
        for top_level in ("public", "truth")
        for path in sorted((dataset_dir / top_level).rglob("*"))
        if path.is_file()
    }


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _build_dataset(
    root: Path,
    category: str,
    replicate: int,
) -> tuple[dict, Path]:
    validation_metrics = {"auc": 0.6, "brier": 0.2, "log_loss": 0.5}
    utility_components = {"auc": 0.2, "brier": 0.2, "log_loss": 2.0 / 7.0}
    selection_utility = 38.0 / 175.0
    dataset_id, dataset_path = _identity(category, replicate)
    dataset_dir = root / dataset_path
    public = dataset_dir / "public"
    truth = dataset_dir / "truth"
    for name in PUBLIC_FILES:
        content = (
            FAMILY + "\n"
            if name == "family.txt"
            else f"fixture:{replicate}:{name}\n"
        )
        _write(public / name, content)
    _write(truth / "y_test.csv", "sample_id,y\ns0,0\ns1,1\n")
    _write(truth / "truth.npz", b"schema-v6-truth-fixture")

    data_hashes = _dataset_hashes(dataset_dir)
    anchors = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "id": dataset_id,
        "path": dataset_path,
        "category": category,
        "replicate": replicate,
        "family": FAMILY,
        "weight": DATASET_WEIGHT,
        "metrics_naive": {"auc": 0.6, "brier": 0.25, "log_loss": 0.7},
        "metrics_reference": {"auc": 0.7, "brier": 0.2, "log_loss": 0.6},
        "metrics_ref_naive_se": {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        "metrics_naive_se": {"auc": 0.01, "brier": 0.01, "log_loss": 0.01},
        "time_reference": 10.0,
        "time_calibration": 1.0,
        "time_runtime_anchor": 2.0,
        "time_runtime_calibration": 1.0,
        "reference_fit": {
            "implementation": REFERENCE_IMPLEMENTATION,
            "entrypoint": REFERENCE_ENTRYPOINT,
            "fit_protocol": REFERENCE_FIT_PROTOCOL,
            "python_executable": "/opt/svpgs-venv/bin/python",
            "numpy_version": "2.1.3",
            "public_source_sha256": "a" * 64,
            "converged": True,
        },
        "reference_protocol": {
            "protocol_version": REFERENCE_PROTOCOL_VERSION,
            "reference_method": "hierarchical_eb",
            "converged": True,
            "fit_protocol": REFERENCE_FIT_PROTOCOL,
            "wall_time_s": 1.0,
            "fit_exit_code": 0,
            "fit_source_sha256": "3" * 64,
            "fit_python_executable": "/opt/svpgs-venv/bin/python",
            "fit_numpy_version": "2.1.3",
            "reference_total_fit_wall_s": 1.0,
            "reference_full_fit_wall_s": 0.5,
            "reference_program_reported_wall_s": 0.9,
            "selection_wall_s": 0.4,
            "calibration_intercept": -0.1,
            "calibration_slope": 0.8,
            "calibration_iterations": 4,
            "calibration_relative_gradient": 1e-10,
            "inner_selection_auc": {
                "hierarchical_eb": 0.7, "ridge_logistic": 0.65},
            "inner_ridge_trials": [
                {"penalty": penalty, "converged": True,
                 "auc": 0.65 - 0.001 * index}
                for index, penalty in enumerate(
                    (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
                )
            ],
            "hierarchical_outer_iterations": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_inner_global_scale": 0.5,
            "hierarchical_iterations_run": REFERENCE_HIERARCHICAL_OUTER_ITERATIONS,
            "hierarchical_global_scale": 0.5,
            "prediction_exit_code": 0,
            "prediction_wall_s": 0.1,
            "prediction_output_sha256": "2" * 64,
            "prediction_output_byte_count": 64,
            "prediction_model_sha256": "1" * 64,
            "prediction_source_sha256": "3" * 64,
            "classes_present": ["snv", "deletion_long"],
            "training_output": {
                "model_sha256": "1" * 64,
                "model_byte_count": 1024,
            },
            "execution_order": [
                "all_anchor_fits",
                "prediction_data_load",
                "all_anchor_predictions",
                "truth_load",
            ],
        },
        "provenance": {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "code_version": CORPUS_BUILDER_CODE_VERSION,
            "builder_sha256": hashlib.sha256(
                Path(build_corpus.__file__).read_bytes()
            ).hexdigest(),
            "purpose": CORPUS_PURPOSE,
            "generation_pipeline_sha256": PIPELINE_SHA256,
            "id": dataset_id,
            "path": dataset_path,
            "category": category,
            "replicate": replicate,
            "N": SHIPPED_REQUESTED_N,
            "P": SHIPPED_REQUESTED_P,
            "family": FAMILY,
            "prior_cols": ["variant_class"],
            "cfg": {"n_samples": SHIPPED_REQUESTED_N},
            "file_sha256": data_hashes,
        },
        "_n_train": 10,
        "_n_test": 2,
    }
    anchors[ANCHOR_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        anchors,
        exclude_field=ANCHOR_HMAC_FIELD,
        domain=ANCHOR_HMAC_DOMAIN,
    )
    _write(truth / "anchors.json", json.dumps(anchors, indent=2))
    return {
        "id": dataset_id,
        "path": dataset_path,
        "category": category,
        "replicate": replicate,
        "family": FAMILY,
        "weight": DATASET_WEIGHT,
        "sha256": _dataset_hashes(dataset_dir),
    }, dataset_dir


def _build_valid_corpus(root: Path) -> tuple[dict, Path]:
    datasets = []
    dataset_dirs = []
    for category in CATEGORIES:
        for replicate in range(REPLICATES_PER_CATEGORY):
            entry, dataset_dir = _build_dataset(root, category, replicate)
            datasets.append(entry)
            dataset_dirs.append(dataset_dir)
    assert len(datasets) == SHIPPED_DATASET_COUNT
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "datasets": datasets,
        "meta": {
            "n_datasets": len(datasets),
            "weight_rule": WEIGHT_RULE,
            "categories": list(CATEGORIES),
            "replicates_per_category": REPLICATES_PER_CATEGORY,
            "requested_n": SHIPPED_REQUESTED_N,
            "requested_p": SHIPPED_REQUESTED_P,
            "development_report_sha256": "d" * 64,
            "purpose": CORPUS_PURPOSE,
            "key_id": corpus_key_id(AUTH_KEY),
            "generation_pipeline_sha256": PIPELINE_SHA256,
        },
    }
    _refresh_manifest_hmac(manifest)
    return manifest, dataset_dirs[0]


def _validate(root: Path, manifest: dict) -> list[str]:
    # Project the in-memory manifest onto disk first: the shipped corpus root IS the
    # manifest plus the declared category tree, and _validate_corpus enforces that
    # exactly (require_manifest_file defaults True). Validating with the file absent
    # would exercise a shape the corpus never has, so tamper tests below stay honest
    # only if disk and memory agree at call time.
    (root / "manifest.json").write_text(json.dumps(manifest))
    return _validate_corpus(str(root), manifest, manifest["datasets"], AUTH_KEY)


def _refresh_manifest_hashes(manifest: dict, dataset_dir: Path) -> None:
    manifest["datasets"][0]["sha256"] = _dataset_hashes(dataset_dir)
    _refresh_manifest_hmac(manifest)


def _refresh_anchor_hmac(anchors: dict) -> None:
    anchors[ANCHOR_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        anchors,
        exclude_field=ANCHOR_HMAC_FIELD,
        domain=ANCHOR_HMAC_DOMAIN,
    )


def _refresh_manifest_hmac(manifest: dict) -> None:
    manifest[MANIFEST_HMAC_FIELD] = json_hmac_sha256(
        AUTH_KEY,
        manifest,
        exclude_field=MANIFEST_HMAC_FIELD,
        domain=MANIFEST_HMAC_DOMAIN,
    )


def test_schema_v4_valid_contract_is_accepted(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    assert _validate(tmp_path, manifest) == [
        _identity(category, replicate)[0]
        for category in CATEGORIES
        for replicate in range(REPLICATES_PER_CATEGORY)
    ]


def test_previous_schema_manifest_is_rejected_without_compatibility(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["schema_version"] = CORPUS_SCHEMA_VERSION - 1
    with pytest.raises(
        CorpusValidationError,
        match=f"schema_version must be {CORPUS_SCHEMA_VERSION}",
    ):
        _validate(tmp_path, manifest)


def test_wrong_key_cannot_authenticate_manifest(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    with pytest.raises(CorpusValidationError, match="manifest HMAC mismatch"):
        _validate_corpus(
            str(tmp_path),
            manifest,
            manifest["datasets"],
            b"x" * 32,
        )


def test_manifest_tamper_fails_before_semantic_validation(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["meta"]["requested_n"] = 4_000
    with pytest.raises(CorpusValidationError, match="manifest HMAC mismatch"):
        _validate(tmp_path, manifest)


def test_identity_and_rng_streams_are_opaque_and_domain_separated() -> None:
    dataset_id, dataset_path = _identity(CATEGORY, REPLICATE)
    assert len(dataset_id) == 34
    assert dataset_id.startswith("d_")
    assert all(character in "0123456789abcdef" for character in dataset_id[2:])
    assert dataset_path == f"{CATEGORY}/{dataset_id}"
    seeds = {
        derive_stream_seed(
            AUTH_KEY,
            purpose=CORPUS_PURPOSE,
            pipeline_sha256=PIPELINE_SHA256,
            category=CATEGORY,
            replicate=REPLICATE,
            stream=stream,
        )
        for stream in ("dgp", "materialize", "bootstrap", "dataset-id")
    }
    assert len(seeds) == 4
    assert all(0 <= seed < 2**128 for seed in seeds)


@pytest.mark.parametrize(
    ("category", "replicate"),
    [
        ("not_a_shipping_category", 0),
        (next(iter(build_corpus.CATEGORIES)), -1),
        (next(iter(build_corpus.CATEGORIES)), REPLICATES_PER_CATEGORY),
        (next(iter(build_corpus.CATEGORIES)), True),
    ],
)
def test_builder_rejects_cells_outside_the_exact_shipping_grid(
    category: str,
    replicate: int,
) -> None:
    with pytest.raises(ValueError):
        build_corpus._dataset_id(
            category,
            replicate,
            AUTH_KEY,
            PIPELINE_SHA256,
        )


def test_schema_v4_serializes_replicates_but_no_generation_seed(tmp_path: Path) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor = json.loads((dataset_dir / "truth" / "anchors.json").read_text())
    assert manifest["datasets"][0]["replicate"] == 0
    assert "seed" not in manifest["datasets"][0]
    assert "seed" not in manifest["meta"]
    assert anchor["replicate"] == 0
    assert "seed" not in anchor
    assert "seed" not in anchor["provenance"]
    assert "seed" not in anchor["provenance"]["cfg"]


def test_key_loader_requires_private_external_regular_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    key_path = tmp_path / "corpus.key"
    key_path.write_bytes(AUTH_KEY)
    key_path.chmod(0o600)
    assert load_corpus_key(
        str(key_path.resolve()),
        repository_root=str(repository),
    ) == AUTH_KEY

    key_path.chmod(0o640)
    with pytest.raises(ValueError, match="mode must be exactly 0600"):
        load_corpus_key(str(key_path.resolve()), repository_root=str(repository))


def test_key_loader_rejects_symlink_wrong_size_and_repository_key(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    key_path = tmp_path / "corpus.key"
    key_path.write_bytes(AUTH_KEY)
    key_path.chmod(0o600)

    link_path = tmp_path / "key-link"
    link_path.symlink_to(key_path)
    with pytest.raises(ValueError, match="cannot securely open"):
        load_corpus_key(str(link_path.absolute()), repository_root=str(repository))

    hard_link = tmp_path / "key-hard-link"
    hard_link.hardlink_to(key_path)
    with pytest.raises(ValueError, match="exactly one hard link"):
        load_corpus_key(str(key_path.resolve()), repository_root=str(repository))
    hard_link.unlink()

    key_path.write_bytes(AUTH_KEY + b"x")
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        load_corpus_key(str(key_path.resolve()), repository_root=str(repository))

    internal_key = repository / "corpus.key"
    internal_key.write_bytes(AUTH_KEY)
    internal_key.chmod(0o600)
    with pytest.raises(ValueError, match="outside the repository"):
        load_corpus_key(str(internal_key.resolve()), repository_root=str(repository))


def test_manifest_and_anchor_require_full_shipped_dimensions(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["meta"]["requested_n"] = 4_000
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="requested dimensions"):
        _validate(tmp_path, manifest)

    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["provenance"]["P"] = 400
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="provenance P"):
        _validate(tmp_path, manifest)


@pytest.mark.parametrize(
    "relative_path",
    ["public/formula.txt", "truth/y_test.csv", "truth/anchors.json"],
)
def test_any_public_or_truth_tamper_is_rejected(
    tmp_path: Path,
    relative_path: str,
) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    with (dataset_dir / relative_path).open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(CorpusValidationError, match="SHA-256 mismatch"):
        _validate(tmp_path, manifest)


def test_undeclared_dataset_file_is_rejected(tmp_path: Path) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    _write(dataset_dir / "truth" / "extra.txt", "not declared\n")
    with pytest.raises(CorpusValidationError, match="digest file set mismatch"):
        _validate(tmp_path, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "wrong", "dataset id mismatch"),
        ("path", "wrong/path", "dataset path mismatch"),
        ("category", "other", "order/grid mismatch"),
        ("replicate", 1, "order/grid mismatch"),
    ],
)
def test_manifest_identity_fields_are_exact(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["datasets"][0][field] = value
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match=message):
        _validate(tmp_path, manifest)


def test_schema_version_and_equal_weight_rule_are_required(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["schema_version"] = 1
    with pytest.raises(CorpusValidationError, match="schema_version"):
        _validate(tmp_path, manifest)

    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["meta"]["weight_rule"] = "outcome_gap_weighted"
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="weight_rule"):
        _validate(tmp_path, manifest)

    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["datasets"][0]["weight"] = 0.5
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="weight must equal"):
        _validate(tmp_path, manifest)


def test_unconverged_reference_is_rejected_even_with_fresh_digest(tmp_path: Path) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["reference_fit"]["converged"] = False
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="reference fit is unconverged"):
        _validate(tmp_path, manifest)


def test_every_reference_metric_must_pass_at_least_one_reliability_gate(
    tmp_path: Path,
) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["metrics_reference"] = {
        "auc": anchors["metrics_naive"]["auc"] + 2e-9,
        "brier": anchors["metrics_naive"]["brier"] - 2e-9,
        "log_loss": anchors["metrics_naive"]["log_loss"] - 2e-9,
    }
    anchors["metrics_ref_naive_se"] = {
        "auc": 1.0,
        "brier": 1.0,
        "log_loss": 1.0,
    }
    anchors["metrics_naive_se"] = {
        "auc": 1.0,
        "brier": 1.0,
        "log_loss": 1.0,
    }
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="every reference metric"):
        _validate(tmp_path, manifest)


def test_zero_paired_gap_se_is_rejected_by_corpus_preflight(
    tmp_path: Path,
) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["metrics_ref_naive_se"]["auc"] = 0.0
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="invalid SE auc=0.0"):
        _validate(tmp_path, manifest)


def test_dataset_with_only_unweighted_metric_adequate_is_rejected(
    tmp_path: Path,
) -> None:
    # Scoring is AUC-only (METRIC_WEIGHTS={"auc":1.0}). A dataset whose AUC gap is
    # inadequate but whose brier gap is adequate would otherwise clear preflight yet
    # score 0 for EVERY submission (reference included). The guard must reject it.
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    # AUC gap tiny relative to a large SE -> AUC inadequate/unreliable.
    anchors["metrics_reference"]["auc"] = anchors["metrics_naive"]["auc"] + 2e-9
    # brier gap large and reliable -> brier is active (but unweighted).
    anchors["metrics_reference"]["brier"] = anchors["metrics_naive"]["brier"] - 0.05
    anchors["metrics_ref_naive_se"]["auc"] = 1.0
    anchors["metrics_naive_se"]["auc"] = 1.0
    anchors["metrics_ref_naive_se"]["brier"] = 1e-4
    anchors["metrics_naive_se"]["brier"] = 1e-4
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="no SCORED metric"):
        _validate(tmp_path, manifest)


def test_losing_metric_is_excluded_when_another_metric_is_reliably_active(
    tmp_path: Path,
) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["metrics_reference"]["brier"] = (
        anchors["metrics_naive"]["brier"] + 0.01
    )
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)

    _validate(tmp_path, manifest)


def test_category_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    manifest, _dataset_dir = _build_valid_corpus(backing)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / CATEGORY).symlink_to(backing / CATEGORY, target_is_directory=True)
    with pytest.raises(CorpusValidationError, match="category is not a directory"):
        _validate(corpus, manifest)


def test_anchor_symlink_is_rejected_even_when_digest_is_declared(tmp_path: Path) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    target_path = tmp_path / "anchor-target.json"
    target_path.write_bytes(anchor_path.read_bytes())
    anchor_path.unlink()
    anchor_path.symlink_to(target_path)
    manifest["datasets"][0]["sha256"]["truth/anchors.json"] = _sha256(target_path)
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="not a regular file"):
        _validate(tmp_path, manifest)


def test_generation_pipeline_digest_is_required(tmp_path: Path) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["provenance"]["generation_pipeline_sha256"] = "invalid"
    _refresh_anchor_hmac(anchors)
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="generation pipeline digest"):
        _validate(tmp_path, manifest)


def test_modified_anchor_payload_cannot_be_reblessed_by_manifest_hmac(
    tmp_path: Path,
) -> None:
    manifest, dataset_dir = _build_valid_corpus(tmp_path)
    anchor_path = dataset_dir / "truth" / "anchors.json"
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors["time_reference"] += 1.0
    _write(anchor_path, json.dumps(anchors, indent=2))
    _refresh_manifest_hashes(manifest, dataset_dir)
    with pytest.raises(CorpusValidationError, match="anchor HMAC mismatch"):
        _validate(tmp_path, manifest)


def test_schema_version_is_an_exact_integer(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["schema_version"] = 2.0
    with pytest.raises(CorpusValidationError, match="schema_version"):
        _validate(tmp_path, manifest)


def test_declared_grid_requires_five_replicates_per_category(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["datasets"] = manifest["datasets"][:4]
    manifest["meta"]["n_datasets"] = 4
    manifest["meta"]["replicates_per_category"] = 4
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="replicates_per_category"):
        _validate(tmp_path, manifest)


def test_declared_grid_requires_exactly_fifteen_categories(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["datasets"] = manifest["datasets"][:-REPLICATES_PER_CATEGORY]
    manifest["meta"]["n_datasets"] = len(manifest["datasets"])
    manifest["meta"]["categories"] = manifest["meta"]["categories"][:-1]
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="canonical ordered"):
        _validate(tmp_path, manifest)


def test_manifest_category_names_and_order_are_exact(tmp_path: Path) -> None:
    manifest, _dataset_dir = _build_valid_corpus(tmp_path)
    manifest["meta"]["categories"][0:2] = reversed(
        manifest["meta"]["categories"][0:2]
    )
    _refresh_manifest_hmac(manifest)
    with pytest.raises(CorpusValidationError, match="canonical ordered"):
        _validate(tmp_path, manifest)
