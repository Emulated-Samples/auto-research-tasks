"""Grade a submission over the corpus. Outcome env: reward = signed accuracy skill,
tail-aware category aggregation, naive=0 / principled reference=1.

Per dataset:
  1. run submission fit/predict on public/ -> pred.csv (submission_runner)
  2. score pred vs truth/y_test.csv with family metrics (metrics.py)
  3. accuracy skill vs anchors (truth/anchors.json: naive + reference metric
     values + reference fit+predict time) (skill.py)
  4. report runtime headroom/efficiency (perf.py); hard caps enforce speed
  5. reward = signed accuracy skill ; validity failure -> INVALID_REWARD
Aggregate: 60% category-equal mean + 40% weakest-fifth category mean.

Both manifest.json and anchors.json use corpus schema version 8. The manifest
pins every public/truth file by SHA-256 and carries keyed opaque identities and
HMACs. Public replicate ordinals never expose the private generation streams.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat

import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader.metrics import compute_metrics
from grader.truth import read_binary_truth_csv
from grader.contract import (
    ANCHOR_HMAC_FIELD,
    CORPUS_SCHEMA_VERSION,
    CORPUS_PURPOSE,
    CORPUS_BUILDER_CODE_VERSION,
    DATASET_WEIGHT,
    MANIFEST_HMAC_FIELD,
    MASTERY_THRESHOLD,
    MASTERY_TAIL_THRESHOLD,
    METRIC_WEIGHTS,
    REPLICATES_PER_CATEGORY,
    SHIPPED_CATEGORIES,
    SHIPPED_CATEGORY_COUNT,
    SHIPPED_DATASET_COUNT,
    SHIPPED_REQUESTED_N,
    SHIPPED_REQUESTED_P,
    WEIGHT_RULE,
)
from grader.corpus_auth import (
    ANCHOR_HMAC_DOMAIN,
    MANIFEST_HMAC_DOMAIN,
    corpus_key_id,
    load_corpus_key,
    opaque_dataset_id,
    verify_json_hmac,
)
from grader.skill import (INVALID_REWARD, K_SE, RESOLUTION_K, SKILL_HI, SKILL_LO,
                          accuracy_skill, apply_calibration_factor,
                          calibration_factor, category_aggregation, clamp_report,
                          reward_reason)
from grader.perf import (perf_factor, efficiency_ratio, calibration_seconds,
                         normalized_reference_time)
from grader.submission_runner import (run_on_dataset, build_submission,
                                      SandboxUnavailable, _rmtree_ro,
                                      PUBLIC_FILES)
from reference.protocol import (
    REFERENCE_ANCHOR_EXECUTION_ORDER,
    REFERENCE_ENTRYPOINT,
    REFERENCE_FIT_PROTOCOL,
    REFERENCE_IMPLEMENTATION,
    REFERENCE_NUMPY_VERSION,
    REFERENCE_PYTHON_EXECUTABLE,
    validate_reference_diagnostics,
)


class CorpusValidationError(RuntimeError):
    """The trusted corpus/anchor bundle is malformed and must not be scored."""


def _hib_map(family):
    """higher_is_better per metric name, to re-attach to anchor scalar values."""
    if family != "binomial-logit":
        raise CorpusValidationError(
            f"unsupported family {family!r}; expected 'binomial-logit'")
    return {"auc": True, "brier": False, "log_loss": False}


def _require_regular(path, label):
    try:
        st = os.lstat(path)
    except OSError as e:
        raise CorpusValidationError(f"{label} missing/unreadable at {path}: {e}") from e
    if not stat.S_ISREG(st.st_mode):
        raise CorpusValidationError(f"{label} is not a regular file: {path}")


def _require_directory(path, label):
    try:
        st = os.lstat(path)
    except OSError as e:
        raise CorpusValidationError(f"{label} missing/unreadable at {path}: {e}") from e
    if not stat.S_ISDIR(st.st_mode):
        raise CorpusValidationError(f"{label} is not a directory: {path}")


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hashes(dataset_dir):
    """Hash every regular public/truth file and reject symlinks/special files."""
    hashes = {}
    for top_level in ("public", "truth"):
        root = os.path.join(dataset_dir, top_level)
        _require_directory(root, f"dataset {top_level}")
        for current, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            for dirname in dirnames:
                _require_directory(
                    os.path.join(current, dirname),
                    f"dataset nested directory {dirname}",
                )
            for filename in filenames:
                path = os.path.join(current, filename)
                relative = os.path.relpath(path, dataset_dir).replace(os.sep, "/")
                _require_regular(path, f"dataset file {relative}")
                hashes[relative] = _hash_file(path)
    return dict(sorted(hashes.items()))


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, ValueError):
        return False


def _reject_nonfinite_json(token):
    raise ValueError(f"non-finite JSON constant {token!r}")


def _validate_corpus(
    corpus_dir,
    manifest,
    entries,
    auth_key,
    *,
    require_manifest_file=True,
):
    """Validate every trusted dataset and anchor before running untrusted code.

    A broken truth file, stale anchor, or malformed manifest is infrastructure
    failure, never a reward-zero submission. This preflight also prevents a
    partially copied corpus from being silently averaged into the score.
    """
    if not isinstance(manifest, dict):
        raise CorpusValidationError("manifest must be a JSON object")
    if (type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != CORPUS_SCHEMA_VERSION):
        raise CorpusValidationError(
            f"manifest schema_version must be {CORPUS_SCHEMA_VERSION}")
    required_manifest_fields = {
        "schema_version",
        "datasets",
        "meta",
        MANIFEST_HMAC_FIELD,
    }
    if set(manifest) != required_manifest_fields:
        raise CorpusValidationError(
            f"manifest fields do not match schema v{CORPUS_SCHEMA_VERSION}: "
            f"expected={sorted(required_manifest_fields)}, got={sorted(manifest)}"
        )
    if not verify_json_hmac(
        auth_key,
        manifest,
        field=MANIFEST_HMAC_FIELD,
        domain=MANIFEST_HMAC_DOMAIN,
    ):
        raise CorpusValidationError("manifest HMAC mismatch")
    _require_directory(corpus_dir, "corpus root")
    if not isinstance(entries, list) or not entries:
        raise CorpusValidationError("manifest.datasets must be a non-empty list")
    meta = manifest.get("meta")
    if not isinstance(meta, dict):
        raise CorpusValidationError("manifest.meta must be an object")
    required_meta_fields = {
        "n_datasets",
        "weight_rule",
        "categories",
        "replicates_per_category",
        "requested_n",
        "requested_p",
        "development_report_sha256",
        "purpose",
        "key_id",
        "generation_pipeline_sha256",
    }
    if set(meta) != required_meta_fields:
        raise CorpusValidationError(
            f"manifest.meta fields do not match schema v{CORPUS_SCHEMA_VERSION}: "
            f"expected={sorted(required_meta_fields)}, got={sorted(meta)}"
        )
    declared = meta.get("n_datasets")
    if type(declared) is not int or declared != len(entries):
        raise CorpusValidationError(
            f"manifest dataset count mismatch: meta={declared!r}, actual={len(entries)}")
    if meta["purpose"] != CORPUS_PURPOSE:
        raise CorpusValidationError(
            f"manifest purpose must equal {CORPUS_PURPOSE!r}")
    if meta["key_id"] != corpus_key_id(auth_key):
        raise CorpusValidationError("manifest key_id does not match the corpus key")
    if not _valid_sha256(meta["generation_pipeline_sha256"]):
        raise CorpusValidationError(
            "manifest generation_pipeline_sha256 must be a lowercase SHA-256 digest"
        )
    if (
        type(meta["requested_n"]) is not int
        or meta["requested_n"] != SHIPPED_REQUESTED_N
        or type(meta["requested_p"]) is not int
        or meta["requested_p"] != SHIPPED_REQUESTED_P
    ):
        raise CorpusValidationError(
            "manifest requested dimensions must equal the shipped contract "
            f"N={SHIPPED_REQUESTED_N}, P={SHIPPED_REQUESTED_P}"
        )
    if not _valid_sha256(meta["development_report_sha256"]):
        raise CorpusValidationError(
            "manifest development_report_sha256 must be a lowercase SHA-256 digest"
        )
    categories = meta["categories"]
    if categories != list(SHIPPED_CATEGORIES):
        raise CorpusValidationError(
            "manifest categories must exactly match the canonical ordered "
            f"{SHIPPED_CATEGORY_COUNT}-category matrix"
        )
    replicates_per_category = meta["replicates_per_category"]
    if (type(replicates_per_category) is not int
            or replicates_per_category != REPLICATES_PER_CATEGORY):
        raise CorpusValidationError(
            "manifest replicates_per_category must equal "
            f"{REPLICATES_PER_CATEGORY}"
        )
    expected_grid = [
        (category, replicate)
        for category in categories
        for replicate in range(replicates_per_category)
    ]
    if len(entries) != len(expected_grid):
        raise CorpusValidationError(
            "manifest datasets do not fill the declared category/replicate grid")
    if len(entries) != SHIPPED_DATASET_COUNT:
        raise CorpusValidationError(
            f"shipping manifest must contain exactly {SHIPPED_DATASET_COUNT} datasets"
        )
    if meta.get("weight_rule") != WEIGHT_RULE:
        raise CorpusValidationError(
            f"manifest weight_rule must be {WEIGHT_RULE!r}")

    seen_paths = set()
    seen_ids = set()
    dataset_ids = []
    required_entry_fields = {
        "id", "path", "category", "replicate", "family", "weight", "sha256",
    }
    for entry, expected_grid_cell in zip(entries, expected_grid, strict=True):
        if not isinstance(entry, dict):
            raise CorpusValidationError("manifest dataset entries must be objects")
        if set(entry) != required_entry_fields:
            raise CorpusValidationError(
                f"manifest dataset entry fields do not match schema v{CORPUS_SCHEMA_VERSION}: "
                f"expected={sorted(required_entry_fields)}, got={sorted(entry)}"
            )

        category = entry["category"]
        replicate = entry["replicate"]
        if (not isinstance(category, str) or not category
                or category in {".", ".."} or "/" in category or "\\" in category):
            raise CorpusValidationError(f"invalid dataset category {category!r}")
        if type(replicate) is not int or replicate < 0:
            raise CorpusValidationError(f"invalid dataset replicate {replicate!r}")
        if (category, replicate) != expected_grid_cell:
            raise CorpusValidationError(
                "manifest dataset order/grid mismatch: "
                f"expected={expected_grid_cell!r}, got={(category, replicate)!r}"
            )
        expected_id = opaque_dataset_id(
            auth_key,
            purpose=CORPUS_PURPOSE,
            pipeline_sha256=meta["generation_pipeline_sha256"],
            category=category,
            replicate=replicate,
        )
        expected_path = f"{category}/{expected_id}"
        dataset_id = entry["id"]
        rel = entry["path"]
        if dataset_id != expected_id:
            raise CorpusValidationError(
                f"dataset id mismatch: expected {expected_id!r}, got {dataset_id!r}")
        if rel != expected_path:
            raise CorpusValidationError(
                f"dataset path mismatch: expected {expected_path!r}, got {rel!r}")
        if rel in seen_paths:
            raise CorpusValidationError(f"duplicate dataset path {rel!r}")
        if dataset_id in seen_ids:
            raise CorpusValidationError(f"duplicate dataset id {dataset_id!r}")
        seen_paths.add(rel)
        seen_ids.add(dataset_id)
        dataset_ids.append(dataset_id)

        if not _valid_number(entry["weight"]):
            raise CorpusValidationError(f"{dataset_id}: invalid weight")
        weight = float(entry["weight"])
        if weight != DATASET_WEIGHT:
            raise CorpusValidationError(
                f"{dataset_id}: schema-v{CORPUS_SCHEMA_VERSION} weight must equal "
                f"{DATASET_WEIGHT}")

        category_dir = os.path.join(corpus_dir, category)
        _require_directory(category_dir, f"{category} category")
        dataset_dir = os.path.join(category_dir, dataset_id)
        _require_directory(dataset_dir, f"{dataset_id} dataset")
        declared_hashes = entry["sha256"]
        if not isinstance(declared_hashes, dict) or not declared_hashes:
            raise CorpusValidationError(f"{dataset_id}: sha256 must be a non-empty object")
        for file_path, digest in declared_hashes.items():
            if (not isinstance(file_path, str) or "\\" in file_path
                    or os.path.normpath(file_path) != file_path
                    or not file_path.startswith(("public/", "truth/"))
                    or file_path.startswith(("public/../", "truth/../"))):
                raise CorpusValidationError(
                    f"{dataset_id}: unsafe digest path {file_path!r}")
            if not _valid_sha256(digest):
                raise CorpusValidationError(
                    f"{dataset_id}: invalid SHA-256 for {file_path!r}")
        actual_hashes = _dataset_hashes(dataset_dir)
        if set(actual_hashes) != set(declared_hashes):
            missing = sorted(set(actual_hashes) - set(declared_hashes))
            unexpected = sorted(set(declared_hashes) - set(actual_hashes))
            raise CorpusValidationError(
                f"{dataset_id}: digest file set mismatch "
                f"(undeclared={missing}, absent={unexpected})")
        for file_path, expected_digest in declared_hashes.items():
            if actual_hashes[file_path] != expected_digest:
                raise CorpusValidationError(
                    f"{dataset_id}: SHA-256 mismatch for {file_path}")

        public = os.path.join(dataset_dir, "public")
        truth = os.path.join(dataset_dir, "truth")
        for name in PUBLIC_FILES:
            plain = os.path.join(public, name)
            gz = plain + ".gz"
            present = [path for path in (plain, gz) if os.path.lexists(path)]
            if len(present) != 1:
                raise CorpusValidationError(
                    f"{dataset_id}: expected exactly one of {name} or {name}.gz")
            _require_regular(present[0], f"{dataset_id} public {name}")

        family_path = os.path.join(public, "family.txt")
        _require_regular(family_path, f"{dataset_id} family")
        with open(family_path, encoding="utf-8") as family_file:
            family = family_file.read().strip()
        if family != entry["family"]:
            raise CorpusValidationError(
                f"{dataset_id}: family mismatch manifest={entry['family']!r} public={family!r}")
        hib = _hib_map(family)

        anchor_path = os.path.join(truth, "anchors.json")
        y_path = os.path.join(truth, "y_test.csv")
        _require_regular(anchor_path, f"{dataset_id} anchors")
        _require_regular(y_path, f"{dataset_id} held-out labels")
        try:
            with open(anchor_path, encoding="utf-8") as anchor_file:
                anchors = json.load(
                    anchor_file,
                    parse_constant=_reject_nonfinite_json,
                )
        except (OSError, ValueError) as e:
            raise CorpusValidationError(f"{dataset_id}: invalid anchors.json: {e}") from e
        if not isinstance(anchors, dict):
            raise CorpusValidationError(f"{dataset_id}: anchors.json must be an object")
        if not verify_json_hmac(
            auth_key,
            anchors,
            field=ANCHOR_HMAC_FIELD,
            domain=ANCHOR_HMAC_DOMAIN,
        ):
            raise CorpusValidationError(f"{dataset_id}: anchor HMAC mismatch")
        if (type(anchors.get("schema_version")) is not int
                or anchors["schema_version"] != CORPUS_SCHEMA_VERSION):
            raise CorpusValidationError(
                f"{dataset_id}: anchor schema is not v{CORPUS_SCHEMA_VERSION}")
        for identity_field in ("id", "path", "category", "replicate", "family"):
            if anchors.get(identity_field) != entry[identity_field]:
                raise CorpusValidationError(
                    f"{dataset_id}: anchor {identity_field} mismatch")
        if "weight" not in anchors or not _valid_number(anchors["weight"]):
            raise CorpusValidationError(f"{dataset_id}: invalid anchor weight")
        anchor_weight = float(anchors["weight"])
        if anchor_weight != DATASET_WEIGHT:
            raise CorpusValidationError(
                f"{dataset_id}: anchor weight must equal {DATASET_WEIGHT}")

        reference_fit = anchors.get("reference_fit")
        required_reference_fields = {
            "implementation", "entrypoint", "fit_protocol",
            "python_executable", "numpy_version",
            "public_source_sha256",
            "converged",
        }
        if not isinstance(reference_fit, dict) or set(reference_fit) != required_reference_fields:
            raise CorpusValidationError(
                f"{dataset_id}: invalid reference_fit provenance fields")
        if (
            reference_fit["implementation"] != REFERENCE_IMPLEMENTATION
            or reference_fit["entrypoint"] != REFERENCE_ENTRYPOINT
            or reference_fit["fit_protocol"] != REFERENCE_FIT_PROTOCOL
            or reference_fit["python_executable"] != REFERENCE_PYTHON_EXECUTABLE
            or reference_fit["numpy_version"] != REFERENCE_NUMPY_VERSION
            or reference_fit["converged"] is not True
            or not _valid_sha256(reference_fit["public_source_sha256"])
        ):
            raise CorpusValidationError(
                f"{dataset_id}: reference fit is unconverged or has invalid provenance")

        n_train = anchors.get("_n_train")
        n_test = anchors.get("_n_test")
        if (
            type(n_train) is not int
            or n_train <= 0
            or type(n_test) is not int
            or n_test <= 0
        ):
            raise CorpusValidationError(
                f"{dataset_id}: anchor train/test dimensions are invalid"
            )
        reference_protocol = anchors.get("reference_protocol")
        try:
            validate_reference_diagnostics(
                reference_protocol,
                training_count=n_train,
                execution_order=REFERENCE_ANCHOR_EXECUTION_ORDER,
            )
        except ValueError as exc:
            raise CorpusValidationError(
                f"{dataset_id}: invalid strong-reference protocol diagnostics"
            ) from exc

        provenance = anchors.get("provenance")
        if not isinstance(provenance, dict):
            raise CorpusValidationError(f"{dataset_id}: missing anchor provenance")
        if (type(provenance.get("schema_version")) is not int
                or provenance["schema_version"] != CORPUS_SCHEMA_VERSION):
            raise CorpusValidationError(
                f"{dataset_id}: provenance schema_version mismatch")
        for identity_field in (
            "schema_version",
            "id",
            "path",
            "category",
            "replicate",
            "family",
        ):
            expected = (
                CORPUS_SCHEMA_VERSION
                if identity_field == "schema_version"
                else entry[identity_field]
            )
            if provenance.get(identity_field) != expected:
                raise CorpusValidationError(
                    f"{dataset_id}: provenance {identity_field} mismatch")
        if provenance.get("code_version") != CORPUS_BUILDER_CODE_VERSION:
            raise CorpusValidationError(f"{dataset_id}: provenance code_version mismatch")
        expected_builder_sha256 = _hash_file(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datagen",
            "build_corpus.py",
        ))
        if provenance.get("builder_sha256") != expected_builder_sha256:
            raise CorpusValidationError(
                f"{dataset_id}: provenance builder digest does not match the live builder"
            )
        if provenance.get("purpose") != CORPUS_PURPOSE:
            raise CorpusValidationError(f"{dataset_id}: invalid provenance purpose")
        if not _valid_sha256(provenance.get("generation_pipeline_sha256")):
            raise CorpusValidationError(
                f"{dataset_id}: invalid provenance generation pipeline digest")
        if (
            provenance["generation_pipeline_sha256"]
            != meta["generation_pipeline_sha256"]
        ):
            raise CorpusValidationError(
                f"{dataset_id}: provenance generation pipeline mismatch"
            )
        if "seed" in anchors or "seed" in provenance or "eff_seed" in provenance:
            raise CorpusValidationError(
                f"{dataset_id}: schema-v{CORPUS_SCHEMA_VERSION} corpus must not "
                "serialize generation seeds"
            )
        expected_dimensions = {
            "N": SHIPPED_REQUESTED_N,
            "P": SHIPPED_REQUESTED_P,
        }
        for dimension, expected_dimension in expected_dimensions.items():
            if (
                type(provenance.get(dimension)) is not int
                or provenance[dimension] != expected_dimension
                or provenance[dimension] != meta[f"requested_{dimension.lower()}"]
            ):
                raise CorpusValidationError(
                    f"{dataset_id}: provenance {dimension} does not match the "
                    "shipped dimension contract"
                )
        prior_cols = provenance.get("prior_cols")
        if (not isinstance(prior_cols, list)
                or any(not isinstance(column, str) or not column for column in prior_cols)
                or len(prior_cols) != len(set(prior_cols))):
            raise CorpusValidationError(f"{dataset_id}: invalid provenance prior_cols")
        if (
            not isinstance(provenance.get("cfg"), dict)
            or "seed" in provenance["cfg"]
        ):
            raise CorpusValidationError(f"{dataset_id}: invalid provenance cfg")
        expected_data_hashes = {
            path: digest
            for path, digest in declared_hashes.items()
            if path != "truth/anchors.json"
        }
        if provenance.get("file_sha256") != expected_data_hashes:
            raise CorpusValidationError(
                f"{dataset_id}: provenance file digests do not match manifest")

        naive = anchors.get("metrics_naive")
        reference = anchors.get("metrics_reference")
        if not isinstance(naive, dict) or not isinstance(reference, dict):
            raise CorpusValidationError(f"{dataset_id}: missing metric anchors")
        if set(naive) != set(hib) or set(reference) != set(hib):
            raise CorpusValidationError(
                f"{dataset_id}: metric keys do not match {family} contract")
        for name, higher in hib.items():
            if not (_valid_number(naive[name]) and _valid_number(reference[name])):
                raise CorpusValidationError(
                    f"{dataset_id}: non-finite/non-numeric {name} anchor")
        for key in (
            "time_reference", "time_calibration",
            "time_runtime_anchor", "time_runtime_calibration",
        ):
            if key not in anchors or not _valid_number(anchors[key]):
                raise CorpusValidationError(f"{dataset_id}: invalid {key}")
            value = float(anchors[key])
            if value <= 0:
                raise CorpusValidationError(f"{dataset_id}: invalid {key}={value!r}")
        se = anchors.get("metrics_ref_naive_se")
        if not isinstance(se, dict) or set(se) != set(hib):
            raise CorpusValidationError(
                f"{dataset_id}: metrics_ref_naive_se keys do not match contract")
        for name, sval in se.items():
            if not _valid_number(sval):
                raise CorpusValidationError(
                    f"{dataset_id}: non-finite/non-numeric SE for {name}")
            sv = float(sval)
            if sv <= 0:
                raise CorpusValidationError(f"{dataset_id}: invalid SE {name}={sv!r}")
        # The ADEQUACY yardstick (skill.RESOLUTION_K). Required, not optional: an
        # anchor that cannot show its gap is large enough to be a skill
        # denominator is a broken anchor, and metric_skill's se_naive=None branch
        # would silently skip the adequacy test rather than fail. Fail closed here
        # so that branch is never reachable from a shipped corpus.
        naive_se = anchors.get("metrics_naive_se")
        if not isinstance(naive_se, dict) or set(naive_se) != set(hib):
            raise CorpusValidationError(
                f"{dataset_id}: metrics_naive_se keys do not match contract")
        for name, sval in naive_se.items():
            if not _valid_number(sval) or float(sval) <= 0:
                raise CorpusValidationError(
                    f"{dataset_id}: invalid naive SE {name}={sval!r}")
        # A metric is active only if its gap is BOTH reliable (>= K_SE*paired_se)
        # AND adequate (>= RESOLUTION_K*naive_se) -- the exact rule metric_skill
        # applies at scoring time. Checking only reliability here let a corpus pass
        # preflight and then score every metric NaN (accuracy_skill -> 0 for even the
        # reference). See grader/skill.py:RESOLUTION_K.
        active_metrics = []
        for name, higher in hib.items():
            naive_value = float(naive[name])
            reference_value = float(reference[name])
            gap = (
                reference_value - naive_value
                if higher
                else naive_value - reference_value
            )
            if (gap > 1e-9 and gap >= K_SE * float(se[name])
                    and gap >= RESOLUTION_K * float(naive_se[name])):
                active_metrics.append(name)
        if not active_metrics:
            raise CorpusValidationError(
                f"{dataset_id}: every reference metric fails the reliability gate")
        # The guard must match the SCORING predicate, not merely "some metric is
        # gradable". accuracy_skill weights only metrics in METRIC_WEIGHTS (AUC), so a
        # dataset whose AUC gap fails adequacy but whose brier gap passes would clear
        # the check above yet score 0 for EVERY submission -- reference included --
        # when metric_skill NaNs the AUC and the weighted sum collapses to 0
        # (skill.py:253). Require at least one *weighted* metric to be adequate, so
        # preflight tests the adequacy of what is actually divided by. See MEMORY:
        # guards must test adequacy, not existence.
        weighted_active = [m for m in active_metrics if METRIC_WEIGHTS.get(m, 0.0) > 0.0]
        if not weighted_active:
            raise CorpusValidationError(
                f"{dataset_id}: no SCORED metric (weight>0) has a reliable, adequate "
                f"gap; active but unweighted={active_metrics!r}")

        try:
            _, y = read_binary_truth_csv(
                os.path.join(truth, "y_test.csv")
            )
        except (OSError, ValueError, IndexError) as e:
            raise CorpusValidationError(f"{dataset_id}: invalid y_test.csv: {e}") from e
        if n_test != len(y):
            raise CorpusValidationError(f"{dataset_id}: anchor _n_test mismatch")

    # Reject dead/unmanifested corpus material. Packaging extra dataset directories is
    # not harmless: it leaves hidden truth and public bytes in the released image that
    # no score or provenance record owns. The filesystem must be the exact manifest
    # projection -- one manifest plus the declared category/dataset directory tree.
    root_entries = set(os.listdir(corpus_dir))
    expected_root_entries = set(SHIPPED_CATEGORIES)
    if require_manifest_file:
        expected_root_entries.add("manifest.json")
    if root_entries != expected_root_entries:
        raise CorpusValidationError(
            "corpus root contains undeclared or missing entries: "
            f"extra={sorted(root_entries - expected_root_entries)}, "
            f"missing={sorted(expected_root_entries - root_entries)}"
        )
    declared_by_category = {
        category: {
            entry["id"] for entry in entries if entry["category"] == category
        }
        for category in SHIPPED_CATEGORIES
    }
    for category, declared_ids in declared_by_category.items():
        category_dir = os.path.join(corpus_dir, category)
        actual_ids = set(os.listdir(category_dir))
        if actual_ids != declared_ids:
            raise CorpusValidationError(
                f"{category}: category directory does not match the manifest: "
                f"extra={sorted(actual_ids - declared_ids)}, "
                f"missing={sorted(declared_ids - actual_ids)}"
            )
    return dataset_ids


def grade_one(dataset_dir, submission_dir, *, grade_calibration):
    public = os.path.join(dataset_dir, "public")
    truth = os.path.join(dataset_dir, "truth")
    anchors = json.load(open(os.path.join(truth, "anchors.json")))
    family = anchors["family"]
    y_sids, y_test = read_binary_truth_csv(os.path.join(truth, "y_test.csv"))
    hib = _hib_map(family)
    naive_m = {key: (value, hib[key]) for key, value in anchors["metrics_naive"].items()}
    ref_m = {key: (value, hib[key]) for key, value in anchors["metrics_reference"].items()}
    # Two independent activation yardsticks; a metric must pass BOTH.
    #   ref_naive_se -- paired bootstrap SE of the (reference - naive) gap:
    #     RELIABILITY, "is the gap real?"
    #   naive_se -- bootstrap SE of the single naive estimate:
    #     ADEQUACY, "is the gap large enough to divide by?"
    # A metric failing either is excluded before the remaining weights are
    # renormalized; an active metric keeps its actual gap so reference=1 remains
    # exact. See grader/skill.py:RESOLUTION_K for why the paired SE cannot answer
    # the second question (it shrinks with the gap; corr = +0.925 on the v6 corpus).
    ref_naive_se = dict(anchors["metrics_ref_naive_se"])
    naive_se = dict(anchors["metrics_naive_se"])

    res = run_on_dataset(submission_dir, public, n_test_expected=len(y_test),
                         family=family)
    detail = {"dataset": os.path.basename(dataset_dir), "category": anchors["category"],
              "weight": anchors["weight"], "status": res["status"],
              "t_fit": round(res["t_fit"], 3), "t_predict": round(res["t_predict"], 3)}
    if res["status"] != "ok":
        detail.update(reward=INVALID_REWARD, accuracy=0.0,
                      raw_skill=INVALID_REWARD, perf=0.0, note=res["detail"],
                      reward_reason=reward_reason(res["status"], INVALID_REWARD))
        return {"category": anchors["category"], "weight": anchors["weight"],
                "reward": INVALID_REWARD}, detail

    pred = res["pred"]
    # The public contract requires one prediction per test sample IN INPUT ORDER.
    # An optional sample_id column is useful self-checking metadata, not permission
    # to reorder output behind the contract. The old alignment path silently made
    # an explicitly out-of-order submission valid.
    if "sample_id" in pred and [str(value) for value in pred["sample_id"]] != [
        str(value) for value in y_sids
    ]:
        detail.update(reward=INVALID_REWARD, accuracy=0.0,
                      raw_skill=INVALID_REWARD, perf=0.0,
                      status="bad_pred",
                      note="pred sample_id is not in test input order",
                      reward_reason=reward_reason("bad_pred", INVALID_REWARD))
        return {"category": anchors["category"], "weight": anchors["weight"],
                "reward": INVALID_REWARD}, detail
    sub_metrics = compute_metrics(family, y_test, pred)
    # `auc_skill` is the SIGNED discrimination skill before its positive part is
    # calibration-discounted. Scoring a floored value would collapse every
    # below-naive fit onto an identical 0, so a weak model and a catastrophic one
    # would be indistinguishable to RL and to any calibration study.
    auc_skill, per = accuracy_skill(sub_metrics, naive_m, ref_m, ref_naive_se,
                                    naive_se)
    t_sub = res["t_fit"] + res["t_predict"]
    t_runtime_anchor = float(anchors["time_runtime_anchor"])
    # Rescale the anchor time to THIS host's speed via the required calibration
    # ratio. This feeds the REPORTED efficiency ratio only -- a genuine cross-host
    # comparison against a practical production model.
    t_ref_eff = normalized_reference_time(
        t_runtime_anchor,
        anchors["time_runtime_calibration"],
        grade_calibration)
    # Runtime headroom is REPORT-ONLY. The disclosed 170s/30s hard caps enforce the
    # speed contract. Multiplying a valid fit's scientific skill by runtime made the
    # authenticated reference score below 1 and paid shortcuts more than principled
    # fits; the reference=1 scale is now exact for every valid in-budget submission.
    perf = perf_factor(t_sub)
    # CALIBRATION: a continuous multiplier on POSITIVE AUC credit, not an
    # additive arm and not a discontinuous all-or-zero gate.
    #
    # The prompt tells the agent its predictions must "remain probabilistically
    # calibrated, and avoid severe overconfidence". Nothing funded that promise.
    # Adding brier/log_loss skill is not sound: their naive->reference gaps are
    # tiny, so normalized ratios become clamp arithmetic and a zero-information
    # base-rate predictor receives free credit. They therefore carry no additive
    # weight, but their absolute proper-score regret remains load-bearing here.
    #
    # The old loose gate against naive let a rank-perfect submission compress all
    # probabilities toward prevalence and keep FULL AUC credit. Reference-relative
    # proper-score regret now discounts that credit smoothly. Exact reference
    # predictions have factor 1; severe overconfidence tends toward 0. Negative
    # AUC skill is deliberately unchanged so miscalibration can never raise it.
    calibration = calibration_factor(sub_metrics, ref_m)
    reward = apply_calibration_factor(auc_skill, calibration["factor"])
    accuracy = max(reward, 0.0)
    eff = efficiency_ratio(t_sub, t_ref_eff)
    # Authenticated scientific fields are machine-consumed, not display strings.
    # The wrapper reconstructs each category from these per-dataset rewards and
    # compares it with the full-precision aggregate at 1e-9. Rounding the detail
    # while aggregating the unrounded value made honest grades fail integrity.
    detail.update(reward=float(reward), accuracy=float(accuracy),
                  raw_skill=float(reward), auc_skill=float(auc_skill), perf=float(perf),
                  efficiency=(round(eff, 3) if eff is not None else None),
                  t_ref=round(float(anchors["time_reference"]), 3),
                  t_runtime_anchor=round(t_runtime_anchor, 3),
                  t_ref_eff=round(float(t_ref_eff), 3),
                  per_metric={k: round(v, 3) for k, v in per.items()},
                  # A bound-saturated AUC ratio is censored, not an unbounded
                  # measurement. Report that state without changing the reward.
                  **clamp_report(per),
                  calibration_factor=float(calibration["factor"]),
                  calibration_metric_factors={
                      k: float(v) for k, v in calibration["per_metric_factor"].items()
                  },
                  calibration_regret={
                      k: float(v) for k, v in calibration["regret"].items()
                  },
                  reward_reason=reward_reason(
                      "ok", reward, per,
                      calibration_factor=calibration["factor"]),
                  sub_metrics={k: round(v[0], 4) for k, v in sub_metrics.items()})
    return {"category": anchors["category"], "weight": anchors["weight"],
            "reward": reward}, detail


def _zero_record(entry, status, note):
    """A floor-reward record for a dataset that was never validly graded,
    built from the manifest entry alone (no fit/predict, no truth read)."""
    cat = entry["category"]
    w = entry["weight"]
    agg = {"category": cat, "weight": w, "reward": INVALID_REWARD}
    det = {"dataset": entry["id"], "category": cat, "weight": w,
           "status": status, "reward": INVALID_REWARD, "accuracy": 0.0,
           "raw_skill": INVALID_REWARD, "perf": 0.0,
           "t_fit": 0.0, "t_predict": 0.0, "note": note}
    return agg, det


def _build_reward(per_dataset, details):
    """Assemble the (reward_json, reward_detail) pair for the results so far.
    Pure: does not touch disk."""
    benchmark, per_cat, aggregation = category_aggregation(per_dataset)
    reward_json = {"reward": benchmark}
    reward_detail = {"schema_version": CORPUS_SCHEMA_VERSION,
                     "reward": benchmark,
                     # The pass/mastery event needs BOTH the headline and the
                     # bottom-tail aggregate (a lopsided rollout must not master).
                     # The wrapper gates on both; see contract.py.
                     "mastery_threshold": MASTERY_THRESHOLD,
                     "mastery_tail_threshold": MASTERY_TAIL_THRESHOLD,
                     "mastery_tail_value": aggregation["tail_value"],
                     "score_bounds": {"min_dataset_reward": SKILL_LO,
                                      "max_dataset_reward": SKILL_HI},
                     "additional_data": {
                         "per_category": per_cat,
                         "aggregation": aggregation,
                     },
                     "subscores": [{"name": f"category:{c}", "score": s,
                                    "weight": aggregation["category_coefficients"][c]}
                                   for c, s in sorted(per_cat.items())],
                     "datasets": details}
    return reward_json, reward_detail


def _atomic_write_json(path, obj):
    """Write JSON to `path` atomically (temp file + os.replace) so a reader never
    observes a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _write_partial(out_dir, reward_detail):
    """Snapshot in-progress detail to a partial file that the trusted
    wrapper NEVER reads. It exists only for crash forensics / debugging; because
    the wrapper ignores it, a grader killed mid-corpus cannot have a partial
    score accepted (the trusted reward_detail.json is only written on full,
    successful completion by _finalize)."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    _atomic_write_json(os.path.join(out_dir, "reward_detail.partial.json"),
                       reward_detail)


def _finalize(out_dir, reward_json, reward_detail, *, nonce, dataset_ids,
              manifest_sha256):
    """Write the TRUSTED reward detail, once, atomically, on full completion.

    reward_detail.json carries the integrity fields the wrapper cross-checks
    before it will accept any score:
      complete=true            -- the grading pass ran to the end (not killed)
      grader_nonce             -- proves this file came from THIS grader run,
                                  not a submission-planted forgery
      datasets=[...]           -- one entry per graded dataset id; the wrapper
                                  requires len == the corpus's dataset count
      manifest_sha256          -- ties the file to the exact corpus that was
                                  graded
    """
    reward_detail = dict(reward_detail)
    reward_detail.update(complete=True, grader_nonce=nonce,
                         dataset_ids=list(dataset_ids),
                         manifest_sha256=manifest_sha256)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _atomic_write_json(os.path.join(out_dir, "reward_detail.json"),
                           reward_detail)
    return reward_json, reward_detail


def grade_corpus(
    corpus_dir,
    submission_dir,
    *,
    key_file,
    out_dir=None,
    log=print,
):
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    auth_key = load_corpus_key(key_file, repository_root=repository_root)
    # Read the manifest as raw bytes so its sha256 matches the wrapper's digest of
    # the identical file, tying the trusted reward file to the exact corpus graded.
    with open(os.path.join(corpus_dir, "manifest.json"), "rb") as fh:
        manifest_bytes = fh.read()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(
            manifest_bytes,
            parse_constant=_reject_nonfinite_json,
        )
    except ValueError as exc:
        raise CorpusValidationError(f"invalid manifest.json: {exc}") from exc
    entries = manifest.get("datasets") if isinstance(manifest, dict) else None
    dataset_ids = _validate_corpus(corpus_dir, manifest, entries, auth_key)
    # Per-run nonce. The wrapper supplies exactly 128 random bits and requires the
    # reward file to echo it, binding the artifact to this invocation.
    nonce = os.environ.get("SVPGSBENCH_GRADER_NONCE")
    if (not isinstance(nonce, str) or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)):
        raise CorpusValidationError("missing/invalid 128-bit grader nonce")
    log(f"[grade] grader_nonce={nonce}")
    # Measure this host's speed once (single-threaded env is set by the wrapper);
    # anchors store the reference host's calibration, and the ratio makes the perf
    # factor portable across grading hardware.
    grade_calibration = calibration_seconds()
    log(f"[grade] host calibration = {grade_calibration*1000:.1f} ms")

    # Optional one-time compile step (contract: build.sh runs once before grading),
    # now in an ISOLATED, FROZEN copy sandboxed with no network / no truth mounts;
    # `built_dir` (source + any build artifacts) becomes the source for every
    # fit/predict. A build failure is terminal: every dataset scores the validity
    # floor rather than silently grading an unbuilt submission.
    build_ok, built_dir, build_note = build_submission(submission_dir, log=log)

    # Grade in authenticated manifest order. The runner owns fixed, independent
    # fit and predict caps for every dataset; there is no shared clock, category
    # bank, global bank, or time transfer. A slow dataset can therefore affect
    # only its own record. Any trusted exception or wrapper-level timeout escapes
    # before `_finalize`, so an incomplete corpus can never become a trusted score.
    graded = []
    try:
        for entry in entries:
            if not build_ok:
                record = _zero_record(entry, "build_failed",
                                      build_note or "build.sh failed")
            else:
                ds_dir = os.path.join(corpus_dir, entry["path"])
                record = grade_one(
                    ds_dir,
                    built_dir,
                    grade_calibration=grade_calibration,
                )
            graded.append(record)
            det = record[1]
            log(f"[grade] {det['dataset']:24s} cat={det['category']:16s} "
                f"reward={det['reward']:.3f} acc={det.get('accuracy',0):.3f} "
                f"perf={det.get('perf',0):.3f} status={det['status']}")
            # Incremental snapshot to reward_detail.partial.json only. The trusted file
            # is NOT written here, so a grader killed mid-corpus leaves no
            # acceptable score behind (see _write_partial / _finalize).
            _, partial_detail = _build_reward(
                [r[0] for r in graded], [r[1] for r in graded])
            _write_partial(out_dir, partial_detail)
    finally:
        if built_dir:
            _rmtree_ro(built_dir)

    # Results are graded and reported in manifest order; the wrapper cross-checks
    # datasets[i] against manifest entry i.
    per_dataset = [record[0] for record in graded]
    details = [record[1] for record in graded]

    # Full, successful completion: write the trusted reward_detail.json exactly
    # once, atomically, stamped with the completeness/provenance fields.
    reward_json, reward_detail = _build_reward(per_dataset, details)
    return _finalize(out_dir, reward_json, reward_detail, nonce=nonce,
                     dataset_ids=dataset_ids, manifest_sha256=manifest_sha256)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("submission")
    ap.add_argument("--out", default=None)
    ap.add_argument("--key-file", required=True)
    a = ap.parse_args()
    try:
        rj, rd = grade_corpus(
            a.corpus,
            a.submission,
            key_file=a.key_file,
            out_dir=a.out,
        )
    except SandboxUnavailable as e:
        # Fail closed: do NOT emit a reward file (the wrapper treats a missing
        # reward_detail.json as an errored grade), so no unconfined score exists.
        print(f"ABORT: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(rj, indent=2))
    print("per-category:", json.dumps(rd["additional_data"]["per_category"], indent=2))
