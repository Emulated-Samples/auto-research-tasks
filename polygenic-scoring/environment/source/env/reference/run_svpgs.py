"""Public-data loaders plus the hidden model-zoo SV-PGS adapter.

The loaders enforce the serialized contract used by corpus construction and the
development model zoo: the training loader cannot open prediction tables and
returns an immutable type with no prediction-data fields; trusted call sites invoke
the separate prediction loader only after fitting. ``dgp.json`` selects the
covariate and annotation columns, sample/variant IDs are validated across tables,
and no generator metadata enters a fit.

The SV-PGS estimator in this module is an actively measured hidden model-zoo arm,
not the shipped reference. The shipped anchor is the exact public NumPy program in
``gold/fit``; see ``reference/protocol.py``.
The separate ``fit_svpgs_generated`` entry point exists only for synthetic
validation studies that intentionally operate before materialization.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import gzip
import json
import os
from pathlib import Path
import stat
import sys
import time
from types import MappingProxyType
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from reference.protocol import (  # noqa: E402
    REFERENCE_CONVERGENCE_TOLERANCE,
    REFERENCE_MAX_OUTER_ITERATIONS,
)


_ANNOTATION_TYPES = {
    "variant_class",
    "class_members",
    "class_membership",
    "binary",
    "continuous",
    "categorical",
}


class VariantClass(str, Enum):
    """Public metadata enum; the hidden zoo engine is imported only at fit time."""

    SNV = "snv"
    SMALL_INDEL = "small_indel"
    DELETION_SHORT = "deletion_short"
    DELETION_LONG = "deletion_long"
    DUPLICATION_SHORT = "duplication_short"
    DUPLICATION_LONG = "duplication_long"
    INSERTION_MEI = "insertion_mei"
    INVERSION_BND_COMPLEX = "inversion_bnd_complex"
    STR_VNTR_REPEAT = "str_vntr_repeat"
    OTHER_COMPLEX_SV = "other_complex_sv"
def _validated_annotation_types(
    values: Any,
    *,
    expected_names: tuple[str, ...] | None = None,
) -> dict[str, str]:
    if type(values) is not dict:
        raise ValueError("annotation_types must be a plain dictionary")
    annotation_types = dict(values)
    reserved_type_names = {
        "variant_class": "variant_class",
        "class_members": "prior_class_members",
        "class_membership": "prior_class_membership",
    }
    members_present = "prior_class_members" in annotation_types
    membership_present = "prior_class_membership" in annotation_types
    if (
        not annotation_types
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(annotation_type, str)
            or annotation_type not in _ANNOTATION_TYPES
            for name, annotation_type in annotation_types.items()
        )
        or annotation_types.get("variant_class") != "variant_class"
        or any(
            annotation_type in reserved_type_names
            and name != reserved_type_names[annotation_type]
            for name, annotation_type in annotation_types.items()
        )
        or members_present != membership_present
        or (
            members_present
            and (
                annotation_types["prior_class_members"] != "class_members"
                or annotation_types["prior_class_membership"]
                != "class_membership"
            )
        )
        or (
            expected_names is not None
            and set(annotation_types) != set(expected_names)
        )
    ):
        raise ValueError("annotation_types do not match the public schema")
    return annotation_types


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _immutable_array(values: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return an array backed by immutable bytes, not merely a cleared flag."""
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class PublicPredictorSchema:
    family: str
    covariate_names: tuple[str, ...]
    variant_ids: tuple[str, ...]
    annotation_types: Mapping[str, str]

    def __post_init__(self) -> None:
        covariate_names = tuple(self.covariate_names)
        variant_ids = tuple(self.variant_ids)
        annotation_types = _validated_annotation_types(self.annotation_types)
        if (
            self.family != "binomial-logit"
            or not covariate_names
            or any(
                not isinstance(value, str) or not value
                for value in covariate_names
            )
            or len(set(covariate_names)) != len(covariate_names)
            or not variant_ids
            or any(not isinstance(value, str) or not value for value in variant_ids)
            or len(set(variant_ids)) != len(variant_ids)
        ):
            raise ValueError("public predictor schema is invalid")
        object.__setattr__(self, "covariate_names", covariate_names)
        object.__setattr__(self, "variant_ids", variant_ids)
        object.__setattr__(
            self,
            "annotation_types",
            _immutable_mapping(annotation_types),
        )


@dataclass(frozen=True, slots=True)
class ReferenceVariantRecord:
    """Immutable benchmark-side representation converted at the engine boundary."""

    variant_id: str
    variant_class: VariantClass
    chromosome: str
    position: int
    length: float = 1.0
    allele_frequency: float = 0.01
    quality: float = 1.0
    training_support: int | None = None
    is_repeat: bool = False
    is_copy_number: bool = False
    prior_binary_features: Mapping[str, bool] = field(default_factory=dict)
    prior_continuous_features: Mapping[str, float] = field(default_factory=dict)
    prior_categorical_features: Mapping[str, str] = field(default_factory=dict)
    prior_class_members: tuple[VariantClass, ...] = ()
    prior_class_membership: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        binary_features = dict(self.prior_binary_features)
        continuous_features = dict(self.prior_continuous_features)
        categorical_features = dict(self.prior_categorical_features)
        class_members = tuple(self.prior_class_members)
        class_membership = tuple(float(value) for value in self.prior_class_membership)
        if (
            not isinstance(self.variant_id, str)
            or not self.variant_id
            or type(self.variant_class) is not VariantClass
            or not isinstance(self.chromosome, str)
            or not self.chromosome
            or type(self.position) is not int
            or self.position <= 0
            or not np.isfinite(self.length)
            or self.length <= 0.0
            or not np.isfinite(self.allele_frequency)
            or not 0.0 <= self.allele_frequency <= 1.0
            or not np.isfinite(self.quality)
            or self.quality < 0.0
            or (
                self.training_support is not None
                and (
                    type(self.training_support) is not int
                    or self.training_support < 0
                )
            )
            or type(self.is_repeat) is not bool
            or type(self.is_copy_number) is not bool
            or any(
                not isinstance(name, str)
                or not name
                or type(value) is not bool
                for name, value in binary_features.items()
            )
            or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                for name, value in continuous_features.items()
            )
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
                for name, value in categorical_features.items()
            )
            or any(type(member) is not VariantClass for member in class_members)
            or len(set(class_members)) != len(class_members)
            or len(class_members) != len(class_membership)
            or any(not np.isfinite(value) or value < 0.0 for value in class_membership)
            or (
                class_members
                and not np.isclose(
                    sum(class_membership),
                    1.0,
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            or (not class_members and class_membership)
        ):
            raise ValueError("reference variant record is invalid")
        object.__setattr__(
            self,
            "prior_binary_features",
            _immutable_mapping(binary_features),
        )
        object.__setattr__(
            self,
            "prior_continuous_features",
            _immutable_mapping(continuous_features),
        )
        object.__setattr__(
            self,
            "prior_categorical_features",
            _immutable_mapping(categorical_features),
        )
        object.__setattr__(self, "prior_class_members", class_members)
        object.__setattr__(
            self,
            "prior_class_membership",
            class_membership,
        )


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    schema: PublicPredictorSchema
    sample_ids: tuple[str, ...]
    genotypes: np.ndarray
    covariates: np.ndarray
    targets: np.ndarray
    variant_records: tuple[ReferenceVariantRecord, ...]

    def __post_init__(self) -> None:
        if type(self.schema) is not PublicPredictorSchema:
            raise TypeError("training schema must be PublicPredictorSchema")
        sample_ids = tuple(self.sample_ids)
        raw_targets = np.asarray(self.targets)
        genotypes = _immutable_array(self.genotypes, dtype=np.dtype(np.float32))
        covariates = _immutable_array(self.covariates, dtype=np.dtype(np.float32))
        targets = _immutable_array(self.targets, dtype=np.dtype(np.float32))
        records = tuple(self.variant_records)
        expected_binary = {
            name
            for name, annotation_type in self.schema.annotation_types.items()
            if annotation_type == "binary"
        }
        expected_continuous = {
            name
            for name, annotation_type in self.schema.annotation_types.items()
            if annotation_type == "continuous"
        }
        expected_categorical = {
            name
            for name, annotation_type in self.schema.annotation_types.items()
            if annotation_type == "categorical"
        }
        expects_soft_membership = (
            self.schema.annotation_types.get("prior_class_members")
            == "class_members"
        )
        if (
            not sample_ids
            or any(
                not isinstance(sample_id, str) or not sample_id
                for sample_id in sample_ids
            )
            or len(set(sample_ids)) != len(sample_ids)
            or genotypes.ndim != 2
            or covariates.ndim != 2
            or raw_targets.ndim != 1
            or genotypes.shape[0] != len(sample_ids)
            or covariates.shape != (len(sample_ids), len(self.schema.covariate_names))
            or targets.shape != (len(sample_ids),)
            or genotypes.shape[1] != len(self.schema.variant_ids)
            or len(records) != len(self.schema.variant_ids)
            or any(type(record) is not ReferenceVariantRecord for record in records)
            or tuple(record.variant_id for record in records)
            != self.schema.variant_ids
            or any(
                set(record.prior_binary_features) != expected_binary
                or set(record.prior_continuous_features) != expected_continuous
                or set(record.prior_categorical_features) != expected_categorical
                or bool(record.prior_class_members) is not expects_soft_membership
                for record in records
            )
            or not np.all(np.isfinite(genotypes))
            or not np.all((genotypes == 0.0) | (genotypes == 1.0) | (genotypes == 2.0))
            or not np.all(np.isfinite(covariates))
            or set(np.unique(targets)) != {0.0, 1.0}
        ):
            raise ValueError("training dataset does not align with its predictor schema")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "genotypes", genotypes)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "variant_records", records)


@dataclass(frozen=True, slots=True)
class PredictionDataset:
    schema: PublicPredictorSchema
    sample_ids: tuple[str, ...]
    genotypes: np.ndarray
    covariates: np.ndarray

    def __post_init__(self) -> None:
        if type(self.schema) is not PublicPredictorSchema:
            raise TypeError("prediction schema must be PublicPredictorSchema")
        sample_ids = tuple(self.sample_ids)
        genotypes = _immutable_array(self.genotypes, dtype=np.dtype(np.float32))
        covariates = _immutable_array(self.covariates, dtype=np.dtype(np.float32))
        if (
            not sample_ids
            or any(
                not isinstance(sample_id, str) or not sample_id
                for sample_id in sample_ids
            )
            or len(set(sample_ids)) != len(sample_ids)
            or genotypes.ndim != 2
            or covariates.ndim != 2
            or genotypes.shape != (len(sample_ids), len(self.schema.variant_ids))
            or covariates.shape != (len(sample_ids), len(self.schema.covariate_names))
            or not np.all(np.isfinite(genotypes))
            or not np.all((genotypes == 0.0) | (genotypes == 1.0) | (genotypes == 2.0))
            or not np.all(np.isfinite(covariates))
        ):
            raise ValueError("prediction dataset does not align with its predictor schema")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "genotypes", genotypes)
        object.__setattr__(self, "covariates", covariates)


def _public_path(public_dir: str | os.PathLike[str], name: str) -> Path:
    base = Path(public_dir) / name
    candidates = [
        path
        for path in (base, Path(str(base) + ".gz"))
        if path.exists() or path.is_symlink()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one regular public file for {name!r}; found {candidates}"
        )
    candidate = candidates[0]
    if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
        raise ValueError(f"public file must be a plain regular file: {candidate}")
    return candidate


def _open_public(public_dir: str | os.PathLike[str], name: str):
    path = _public_path(public_dir, name)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _unique_strings(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    resolved = tuple(values)
    if any(not isinstance(value, str) or not value for value in resolved):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} contains duplicates")
    return resolved


def _reject_nonfinite_json(token: str):
    raise ValueError(f"non-finite JSON constant {token!r}")


def _load_public_spec(public_dir: str | os.PathLike[str]):
    with _open_public(public_dir, "dgp.json") as handle:
        spec = json.load(handle, parse_constant=_reject_nonfinite_json)
    if set(spec) != {"family", "formula"}:
        raise ValueError("dgp.json must contain exactly family and formula")
    if spec["family"] != "binomial-logit":
        raise ValueError("reference supports only family='binomial-logit'")
    formula = spec["formula"]
    required_formula_fields = {
        "response",
        "pgs_annotations",
        "annotation_types",
        "covariates",
    }
    if not isinstance(formula, dict) or set(formula) != required_formula_fields:
        raise ValueError("dgp.json formula does not match the public contract")
    if formula["response"] != "y":
        raise ValueError("dgp.json formula response must be 'y'")
    annotations = _unique_strings(formula["pgs_annotations"], "pgs_annotations")
    covariates = _unique_strings(formula["covariates"], "covariates")
    try:
        annotation_types = _validated_annotation_types(
            formula["annotation_types"],
            expected_names=annotations,
        )
    except ValueError as exc:
        raise ValueError(
            "annotation_types must type every pgs annotation exactly"
        ) from exc
    with _open_public(public_dir, "family.txt") as handle:
        if handle.read().strip() != spec["family"]:
            raise ValueError("family.txt disagrees with dgp.json")
    expected_formula = (
        f"y ~ pgs({' + '.join(annotations)}) + "
        f"covariate({', '.join(covariates)})"
    )
    with _open_public(public_dir, "formula.txt") as handle:
        if handle.read().strip() != expected_formula:
            raise ValueError("formula.txt disagrees with dgp.json")
    return spec["family"], annotations, covariates, dict(annotation_types)


def _read_covariates(
    public_dir: str | os.PathLike[str],
    name: str,
    selected_columns: tuple[str, ...],
    *,
    require_targets: bool,
):
    with _open_public(public_dir, name) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{name} is empty") from exc
        if not header or header[0] != "sample_id" or len(set(header)) != len(header):
            raise ValueError(f"{name} has an invalid or duplicate header")
        if require_targets and "y" not in header:
            raise ValueError(f"{name} is missing training target y")
        if not require_targets and "y" in header:
            raise ValueError(f"{name} exposes a held-out target")
        missing = [column for column in selected_columns if column not in header]
        if missing:
            raise ValueError(f"{name} is missing formula covariates {missing}")
        selected_indices = [header.index(column) for column in selected_columns]
        target_index = header.index("y") if require_targets else None
        sample_ids: list[str] = []
        rows: list[list[float]] = []
        targets: list[float] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{name}:{line_number} has the wrong number of fields")
            sample_id = row[0]
            if not sample_id:
                raise ValueError(f"{name}:{line_number} has an empty sample_id")
            sample_ids.append(sample_id)
            try:
                rows.append([float(row[index]) for index in selected_indices])
                if target_index is not None:
                    targets.append(float(row[target_index]))
            except ValueError as exc:
                raise ValueError(f"{name}:{line_number} contains a non-numeric value") from exc
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{name} must contain unique sample IDs")
    matrix = np.asarray(rows, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32) if require_targets else None
    if not np.all(np.isfinite(matrix)) or (
        target_array is not None and not np.all(np.isfinite(target_array))
    ):
        raise ValueError(f"{name} contains non-finite values")
    return tuple(sample_ids), matrix, target_array


def _read_genotypes(public_dir: str | os.PathLike[str], name: str):
    with _open_public(public_dir, name) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{name} is empty") from exc
        if len(header) < 2 or header[0] != "sample_id":
            raise ValueError(f"{name} has an invalid header")
        variant_ids = tuple(header[1:])
        if any(not value for value in variant_ids) or len(set(variant_ids)) != len(variant_ids):
            raise ValueError(f"{name} has empty or duplicate variant IDs")
        sample_ids: list[str] = []
        rows: list[list[int]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{name}:{line_number} has the wrong number of fields")
            sample_ids.append(row[0])
            try:
                dosage = [int(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(f"{name}:{line_number} contains a non-integer dosage") from exc
            if any(value not in (0, 1, 2) for value in dosage):
                raise ValueError(f"{name}:{line_number} contains dosage outside 0/1/2")
            rows.append(dosage)
    if not sample_ids or any(not value for value in sample_ids):
        raise ValueError(f"{name} has empty sample IDs")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{name} has duplicate sample IDs")
    return tuple(sample_ids), variant_ids, np.asarray(rows, dtype=np.float32)


def _strict_bool(value: str, *, column: str) -> bool:
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"binary annotation {column!r} must contain only 0 or 1")


def _read_variant_records(
    public_dir: str | os.PathLike[str],
    variant_ids: tuple[str, ...],
    annotation_types: dict[str, str],
    train_genotypes: np.ndarray,
) -> tuple[ReferenceVariantRecord, ...]:
    with _open_public(public_dir, "variant_metadata.tsv") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("variant_metadata.tsv has an invalid or duplicate header")
        required = {"variant_id", *annotation_types}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"variant_metadata.tsv is missing required columns {missing}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            variant_id = row.get("variant_id", "")
            if not variant_id or variant_id in rows:
                raise ValueError(
                    f"variant_metadata.tsv:{line_number} has an empty or duplicate variant_id"
                )
            rows[variant_id] = row
    if set(rows) != set(variant_ids):
        raise ValueError("variant metadata IDs do not match genotype variant IDs")

    records: list[ReferenceVariantRecord] = []
    for index, variant_id in enumerate(variant_ids):
        row = rows[variant_id]
        try:
            variant_class = VariantClass(row["variant_class"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{variant_id}: invalid variant_class") from exc
        binary: dict[str, bool] = {}
        continuous: dict[str, float] = {}
        categorical: dict[str, str] = {}
        for column, annotation_type in annotation_types.items():
            value = row[column]
            if value == "":
                raise ValueError(f"{variant_id}: annotation {column!r} is empty")
            if annotation_type == "binary":
                binary[column] = _strict_bool(value, column=column)
            elif annotation_type == "continuous":
                parsed = float(value)
                if not np.isfinite(parsed):
                    raise ValueError(f"{variant_id}: annotation {column!r} is non-finite")
                continuous[column] = parsed
            elif annotation_type == "categorical":
                categorical[column] = value

        class_members: tuple[VariantClass, ...] = ()
        class_membership: tuple[float, ...] = ()
        if "prior_class_members" in annotation_types:
            class_members = tuple(
                VariantClass(value.strip())
                for value in row["prior_class_members"].split(",")
                if value.strip()
            )
            class_membership = tuple(
                float(value.strip())
                for value in row["prior_class_membership"].split(",")
                if value.strip()
            )
            if (
                not class_members
                or len(class_members) != len(class_membership)
                or any(not np.isfinite(value) or value < 0 for value in class_membership)
                or not np.isclose(sum(class_membership), 1.0, rtol=0.0, atol=1e-6)
            ):
                raise ValueError(f"{variant_id}: invalid soft class membership")

        observed_allele_frequency = float(
            np.mean(train_genotypes[:, index], dtype=np.float64) / 2.0
        )
        allele_frequency = (
            continuous["allele_frequency"]
            if "allele_frequency" in annotation_types
            else observed_allele_frequency
        )
        records.append(
            ReferenceVariantRecord(
                variant_id=variant_id,
                variant_class=variant_class,
                chromosome="1",
                position=index + 1,
                allele_frequency=allele_frequency,
                training_support=int(np.count_nonzero(train_genotypes[:, index])),
                is_repeat=binary["repeat_overlap"],
                prior_binary_features=binary,
                prior_continuous_features=continuous,
                prior_categorical_features=categorical,
                prior_class_members=class_members,
                prior_class_membership=class_membership,
            )
        )
    return tuple(records)


def load_training_dataset(public_dir: str | os.PathLike[str]) -> TrainingDataset:
    family, _, covariates, annotation_types = _load_public_spec(public_dir)
    train_ids, train_covariates, train_targets = _read_covariates(
        public_dir,
        "covariates_train.csv",
        covariates,
        require_targets=True,
    )
    train_genotype_ids, variant_ids, train_genotypes = _read_genotypes(
        public_dir, "genotypes_train.tsv"
    )
    if train_ids != train_genotype_ids:
        raise ValueError("training covariate and genotype sample IDs/orders do not match")
    assert train_targets is not None
    if set(np.unique(train_targets)) != {0.0, 1.0}:
        raise ValueError("binary training targets must contain both 0 and 1")
    records = _read_variant_records(
        public_dir,
        variant_ids,
        annotation_types,
        train_genotypes,
    )
    schema = PublicPredictorSchema(
        family=family,
        covariate_names=covariates,
        variant_ids=variant_ids,
        annotation_types=annotation_types,
    )
    return TrainingDataset(
        schema=schema,
        sample_ids=train_ids,
        genotypes=train_genotypes,
        covariates=train_covariates,
        targets=train_targets,
        variant_records=records,
    )


def load_prediction_dataset(
    public_dir: str | os.PathLike[str],
    *,
    training: TrainingDataset,
) -> PredictionDataset:
    if type(training) is not TrainingDataset:
        raise TypeError("training must be exactly TrainingDataset")
    expected_schema = training.schema
    family, _, covariates, annotation_types = _load_public_spec(public_dir)
    if (
        family != expected_schema.family
        or covariates != expected_schema.covariate_names
        or dict(annotation_types) != dict(expected_schema.annotation_types)
    ):
        raise ValueError("prediction files do not match the fitted training schema")
    test_ids, test_covariates, _ = _read_covariates(
        public_dir,
        "covariates_test.csv",
        covariates,
        require_targets=False,
    )
    test_genotype_ids, test_variant_ids, test_genotypes = _read_genotypes(
        public_dir, "genotypes_test.tsv"
    )
    if test_ids != test_genotype_ids:
        raise ValueError("prediction covariate and genotype sample IDs/orders do not match")
    if test_variant_ids != expected_schema.variant_ids:
        raise ValueError("prediction variant IDs/order do not match the fitted training schema")
    if set(training.sample_ids) & set(test_ids):
        raise ValueError("training and prediction sample IDs overlap")
    return PredictionDataset(
        schema=expected_schema,
        sample_ids=test_ids,
        genotypes=test_genotypes,
        covariates=test_covariates,
    )


class _PosteriorPredictiveSVPGS:
    """Score with uncertainty-aware SV-PGS posterior-predictive logits."""

    __slots__ = ("_model",)

    def __init__(self, model) -> None:
        self._model = model

    def decision_function(self, genotypes, covariates):
        return self._model.posterior_predictive_logit(genotypes, covariates)


# How far below its OWN pre-data class prior the reference's empirical-Bayes fit
# may shrink its global effect scale. See _class_prior_global_scale_floor.
REFERENCE_GLOBAL_SCALE_PRIOR_SLACK = 2.0


def _class_prior_global_scale(
    records: Sequence[ReferenceVariantRecord],
    config: ModelConfig,
) -> float:
    """SV-PGS's own pre-data belief about the global effect scale.

    Reproduces ``sv_pgs.mixture_inference._initialize_scale_model``:
    ``exp(mean_v(sum_c membership[v, c] * class_log_baseline_scale[c]))``. Class
    membership weights are mixture proportions summing to 1.0 (VariantRecord
    enforces this); an omitted membership means the variant belongs wholly to its
    own ``variant_class``.

    This is exact up to the engine's tie-map reduction, which dedups identical
    genotype columns before building the prior design and so averages over a
    slightly different variant set where exact ties exist. The 2x slack in
    _class_prior_global_scale_floor absorbs that difference; nothing here needs
    the value to be bit-exact, only to be the right order of magnitude.

    Uses ONLY public variant metadata -- no genotypes, no targets, no truth.
    """
    baseline = config.class_log_baseline_scales()
    per_variant = []
    for record in records:
        if record.prior_class_members:
            per_variant.append(
                float(
                    sum(
                        float(weight) * baseline[member]
                        for member, weight in zip(
                            record.prior_class_members,
                            record.prior_class_membership,
                            strict=True,
                        )
                    )
                )
            )
        else:
            per_variant.append(float(baseline[record.variant_class]))
    if not per_variant:
        raise ValueError("model-zoo SV-PGS requires at least one variant record")
    return float(np.exp(float(np.mean(per_variant))))


def _class_prior_global_scale_floor(
    records: Sequence[ReferenceVariantRecord],
    config: ModelConfig,
) -> float:
    """The reference's global-scale floor: half its own class prior.

    WHY THIS IS SET AT ALL. SV-PGS's variational EM has a documented degenerate
    attractor. ``mixture_inference._update_scale_model`` says so itself:

        "On problems with few signal variants among many noise variants, the
        Newton step averages 1/p Sigma E[beta^2]/(s^2 lambda) over noise
        coordinates and collapses sigma_g toward the noise floor BEFORE the
        local-scale GIG updates can identify the signal."

    Its mitigation clips the per-iteration log-decrease to ~22%. That bounds the
    SPEED of the collapse, not its DESTINATION: 0.78^168 ~ 1e-18 inside the
    reference's 400-iteration budget, so the collapse still completes -- it merely
    walks down slowly. Measured on the v6 corpus, the fits that collapse burn
    81-168 outer iterations to get there while healthy fits converge in 36-54.
    The iteration count IS the collapse signature.

    WHAT IT COST. On five of fifteen categories the reference's fitted global
    scale landed 3-20x BELOW its own data-free class-prior init and the anchor
    died with it: naive->reference AUC gaps of 0.0047-0.0208, a reference BEATEN
    on raw AUC by a 1.5-second marginal prune+threshold cheat on 9/15 categories,
    and skill scores that were winsorization arithmetic rather than measurements.
    Full record: rollout_analysis/ANCHOR_COLLAPSE_2026-07-17.md.

    WHY HALF THE CLASS PRIOR, AND WHY THAT IS NOT TUNING. The rule is stated in
    the model's own terms and was fixed BEFORE it was measured: an empirical-Bayes
    fit may learn that effects are smaller than its prior expected, but it may not
    conclude they are ~zero when its own pre-data belief -- DEFAULT_CLASS_LOG_BASELINE_SCALE,
    the library's statement of the per-class effect scale it expects before seeing
    any data -- says otherwise. One doubling of slack is the allowance. The value
    was NOT swept: sweeping a reference until it beats the arms would overfit the
    anchor to this corpus, which this repo forbids.

    IT IS A TARGETED FIX, AND THAT WAS VERIFIED, NOT ASSUMED. On the healthy
    categories the floor is a bit-exact no-op -- svld_strong measured identically
    with and without it (AUC 0.6426, global_scale 0.01669, 41 iterations, both
    arms), because a fit that never collapses never reaches the floor. It binds
    only where the EM was already walking into the degenerate solution, and the
    AUC it recovers scales monotonically with how far the fit had collapsed.

    This historical correction keeps the hidden SV-PGS model-zoo arm scientifically
    usable; it does not define the shipped reference or score-1 implementation.

    Computed from public variant metadata alone, deterministically. No genotypes,
    no targets, no truth -- so this cannot leak held-out information into an
    anchor.
    """
    return _class_prior_global_scale(records, config) / REFERENCE_GLOBAL_SCALE_PRIOR_SLACK


def _fit_svpgs(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[ReferenceVariantRecord],
    *,
    max_outer_iterations: int,
    convergence_tolerance: float,
) -> dict[str, Any]:
    # This hidden JAX engine is an active DEVELOPMENT model-zoo arm only. Public
    # data loading and shipped corpus anchors do not require the sibling checkout.
    svpgs_root = Path(__file__).resolve().parents[2] / "SV-PGS"
    if not svpgs_root.is_dir():
        raise ImportError(f"model-zoo SV-PGS checkout is absent at {svpgs_root}")
    if str(svpgs_root) not in sys.path:
        sys.path.insert(0, str(svpgs_root))
    from sv_pgs.config import (  # noqa: PLC0415
        ModelConfig,
        TraitType,
        VariantClass as EngineVariantClass,
    )
    from sv_pgs.data import VariantRecord  # noqa: PLC0415
    from sv_pgs.model import BayesianPGS  # noqa: PLC0415

    immutable_records = tuple(records)
    if any(type(record) is not ReferenceVariantRecord for record in immutable_records):
        raise TypeError(
            "model-zoo SV-PGS records must be immutable ReferenceVariantRecord values")
    engine_records = [
        VariantRecord(
            variant_id=record.variant_id,
            variant_class=EngineVariantClass(record.variant_class.value),
            chromosome=record.chromosome,
            position=record.position,
            length=record.length,
            allele_frequency=record.allele_frequency,
            quality=record.quality,
            training_support=record.training_support,
            is_repeat=record.is_repeat,
            is_copy_number=record.is_copy_number,
            prior_binary_features=dict(record.prior_binary_features),
            prior_continuous_features=dict(record.prior_continuous_features),
            prior_categorical_features=dict(record.prior_categorical_features),
            prior_class_members=tuple(
                EngineVariantClass(member.value)
                for member in record.prior_class_members
            ),
            prior_class_membership=record.prior_class_membership,
        )
        for record in immutable_records
    ]
    # The floor is derived from the class prior, which needs a config to read
    # class_log_baseline_scales from; those scales are config-independent
    # defaults, so a probe config gives the same answer as the real one.
    global_scale_floor = _class_prior_global_scale_floor(
        immutable_records, ModelConfig(trait_type=TraitType.BINARY)
    )
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=max_outer_iterations,
        convergence_tolerance=convergence_tolerance,
        minimum_minor_allele_frequency=0.0,
        use_ld_blocks=False,
        stochastic_variational_updates=False,
        # Bound the DESTINATION of the EM's documented global-scale collapse; the
        # library only bounds its rate. See _class_prior_global_scale_floor.
        global_scale_floor=global_scale_floor,
    )
    model = BayesianPGS(config)
    t0 = time.perf_counter()
    model.fit(
        np.asarray(genotypes, dtype=np.float32),
        np.asarray(covariates, dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1),
        engine_records,
    )
    wall = time.perf_counter() - t0

    state = model.state
    if state is None:
        raise RuntimeError("SV-PGS fit completed without a fitted state")
    fit_result = state.fit_result
    beta_full = np.asarray(state.full_coefficients, dtype=np.float64)
    names = list(fit_result.scale_model_feature_names)
    coefficients = np.asarray(fit_result.scale_model_coefficients, dtype=np.float64)
    coefficient_by_name = dict(zip(names, coefficients))
    variant_classes = np.asarray(
        [record.variant_class.value for record in immutable_records]
    )
    classes_present = sorted(set(variant_classes.tolist()))
    encoded_class_values = set(classes_present)
    for record in immutable_records:
        encoded_class_values.update(member.value for member in record.prior_class_members)
    class_membership_totals = {value: 0.0 for value in sorted(encoded_class_values)}
    for record in immutable_records:
        if record.prior_class_members:
            for member, membership in zip(
                record.prior_class_members,
                record.prior_class_membership,
                strict=True,
            ):
                class_membership_totals[member.value] += float(membership)
        else:
            class_membership_totals[record.variant_class.value] += 1.0
    reference_class = max(
        sorted(encoded_class_values),
        key=lambda value: class_membership_totals[value],
    )
    type_offset_coef: dict[str, float] = {}
    for value in classes_present:
        coefficient_name = f"type_offset::{VariantClass(value).value}"
        if coefficient_name in coefficient_by_name:
            type_offset_coef[value] = float(coefficient_by_name[coefficient_name])
        elif value == reference_class:
            type_offset_coef[value] = 0.0
        else:
            raise RuntimeError(
                f"SV-PGS omitted non-reference type contrast {coefficient_name!r}"
            )
    prior_scales = np.asarray(fit_result.prior_scales, dtype=np.float64)
    original_to_reduced = np.asarray(state.tie_map.original_to_reduced)
    full_prior_variance = np.full(len(immutable_records), np.nan, dtype=np.float64)
    active = original_to_reduced >= 0
    full_prior_variance[active] = prior_scales[original_to_reduced[active]]
    per_class_mean_abs_beta = {}
    per_class_mean_prior_var = {}
    for value in classes_present:
        mask = variant_classes == value
        per_class_mean_abs_beta[value] = float(np.mean(np.abs(beta_full[mask])))
        finite_prior = full_prior_variance[mask]
        finite_prior = finite_prior[np.isfinite(finite_prior)]
        per_class_mean_prior_var[value] = (
            float(np.mean(finite_prior)) if finite_prior.size else float("nan")
        )
    return {
        "model": _PosteriorPredictiveSVPGS(model),
        "global_scale": float(fit_result.global_scale),
        # Recorded so an auditor can tell a floored anchor from a collapsed one,
        # and can see whether the fit PINNED at the floor (global_scale == floor,
        # i.e. the EM was still trying to collapse) or settled above it freely.
        # The v6 saturation went undiagnosed for two sessions partly because this
        # configuration was not in the artifact.
        "global_scale_floor": float(global_scale_floor),
        "scale_model_feature_names": names,
        "scale_model_coef_by_name": coefficient_by_name,
        "type_offset_coef": type_offset_coef,
        "per_class_mean_abs_beta": per_class_mean_abs_beta,
        "per_class_mean_prior_var": per_class_mean_prior_var,
        "converged": bool(fit_result.converged),
        "convergence_tolerance": float(convergence_tolerance),
        "iterations_run": len(fit_result.objective_history),
        "final_parameter_change": float(fit_result.final_parameter_change),
        "final_predictor_change": float(fit_result.final_predictor_change),
        "final_objective_change": float(fit_result.final_objective_change),
        "final_hyperparameter_change": float(fit_result.final_hyperparameter_change),
        "wall_time_s": wall,
        "classes_present": classes_present,
    }


def _build_generated_variant_records(
    metadata: dict[str, Any],
) -> tuple[ReferenceVariantRecord, ...]:
    """Build records for pre-materialization validation studies only."""
    records: list[ReferenceVariantRecord] = []
    for index, variant_id in enumerate(metadata["variant_id"]):
        kwargs: dict[str, Any] = {}
        if "prior_class_members" in metadata:
            members = tuple(
                VariantClass(value)
                for value in str(metadata["prior_class_members"][index]).split(",")
                if value
            )
            weights = tuple(
                float(value)
                for value in str(metadata["prior_class_membership"][index]).split(",")
                if value
            )
            kwargs["prior_class_members"] = members
            kwargs["prior_class_membership"] = weights
        records.append(
            ReferenceVariantRecord(
                variant_id=str(variant_id),
                variant_class=VariantClass(str(metadata["variant_class"][index])),
                chromosome="1",
                position=index + 1,
                length=float(metadata["length"][index]),
                allele_frequency=float(metadata["allele_frequency"][index]),
                is_repeat=bool(metadata["is_repeat"][index]),
                is_copy_number=bool(metadata["is_copy_number"][index]),
                prior_continuous_features={
                    "sv_log_length": float(
                        np.log10(max(float(metadata["length"][index]), 1.0))
                    )
                },
                prior_binary_features={
                    "repeat_overlap": bool(metadata["is_repeat"][index])
                },
                **kwargs,
            )
        )
    return tuple(records)


def fit_svpgs_generated(
    dataset: dict[str, Any],
    *,
    max_outer_iterations: int = 20,
    convergence_tolerance: float = REFERENCE_CONVERGENCE_TOLERANCE,
) -> dict[str, Any]:
    """Fit a generated in-memory dataset for non-corpus validation studies."""
    covariates = np.asarray(dataset["cov"], dtype=np.float32)
    if covariates.shape[1] and np.allclose(covariates[:, 0], 1.0):
        covariates = covariates[:, 1:]
    return _fit_svpgs(
        np.asarray(dataset["G"], dtype=np.float32),
        covariates,
        np.asarray(dataset["y"], dtype=np.float32),
        _build_generated_variant_records(dataset["meta"]),
        max_outer_iterations=max_outer_iterations,
        convergence_tolerance=convergence_tolerance,
    )


if __name__ == "__main__":
    import tempfile

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datagen.dgp import DGPConfig, generate
    from datagen.materialize import materialize

    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_variants = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    print(f"[smoke] generating binary N={n_samples} P={n_variants}", flush=True)
    generated = generate(
        DGPConfig(seed=1, n_samples=n_samples, n_variants=n_variants)
    )
    with tempfile.TemporaryDirectory(prefix="svpgs-reference-") as directory:
        info = materialize(
            generated,
            directory,
            family="binomial-logit",
            frac_train=generated["cfg"].frac_train_hint,
        )
        print("[smoke] fitting SV-PGS from materialized public files", flush=True)
        training = load_training_dataset(info["public"])
        fitted = _fit_svpgs(
            training.genotypes,
            training.covariates,
            training.targets,
            training.variant_records,
            max_outer_iterations=REFERENCE_MAX_OUTER_ITERATIONS,
            convergence_tolerance=REFERENCE_CONVERGENCE_TOLERANCE,
        )
    print(
        f"[smoke] done in {fitted['wall_time_s']:.1f}s "
        f"converged={fitted['converged']} global_scale={fitted['global_scale']:.4g}",
        flush=True,
    )
