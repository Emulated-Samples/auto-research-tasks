from __future__ import annotations

import json
from pathlib import Path

from datagen.categories import CATEGORIES
from datagen.dgp import generate
from datagen.materialize import _write_dgp_json, materialize


def _annotation_types(columns):
    special = {
        "variant_class": "variant_class",
        "prior_class_members": "class_members",
        "prior_class_membership": "class_membership",
        "repeat_overlap": "binary",
    }
    return {name: special.get(name, "continuous") for name in columns}


def test_public_spec_exposes_only_observable_model_contract(tmp_path):
    spec = CATEGORIES["sparse_heavy_tail"]
    path = tmp_path / "dgp.json"
    _write_dgp_json(
        path,
        family=spec["family"],
        prior_cols=spec["prior_cols"],
        annotation_types=_annotation_types(spec["prior_cols"]),
        cov_cols=("age", "PC1"),
    )

    public_spec = json.loads(path.read_text())
    assert set(public_spec) == {"family", "formula"}
    serialized = path.read_text().lower()
    for latent_name in (
        "architecture",
        "sparsity",
        "tail",
        "ld",
        "prevalence",
        "shift",
        "p_gt_n",
    ):
        assert latent_name not in serialized


def test_rare_variant_annotation_remains_observable_without_truth_hint(tmp_path):
    spec = CATEGORIES["rare_variant_maf"]
    assert "allele_frequency" in spec["prior_cols"]

    dataset = generate(spec["make_cfg"](7, 80, 24))
    output = materialize(
        dataset,
        tmp_path / "rare_materialized",
        family=spec["family"],
        frac_train=dataset["cfg"].frac_train_hint,
        prior_cols=spec["prior_cols"],
        seed=7,
    )
    public_dir = Path(output["public"])
    with open(f"{public_dir}/dgp.json") as handle:
        public_spec = json.load(handle)
    assert "allele_frequency" in public_spec["formula"]["pgs_annotations"]
    with open(f"{public_dir}/variant_metadata.tsv") as handle:
        assert "allele_frequency" in handle.readline().rstrip("\n").split("\t")


def test_public_files_and_headers_do_not_label_feature_utility(tmp_path):
    spec = CATEGORIES["decoy_annotations"]
    dataset = generate(spec["make_cfg"](13, 100, 30))
    output = materialize(
        dataset,
        tmp_path / "neutral_names",
        family=spec["family"],
        frac_train=dataset["cfg"].frac_train_hint,
        prior_cols=spec["prior_cols"],
        seed=13,
    )
    public_dir = Path(output["public"])
    assert "dgp.md" not in {path.name for path in public_dir.iterdir()}

    public_text = "\n".join(
        path.read_text(errors="replace").lower()
        for path in public_dir.iterdir()
        if path.suffix != ".gz"
    )
    for leaked_role in (
        "decoy",
        "nuisance",
        "misleading",
        "no signal",
        "batch_artifact",
        "assay_noise",
    ):
        assert leaked_role not in public_text

    with open(public_dir / "variant_metadata.tsv") as handle:
        metadata_header = handle.readline().rstrip("\n").split("\t")
    assert {"annotation_1", "annotation_2", "annotation_3"} <= set(metadata_header)

    with open(public_dir / "covariates_train.csv") as handle:
        covariate_header = handle.readline().rstrip("\n").split(",")
    assert {"feature_1", "feature_2"} <= set(covariate_header)


def test_sparse_heavy_tail_is_privately_p_greater_than_training_n_and_bounded():
    from grader.contract import SHIPPED_FRAC_TRAIN

    make_cfg = CATEGORIES["sparse_heavy_tail"]["make_cfg"]

    shipped = make_cfg(200, 12_000, 800)
    shipped_train_n = int(SHIPPED_FRAC_TRAIN * shipped.n_samples)
    assert shipped.n_variants > shipped_train_n
    assert shipped.n_variants / shipped_train_n >= 1.2
    assert shipped.n_samples * shipped.n_variants <= 30_000_000

    local = make_cfg(1, 200, 40)
    local_train_n = int(SHIPPED_FRAC_TRAIN * local.n_samples)
    assert local.n_variants > local_train_n
    assert local.n_samples * local.n_variants <= 30_000


def test_very_strong_ld_keeps_the_full_cohort_but_caps_the_production_dimension():
    make_cfg = CATEGORIES["svld_strong"]["make_cfg"]

    shipped = make_cfg(0, 12_000, 2_500)
    local = make_cfg(0, 200, 40)

    assert shipped.n_samples == 12_000
    assert shipped.n_variants == 2_000
    assert shipped.heritability == 0.95
    assert local.n_variants == 40


def test_sparse_dense_mix_keeps_the_full_cohort_but_caps_the_production_dimension():
    make_cfg = CATEGORIES["sparse_dense_mix"]["make_cfg"]

    shipped = make_cfg(0, 12_000, 2_500)
    local = make_cfg(0, 200, 40)

    assert shipped.n_samples == 12_000
    assert shipped.n_variants == 2_000
    assert shipped.heritability == 0.95
    assert local.n_variants == 40
