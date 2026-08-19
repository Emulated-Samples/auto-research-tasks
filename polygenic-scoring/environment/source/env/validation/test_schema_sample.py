"""The workspace schema sample must carry the CORPUS's schema, and must not be
mistakable for a development corpus.

Why this file exists (measured, run_019f6689, 2026-07-15): the workspace used to
be EMPTY, so the solver fabricated its own fixture -- a 61-byte formula.txt
against the corpus's ~145 -- whose `pgs(...)` contained no categorical
annotation. Its reader passed against its own fiction, then raised
`float('snv')` on the real formula and scored the INVALID FLOOR on all 45
datasets. A 45-dataset corpus measured one `float()` call.

The sample fixes that ONLY if it keeps two properties, and both are pinned here:

  * it carries the REAL schema, INCLUDING the hazard (a categorical
    variant_class inside pgs()). A sample that omits the hazard is the
    fabricated fixture again, just written by us;
  * it stays a SCHEMA sample -- tiny, public-only, unfittable. The moment it is
    big enough to tune against it leaks the DGP, which is the one thing the
    private corpus exists to withhold.

The sample and the corpus are TWO AUTHORITIES on "what a dataset looks like".
Generating both through datagen.materialize is what binds them; these tests are
what prove the binding held.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datagen.build_sample import (  # noqa: E402
    SAMPLE_DIRNAME,
    SAMPLE_N,
    SAMPLE_P,
    build_sample,
)


@pytest.fixture(scope="module")
def sample_public():
    with tempfile.TemporaryDirectory(prefix="svpgs_sample_") as tmp:
        yield os.path.join(build_sample(tmp), "public")


def _corpus_public():
    hits = sorted(glob.glob(str(REPO_ROOT / "corpus" / "*" / "d_*" / "public")))
    return hits[0] if hits else None


def test_sample_ships_only_public_files(sample_public):
    """A sample carrying truth would hand the solver labels. It shows the FORMAT."""
    root = os.path.dirname(sample_public)
    assert sorted(os.listdir(root)) == ["public"], "the sample must not ship truth/"


def test_sample_carries_the_hazard_that_a_fabricated_fixture_omits(sample_public):
    """variant_class must be categorical AND inside pgs().

    This is the single property that makes the sample worth shipping: it is the
    exact case run_019f6689's invented fixture omitted and its code could not
    handle.
    """
    formula = open(os.path.join(sample_public, "formula.txt")).read()
    inner = re.search(r"pgs\(([^)]*)\)", formula).group(1)
    assert "variant_class" in inner, "the sample's pgs() must name variant_class"

    with open(os.path.join(sample_public, "variant_metadata.tsv")) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    classes = {r["variant_class"] for r in rows}
    assert len(classes) >= 2, "variant_class must be genuinely multi-valued"
    for value in classes:
        with pytest.raises(ValueError):
            float(value)  # it is CATEGORICAL: float() must raise, as it did in the rollout


def test_sample_schema_matches_the_shipped_corpus(sample_public):
    """The binding. The sample and the corpus must not drift.

    Both come from datagen.materialize, so this asserts the property that makes
    that worth doing: same file set, same formula grammar, same annotation
    columns. If this ever fails, a solver is being taught a contract the grader
    does not use -- the two-authority disease, in the fixture layer.
    """
    corpus_public = _corpus_public()
    if corpus_public is None:
        pytest.skip("no built corpus in this tree")

    def files(d):
        return {f[:-3] if f.endswith(".gz") else f for f in os.listdir(d)}

    assert files(sample_public) == files(corpus_public), "file set drifted from the corpus"

    def terms(path):
        text = open(os.path.join(path, "formula.txt")).read()
        return (
            {t.strip() for t in re.split(r"[+,]", re.search(r"pgs\(([^)]*)\)", text).group(1))},
            {t.strip() for t in re.split(r"[+,]", re.search(r"covariate\(([^)]*)\)", text).group(1))},
        )

    sample_pgs, sample_cov = terms(sample_public)
    corpus_pgs, corpus_cov = terms(corpus_public)
    assert sample_pgs == corpus_pgs, "pgs() annotation columns drifted from the corpus"
    assert sample_cov == corpus_cov, "covariate() columns drifted from the corpus"


def test_sample_is_too_small_to_be_a_dev_corpus():
    """Guard the 'do not grow this' contract mechanically, not just in a comment.

    A comment saying "keep it tiny" is not a constraint. If someone scales this
    up it stops being a schema sample and starts being a tuning set that leaks
    the DGP -- so the size bound is asserted.
    """
    assert SAMPLE_N <= 200, "the schema sample is growing into a dev corpus"
    assert SAMPLE_P <= 100, "the schema sample is growing into a dev corpus"


def test_sample_is_not_a_graded_dataset(sample_public):
    """It must never overlap the grading corpus.

    Corpus datasets derive their seeds from the HMAC key via derive_stream_seed,
    and live under opaque d_<hash> ids; the sample uses a literal seed and a
    plain directory name, so they cannot collide. Pin the observable form of
    that: the sample is not addressable as a corpus dataset id.
    """
    assert not os.path.basename(os.path.dirname(sample_public)).startswith("d_")
    assert os.path.basename(os.path.dirname(sample_public)) == SAMPLE_DIRNAME
