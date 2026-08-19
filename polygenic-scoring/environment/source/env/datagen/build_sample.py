"""Generate the tiny SCHEMA SAMPLE that ships in the solver's workspace.

WHAT THIS IS FOR, so nobody later "helpfully" grows it
------------------------------------------------------
This is a **schema sample**, not a development corpus. Its ONLY job is to let a
solver execute its own reader against the real file format before grading. It is
deliberately too small and too weak to fit anything: a handful of samples over a
handful of variants, with effects that carry no usable structure.

**Do NOT grow it into a dev corpus.** Do not add replicates, do not scale it up,
do not make its signal learnable, and do not add categories. The moment it is big
enough to tune against, it leaks the DGP -- which is the whole thing the private
corpus exists to withhold. If a future maintainer wants a dev corpus, that is a
benchmark-scope decision, not an edit to this file.

WHY IT EXISTS AT ALL (measured, 2026-07-15)
-------------------------------------------
`environment/src/index.ts` used to hand the solver an EMPTY workspace. Rollout
`run_019f6689` (Haiku 4.5) therefore fabricated its own fixture -- a 61-byte
`formula.txt` against the corpus's ~145 -- whose `pgs(...)` contained no
categorical annotation. It ran its pipeline against its own fiction, PASSED, and
reported "produces valid finite probabilities for every prediction": true of the
data it invented, false of the corpus. On the real formula
`pgs(variant_class + ...)` its `construct_pgs_features` hit `float('snv')` and
raised, scoring the INVALID FLOOR on all 45 datasets, 0/16 tests.

So a 45-dataset corpus, a 170 s fit budget and 15 capability categories all
measured ONE `float()` call. That is schema roulette: it dominates the score and
is uncorrelated with the capability under test. The fabricated fixture's omission
was not bad luck either -- anyone writing a toy fixture writes numeric
annotations, i.e. exactly the case the buggy code handles.

WHY IT DOESN'T MAKE THE TASK EASIER
-----------------------------------
The schema is ALREADY disclosed, in the prompt's prose AND in `dgp.json`'s
`annotation_types` mapping. This only makes it EXECUTABLE rather than described.
It discloses no method, no prior, no LD structure, and no answer; all five real
sub-problems (SV dosage handling, LD, effect estimation, the prior machinery,
validation discipline) are untouched. A demigod who reads `dgp.json` reaches the
same score either way.

INVARIANTS (each one is load-bearing; `validation/test_schema_sample.py` pins them)
-----------------------------------------------------------------------------------
1. Generated through `datagen.materialize` -- the SAME code path as the graded
   corpus. A hand-written sample would be a SECOND AUTHORITY on the schema and
   would drift from the corpus silently, which is the exact disease this fixes.
2. A distinct, dedicated seed. It must NEVER be one of the graded datasets:
   overlapping the grading corpus would overfit the solver onto data we grade on.
3. It carries the REAL schema's hazards -- a categorical `variant_class` inside
   `pgs(...)`, and the same annotation columns -- because a sample that omits the
   hazard is the fabricated fixture again, just written by us.
"""
from __future__ import annotations

import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datagen.dgp import DGPConfig, generate  # noqa: E402
from datagen.materialize import materialize  # noqa: E402

# A seed reserved for the sample alone. The graded corpus derives its seeds from
# the HMAC key via derive_stream_seed(), so this literal cannot collide with any
# shipped dataset -- they do not live in the same space at all.
SAMPLE_SEED = 20260715

# Deliberately tiny. Big enough to exercise every reader path (multi-class
# annotations, both separators, train/test split, covariates); far too small to
# fit anything. If you are tempted to raise these, re-read the docstring.
SAMPLE_N = 60
SAMPLE_P = 25
SAMPLE_FRAC_TRAIN = 0.5

SAMPLE_DIRNAME = "sample_dataset"


def build_sample(out_dir: str) -> str:
    """Materialize the schema sample into `out_dir/sample_dataset/public`.

    Only the PUBLIC files are kept: a sample carrying truth would hand the solver
    labels for data it can measure itself against, which is a different (and
    unauthorized) thing from showing it the file format.
    """
    cfg = DGPConfig(
        seed=SAMPLE_SEED,
        n_samples=SAMPLE_N,
        n_variants=SAMPLE_P,
        heritability=0.5,
        prevalence=0.3,
        snv_frac=0.6,
        small_indel_frac=0.2,
        snv_causal_frac=0.5,
        causal_maf_max=0.5,
        ld_rho_lo=0.1,
        ld_rho_hi=0.3,
        class_scale_spread=1.0,
        tail_temper=1.0,
        frac_train_hint=SAMPLE_FRAC_TRAIN,
    )
    dataset = generate(cfg)
    staging = os.path.join(out_dir, SAMPLE_DIRNAME)
    shutil.rmtree(staging, ignore_errors=True)
    materialize(
        dataset,
        staging,
        family="binomial-logit",
        frac_train=SAMPLE_FRAC_TRAIN,
        seed=SAMPLE_SEED,
    )
    # Drop truth: the sample shows the FORMAT, never the answers.
    shutil.rmtree(os.path.join(staging, "truth"), ignore_errors=True)
    return staging


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: datagen/build_sample.py <out_dir>")
    path = build_sample(sys.argv[1])
    print(f"[sample] wrote {path}")
    for name in sorted(os.listdir(os.path.join(path, "public"))):
        print(f"  public/{name}")
