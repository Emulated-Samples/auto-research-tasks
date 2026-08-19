"""Canonical source of the agent-facing task text.

Every human-readable prompt surface is RENDERED from this module:

  * ``environment/problems.yaml``            (loaded by the Hyperfocal wrapper)
  * ``tasks/from_scratch_svpgs/instruction.md``

Both were previously hand-maintained and had silently drifted apart -- they
disagreed on the formula grammar, decoy columns, ``dgp.json`` and the speed
term, so an agent and a maintainer could reason about different contracts.
Rendering both from one string makes that class of drift unrepresentable, and
``validation/test_prompt_contract.py`` fails if either file is stale.

What the prompt deliberately does NOT contain: a recipe. It states the
observable interface (files, grammar, entry points, and resource rules) and
truthful instance facts, and stops there. Choosing the estimator is the task,
not a hint to be handed over.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grader.contract import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    FIT_TIMEOUT_S,
    PREDICT_TIMEOUT_S,
)

# The prompt describes the schema-v8 public contract (dgp.json, formula-scoped
# columns, and comma-or-plus grammar). It is versioned
# WITH the corpus it describes: a corpus schema bump must re-examine this text.
PROMPT_CONTRACT_VERSION = CORPUS_SCHEMA_VERSION

PROBLEM_ID = "from_scratch_svpgs"
PROBLEM_TITLE = "Polygenic scoring from scratch"

_BODY = """\
I have a research question I want you to investigate. I'm building a polygenic-scoring (genotype -> phenotype prediction) library from scratch, using the fixed toolchain described below.

You are given, per problem, a cohort of samples with covariates and a genotype dosage vector over many variants, plus per-variant annotations. Your program must learn to predict a phenotype for held-out samples as accurately as possible, and quickly.

You are not being asked to implement a named method or reproduce a published tool. Design any estimator that satisfies the program contract below and performs well on the quality bar.

Program contract:

The current working directory is your submission root. Provide two executable phase entry points there (`chmod +x`; if scripts, use an absolute interpreter path in the shebang, `#!/opt/svpgs-venv/bin/python`):

- `./fit`
- `./predict`

If your language needs a compile step, put it in an optional `./build.sh` (runs once, offline).

`fit` reads (from the current directory):

- `covariates_train.csv` - header row; `sample_id`, covariate columns, and the response column `y`.
- `genotypes_train.tsv` - header row; `sample_id` then one column per variant id; integer dosages `{0,1,2}`. Rows align with `covariates_train.csv` by `sample_id`.
- `variant_metadata.tsv` - keyed by `variant_id`; a `variant_class` column plus additional annotation columns (see `formula.txt`).
- `formula.txt`, `family.txt`, `dgp.json` - the text and machine-readable model spec (below).

`fit` writes `model.out` (your serialized fitted model, any format).

`sample_dataset/public/` holds one small example cohort in exactly this format, so you can run your program end to end before you are finished. It is an example of the FORMAT only: it is far too small to learn anything from, it ships no response for its test rows, and the target cohorts are larger and more varied. Nothing about it is guaranteed beyond the contract described here.

`predict` reads `model.out`, `covariates_test.csv` (same covariate columns, no `y`), `genotypes_test.tsv`, and `variant_metadata.tsv`; it writes `pred.csv` with one row per test sample, in input order.

`family.txt` is always `binomial-logit`. `pred.csv` must contain a `mean` column: the predicted probability of the positive class, in `[0, 1]` and finite for every row.

Your program must run non-interactively, read no network, and exit 0 on success.

Formula / family:

`family.txt` is `binomial-logit`.

`formula.txt` has the form

    y ~ pgs(<annotation columns>) + covariate(<covariate columns>)

`dgp.json` carries the same lists and an explicit `annotation_types` mapping; the two specs agree exactly.

Grammar: inside `pgs(...)` and `covariate(...)` the listed names are separated by `+` or `,` (either separator may appear, and both may appear in the same corpus; treat them the same and strip surrounding whitespace). Parse the formula - do not assume a fixed column layout, and do not assume every column you can see is yours to use.

- `pgs(...)` lists the `variant_metadata.tsv` annotation columns available to your model. Use only those columns and ignore unlisted metadata columns.
- `covariate(...)` lists the covariate columns to adjust for (e.g. age, sex, ancestry PCs). Use exactly the listed covariate columns and ignore any unlisted columns. Covariates are ordinary fixed effects; they are not the object of interest but must be handled correctly.

Constraints:

- Implement the statistical machinery yourself. The image guarantees GCC/G++/GFortran, Make, CMake, OpenBLAS/LAPACK development libraries, and Python 3.12 at `/opt/svpgs-venv/bin/python` with NumPy 2.1.3 and SciPy 1.14.1. Those generic numeric tools are fair game, but no polygenic-scoring / GWAS / GAM / stock penalized-regression end-to-end tools. Build the estimator (basis/penalty/optimizer/derivatives) yourself. Work fully offline; install nothing.
- `./` must contain only regular files and directories - no symlinks.
- Execution provides 4 vCPUs. The optional build has a 60-minute wall limit; each dataset gives `fit` __FIT_TIMEOUT_S__ seconds and `predict` __PREDICT_TIMEOUT_S__ seconds. Each untrusted phase has a 6 GiB address-space limit.

Quality bar:

Predictions should discriminate cases from controls, remain probabilistically calibrated, and avoid severe overconfidence. Fit and predict must each complete inside their stated limits; unused time cannot compensate for weak predictions. Non-finite or malformed output and violated program contracts are invalid.

You have 24 hours; get something in before then.
"""


def render_prompt() -> str:
    """Return the canonical solver-facing task text.

    The per-phase wall budgets are INTERPOLATED from ``grader.contract`` rather
    than written into ``_BODY``. They were hand-written once and drifted: the
    contract was rebalanced while the prompt kept promising 72/48, so
    a submission that sized ``predict`` to the advertised 48 s was SIGKILLed at
    20 s and scored ``predict_failed`` -- a zero earned by believing the prompt.
    Rendering the numbers from the enforced constants makes that unrepresentable.
    """
    return (
        _BODY
        .replace("__FIT_TIMEOUT_S__", str(FIT_TIMEOUT_S))
        .replace("__PREDICT_TIMEOUT_S__", str(PREDICT_TIMEOUT_S))
    )


def prompt_sha256() -> str:
    return hashlib.sha256(render_prompt().encode()).hexdigest()


def render_problems_yaml() -> str:
    """environment/problems.yaml -- the file the Hyperfocal wrapper loads."""
    indented = "\n".join(
        ("    " + line) if line.strip() else "" for line in render_prompt().splitlines()
    )
    return (
        "# GENERATED FROM tasks/prompt_spec.py -- DO NOT EDIT BY HAND.\n"
        "# Re-render with:  python tasks/prompt_spec.py --write\n"
        "# validation/test_prompt_contract.py fails if this file is stale.\n"
        f"# prompt_contract_version: {PROMPT_CONTRACT_VERSION}\n"
        f"# prompt_sha256: {prompt_sha256()}\n"
        "\n"
        f"- id: {PROBLEM_ID}\n"
        "  default: true\n"
        f'  title: "{PROBLEM_TITLE}"\n'
        "  prompt: |\n"
        f"{indented}\n"
    )


def render_instruction_md() -> str:
    """tasks/from_scratch_svpgs/instruction.md -- the same text, as a doc."""
    return (
        "<!-- GENERATED FROM tasks/prompt_spec.py -- DO NOT EDIT BY HAND.\n"
        "     Re-render with:  python tasks/prompt_spec.py --write\n"
        f"     prompt_contract_version: {PROMPT_CONTRACT_VERSION}\n"
        f"     prompt_sha256: {prompt_sha256()} -->\n\n"
        + render_prompt()
    )


SURFACES = {
    REPO_ROOT / "environment" / "problems.yaml": render_problems_yaml,
    REPO_ROOT / "tasks" / "from_scratch_svpgs" / "instruction.md": render_instruction_md,
}


def write_surfaces() -> list[Path]:
    written = []
    for path, render in SURFACES.items():
        text = render()
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            written.append(path)
    return written


def stale_surfaces() -> list[Path]:
    return [p for p, render in SURFACES.items()
            if not p.exists() or p.read_text() != render()]


if __name__ == "__main__":
    if "--write" in sys.argv:
        for path in write_surfaces():
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        print(f"prompt_sha256 {prompt_sha256()}")
    else:
        stale = stale_surfaces()
        for path in stale:
            print(f"STALE {path.relative_to(REPO_ROOT)}")
        print(f"prompt_sha256 {prompt_sha256()}")
        sys.exit(1 if stale else 0)
