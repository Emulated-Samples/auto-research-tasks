"""The agent-facing prompt must not drift or disclose evaluation mechanics.

The evaluated rollout ran against two hand-maintained prompt surfaces that
disagreed on the formula grammar, decoys, dgp.json and the speed term. Both are
now rendered from tasks/prompt_spec.py; these tests fail if either surface is
stale, if evaluation-facing language returns, or if an algorithm recipe creeps
back into the task text.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grader.contract import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    FIT_TIMEOUT_S,
    PREDICT_TIMEOUT_S,
)
from tasks import prompt_spec  # noqa: E402


EVALUATION_MECHANICS = (
    "hidden grader", "grader", "evaluation harness", "scoring summary",
    "reward", "strong reference", "naive baseline", "baseline =",
    "per-category scores", "scored continuously", "held-out evaluation",
    "AUC (", "Brier score (", "log-loss (",
)


def test_prompt_surfaces_are_not_stale():
    stale = [str(p.relative_to(REPO_ROOT)) for p in prompt_spec.stale_surfaces()]
    assert stale == [], (
        f"prompt surfaces are stale: {stale}. "
        "Re-render with `python tasks/prompt_spec.py --write`."
    )


def test_both_surfaces_carry_the_same_prompt_body():
    body = prompt_spec.render_prompt()
    md = (REPO_ROOT / "tasks" / "from_scratch_svpgs" / "instruction.md").read_text()
    yaml_text = (REPO_ROOT / "environment" / "problems.yaml").read_text()
    assert body in md
    # The YAML surface carries the same body, indented into a block scalar.
    for line in body.splitlines():
        if line.strip():
            assert ("    " + line) in yaml_text


def test_prompt_hash_is_pinned_to_the_grader_contract_version():
    digest = prompt_spec.prompt_sha256()
    assert prompt_spec.PROMPT_CONTRACT_VERSION == CORPUS_SCHEMA_VERSION
    for path in prompt_spec.SURFACES:
        text = path.read_text()
        assert digest in text, f"{path.name} does not record the prompt hash"
        assert f"prompt_contract_version: {CORPUS_SCHEMA_VERSION}" in text


def test_prompt_states_outcomes_without_evaluation_mechanics():
    body = prompt_spec.render_prompt()
    for outcome in ("discriminate cases", "probabilistically calibrated",
                    "inside their stated limits"):
        assert outcome in body
    # Runtime is a CONSTRAINT (hard fit/predict caps), never a reward term. The
    # prompt must not imply that finishing faster earns anything, or an agent will
    # trade away accuracy -- the only thing actually scored -- to buy speed that
    # cannot pay. See grader/perf.py: perf_factor is report-only.
    for speed_reward in ("speed bonus", "faster is better", "the faster",
                         "reward for speed", "speed is rewarded"):
        assert speed_reward.lower() not in body.lower(), (
            f"prompt implies a speed reward that grading does not pay: {speed_reward!r}"
        )
    for phrase in EVALUATION_MECHANICS:
        assert phrase.lower() not in body.lower(), (
            f"solver prompt exposes evaluation-facing phrase: {phrase!r}"
        )


def test_display_metadata_states_the_work_not_the_evaluation():
    config = (REPO_ROOT / "hyperfocal.yaml").read_text().lower()
    task = tomllib.loads(
        (REPO_ROOT / "tasks" / "from_scratch_svpgs" / "task.toml").read_text()
    )
    surfaces = (
        ("hyperfocal.yaml", config),
        ("task.description", task["task"]["description"].lower()),
        ("metadata.display_description", task["metadata"]["display_description"].lower()),
        ("task.keywords", " ".join(task["task"]["keywords"]).lower()),
    )
    for label, text in surfaces:
        for phrase in EVALUATION_MECHANICS:
            assert phrase.lower() not in text, (
                f"{label} exposes evaluation-facing phrase: {phrase!r}"
            )


def test_prompt_states_the_load_bearing_contract_facts():
    body = prompt_spec.render_prompt()
    required = [
        "dgp.json",                 # machine-readable spec exists
        "`+` **or** `,`",           # both separators are legal
        "pred.csv",
        # Runtime is enforced by the disclosed caps and is NOT a reward term, so the
        # prompt promises no speed credit -- only that leftover budget buys nothing.
        "unused time cannot compensate",
        "no symlinks",
        "4 vCPUs",
        "60-minute wall limit",
        "6 GiB address-space limit",
        "Use only those columns and ignore unlisted metadata columns",
        "ignore any unlisted columns",
    ]
    for fact in required:
        assert fact in body, f"prompt no longer states: {fact}"


def test_prompt_advertises_the_wall_budgets_the_runner_actually_enforces():
    """The advertised per-phase budgets must BE the enforced ones.

    Regression: the prompt promised `fit` 72 s / `predict` 48 s after
    grader.contract had been rebalanced. submission_runner._run SIGKILLs
    predict at PREDICT_TIMEOUT_S, so a submission that sized predict to the
    advertised 48 s returned status=predict_failed -> INVALID_REWARD. That is a
    false negative manufactured by the prompt: the agent is penalized for
    believing a truthful-looking instance fact. The old assertion pinned the
    stale literals, so the test suite enforced the mismatch instead of catching
    it. Compare against the constants, never against a copied number.
    """
    words = " ".join(prompt_spec.render_prompt().split())
    assert f"`fit` {FIT_TIMEOUT_S} seconds" in words
    assert f"`predict` {PREDICT_TIMEOUT_S} seconds" in words
    # The superseded split must not survive anywhere in the agent-facing text.
    for stale in ("72 seconds", "48 seconds"):
        if stale not in (f"{FIT_TIMEOUT_S} seconds", f"{PREDICT_TIMEOUT_S} seconds"):
            assert stale not in words, f"prompt still advertises stale budget {stale!r}"


def test_agent_facing_surfaces_do_not_advertise_the_private_task_matrix():
    task = tomllib.loads(
        (REPO_ROOT / "tasks" / "from_scratch_svpgs" / "task.toml").read_text()
    )
    surfaces = {
        "prompt": prompt_spec.render_prompt().lower(),
        "hyperfocal.description": (
            REPO_ROOT / "hyperfocal.yaml"
        ).read_text().lower(),
        "task metadata": " ".join(
            (
                task["task"]["description"],
                " ".join(task["task"]["keywords"]),
                task["metadata"]["display_description"],
            )
        ).lower(),
    }
    leaked_hints = (
        "instance you are holding",
        "sparse and dense effects",
        "signed/very-strong ld",
        "rare outcomes or variants",
        "nonlinear or interacting annotations",
        "distribution shift between the training cohort",
        "hurt held-out accuracy",
        "some candidates may carry no signal",
        "sv_log_length",
        "repeat_overlap",
        "genetic architectures",
        "annotation regimes",
        "cohort shifts",
        "shrinkage",
        "empirical-bayes",
    )
    for label, surface in surfaces.items():
        for leaked_hint in leaked_hints:
            assert leaked_hint not in surface, f"{label} leaks {leaked_hint!r}"


ALGORITHM_LEAKS = [
    # The prompt must state the observable contract, not prescribe the winning
    # estimator. Each of these was in the evaluated prompt's "suggested build
    # order", which handed the agent the entire solution.
    r"penalized logistic",
    r"elastic[- ]net",
    r"\bridge\b",
    r"\blasso\b",
    r"adaptive (?:shrinkage|penalt|regulariz)",
    r"annotation-aware",
    r"heavy-tailed (?:prior|shrinkage)",
    r"global penalty",
    r"jointly where LD",
    r"sparse prior",
    r"calibrate the predicted probabilities",
]


@pytest.mark.parametrize("pattern", ALGORITHM_LEAKS)
def test_prompt_does_not_prescribe_the_solution(pattern):
    body = prompt_spec.render_prompt()
    hit = re.search(pattern, body, flags=re.IGNORECASE)
    assert hit is None, (
        f"prompt leaks the intended algorithm ({pattern!r} -> {hit.group(0)!r}). "
        "State the observable contract and truthful instance facts; the choice of "
        "estimator is the task."
    )
