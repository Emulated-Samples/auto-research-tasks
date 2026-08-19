"""
Grader-side HIDDEN evaluation harness for the varseq seq-design environment.

Never present in the agent's workspace. Runs host-side (a fresh ``python -m pytest`` spawned by
``grader/pytest-suite.ts``) against the AGENT's workspace varseq, imported directly. The runner passes:

  WORKSPACE_PATH = <env>/workspace       (orchestrator-provided)
  PYTHONPATH     = <env>/workspace/src   (so the workspace copy wins on sys.path)

The sole graded deliverable is ``varseq.design_task.design``. Everything the grade is measured
against (evaluators, z-norm, natural set, oracle weights) lives in ``fixtures/`` and is duplicated
grader-side, never imported from the workspace.
"""
import os
import sys
from pathlib import Path

import pytest


def _workspace_root() -> Path:
    ws = os.environ.get("WORKSPACE_PATH")
    if not ws:
        pytest.fail("WORKSPACE_PATH not set — the grader must pass the agent workspace path")
    return Path(ws).resolve()


# Workspace src wins on sys.path so the agent's copy cannot be shadowed by a site-packages varseq.
_WS_SRC = _workspace_root() / "src"
if str(_WS_SRC) not in sys.path:
    sys.path.insert(0, str(_WS_SRC))

# Grader-owned fixtures package (MeteredOracle, evaluators, z-norm, regimes) — never in the workspace.
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))


@pytest.fixture(scope="session")
def design_fn():
    """The agent's graded deliverable, imported from the workspace."""
    from varseq.design_task import design

    return design


@pytest.fixture(scope="session")
def F():
    """The grader-owned harness module (build_oracle, heldout_z, validate, REGIMES, emit_metric)."""
    import seqdesign_fixtures

    return seqdesign_fixtures
