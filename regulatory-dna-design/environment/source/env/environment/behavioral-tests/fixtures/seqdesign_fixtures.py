"""Grader-OWNED machinery for the seq-design held-out evaluation harness.

Everything the grade depends on lives here, NEVER imported from the agent's workspace:
  * the oracle net (DeepSTARR) + authoritative weights — the agent is *given* the oracle, but the
    grader builds its own so a tampered workspace copy cannot move the grade;
  * the two held-out evaluators (EvalCNN, different arch/seed/split) — the answer key;
  * the per-(model,track) z-norm stats + the 2000-natural-seq set — the frozen scale + novelty ref.

The sole variable under test is `varseq.design_task.design`, imported from the workspace.
The grade is the held-out activity of the sequences that function designs — a number in a model
the agent never sees.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.realpath(os.path.dirname(__file__))
_B2I = {b: i for i, b in enumerate("ACGT")}
TRACKS = ("Dev", "Hk")
INPUT_LENGTH = 249

# Side-channel for continuous metrics (mirrors the epistasis emit_metric mechanism; grader-side only).
_METRICS_FILE = os.environ.get("SEQDESIGN_METRICS_FILE")


def emit_metric(name, value):
    if not _METRICS_FILE:
        return
    with open(_METRICS_FILE, "a") as f:
        f.write(json.dumps({"name": name, "value": float(value)}) + "\n")


# ------------------------------------------------------------------ nets (grader-owned copies)
class DeepSTARR(nn.Module):
    """Oracle net: DeepSTARR (Basset-style 4-conv + 2-FC), 249bp -> (Dev, Hk)."""

    def __init__(self, length=INPUT_LENGTH):
        super().__init__()

        def blk(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]

        self.body = nn.Sequential(*blk(4, 256, 7), *blk(256, 60, 3), *blk(60, 60, 5), *blk(60, 120, 3))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(120 * (length // 16), 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.body(x))


class EvalCNN(nn.Module):
    """Held-out evaluator A/B: a maxpool conv tower (wider kernels, global-avg-pool, no big FC)."""

    def __init__(self, ch=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(4, ch, 11, padding=5), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(ch, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch, 2))

    def forward(self, x):
        return self.head(self.body(x))


class _ResBlock(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=d, dilation=d); self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=d, dilation=d); self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return torch.relu(x + h)


class DilatedResNet(nn.Module):
    """Held-out evaluator C: a GENUINELY different family (dilated residual, no maxpool). Its distinct
    receptive-field structure does NOT share the maxpool-CNN blind spot, so including it in the
    ensemble-min kills aggressive-optimization designs that only fool the maxpool towers."""

    def __init__(self, ch=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(ch, d) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 2))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def _onehot(seqs):
    x = np.zeros((len(seqs), 4, INPUT_LENGTH), np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:INPUT_LENGTH]):
            if c in _B2I:
                x[i, _B2I[c], j] = 1.0
    return torch.from_numpy(x)


@torch.no_grad()
def _raw_score(model, seqs, bs=512):
    out = []
    for i in range(0, len(seqs), bs):
        out.append(model(_onehot(list(seqs[i:i + bs]))).numpy())
    return np.concatenate(out, axis=0)  # (N, n_tracks)


# ------------------------------------------------------------------ loaders (cached)
_CACHE = {}


def _load():
    if _CACHE:
        return _CACHE
    torch.set_num_threads(1)
    oracle = DeepSTARR()
    oracle.load_state_dict(torch.load(os.path.join(_HERE, "oracle_weights.pth"), map_location="cpu"))
    oracle.eval()
    evs = []
    for name, cls in (("evaluatorA.pt", EvalCNN), ("evaluatorB.pt", EvalCNN), ("evaluatorC.pt", DilatedResNet)):
        m = cls()
        m.load_state_dict(torch.load(os.path.join(_HERE, name), map_location="cpu"))
        m.eval()
        evs.append((name[:-3].replace("evaluator", "eval"), m))
    with open(os.path.join(_HERE, "znorm_stats.json")) as f:
        znorm = json.load(f)["stats"]
    natural = np.load(os.path.join(_HERE, "natural_peaks.npy")).tolist()
    _CACHE.update(oracle=oracle, evals=evs, znorm=znorm, natural=natural)
    return _CACHE


# ------------------------------------------------------------------ metered oracle
class BudgetExceeded(Exception):
    pass


class MeteredOracle:
    """The callable handed to design(). Counts every scored sequence against `budget`. Exposes the
    same introspection interface as the dev oracle (varseq.oracle.Oracle): input_length, n_tracks."""

    def __init__(self, model, budget):
        self._model = model
        self._budget = budget
        self.used = 0
        self.input_length = INPUT_LENGTH
        self.n_tracks = model.head[-1].out_features
        self.tracks = TRACKS

    def __call__(self, seqs, batch_size=512):
        if isinstance(seqs, str):
            raise TypeError("oracle expects a list of sequences, not a single string")
        n = len(seqs)
        if self.used + n > self._budget:
            raise BudgetExceeded(f"used {self.used}+{n} > budget {self._budget}")
        self.used += n
        return _raw_score(self._model, list(seqs), batch_size)


# ------------------------------------------------------------------ scoring the designs
def _z(model_key, track_idx, raw_col):
    s = _load()["znorm"][model_key][TRACKS[track_idx]]
    return (raw_col - s["mean"]) / s["std"]


def heldout_z(designs, track_idx):
    """Ensemble-min held-out z over 3 evaluators (2 maxpool-CNN + 1 dilated-residual), mean over
    designs. The min across genuinely different architectures is what makes aggressive oracle-gaming
    fail: a design must transfer to ALL THREE, not just the maxpool family that shares the oracle's
    blind spot."""
    c = _load()
    per_design_min = None
    for ek, model in c["evals"]:
        z = _z(ek, track_idx, _raw_score(model, designs)[:, track_idx])
        per_design_min = z if per_design_min is None else np.minimum(per_design_min, z)
    return float(np.mean(per_design_min))


def oracle_z(designs, track_idx):
    return float(np.mean(_z("oracle", track_idx, _raw_score(_load()["oracle"], designs)[:, track_idx])))


def build_oracle(budget):
    return MeteredOracle(_load()["oracle"], budget)


# ------------------------------------------------------------------ session meter (root-owned)
# The per-call MeteredOracle above only bounds what design() spends INSIDE the grade. The budget
# that matters for the task's premise -- sample efficiency -- is the whole session: before this,
# the model weights sat in the agent's workspace, so an agent could run an unbounded offline search
# and return constants, satisfying every gate while spending zero metered queries.
#
# The weights now live behind a root-owned scoring daemon that counts every sequence it scores into
# a root-owned counter (0600, inside a 0700 dir -- the agent uid can neither read nor write it).
# This is the number the budget assertion trusts; the workspace-side meter is advisory.
_COUNTER_PATH = "/opt/seqdesign/counter"

# Sequences the scoring service may be asked for across one whole session, development included.
# Held above the graded per-call budget (40k) so honest iteration fits -- smoke runs at small budgets
# plus ~4 full-scale rehearsals -- and low enough that precomputing an answer offline buys almost
# nothing. MEASURED (box7, 2026-08-19, reference design() at seed 101, held-out ensemble-min z):
#   budget    40k -> 18.5665 | 80k -> 18.9278 | 120k -> 18.9278 | 200k -> 18.9278 | 400k -> 18.8231
# i.e. the search converges by ~2x budget; the whole 5x session allowance is worth +0.36 held-out z
# (+0.014 grade) over an honest in-budget run, and 10x is WORSE than 2x. The hardcode-a-constant hack
# the old workspace-resident weights enabled is therefore priced out, not merely discouraged.
SESSION_BUDGET = 200000
_session_override = os.environ.get("SEQDESIGN_SESSION_BUDGET_DEBUG")
if _session_override:
    SESSION_BUDGET = int(_session_override)


def session_queries():
    """Total sequences scored through the scoring service this session, or None if the meter is
    absent (the daemon never came up -- an infra fault, not an agent one)."""
    try:
        with open(_COUNTER_PATH) as f:
            return int((f.read() or "0").strip() or 0)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ validation / novelty
def validate(designs, length, n_designs):
    """Return (ok, reason). Enforces the catastrophic output contract (verifier §4.1)."""
    if not isinstance(designs, (list, tuple)):
        return False, f"design() returned {type(designs).__name__}, not a list"
    if len(designs) != n_designs:
        return False, f"returned {len(designs)} designs, expected {n_designs}"
    for s in designs:
        if not isinstance(s, str):
            return False, f"non-string design: {type(s).__name__}"
        if len(s) != length:
            return False, f"length {len(s)} != required {length}"
        if any(c not in _B2I for c in s):
            return False, "sequence has characters outside A/C/G/T"
    return True, "ok"


def min_edit_to_natural(designs, sample=400):
    """Diagnostic novelty: min Hamming (equal length) to a sample of natural peaks."""
    nat = _load()["natural"][:sample]
    best = INPUT_LENGTH
    for d in designs:
        for n in nat:
            h = sum(a != b for a, b in zip(d, n))
            if h < best:
                best = h
    return best


# ------------------------------------------------------------------ regimes (DeepSTARR: Dev only, v3)
# 3-WAY ensemble (evalA,B maxpool-CNN + evalC dilated-residual) revealed that Hk's high 2-way scores
# were a SHARED maxpool-CNN artifact: population search Hk 2-way 22.1 -> 3-way -0.21 (collapses on the
# different arch). Dev is GENUINE: population 2-way 16.6 -> 3-way 16.6 (transfers to all 3 arches).
# Ladder (3-way ens-min, budget 40k, held-out z): overfitter -0.61, random 0.09, natural 1.37,
# careless greedy 2.79, competent greedy 5.13, gold design() reference 18.57. Grade anchors
# (floor 0.5 / SATURATION 1.27, gold reference -> ~0.80 overall) live in grader/pytest-suite.ts.
# Grade Dev only. Hk demoted (3-way: only conservative greedy clears natural, tiny gradient) — kept as
# a possible future "restraint" regime, not graded in v3.
REGIMES = {
    "Dev": dict(track=0, budget=40000, length=INPUT_LENGTH, n_designs=4, seed=101),
}

# Grader-only debug hook: shrink the budget for a fast wiring smoke test. Never set in production
# (the grader does not export it); the agent cannot reach it.
_budget_override = os.environ.get("SEQDESIGN_BUDGET")
if _budget_override:
    for _r in REGIMES.values():
        _r["budget"] = int(_budget_override)
