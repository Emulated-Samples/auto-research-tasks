"""
Shared fixtures for the Variant Effect Prediction (VEP) behavioral tests.

Promoted from the Stage-1 oracle de-risk spikes. Provides:
  * a deterministic seeded 2xkernel-5 ConvModel oracle (non-degenerate dinuc-shuffle
    background -- a kernel-1 composition model collapses the background to sd=0),
  * the committed synthetic genome (tests/files/vep_synth.fa, CustomGenome; no downloads),
  * four homogeneous-allele-length variant sets (SNV / insertion / deletion / MNV) whose
    reference alleles match the genome at check_reference's window so the healthy path
    emits no warnings.

NB varseq requires allele-length-homogeneous batches per call (strings_to_indices stacks
alleles), so each set is a separate call; mixing lengths raises AssertionError.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from varseq.lightning import LightningModel

# --- determinism knobs (confirm at frozen-oracle audit) -------------------------------
SEQ_LEN = 21
N_SHUFFLES = 30
SEED = 7

_HERE = os.path.realpath(os.path.dirname(__file__))
GENOME_PATH = os.path.join(_HERE, "files", "vep_synth.fa")


def _genome_string() -> str:
    """Read the single-chromosome synthetic genome as a plain string."""
    lines = []
    with open(GENOME_PATH) as fh:
        for line in fh:
            if not line.startswith(">"):
                lines.append(line.strip())
    return "".join(lines)


CHROM_SEQ = _genome_string()

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _ref_match(pos: int, ref_len: int) -> str:
    """
    The reference string that check_reference extracts for a variant at 1-based `pos`
    with a reference allele of length `ref_len` (window start = pos - ceil(ref_len/2)).
    Building refs this way keeps the healthy path warning-free.
    """
    start = pos - int(np.ceil(ref_len / 2))
    return CHROM_SEQ[start:start + ref_len]


def build_oracle_model() -> LightningModel:
    """
    Seeded fixed-weight ConvModel: n_tasks=1, two k=5 conv layers (stem + 1 block, RF 9),
    relu, avg final pool. The head is forced strictly positive so log2FC is well-defined.
    Deterministic given torch.manual_seed(0).
    """
    torch.manual_seed(0)
    model = LightningModel(
        model_params={
            "model_type": "ConvModel",
            "n_tasks": 1,
            "stem_channels": 8,
            "stem_kernel_size": 5,
            "n_conv": 2,  # counts the stem: Stem + 1 ConvBlock = two k=5 conv layers
            "channel_init": 8,
            "kernel_size": 5,
            "act_func": "relu",
            "norm": False,
            "residual": False,
            "dropout": 0.0,
            "final_pool_func": "avg",
        },
        train_params={"task": "regression", "loss": "MSE"},
    )
    model.data_params = {"train": {"seq_len": SEQ_LEN}, "tasks": {"name": ["t0"]}}
    model.eval()

    head_conv = None
    for _, mod in model.model.head.named_modules():
        if isinstance(mod, nn.Conv1d):
            head_conv = mod
    with torch.no_grad():
        head_conv.weight.data = head_conv.weight.data.abs() + 0.05
        if head_conv.bias is not None:
            head_conv.bias.data = head_conv.bias.data.abs() + 1.0
        else:
            head_conv.bias = nn.Parameter(torch.ones(head_conv.out_channels))
    return model


def variant_sets() -> "dict[str, pd.DataFrame]":
    """
    Four homogeneous-allele-length variant sets on chr1 of the synthetic genome.
    Each is a separate score_variant_effects call (allele lengths must match within a call).
    """
    sets = {
        "snv": [(50, 1), (90, 1)],   # |ref|=1, |alt|=1
        "ins": [(130, 1), (170, 1)],  # |ref|=1, |alt|=3
        "del": [(210, 3), (250, 3)],  # |ref|=3, |alt|=1
        "mnv": [(290, 2), (330, 2)],  # |ref|=2, |alt|=2
    }
    out = {}
    for kind, spec in sets.items():
        rows = []
        for pos, ref_len in spec:
            ref = _ref_match(pos, ref_len)
            if kind == "snv":
                alt = _COMP[ref]
            elif kind == "ins":
                alt = ref + "GC"                       # 1 -> 3
            elif kind == "del":
                alt = ref[0]                           # 3 -> 1
            else:  # mnv: complement (not reverse-complement, to avoid palindromes == ref)
                alt = "".join(_COMP[b] for b in ref)   # 2 -> 2
            rows.append({"chrom": "chr1", "pos": pos, "ref": ref, "alt": alt})
        out[kind] = pd.DataFrame(rows)
    return out
