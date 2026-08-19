#!/usr/bin/env python3
"""Write a structureless format stub into the agent's workspace.

This is NOT sample data. It is Gaussian noise at a shape the prompt already
discloses, so a solver can exercise its own `fit`/`transform` -- the config keys,
the output schema, and the wall-clock cost of a fit at a realistic shape -- and
learn nothing whatsoever about the generative model.

It exists because both Haiku rollouts on this env scored exactly 0.000 without it
and neither zero measured scientific ability. one run read
`config["output_file"]` (the key is `predictions_file`) and crashed on all nine
suites; another blew the disclosed 75s fit budget on all nine. With no data
of any shape in the workspace, a submission could not be run even once before it
was graded, so an exact multi-array contract had to be guessed from prose. That
is variance uncorrelated with capability.

There is no structure here to leak: no latent objects, no sparse support, no
manifold geometry, no correlations. The shape is the smallest production case and
both numbers are public in every real `config.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

# The smallest production case. Both numbers are disclosed at run time anyway.
AMBIENT_DIM = 36
MAX_FEATURES = 22
TRAIN_ROWS = 1_900
EVAL_ROWS = 720

README = """# Format stub -- structureless

These arrays are Gaussian noise. There are no latent objects in them, no sparse
support, no manifold geometry and no correlations.

They exist so you can run your own `fit` and `transform` before you are done:
check that you read the config keys you expect, that you write a well-formed
`predictions.npz`, and that your `fit` finishes inside its budget at this shape.
Fitting them will teach you nothing about the real observations, which are drawn
fresh and arrive only when your programs are invoked.

The shape here is the smallest case. Larger cases are wider and have more rows;
the `config.json` your programs receive states the real numbers at run time.
"""


def write_stub(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(root / "train.npz", x=rng.standard_normal((TRAIN_ROWS, AMBIENT_DIM)).astype(np.float32))
    np.savez(root / "eval.npz", x=rng.standard_normal((EVAL_ROWS, AMBIENT_DIM)).astype(np.float32))
    (root / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "ambient_dim": AMBIENT_DIM,
                "max_features": MAX_FEATURES,
                "train_rows": TRAIN_ROWS,
                "predictions_file": "predictions.npz",
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "README.md").write_text(README)
    return root


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_format_stub.py <workspace-directory>")
    print(f"wrote format stub to {write_stub(Path(sys.argv[1]) / 'format_stub')}")
