# Third-party notices for this task

Attribution and licence notices for third-party material redistributed in this task.
This file is intentionally placed beside `task.toml`, outside `environment/source/env/`,
so that it travels with the distribution without becoming visible to the model under
evaluation.

## SauersML components

This task incorporates code and/or specifications from repositories authored by SauersML.
Emulated holds a written licence from the author covering redistribution as part of this dataset
and use in commercial AI model training, fine-tuning and evaluation. See `LICENSING.md` at the
root of this delivery.

Specifically, `environment/source/env/grader/zoo.py` records that it is "a benchmark-focused
adaptation of the local manifold-zoo generators in `Manifold-SAE/experiments/amm_zoo` and
`gam/bench/bsf_manifold_zoo.py`" — that is, an **adaptation of `SauersML/Manifold-SAE`**
(AGPL-3.0 publicly) with the invariants retained. The calibration targets under
`environment/source/env/calibration/` inherit the same lineage.

## Neural coordinate bank (CC-BY-4.0)

`environment/source/env/grader/data/neural_manifolds_v1.npz` (with its manifest
`neural_manifolds_v1.json`) is an inference-free derivative of two public, **CC-BY-4.0** activation
datasets. It contains only four-dimensional coordinate clouds — no text, images, model weights or
full-width activations — and its provenance is recorded in
`environment/source/env/grader/data/README.md`. That attribution lives inside the sealed
environment tree; the licence requires it to travel with the redistributed derivative, so it is
repeated here:

| Source dataset | Revision | What the bank uses |
|---|---|---|
| `scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations` | `9682f489707fd078f5ee14730b5f5ce0c7604aed` | final-token residual streams from layers 4, 8, 12, 15 of `meta-llama/Llama-3.2-1B-Instruct` run over LMSYS-Chat-1M |
| `mpg-ranch/drone-lsr` | `e8ee1381cd8f0dd81f590884a31816e6ef47069e` | three DINOv2-base CLS views and the morning DINOv3 CLS view of forest-plot drone imagery |

Both are declared CC-BY-4.0 by their dataset cards; full text at `LICENSES/CC-BY-4.0.txt`. The
reduction is reproducible offline via `tools/build_neural_bank.py`, and the loader verifies the
artifact SHA-256 at grade time.

## Everything else

The workspace ships only a Gaussian-noise format stub; all graded observations are simulated fresh
at evaluation time. The task runtime is NumPy (BSD-3-Clause); the Node harness declares its npm
dependencies in lockfiles (Apache-2.0 / MIT / ISC / BSD / BlueOak). No third-party library is
vendored and no model weights ship.
