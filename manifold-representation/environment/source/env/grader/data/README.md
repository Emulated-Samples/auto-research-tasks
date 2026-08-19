# Neural coordinate-bank provenance

`neural_manifolds_v1.npz` is an inference-free derivative of two public,
CC-BY-4.0 activation datasets. It contains only four-dimensional coordinate
clouds; it contains no text, images, model weights, or full-width activations.

Sources:

- [`scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations`](https://huggingface.co/datasets/scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations), revision
  `9682f489707fd078f5ee14730b5f5ce0c7604aed`, produced from
  `meta-llama/Llama-3.2-1B-Instruct` on LMSYS-Chat-1M. The bank uses final-token
  residual streams from layers 4, 8, 12, and 15.
- [`mpg-ranch/drone-lsr`](https://huggingface.co/datasets/mpg-ranch/drone-lsr), revision
  `e8ee1381cd8f0dd81f590884a31816e6ef47069e`, containing DINOv2-base and
  satellite-pretrained DINOv3 representations of morning, noon, and afternoon
  drone imagery from forest plots. The bank uses the three DINOv2 CLS views and
  the morning DINOv3 CLS view.

The offline maintainer command `python tools/build_neural_bank.py` performs the
complete reproducible reduction. It partitions source examples before fitting,
uses the same partition for every layer/view from a source, fits deterministic
four-dimensional randomized PCA only on training rows, and applies that fixed
chart to disjoint match and score rows. The adjacent JSON manifest records the
exact source columns, revisions, row counts, explained variance, reducer seed,
and artifact SHA-256. Runtime loading independently checks the checksum, schema,
finite values, and row separation.

The derived coordinate bank retains the source datasets' CC-BY-4.0 attribution
requirements. The benchmark's additive mixtures are newly simulated from these
coordinates and do not assert that the source networks themselves use an
additive factorization.
