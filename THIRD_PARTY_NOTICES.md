# Third-party notices

This repository redistributes third-party open-source material. This file provides the
attribution and notices required by those licences. Full licence texts are in `LICENSES/`.
Per-task detail is in `<task>/THIRD_PARTY_NOTICES.md`.

These notices are deliberately placed **outside** each task's `environment/source/env/workspace/`
tree. The workspaces are decontaminated of upstream identifiers so that the model under evaluation
cannot retrieve an upstream solution; keeping the notices outside that tree preserves both the
integrity of the evaluation and the attribution the licences require.

## Copyleft components

| Component | Licence | Tasks | Modified? |
|---|---|---|---|
| ENCODE blacklist v2 (Boyle Lab) | GPL-3.0-only | `regulatory-dna-design` | No |
| `SauersML/Manifold-SAE` | AGPL-3.0 publicly; redistributed under written grant | `manifold-representation` | **Yes** — adapted into the grader |
| `SauersML/SV-PGS` | AGPL-3.0 publicly; redistributed under written grant | `polygenic-scoring` | **Yes** — specifications reimplemented for the task |

The ENCODE blacklist BED files are redistributed unmodified, so GPL-3.0 attaches to those data
files and does not extend to the surrounding task. The SauersML components are covered by the author's written grant under the terms
described in `LICENSING.md`.

## Permissively licensed components requiring attribution

| Component | Licence | Tasks | Modified? |
|---|---|---|---|
| cherimoya (c) 2026 Jacob Schreiber | MIT | `cheri-triton-training` | **Yes** — functionality withheld for the task; identifiers decontaminated |
| gReLU (c) 2024 Genentech, Inc. | MIT | `regulatory-dna-design` | **Yes** — package renamed `grelu` → `varseq`; identifiers decontaminated |

## Data components

| Component | Licence / terms | Tasks | Modified? |
|---|---|---|---|
| JASPAR 2024 consensus motifs | CC-BY-4.0 | `regulatory-dna-design` | No |
| HOCOMOCO v12 / v13 CORE motifs | WTFPL (upstream permits treating as CC-BY) | `regulatory-dna-design` | No |
| ENCODE blacklist v2 BEDs (hg19/hg38/mm10) | GPL-3.0-only | `regulatory-dna-design` | No |
| Mitochondrial-homology blacklist BEDs | redistributed with attribution | `regulatory-dna-design` | No |
| Activity-model weights and seed enhancer sequences | shipped as task fixtures | `regulatory-dna-design` | n/a |
| Neural coordinate bank (`neural_manifolds_v1.npz`), derived from `scaleinvariant/llama-3.2-1b-instruct-lmsys-chat-1m-activations` and `mpg-ranch/drone-lsr` | CC-BY-4.0 per source dataset cards — attribution carried here and in the task notice | `manifold-representation` | **Yes** — inference-free reduction to 4-D coordinates |
| Synthetic genotype / phenotype corpora | generated in-repo, no third-party terms | `polygenic-scoring` | n/a |

## Build-time dependencies

Each task's `environment/Dockerfile` builds from a public base (`amazonlinux:2023`) and installs
its Python and Node dependencies from PyPI and npm at build time on the buyer's machine —
including `torch`, `triton`, `bpnet-lite`, `tangermeme`, `modisco-lite`, `macs3`, `numpy`,
`scipy`, `scikit-learn` and `pytorch-lightning` (Apache-2.0, BSD and MIT). Those packages are
**not redistributed by this repository**; each retains its own in-tree licence and per-file
headers.

## Sample trajectories

`trajectory/` carries sample rollouts recorded from Claude Opus 5, GPT-5.6 Sol and Gemini 3.7
Flash. See the repository README for the layout.

