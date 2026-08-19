# Third-party notices for this task

Attribution and licence notices for third-party material redistributed in this task.
This file is intentionally placed beside `task.toml`, outside `environment/source/env/`,
so that it travels with the distribution without becoming visible to the model under
evaluation.

## gReLU (MIT)

The `workspace/` tree of this task is derived from **gReLU**
(https://github.com/Genentech/gReLU), **Copyright (c) 2024 Genentech, Inc.**, MIT licensed.
Full text at `LICENSES/MIT-gReLU-Genentech.txt`; the in-tree copy is
`environment/source/env/workspace/LICENSE.txt`.

**Modifications:** the source has been modified by Emulated for this evaluation environment and
the package renamed (`grelu` → `varseq`). Upstream identifiers were removed from the in-image
copy so that the model under evaluation cannot retrieve the upstream solution; this notice
preserves the attribution required by the MIT licence for the distributed work.

## ENCODE blacklist v2 (GPL-3.0-only)

This task redistributes blacklist BED files from the Boyle Lab ENCODE Blacklist project
(https://github.com/Boyle-Lab/Blacklist), licensed **GPL-3.0**, at
`workspace/src/varseq/resources/blacklists/encode/` (hg19, hg38, mm10 v2). Full text at
`LICENSES/GPL-3.0.txt`. The data files are redistributed **unmodified**.

Cite: Amemiya, Kundaje & Boyle, "The ENCODE Blacklist: Identification of Problematic Regions of
the Genome", Sci Rep 9, 9354 (2019).

## Mitochondrial-homology blacklists — provenance open

`workspace/src/varseq/resources/blacklists/mito_combined/` carries five BED files (hg19, hg38,
mm9, mm10 and a combined hg19/mm10 track) that arrive through gReLU and carry no upstream
provenance or licence statement in the tree. The closest public source, the mitochondrial
blacklist project at https://github.com/caleblareau/mitoblacklist, **publishes no licence at
all**.

## Motif databases

| File | Source | Terms |
|---|---|---|
| `jaspar_2024_consensus.meme` | JASPAR 2024 (https://jaspar.elixir.no) | **CC-BY-4.0** — attribution required; text at `LICENSES/CC-BY-4.0.txt` |
| `H12CORE_meme_format.meme` | HOCOMOCO v12 (https://hocomoco14.autosome.org) | **WTFPL**, which upstream states may be treated as CC-BY; texts at `LICENSES/WTFPL-2.0.txt` and `LICENSES/CC-BY-4.0.txt` |
| `H13CORE_meme_format.meme` | HOCOMOCO v13 | as above |

Cite for JASPAR: Rauluseviciute et al., "JASPAR 2024: 20th anniversary of the open-access
database of transcription factor binding profiles", Nucleic Acids Res 52, D174–D182 (2024).
Cite for HOCOMOCO: Vorontsov et al., "HOCOMOCO in 2024: a rebuild of the curated collection of
binding models for human and mouse transcription factors", Nucleic Acids Res 52, D154–D163 (2024).

All three files are redistributed **unmodified**.

## Activity-model weights and seed sequences — provenance open

Two binary artifacts carry the science of this task and have no provenance recorded in the tree:

- `environment/source/seqdesign/oracle.pth` (and the grader-owned copies under
  `environment/source/env/environment/behavioral-tests/fixtures/`) — weights for a **DeepSTARR**
  network (de Almeida, Reiter, Pagani & Stark, Nat Genet 54, 613–624, 2022). The **architecture**
  is re-implemented in this repository from the published description; the weights are trained
  artifacts.
- `workspace/src/varseq/_assets/seed_sequences.npy` — 2,048 "real regulatory sequences (natural
  enhancers)" offered to the agent as design material.

