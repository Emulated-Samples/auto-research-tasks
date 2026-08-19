# Licensing

This file records the licence position for third-party material redistributed in this
repository. It is a companion to `THIRD_PARTY_NOTICES.md`, which carries the per-component
attribution, and to the per-task `<task>/THIRD_PARTY_NOTICES.md` files.

## SauersML components: written grant

The following upstream repositories, authored by SauersML, are incorporated into this
repository:

| Repository | Public licence on the repo | Components used |
|---|---|---|
| `SauersML/Manifold-SAE` (with `SauersML/gam`) | AGPL-3.0 | Data simulator adapted into `manifold-representation`'s grader |
| `SauersML/SV-PGS` | AGPL-3.0 | Algorithm specifications for `polygenic-scoring` |

Both tasks record SauersML as the benchmark author. These components are redistributed under a
written licence grant obtained from the author, covering redistribution as part of this
dataset and use in commercial AI model training, fine-tuning and evaluation. The grant
supersedes the public licences listed above for the purposes of this repository. The AGPL-3.0
text those repositories publish is at `LICENSES/AGPL-3.0.txt` for reference.

## MIT upstreams

`cheri-triton-training` is built on cherimoya, MIT, Copyright (c) 2026 Jacob Schreiber.
`regulatory-dna-design` is built on gReLU, MIT, Copyright (c) 2024 Genentech, Inc. Both are
modified for their tasks; the per-task notices carry the details. Full licence texts are at
`LICENSES/MIT-cherimoya-jmschrei.txt` and `LICENSES/MIT-gReLU-Genentech.txt`.

## Data components

`regulatory-dna-design` redistributes the ENCODE blacklist v2 (Boyle Lab, GPL-3.0), JASPAR
2024 consensus motifs (CC-BY-4.0) and HOCOMOCO v12/v13 CORE motifs (WTFPL). GPL-3.0 attaches
to the blacklist data files themselves and does not extend to the surrounding task.
`manifold-representation` includes a coordinate bank derived from CC-BY-4.0 datasets; the
attribution is carried in the task notice. `polygenic-scoring`'s corpora are synthetic and
generated in-repo. See `THIRD_PARTY_NOTICES.md` for the full data table.

## Everything else

All other third-party material is redistributed under its own licence. See
`THIRD_PARTY_NOTICES.md` for the component-level attribution and `LICENSES/` for full texts.
