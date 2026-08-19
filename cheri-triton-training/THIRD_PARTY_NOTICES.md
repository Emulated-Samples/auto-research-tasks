# Third-party notices for this task

Attribution and licence notices for third-party material redistributed in this task.
This file is intentionally placed beside `task.toml`, outside `environment/source/env/`,
so that it travels with the distribution without becoming visible to the model under
evaluation.

## cherimoya (MIT)

The `workspace/` tree of this task is derived from **cherimoya**
(https://github.com/jmschrei/cherimoya), **Copyright (c) 2026 Jacob Schreiber**, MIT licensed.
Full text at `LICENSES/MIT-cherimoya-jmschrei.txt`. The `LICENSE` file shipped inside the
workspace carries the decontaminated copyright line ("The Cherimoya Authors"); the upstream
attribution above is the one the licence requires and is recorded here.

**Modifications:** the source has been modified by Emulated for this evaluation environment —
the Triton kernel path and parts of the training loop are withheld for the task. Upstream
identifiers (author names, project URLs) were removed from the in-image copy so that the model
under evaluation cannot retrieve the upstream solution; this notice preserves the attribution
required by the MIT licence for the distributed work.

`cherimoya/io.py` records that its data-loading path is adapted from work by Avanti Shrikumar
and Ziga Avsec (the BPNet / ChromBPNet lineage, MIT); that adaptation reached this delivery
through cherimoya and is covered by the notice above.

## Build-time dependencies

The image installs `bpnet-lite`, `tangermeme`, `modisco-lite`, `macs3`, `bam2bw`, `torch` and
`triton` from PyPI at build time (MIT / BSD-3-Clause / Apache-2.0). Those packages are fetched
on the buyer's machine and are not redistributed by this repository; each retains its own
in-tree licence and per-file headers.

No datasets or model weights are redistributed with this task.
