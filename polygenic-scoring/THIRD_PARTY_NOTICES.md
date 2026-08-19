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

Specifically:

- `environment/source/env/tasks/from_scratch_svpgs/task.toml` records the benchmark's author as
  **SauersML**.
- `environment/source/env/reference/01_inference_engine.md`, `02_model_and_preprocessing.md` and
  `03_priors_annotations_ld.md` are **reverse-engineered specifications of `SauersML/SV-PGS`**
  (AGPL-3.0 publicly), written with `file:line` citations into that source. They are derivative
  documentation rather than copied code, which makes the grant position **more** load-bearing
  here, not less.
- `environment/source/env/datagen/dgp.py` states that its base effect sampler follows the
  hierarchical global-local shrinkage model used by SV-PGS.

`environment/source/env/reference/run_svpgs.py` is an adapter for a private SV-PGS checkout used
as one measured model-zoo arm. **No SV-PGS source ships in this repository**, and the adapter
raises `ImportError` without an external checkout; the shipped anchors and gold solution are
Emulated's own NumPy implementation.

## Datasets

The 45 graded cohorts under `environment/source/env/corpus/` (158 MB) and the calibration reports
under `environment/source/env/validation/` are **entirely synthetic**, generated in-repo by
`datagen/dgp.py` via `datagen/build_corpus.py`. Sample and variant identifiers are sequential
(`s10027…`, `v0…`); there are no rsIDs, no chromosome/position coordinates and no real cohort
identifiers. **No human genomic data is redistributed by this task.**

## Third-party libraries

The task runtime is NumPy 2.1.3 and SciPy 1.14.1 (BSD-3-Clause) installed at build time; the Node
harness declares its npm dependencies in lockfiles (Apache-2.0 / MIT / ISC / BSD / BlueOak). No
third-party library is vendored, no model weights ship, and no GPL-family package is installed or
listed.
