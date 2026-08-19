---
name: openblas-thread-oversubscription
description: "In the svpgs env, NumPy's OpenBLAS sees 64 cores but only 4 vCPUs are allotted, so BLAS calls must be pinned to 4 threads or they run ~18x slower."
metadata: 
  node_type: memory
  type: project
  originSessionId: 17c84d78-009f-480f-93df-a9f37d5aee8a
  modified: 2026-08-18T11:00:03.032Z
---

`/opt/svpgs-venv/bin/python` links scipy-openblas 0.3.27 built with
MAX_THREADS=64, and `nproc` reports 64, but graded execution only provides
4 vCPUs. Left at the default, a float32 `gemv` over 8e7 elements took 100 ms;
with `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=4` the same call took 5.5 ms.

**Why:** OpenBLAS spawns one thread per detected core and the resulting
oversubscription dominates runtime on memory-bound BLAS-2 work, which is what
matrix-free CG solvers are made of.

**How to apply:** set the thread env vars at the very top of an entry-point
script, *before* `import numpy` — afterwards has no effect. Benchmark numbers
collected without pinning are meaningless; re-measure after pinning.
