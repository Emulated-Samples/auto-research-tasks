# Packaging

svpgsbench ships in **one primary format: a Hyperfocal environment.** A second,
partial Harbor-task packaging exists under `tasks/from_scratch_svpgs/` and is
documented here as a follow-up, not a parallel product.

## Primary: Hyperfocal environment

This repo *is* the environment repo. The solver agent builds a from-scratch
polygenic-scoring library (`./fit`, `./predict`, optional `./build.sh`) in
`workspace/`, and the environment adapter grades it with the hidden Python
grader over the corpus.

| Piece | Path | Role |
|-------|------|------|
| Env config | `hyperfocal.yaml` | name, compute sizing, `permissionsMode` |
| Env adapter | `environment/src/index.ts` | `EnvironmentDefinition`; `runTests` shells out to `grader/grade.py` and maps per-category reward to one `TestResult` per category |
| Problems | `environment/problems.yaml` | capability categories |
| Grader | `grader/` | authoritative scorer (`grade.py`, `submission_runner.py`, `metrics.py`, `skill.py`, `perf.py`) |
| Corpus | `corpus/` | fixed held-out datasets (`public/` + hidden `truth/`) |
| Reference / generator | `reference/`, `datagen/` | grader-side only; never shipped to the agent (see `SECURITY.md`) |

### env-base dependency resolution

`environment/package.json` depends on `@hyperfocal/env-base` via
`file:../packages/env-base`. That package is **not vendored in this repo**; it is
resolved as a git submodule (declared in `.gitmodules`, same as the sibling
`imputebench` env). A clean checkout initializes the exact recursive gitlinks
and installs the reviewed lockfiles without mutating them:

```sh
git submodule update --init --recursive
(cd packages/env-base && npm ci && npm run build)
(cd environment && npm ci && npm run build)   # emits environment/dist/
```

If your deployment mirrors the Hyperfocal packages elsewhere, point the
submodule URL (or an npm workspace) at that mirror instead. The TypeScript build
(`tsc` -> `environment/dist/index.js`) requires `env-base` to be present; without
it the env adapter cannot be compiled, but the Python grader is fully standalone
and can be run directly:

```sh
python grader/grade.py corpus workspace --out /tmp/reward --sandbox require
```

## Secondary (follow-up): Harbor task

`tasks/from_scratch_svpgs/` currently contains `instruction.md` and `task.toml`
(agent/verifier/environment sizing, artifacts, timeouts). It is **intentionally
partial**: the Dockerfile, `test.sh`, per-task grader wiring, and a reference
solution are **not** duplicated here. The authoritative grader is `grader/`, and
duplicating it into the task dir would create a second copy to keep in sync.

To complete the Harbor packaging as a follow-up, a `Dockerfile` should install a
generic numeric toolchain (numpy/scipy/BLAS/LAPACK/Eigen/autodiff — but **no**
PGS/GWAS/GAM/penalized-regression library, so "from scratch" is image-enforced,
per `task.toml [verifier.env]`), and `test.sh` should invoke
`python grader/grade.py <corpus> /app/submission --out <reward_dir> --sandbox require`.
Until then, treat the Hyperfocal env above as the shipping format.

## Grader integrity at grade time

Regardless of packaging, the grader confines every untrusted phase itself and
**fails closed** if the sandbox backend (`bwrap`) is unavailable — pass
`--sandbox require` (the default) for a real grade, or `--sandbox off` only for a
local dev run with isolation disabled. See `SECURITY.md`.
