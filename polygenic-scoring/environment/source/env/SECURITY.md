# Deployment / integrity contract

This repository contains the **grader-side** material for svpgsbench: the hidden
held-out labels (`corpus/*/truth/y_test.csv`, `anchors.json`, `truth.npz`), the
full data generator (`datagen/`), and the exact score-1 reference program
(`gold/{fit,predict,pgs_core.py}`). Corpus schema v8 derives generation streams
from an external 32-byte key; neither the key nor the derived seeds is serialized.
The authenticated HMAC domain strings intentionally retain their historical
`schema-v6` namespace and are immutable cryptographic protocol identifiers, not a
claim about the current corpus schema. The
environment's non-gameability depends on the agent **never seeing any of it**.

## What the solver agent may see

Only each dataset's `public/` directory, materialized into a fresh workspace:
`family.txt`, `formula.txt`, `dgp.json`, `variant_metadata.tsv`,
`covariates_{train,test}.csv`, `genotypes_{train,test}.tsv`. Nothing else.

## What the agent must NEVER see

- `corpus/*/truth/**` — held-out labels and the precomputed naive/reference
  anchors. Reading `anchors.json` alone would let a submission tune to hit the
  reference metric values.
- `datagen/**`, `corpus/*/manifest.json`, and the external corpus key — together
  they determine the keyed private generation streams and therefore the test set.
- `gold/`, `reference/`, `grader/`, `validation/`, `rollout_analysis/`, and this
  repository as a whole.

## How that is enforced

- **This repository is private.** It must not enter any agent's context window,
  tool-readable filesystem, or training corpus. Treat the URL as a secret.
- **Maintainer rollout analysis is owner-only.** Setup and grading chmod
  `rollout_analysis/` to `0700` and verify with the fixed unprivileged
  `svpgsub` identity that it is unreadable. Setup never copies it into
  `workspace/`.
- **At grade time**, `grader/submission_runner.py` isolates **every untrusted
  phase — `build.sh`, `fit`, and `predict`** — itself, not merely by assertion:
  - **Mandatory fail-closed sandbox** — `bwrap` is the only execution path. If
    it is missing, the grader aborts; there is no CLI flag, environment variable,
    or development mode that can run untrusted code unconfined.
  - **Sandbox preflight** — before scoring, `preflight_sandbox` runs as the fixed
    unprivileged `svpgsub` uid and probes a sentinel, `/root`, repository `.git`,
    the first corpus truth directory, and network egress. Scoring aborts unless
    the uid drop succeeds and every read/connect fails.
  - **Confined, in-a-copy build** — `build.sh` is no longer run in place with
    the grader's inherited environment. Its source is copied into a private
    build dir and compiled there with a **sanitized env** (no repo `PYTHONPATH`
    → `import grader`/`import datagen` fail) and, under `bwrap`, with **no
    network and NO `corpus/*/truth`, grader, or datagen mounts** — only the build
    dir is visible. The built artifact is then frozen read-only and used as the
    source for `fit`/`predict`. So `build.sh` cannot read or exfiltrate the
    held-out labels, and cannot mutate anything the grader later reads.
  - **Sanitized environment** — each phase gets a minimal env with `HOME`/`TMPDIR`
    pinned inside its run dir and repo-pointing variables (`PYTHONPATH`,
    `VIRTUAL_ENV`, `PYTHONHOME`, …) stripped, so `import grader` / `import datagen`
    can no longer reach the hidden truth or the generator via the grader's
    `sys.path`.
  - **Bounded regular-file tree** — the submission is rejected if it contains a
    symlink, FIFO, socket, device, oversized file, too many files, or exceeds the
    total byte limit. This prevents path escapes and blocking/special-file tricks
    before any recursive copy or build begins.
  - **Mount + network sandbox** — under `bwrap` each phase runs with
    explicit PID/network/IPC/UTS/cgroup namespaces; it sees only fixed
    runtime/toolchain paths, an owner-only `/work`, and (for fit/predict)
    root-frozen `/submission`. The host user namespace is retained so the fixed
    host `svpgsub` identity remains mapped when `setpriv` enters its private
    workdir; the process still drops all groups and runs as that unprivileged uid.
    Whole `/etc` and `/opt` are never exposed. Corpus truth, the grader, and the
    datagen tree are *unreadable*, not merely unreferenced. On a host without the
    namespace tools (e.g. a macOS dev laptop), a real grade **fails closed**. The
    environment provisioner installs `bwrap` before a scored run, and the grader
    also applies fail-closed CPU/address-space/file-size/process/open-file limits
    and retains only a bounded stderr tail while continuously draining child
    output, so resource/output floods stay bounded.
  - Only the `public/` files are staged into the run dir, with the held-out test
    features staged in **after** `fit` exits (`grader/submission_runner.py` also
    decompresses `genotypes_*.tsv.gz` there).

## Corpus authentication key

Provision exactly 32 random bytes as an owner-only (`0600`), single-link regular
file at `/run/secrets/svpgsbench-corpus.key`. It must live outside the repository.
The builder derives independent `dgp`, `materialize`, `bootstrap`, and opaque-ID
streams by HMAC over the pipeline digest, purpose, category, and replicate. The
grader refuses a manifest or anchor whose HMAC, key identifier, pipeline digest,
or file hashes disagree. There is no unkeyed generation or grading mode.
