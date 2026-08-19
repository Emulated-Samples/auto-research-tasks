# Sandbox integration contract

The untrusted-phase isolation primitives live in `grader/submission_runner.py`
and are **always fail-closed**. This note records the contract so `grade.py`
(and any other caller) wires them correctly.

## Primitives (submission_runner.py)

- `resolve_sandbox()` — returns the required `bwrap`/`setpriv` paths after
  validating the fixed `svpgsub` account, or raises `SandboxUnavailable`.
- `preflight_sandbox()` — as the sandboxed identity, probes a secret
  sentinel (placed outside all mounts) + a network connect; raises
  `SandboxUnavailable` unless both fail.
- `build_submission(submission_dir, log=print) -> (ok, built_dir, note)`
  — compiles `build.sh` in an **isolated, frozen copy** under bwrap with no
  network and no corpus/truth/grader mounts + a sanitized env. Build artifacts
  are bounded/regular-only, root-owned, and frozen after the build. The preflight
  runs inside, so protection holds regardless of caller. Its fixed cap is 3,600
  seconds. **Use `built_dir`** (not the original `submission_dir`) as the source
  for `fit`/`predict` so build artifacts are seen; `_rmtree_ro(built_dir)` when
  done.
- `run_on_dataset(...)` — runs fit/predict under the same mandatory sandbox with
  the non-overridable caps from `grader/contract.py`. Time never transfers
  between phases or datasets.

## Caller contract for grade.py

1. Call the sandbox preflight before touching untrusted input.
2. Use the returned `built_dir` for scoring; clean it up with `_rmtree_ro`.
3. On `SandboxUnavailable`, do **not** emit a reward file — let the wrapper
   treat it as an errored grade, so no unconfined score can exist.

Every child also receives fail-closed CPU/address-space/file-size/process/file-
descriptor limits. There is no sandbox policy object and no CLI or environment
override. Regression tests cover missing-tool failure, uid drop, minimal mounts,
resource limits, generated special files, and active leak probes.
