# svpgsbench environment wrapper

This directory holds the TypeScript `EnvironmentDefinition` (`src/index.ts`) and
the task prompt (`problems.yaml`) that expose the svpgsbench grader to the
Hyperfocal/Harbor platform.

## Dependencies

The repository tracks `packages/env-base` and `packages/env-orchestrator` as
submodules. Initialize the exact gitlinks recursively, then install from each
committed lockfile and build in dependency order:

```bash
git submodule update --init --recursive
(cd packages/env-base && npm ci && npm run build)
(cd packages/env-orchestrator && npm ci && npm run build && npm run link)
(cd environment && npm ci && npm run build)
```

Do not substitute `npm install` in a release checkout: `npm ci` fails closed on
package/lock drift and leaves the reviewed dependency graph unchanged.

At runtime, `setupProblem` provisions the fixed numeric toolchain and required
Bubblewrap sandbox once under `/opt`, materializes the corpus's exact 32-byte
authentication key as an owner-only (`0600`), single-link regular file at
`/run/secrets/svpgsbench-corpus.key`, then creates an empty solver workspace.
The sole source credential is `grader/corpus-key.hex`; `grader/` is sealed from
the solver account before the workspace is exposed. The wrapper passes only the
fixed runtime path to the schema-v8 grader. There is no environment-variable or
alternate-key fallback.

Setup also runs `grader/preflight.py` before the solver starts. The preflight
checks Bubblewrap confinement and authenticates the deployed manifest, every
public/truth file, every anchor, the complete category grid, and the pinned
development report. A stale or partial release therefore fails before consuming
agent work rather than becoming an infrastructure-censored score after grading.

The Python grader, corpus, datagen, reference, and validation code do not depend
on `env-base`; only this TypeScript wrapper does.
