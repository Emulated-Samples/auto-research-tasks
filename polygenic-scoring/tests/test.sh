#!/bin/bash
# Harbor verifier entrypoint (generated) — shim into `harbor grade`.
export HYPERFOCAL_LOGS_DIR="$(mktemp -d)"
cd /hyperfocal/env
exec node packages/env-orchestrator/bin/env-orchestrator.js harbor grade --problem from_scratch_svpgs
