#!/bin/bash
export HYPERFOCAL_LOGS_DIR="$(mktemp -d)"
cd /hyperfocal/env
exec node packages/env-orchestrator/bin/env-orchestrator.js harbor grade --problem M6_full
