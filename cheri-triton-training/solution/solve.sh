#!/bin/bash
# Reference solution (oracle, generated): apply the pinned gold diff to the
# live tree.
#
# MUST run with cwd = repo root: patch paths are workspace/...-prefixed and
# `git apply` from a subdirectory silently no-ops (rc 0!) on outside paths.
set -euo pipefail

cd /hyperfocal/env
git apply -p1 --whitespace=nowarn /solution/solution.patch
echo "[solve.sh] solution.patch applied"
