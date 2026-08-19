#!/bin/bash
set -euo pipefail

cd /hyperfocal/env
git apply -p1 --whitespace=nowarn /solution/solution.patch
echo "[solve.sh] solution.patch applied"
