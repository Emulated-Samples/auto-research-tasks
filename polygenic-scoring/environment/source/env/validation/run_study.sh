#!/bin/bash
# Resilient study driver: runs each (category,seed) as a short-lived, retriable
# subprocess so OOM SIGKILLs cost at most one retry and never lose completed work.
# Usage: run_study.sh <N> <P> <seed0> <n_seeds> <outfile> [category ...]
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=${SVPGSBENCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${SVPGSBENCH_PYTHON:-${SVPGS_HOME:-$(dirname "$ROOT")/SV-PGS}/.venv-py312/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
N=${1:-1800}; P=${2:-450}; SEED0=${3:-100}; NSEEDS=${4:-5}; OUT=${5:-$ROOT/validation/study_results.jsonl}
shift 5 || true
CATS=("$@")
if [ ${#CATS[@]} -eq 0 ]; then
  # authoritative category list, straight from the recipes (never goes stale)
  CATS=($(cd "$ROOT" && "$PY" -c "from datagen.categories import CATEGORIES; print(' '.join(CATEGORIES))"))
fi
touch "$OUT"
# the study worker log dir is gitignored/untracked, so a clean checkout lacks it;
# create it before the first redirect (mirrors build_corpus.sh).
mkdir -p "$ROOT/validation/logs"
for cat in "${CATS[@]}"; do
  for s in $(seq "$SEED0" $((SEED0+NSEEDS-1))); do
    if grep -q "\"category\": \"$cat\", \"seed\": $s," "$OUT" 2>/dev/null; then
      echo "[skip] $cat seed=$s (already done)"; continue
    fi
    ok=0
    for attempt in 1 2 3 4 5; do
      echo "[run] $cat seed=$s attempt=$attempt N=$N P=$P"
      cd "$ROOT" && $PY -u validation/robustness.py ONE "$cat" "$s" "$N" "$P" "$OUT" \
        >>"$ROOT/validation/logs/study_worker.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ] && grep -q "\"category\": \"$cat\", \"seed\": $s," "$OUT" 2>/dev/null; then
        echo "[done] $cat seed=$s"; ok=1; break
      fi
      echo "[retry] $cat seed=$s rc=$rc (likely OOM); cooling down"; sleep 8
    done
    [ $ok -eq 0 ] && echo "[FAIL] $cat seed=$s after retries"
  done
done
echo "[STUDY-COMPLETE] $OUT"
