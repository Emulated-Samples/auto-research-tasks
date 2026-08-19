#!/bin/bash
# Fail-closed corpus builder: each dataset gets one declared reference fit.
# Usage: build_corpus.sh --key-file <absolute-path> --python <interpreter>
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=""
KEY_FILE=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) PY=$2; shift 2 ;;
    --key-file) KEY_FILE=$2; shift 2 ;;
    --*) echo "unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"
if [ "$#" -ne 0 ]; then
  echo "usage: build_corpus.sh --key-file <absolute-path> --python <interpreter>" >&2
  exit 2
fi
if [ -z "$KEY_FILE" ]; then
  echo "--key-file is required" >&2
  exit 2
fi
# The build needs an explicitly selected interpreter with sv_pgs importable at
# its declared versions. Choosing a checkout or system interpreter implicitly
# would make corpus provenance depend on ambient machine state.
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "--python must name an executable interpreter with sv_pgs" >&2
  exit 2
fi
if ! "$PY" -c "import sv_pgs" >/dev/null 2>&1; then
  echo "build interpreter cannot import sv_pgs: $PY" >&2
  exit 2
fi
# Authoritative category list, straight from the recipes (never goes stale).
read -r -a CATS <<<"$(cd "$ROOT" && "$PY" -c "from datagen.categories import CATEGORIES; print(' '.join(CATEGORIES))")"
REPLICATES=$(cd "$ROOT" && "$PY" -c "from grader.contract import REPLICATES_PER_CATEGORY; print(REPLICATES_PER_CATEGORY)")
mkdir -p "$ROOT/validation/logs"
LOG="$ROOT/validation/logs/build_worker.log"
for cat in "${CATS[@]}"; do
  for k in $(seq 0 $((REPLICATES-1))); do
    # A partial build has no manifest yet, so resume from a dataset only when its
    # own keyed anchor authenticates the exact source/config and every produced
    # non-anchor byte. FINALIZE independently authenticates the complete grid.
    if cd "$ROOT" && "$PY" datagen/build_corpus.py --key-file "$KEY_FILE" CHECK "$cat" "$k" >>"$LOG" 2>&1; then
      echo "[skip] $cat replicate=$k (provenance match)"; continue
    fi
    echo "[build] $cat replicate=$k"
    if ! (cd "$ROOT" && "$PY" -u datagen/build_corpus.py --key-file "$KEY_FILE" ONE "$cat" "$k" >>"$LOG" 2>&1); then
      echo "[BUILD-FAILED] $cat replicate=$k; NOT finalizing" >&2
      exit 1
    fi
    if ! (cd "$ROOT" && "$PY" datagen/build_corpus.py --key-file "$KEY_FILE" CHECK "$cat" "$k" >>"$LOG" 2>&1); then
      echo "[BUILD-FAILED] $cat replicate=$k did not pass semantic CHECK; NOT finalizing" >&2
      exit 1
    fi
    echo "[done] $cat replicate=$k"
  done
done
echo "[BUILD-COMPLETE] finalizing manifest"
cd "$ROOT" && "$PY" -u datagen/build_corpus.py --key-file "$KEY_FILE" FINALIZE
