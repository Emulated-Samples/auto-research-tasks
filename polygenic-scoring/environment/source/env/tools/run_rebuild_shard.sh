#!/usr/bin/env bash
# Build one authenticated, source-identical shard into a fresh index-specific root.
set -Eeuo pipefail

MODE=${1:?mode must be develop or corpus}
SHARD_INDEX=${2:?missing shard index}
SHARD_COUNT=${3:?missing shard count}
RUN_ID=${4:?missing rebuild run id}
SOURCE_SHA=${5:?missing exact source commit SHA}

SOURCE_ROOT=/hyperfocal/build
BUILDER_PYTHON=/hyperfocal/venv/bin/python
KEY_FILE=/hyperfocal/corpus.key
SHARD_BASE=/hyperfocal/rebuild-shards

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
  echo "invalid rebuild run id: $RUN_ID" >&2
  exit 2
fi
SHARD_ROOT="$SHARD_BASE/$RUN_ID/${MODE}-${SHARD_INDEX}-of-${SHARD_COUNT}"

# This is deliberately before mkdir, unlink, cleanup, toolchain installation, or
# any build.  Missing/mismatched source, key, toolchain, coordinates, or an
# already-used shard root fails without mutating persistent state.
"$BUILDER_PYTHON" -B "$SOURCE_ROOT/tools/rebuild_shards.py" preflight \
  --mode "$MODE" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --source-root "$SOURCE_ROOT" \
  --source-sha "$SOURCE_SHA" \
  --key-file "$KEY_FILE" \
  --builder-python "$BUILDER_PYTHON" \
  --shard-root "$SHARD_ROOT"

cd "$SOURCE_ROOT"
mkdir -p "$SHARD_ROOT/output" "$SHARD_ROOT/logs" "$SHARD_ROOT/work"
"$BUILDER_PYTHON" -B "$SOURCE_ROOT/tools/rebuild_shards.py" tasks \
  --mode "$MODE" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --source-root "$SOURCE_ROOT" \
  --key-file "$KEY_FILE" >"$SHARD_ROOT/tasks"

if [[ "$MODE" == develop ]]; then
  develop_one() {
    local category=$1 replicate=$2 dataset_id=$3 attempt result cell_work
    result="$SHARD_ROOT/output/${dataset_id}.json"
    cell_work="$SHARD_ROOT/work/${category}_${replicate}"
    mkdir -p "$cell_work"
    for attempt in 1 2; do
      find "$cell_work" -mindepth 1 -delete
      if [[ -e "$result" || -L "$result" ]]; then unlink "$result"; fi
      if timeout --signal=TERM --kill-after=30s 7200s \
          "$BUILDER_PYTHON" "$SOURCE_ROOT/validation/model_zoo.py" \
          --key-file "$KEY_FILE" develop-one \
          "$category" "$replicate" "$cell_work" --out "$result" \
          >"$SHARD_ROOT/logs/${category}_${replicate}.${attempt}.log" 2>&1; then
        [[ -s "$result" ]] && return 0
      fi
    done
    return 1
  }
  export -f develop_one
  export SOURCE_ROOT BUILDER_PYTHON KEY_FILE SHARD_ROOT
  # Each public-reference subprocess owns four numeric threads.  Two cells fill
  # an 8-vCPU r5.2xlarge without oversubscribing the scientific computation.
  export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  export VECLIB_MAXIMUM_THREADS=2
  xargs -r -P 2 -n 3 bash -c 'develop_one "$@"' _ <"$SHARD_ROOT/tasks"
else
  corpus_one() {
    local category=$1 replicate=$2 dataset_id=$3 attempt dataset_root
    dataset_root="$SHARD_ROOT/output/$category/$dataset_id"
    mkdir -p "$SHARD_ROOT/output/$category"
    for attempt in 1 2; do
      if [[ -d "$dataset_root" ]]; then find "$dataset_root" -depth -delete; fi
      if timeout --signal=TERM --kill-after=30s 3700s \
          "$BUILDER_PYTHON" -B "$SOURCE_ROOT/tools/rebuild_shards.py" corpus-cell \
          --output-root "$SHARD_ROOT/output" --key-file "$KEY_FILE" \
          --action ONE --category "$category" --replicate "$replicate" \
          >"$SHARD_ROOT/logs/${category}_${replicate}.${attempt}.log" 2>&1
      then
        if "$BUILDER_PYTHON" -B "$SOURCE_ROOT/tools/rebuild_shards.py" corpus-cell \
            --output-root "$SHARD_ROOT/output" --key-file "$KEY_FILE" \
            --action CHECK --category "$category" --replicate "$replicate" \
            >>"$SHARD_ROOT/logs/${category}_${replicate}.${attempt}.log" 2>&1
        then
          return 0
        fi
      fi
    done
    return 1
  }
  export -f corpus_one
  export SOURCE_ROOT BUILDER_PYTHON KEY_FILE SHARD_ROOT
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  xargs -r -P 1 -n 3 bash -c 'corpus_one "$@"' _ <"$SHARD_ROOT/tasks"
fi

"$BUILDER_PYTHON" -B "$SOURCE_ROOT/tools/rebuild_shards.py" seal \
  --mode "$MODE" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --source-root "$SOURCE_ROOT" \
  --source-sha "$SOURCE_SHA" \
  --key-file "$KEY_FILE" \
  --builder-python "$BUILDER_PYTHON" \
  --shard-root "$SHARD_ROOT"
printf '%s\n' "${MODE^^}-SHARD-COMPLETE $SHARD_INDEX/$SHARD_COUNT $SHARD_ROOT"
