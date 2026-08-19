#!/bin/sh
# Container ENTRYPOINT: bring up the root-owned scoring daemon, then hand off to whatever command
# the runtime asked for.
#
# Single container only (harbor-format.md: compose / multi-service is unsupported), so the daemon
# runs as a background process here rather than as its own service. The image sets no CMD-level
# supervisor and harbor's compose template overrides `command:` (sh -c "sleep infinity") but never
# `entrypoint:`, so this is the one hook that survives both the harbor flow and a plain `docker run`.
#
# Never fatal: if the daemon cannot start, the container still comes up (the verifier reports the
# missing meter) instead of failing the whole trial as infra.
set -u

SEQDESIGN_SOCKET="${SEQDESIGN_SOCKET:-/run/seqdesign/oracle.sock}"
SEQDESIGN_STATE_DIR="${SEQDESIGN_STATE_DIR:-/opt/seqdesign}"
SEQDESIGN_PYTHON="${SEQDESIGN_PYTHON:-/hyperfocal/env/environment/.grade-venv/bin/python}"
LOG="${SEQDESIGN_STATE_DIR}/daemon.log"

start_daemon() {
    [ -S "$SEQDESIGN_SOCKET" ] && return 0
    [ -x "$SEQDESIGN_PYTHON" ] || { echo "[entrypoint] no interpreter at $SEQDESIGN_PYTHON" >&2; return 1; }
    mkdir -p "$(dirname "$SEQDESIGN_SOCKET")" 2>/dev/null || true
    "$SEQDESIGN_PYTHON" "${SEQDESIGN_STATE_DIR}/oracled.py" >>"$LOG" 2>&1 &
    i=0
    while [ ! -S "$SEQDESIGN_SOCKET" ] && [ "$i" -lt 120 ]; do
        sleep 0.5
        i=$((i + 1))
    done
    if [ -S "$SEQDESIGN_SOCKET" ]; then
        echo "[entrypoint] scoring daemon up on $SEQDESIGN_SOCKET"
    else
        echo "[entrypoint] scoring daemon failed to bind $SEQDESIGN_SOCKET; see $LOG" >&2
        tail -n 20 "$LOG" >&2 2>/dev/null || true
    fi
}

start_daemon || true

[ "$#" -eq 0 ] && exec sh -c "sleep infinity"
exec "$@"
