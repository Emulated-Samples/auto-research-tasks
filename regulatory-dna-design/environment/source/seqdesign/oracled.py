"""Root-owned scoring daemon for the seq-design task (never readable by the agent uid).

Why this exists
---------------
The task's whole difficulty is *sample efficiency*: design high-activity sequences while scoring at
most `budget` candidates through the activity model. Before this daemon the model weights sat inside
the agent's own workspace (`varseq/_assets/deepstarr_oracle.pth`), so the meter was decorative — an
agent could load the weights directly and run an unbounded offline search, then return constants.

Now the weights live at /opt/seqdesign/oracle.pth inside a 0700 root-owned directory and are only
reachable through this daemon, which listens on a unix socket and counts every sequence it scores
into a root-owned counter file. The counter is what the verifier reads.

Contract
--------
* SOFT meter. The daemon never refuses a request, whatever the count reaches: refusing mid-session
  would turn a budget overrun into an opaque crash. It only records the truth; the verifier judges.
* The counter is persisted after every scored request, so it survives a daemon restart (the
  container entrypoint re-launches the daemon if the socket is gone).
* Weights are never returned over the wire — only per-track activity predictions.

Wire protocol (framed): 4-byte big-endian payload length, then a JSON object.
  request  {"op": "score", "seqs": [...]}   -> {"ok": true, "used": N, "scores": [[f, f], ...]}
           {"op": "info"}                   -> {"ok": true, "used": N, "budget": B,
                                                "input_length": 249, "n_tracks": 2,
                                                "tracks": ["Dev", "Hk"]}
  error                                     -> {"ok": false, "error": "..."}
"""
import json
import os
import socketserver
import struct
import sys
import threading

import numpy as np
import torch
import torch.nn as nn

STATE_DIR = os.environ.get("SEQDESIGN_STATE_DIR", "/opt/seqdesign")
WEIGHTS_PATH = os.path.join(STATE_DIR, "oracle.pth")
COUNTER_PATH = os.path.join(STATE_DIR, "counter")
SOCKET_PATH = os.environ.get("SEQDESIGN_SOCKET", "/run/seqdesign/oracle.sock")

# The session allowance reported to clients through `info`. The daemon does NOT enforce it; the
# verifier does. Kept in the environment so the image build is the single place it is set.
SESSION_BUDGET = int(os.environ.get("SEQDESIGN_SESSION_BUDGET", "200000"))

TRACK_NAMES = ("Dev", "Hk")
INPUT_LENGTH = 249
_B2I = {b: i for i, b in enumerate("ACGT")}
MAX_FRAME = 64 * 1024 * 1024


class _DeepSTARR(nn.Module):
    """DeepSTARR (de Almeida et al. 2022): Basset-style 4-conv + 2-FC net, 249bp -> (Dev, Hk)."""

    def __init__(self, length=INPUT_LENGTH):
        super().__init__()

        def blk(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]

        self.body = nn.Sequential(*blk(4, 256, 7), *blk(256, 60, 3), *blk(60, 60, 5), *blk(60, 120, 3))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(120 * (length // 16), 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.body(x))


def _onehot(seqs):
    x = np.zeros((len(seqs), 4, INPUT_LENGTH), np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:INPUT_LENGTH]):
            if c in _B2I:
                x[i, _B2I[c], j] = 1.0
    return torch.from_numpy(x)


class Meter:
    """The root-owned query counter. Persisted after every increment; read by the verifier."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self.used = self._read()
        self._write()

    def _read(self):
        try:
            with open(self._path) as f:
                return int((f.read() or "0").strip() or 0)
        except (OSError, ValueError):
            return 0

    def _write(self):
        tmp = f"{self._path}.tmp"
        with open(tmp, "w") as f:
            f.write(f"{self.used}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)
        os.chmod(self._path, 0o600)

    def add(self, n):
        with self._lock:
            self.used += int(n)
            self._write()
            return self.used


class Scorer:
    def __init__(self, weights_path, meter):
        torch.set_num_threads(1)
        model = _DeepSTARR()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        self._model = model
        self._lock = threading.Lock()
        self.meter = meter
        self.n_tracks = model.head[-1].out_features

    @torch.no_grad()
    def score(self, seqs, bs=512):
        out = []
        with self._lock:
            for i in range(0, len(seqs), bs):
                out.append(self._model(_onehot(list(seqs[i:i + bs]))).numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_tracks), np.float32)


SCORER = None


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send(conn, obj):
    payload = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _handle(req):
    op = req.get("op")
    if op == "info":
        return {"ok": True, "used": SCORER.meter.used, "budget": SESSION_BUDGET,
                "input_length": INPUT_LENGTH, "n_tracks": SCORER.n_tracks,
                "tracks": list(TRACK_NAMES)}
    if op != "score":
        return {"ok": False, "error": f"unknown op {op!r}"}
    seqs = req.get("seqs")
    if not isinstance(seqs, list) or any(not isinstance(s, str) for s in seqs):
        return {"ok": False, "error": "seqs must be a list of strings"}
    # Count first, then score: a request that dies mid-inference still spent the budget.
    used = SCORER.meter.add(len(seqs))
    scores = SCORER.score(seqs)
    return {"ok": True, "used": used, "scores": [[float(v) for v in row] for row in scores]}


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        conn = self.request
        while True:
            header = _recv_exact(conn, 4)
            if header is None:
                return
            (length,) = struct.unpack(">I", header)
            if length > MAX_FRAME:
                _send(conn, {"ok": False, "error": "request too large"})
                return
            body = _recv_exact(conn, length)
            if body is None:
                return
            try:
                resp = _handle(json.loads(body.decode()))
            except Exception as exc:  # never take the daemon down over one bad client
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                _send(conn, resp)
            except OSError:
                return


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 64


def main():
    global SCORER
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    os.chmod(os.path.dirname(SOCKET_PATH), 0o755)
    SCORER = Scorer(WEIGHTS_PATH, Meter(COUNTER_PATH))
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = Server(SOCKET_PATH, Handler)
    # The socket itself is the only agent-reachable surface: world-writable so the agent uid can
    # connect, while the weights, the daemon source and the counter stay behind a 0700 root dir.
    os.chmod(SOCKET_PATH, 0o666)
    print(f"[seqdesign-oracled] listening on {SOCKET_PATH}; used={SCORER.meter.used}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[seqdesign-oracled] fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
