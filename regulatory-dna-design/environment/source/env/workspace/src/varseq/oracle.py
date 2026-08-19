"""Query-only access to the trained activity model used for sequence design.

`load_oracle()` returns a callable `oracle(seqs) -> np.ndarray` of shape `(len(seqs), n_tracks)`,
the interface your `design()` receives. The model itself is served by a local scoring service, so
every sequence you score here is metered exactly the way the graded oracle is metered — `.used`,
`.budget` and `.remaining` on the returned object report where you stand.
"""
import json
import os
import socket
import struct

import numpy as np

_SEED_PATH = os.path.join(os.path.dirname(__file__), "_assets", "seed_sequences.npy")
_SOCKET_PATH = os.environ.get("SEQDESIGN_SOCKET", "/run/seqdesign/oracle.sock")
TRACK_NAMES = ("Dev", "Hk")
INPUT_LENGTH = 249


def load_seed_sequences():
    """A set of real regulatory sequences (natural enhancers), available as design material."""
    return np.load(_SEED_PATH).tolist()


class OracleUnavailable(RuntimeError):
    """The local scoring service could not be reached."""


def _request(payload):
    body = json.dumps(payload).encode()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(_SOCKET_PATH)
            sock.sendall(struct.pack(">I", len(body)) + body)
            header = _recv_exact(sock, 4)
            (length,) = struct.unpack(">I", header)
            resp = json.loads(_recv_exact(sock, length).decode())
    except (OSError, struct.error) as exc:
        raise OracleUnavailable(
            f"could not reach the scoring service at {_SOCKET_PATH}: {exc}"
        ) from exc
    if not resp.get("ok"):
        raise RuntimeError(f"scoring service error: {resp.get('error')}")
    return resp


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise OSError("scoring service closed the connection early")
        buf.extend(chunk)
    return bytes(buf)


class Oracle:
    """Callable wrapper: `oracle(seqs) -> np.ndarray (len(seqs), n_tracks)`.

    Attributes mirror the oracle passed to `design()` at evaluation time: `input_length`,
    `n_tracks`, `tracks`. `used` / `budget` / `remaining` report this session's metered total.
    """

    def __init__(self, info):
        self.input_length = int(info.get("input_length", INPUT_LENGTH))
        self.n_tracks = int(info.get("n_tracks", len(TRACK_NAMES)))
        self.tracks = tuple(info.get("tracks", TRACK_NAMES))
        self.budget = int(info.get("budget", 0))
        self.used = int(info.get("used", 0))

    @property
    def remaining(self):
        return self.budget - self.used

    def refresh(self):
        """Re-read the meter without scoring anything."""
        self.used = int(_request({"op": "info"})["used"])
        return self.used

    def __call__(self, seqs, batch_size=256):
        if isinstance(seqs, str):
            raise TypeError("oracle expects a list of sequences, not a single string")
        seqs = list(seqs)
        out = []
        for i in range(0, len(seqs), batch_size):
            resp = _request({"op": "score", "seqs": seqs[i:i + batch_size]})
            self.used = int(resp["used"])
            out.append(np.asarray(resp["scores"], dtype=np.float32))
        if not out:
            return np.zeros((0, self.n_tracks), np.float32)
        return np.concatenate(out, axis=0)


def load_oracle(weights_path=None):
    """Connect to the local scoring service and return the metered oracle callable."""
    if weights_path is not None:
        raise ValueError(
            "the activity model is served by the local scoring service; there is no local weight "
            "file to point at. Call load_oracle() with no arguments."
        )
    return Oracle(_request({"op": "info"}))
