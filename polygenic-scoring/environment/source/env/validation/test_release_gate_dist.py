from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import validation.release_gate as release_gate


def test_publisher_rejects_dist_that_does_not_match_compiled_source(
    tmp_path, monkeypatch,
):
    environment = tmp_path / "environment"
    compiler = environment / "node_modules" / ".bin" / "tsc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("compiler marker")
    (environment / "tsconfig.json").write_text("{}")
    shipped = environment / "dist"
    generated = tmp_path / "generated"
    shipped.mkdir()
    generated.mkdir()
    outputs = ("aggregation.js", "index.js", "aggregation.d.ts", "index.d.ts")
    for name in outputs:
        (shipped / name).write_bytes(f"generated {name}".encode())
        (generated / name).write_bytes(f"generated {name}".encode())

    monkeypatch.setattr(release_gate, "ROOT", tmp_path)
    monkeypatch.setattr(
        release_gate.tempfile, "TemporaryDirectory",
        lambda **_kwargs: nullcontext(str(generated)),
    )
    monkeypatch.setattr(
        release_gate.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    release_gate._assert_dist_matches_source()

    (shipped / "index.js").write_text("stale hand-edited runtime")
    with pytest.raises(release_gate.ReleaseGateError, match="index.js is stale"):
        release_gate._assert_dist_matches_source()

