import assert from "node:assert/strict";
import test from "node:test";

import { parsePackagingConfig } from "../dist/config/yaml-config.js";
import { legacyBuildInvocation } from "../dist/packager/build/bake.js";
import { renderDockerfile } from "../dist/packager/render/dockerfile.js";

// ---------------------------------------------------------------------------
// packaging.bakeNeedsGpu — strict yaml parsing
// ---------------------------------------------------------------------------

test("parsePackagingConfig accepts bakeNeedsGpu: true", () => {
  const parsed = parsePackagingConfig({ bakeNeedsGpu: true });
  assert.equal(parsed.bakeNeedsGpu, true);
});

test("parsePackagingConfig accepts bakeNeedsGpu: false and keeps it explicit", () => {
  const parsed = parsePackagingConfig({ bakeNeedsGpu: false });
  assert.equal(parsed.bakeNeedsGpu, false);
});

test("parsePackagingConfig omits bakeNeedsGpu when absent (absent means off)", () => {
  const parsed = parsePackagingConfig({ verifierTimeoutSec: 60 });
  assert.equal("bakeNeedsGpu" in parsed, false);
});

test("parsePackagingConfig rejects non-boolean bakeNeedsGpu", () => {
  assert.throws(
    () => parsePackagingConfig({ bakeNeedsGpu: "yes" }),
    /packaging\.bakeNeedsGpu must be a boolean/
  );
});

test("unknown-packaging-key error names bakeNeedsGpu as a known key", () => {
  // The error text doubles as the author-facing list of valid keys; a key
  // the parser accepts but the message omits would read as unsupported.
  assert.throws(
    () => parsePackagingConfig({ bakeNeedsGpuu: true }),
    /expected .*bakeNeedsGpu/
  );
});

// ---------------------------------------------------------------------------
// Legacy-builder invocation shape (S3-F2: DOCKER_BUILDKIT=0 is load-bearing)
// ---------------------------------------------------------------------------

test("legacyBuildInvocation pins DOCKER_BUILDKIT=0 and the classic build argv", () => {
  const inv = legacyBuildInvocation(
    { imageTag: "reg/env:p1-abc-def", dockerfilePath: "/ctx/Dockerfile.p1" },
    "/ctx"
  );
  assert.equal(inv.command, "docker");
  assert.deepEqual(inv.args, [
    "build",
    "-t",
    "reg/env:p1-abc-def",
    "-f",
    "/ctx/Dockerfile.p1",
    "/ctx",
  ]);
  // The whole point of the path: the classic builder honors the daemon's
  // nvidia default-runtime for RUN steps; BuildKit does not (S3-F2).
  assert.equal(inv.env.DOCKER_BUILDKIT, "0");
  // No buildx/bake flags may leak in — the legacy builder rejects them.
  assert.ok(!inv.args.includes("--provenance=false"));
});

// ---------------------------------------------------------------------------
// Rendered Dockerfile stays legacy-builder compatible
// ---------------------------------------------------------------------------

/**
 * The legacy builder understands the classic instruction set only. Guard
 * the rendered output against BuildKit-only syntax creeping in: a `#
 * syntax=` directive, RUN --mount/--network/--security flags, or heredoc
 * RUN bodies would all break every bakeNeedsGpu env at once.
 */
function assertLegacyCompatible(dockerfile) {
  assert.ok(!/^#\s*syntax=/im.test(dockerfile), "no # syntax= directive");
  assert.ok(
    !/^RUN\s+--(mount|network|security|device)/m.test(dockerfile),
    "no BuildKit-only RUN flags"
  );
  assert.ok(!/<<-?\s*['"]?EOF/m.test(dockerfile), "no heredoc bodies");
  const instructions = dockerfile
    .split("\n")
    .filter((l) => /^[A-Z]+ /.test(l))
    .map((l) => l.split(" ")[0]);
  for (const inst of instructions) {
    assert.ok(
      ["FROM", "ENV", "COPY", "WORKDIR", "RUN", "CMD"].includes(inst),
      `legacy-safe instruction set only (got ${inst})`
    );
  }
}

test("rendered Dockerfile with the CUDA dockerfileLines-FROM override is legacy-builder compatible", () => {
  // Mirrors the flashinfer fixture: base-image override via the
  // dockerfileLines escape hatch (S3-F4 idiom) on top of the fleet base.
  const dockerfile = renderDockerfile({
    problemId: "restore-sampling",
    baseImage: "123.dkr.ecr.us-east-1.amazonaws.com/hyperfocal-task-base:latest",
    dockerfileLines: [
      "FROM nvidia/cuda:12.8.1-devel-ubuntu22.04",
      "RUN apt-get update && apt-get install -y nodejs",
    ],
    workspacePath: "workspace",
  });
  assertLegacyCompatible(dockerfile);
  // The override FROM must come after the fleet base stage (orphaning it)
  // and before the repo COPY, so the provision RUN lands on the CUDA stage.
  const fromIdx = dockerfile.indexOf("FROM nvidia/cuda");
  const baseIdx = dockerfile.indexOf("FROM 123.dkr.ecr");
  const copyIdx = dockerfile.indexOf("COPY env/");
  assert.ok(baseIdx >= 0 && fromIdx > baseIdx && copyIdx > fromIdx);
});

test("rendered fallback-base Dockerfile is legacy-builder compatible", () => {
  const dockerfile = renderDockerfile({
    problemId: "p1",
    workspacePath: "workspace",
  });
  assertLegacyCompatible(dockerfile);
});
