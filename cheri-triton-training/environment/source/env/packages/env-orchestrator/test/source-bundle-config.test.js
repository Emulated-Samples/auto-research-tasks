import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { doctor } from "../dist/packager/checks/doctor.js";
import { packageRelease } from "../dist/packager/package.js";
import { parsePackagingConfig } from "../dist/config/yaml-config.js";
import { renderEnvReadme } from "../dist/packager/emit/readme.js";

function g(cwd, ...args) {
  return execFileSync("git", args, { cwd, encoding: "utf-8" }).trim();
}

/**
 * Minimal packageable env repo (same shape as doctor-source-bundle.test.js):
 * hyperfocal.yaml + environment/problems.yaml, no environment/package.json,
 * one commit on main. `packagingLines` lets a test add a packaging: block.
 */
async function makeMiniEnvRepo(name, packagingLines = []) {
  const root = await mkdtemp(path.join(os.tmpdir(), "source-bundle-config-"));
  const repo = path.join(root, name);
  fs.mkdirSync(path.join(repo, "environment"), { recursive: true });
  fs.writeFileSync(
    path.join(repo, "hyperfocal.yaml"),
    [
      'version: "1.0"',
      "environment:",
      `  name: ${name}`,
      '  description: "packaging.sourceBundle test fixture"',
      "paths:",
      "  root: /hyperfocal/env",
      "  environmentDist: environment/dist",
      "  workspace: workspace",
      "agent:",
      "  awsAccess: false",
      ...packagingLines,
      "",
    ].join("\n")
  );
  fs.writeFileSync(
    path.join(repo, "environment", "problems.yaml"),
    ["- id: default", "  default: true", "  prompt: |", "    Test problem.", ""].join("\n")
  );
  g(repo, "init", "-q", "-b", "main");
  g(repo, "config", "user.email", "t@t");
  g(repo, "config", "user.name", "t");
  g(repo, "add", "-A");
  g(repo, "commit", "-qm", "fixture");
  return { root, repo };
}

/** The one emitted task dir under <outDir>/tasks. */
function emittedTaskDir(outDir) {
  const tasksDir = path.join(outDir, "tasks");
  const names = fs
    .readdirSync(tasksDir)
    .filter((n) => fs.existsSync(path.join(tasksDir, n, "task.toml")));
  assert.equal(names.length, 1, `expected exactly one task dir, got: ${names}`);
  return path.join(tasksDir, names[0]);
}

test("parsePackagingConfig accepts sourceBundle booleans and rejects non-booleans", () => {
  assert.deepEqual(parsePackagingConfig({ sourceBundle: false }), {
    sourceBundle: false,
  });
  assert.deepEqual(parsePackagingConfig({ sourceBundle: true }), {
    sourceBundle: true,
  });
  // Absent => not in the parsed config at all (default handled downstream).
  assert.deepEqual(parsePackagingConfig({}), {});
  assert.throws(
    () => parsePackagingConfig({ sourceBundle: "false" }),
    /packaging\.sourceBundle must be a boolean/
  );
});

test("packaging.sourceBundle: false — a FLAGLESS package (the builder path) skips source emission, the 100MB gate, and swaps the README's from-source section for the honest note", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-yaml-optout", [
    "packaging:",
    "  sourceBundle: false",
  ]);
  // 101 MB working-tree file: would trip the 100 MB per-file ship gate if
  // source were emitted — exactly the heavy-history failure the builders hit
  // (they pass NO flags; only the yaml key can save them).
  fs.writeFileSync(path.join(repo, "blob.bin"), Buffer.alloc(101 * 1024 * 1024));
  g(repo, "add", "-A");
  g(repo, "commit", "-qm", "heavy blob");
  const outDir = path.join(root, "out");
  try {
    // NO sourceBundle option: this is what a builder invocation looks like.
    await packageRelease({
      envRepo: repo,
      outDir,
      build: false,
      packagesOverlay: false,
    });
    const taskDir = emittedTaskDir(outDir);
    assert.ok(
      !fs.existsSync(path.join(taskDir, "environment", "source")),
      "yaml opt-out must emit no environment/source"
    );
    assert.ok(
      !fs.existsSync(path.join(taskDir, "environment", "Dockerfile")),
      "yaml opt-out must emit no standalone environment/Dockerfile"
    );
    const readme = fs.readFileSync(path.join(outDir, "README.md"), "utf-8");
    assert.match(
      readme,
      /Source omitted: this repository's git history exceeds public-hosting file limits/,
      "README must carry the honest pin-only note"
    );
    assert.ok(
      !readme.includes("sed -i"),
      "README must NOT carry the delete-the-pin from-source instructions"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("default (no yaml key, no flag) still emits source + the from-source README section", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-yaml-default");
  const outDir = path.join(root, "out");
  try {
    await packageRelease({
      envRepo: repo,
      outDir,
      build: false,
      packagesOverlay: false,
    });
    const taskDir = emittedTaskDir(outDir);
    assert.ok(
      fs.existsSync(path.join(taskDir, "environment", "source", "repo.bundle")),
      "default must keep emitting environment/source/repo.bundle"
    );
    assert.ok(
      fs.existsSync(path.join(taskDir, "environment", "Dockerfile")),
      "default must keep emitting the standalone environment/Dockerfile"
    );
    const readme = fs.readFileSync(path.join(outDir, "README.md"), "utf-8");
    assert.match(
      readme,
      /build it from source \(environment\/source\/\)/,
      "README must keep the from-source instructions by default"
    );
    assert.ok(
      readme.includes("sed -i '/^docker_image/d'"),
      "README must keep the delete-the-pin recipe by default"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("precedence: explicit CLI-style sourceBundle: false wins over yaml sourceBundle: true", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-precedence", [
    "packaging:",
    "  sourceBundle: true",
  ]);
  const outDir = path.join(root, "out");
  try {
    await packageRelease({
      envRepo: repo,
      outDir,
      build: false,
      packagesOverlay: false,
      sourceBundle: false, // what the CLI passes for --no-source-bundle
    });
    const taskDir = emittedTaskDir(outDir);
    assert.ok(
      !fs.existsSync(path.join(taskDir, "environment", "source")),
      "explicit --no-source-bundle must win over yaml sourceBundle: true"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor honors packaging.sourceBundle: false with NO flag (env doctors like it builds)", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-doctor-yaml", [
    "packaging:",
    "  sourceBundle: false",
  ]);
  fs.writeFileSync(path.join(repo, "blob.bin"), Buffer.alloc(101 * 1024 * 1024));
  g(repo, "add", "-A");
  g(repo, "commit", "-qm", "heavy blob");
  try {
    // Would throw on the 100 MB gate if the yaml key were ignored.
    await doctor({ envRepo: repo, trials: 1, build: false });
    const taskDir = emittedTaskDir(path.join(repo, "harbor-doctor"));
    assert.ok(
      !fs.existsSync(path.join(taskDir, "environment", "source")),
      "doctor with yaml opt-out must emit no environment/source"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("renderEnvReadme: sourceBundle false swaps the from-source section and the not-yet-built hint", () => {
  const params = {
    envName: "mini",
    buildTimeoutSec: 2700,
    tasks: [
      {
        name: "mini__main__default",
        gpus: 0,
        hasSolution: false,
        status: "image not yet built",
        checkResults: "—",
      },
    ],
  };
  const withSource = renderEnvReadme({ ...params, sourceBundle: true });
  assert.match(withSource, /build it from source/);
  assert.match(withSource, /build from source, or check back/);

  const omitted = renderEnvReadme({ ...params, sourceBundle: false });
  assert.match(
    omitted,
    /Source omitted: this repository's git history exceeds public-hosting file limits/
  );
  assert.ok(!omitted.includes("sed -i"));
  assert.match(omitted, /yet — check back\.\)/);

  // Param absent => legacy callers (control-plane publish re-render) keep
  // the from-source section.
  const legacy = renderEnvReadme(params);
  assert.match(legacy, /build it from source/);
});
