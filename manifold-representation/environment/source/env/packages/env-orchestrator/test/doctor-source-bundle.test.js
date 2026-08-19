import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { doctor } from "../dist/packager/checks/doctor.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CLI_BIN = path.join(HERE, "..", "bin", "env-orchestrator.js");

function g(cwd, ...args) {
  return execFileSync("git", args, { cwd, encoding: "utf-8" }).trim();
}

/**
 * A minimal packageable env repo: hyperfocal.yaml + environment/problems.yaml,
 * no environment/package.json (so the hook build is skipped), one commit on
 * main. Exactly the "minimal/fixture repo" shape hookBuild.ts documents.
 */
async function makeMiniEnvRepo(name) {
  const root = await mkdtemp(path.join(os.tmpdir(), "doctor-source-bundle-"));
  const repo = path.join(root, name);
  fs.mkdirSync(path.join(repo, "environment"), { recursive: true });
  fs.writeFileSync(
    path.join(repo, "hyperfocal.yaml"),
    [
      'version: "1.0"',
      "environment:",
      `  name: ${name}`,
      '  description: "doctor --no-source-bundle test fixture"',
      "paths:",
      "  root: /hyperfocal/env",
      "  environmentDist: environment/dist",
      "  workspace: workspace",
      "agent:",
      "  awsAccess: false",
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

/** The one emitted task dir under <repo>/harbor-doctor/tasks. */
function emittedTaskDir(repo) {
  const tasksDir = path.join(repo, "harbor-doctor", "tasks");
  const names = fs
    .readdirSync(tasksDir)
    .filter((n) => fs.existsSync(path.join(tasksDir, n, "task.toml")));
  assert.equal(names.length, 1, `expected exactly one task dir, got: ${names}`);
  return path.join(tasksDir, names[0]);
}

test("doctor default still emits and gates the source bundle", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-doctor-default");
  try {
    await doctor({ envRepo: repo, trials: 1, build: false });
    const taskDir = emittedTaskDir(repo);
    assert.ok(
      fs.existsSync(path.join(taskDir, "environment", "source", "repo.bundle")),
      "default doctor run must keep emitting environment/source/repo.bundle"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor without the opt-out fails an over-100MB source file; --no-source-bundle skips the source stage and passes", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-doctor-heavy");
  // 101 MB working-tree file: the staged source copy trips the same 100 MB
  // per-file ship gate that heavy repo.bundle histories trip (decision 7.12).
  fs.writeFileSync(path.join(repo, "blob.bin"), Buffer.alloc(101 * 1024 * 1024));
  g(repo, "add", "-A");
  g(repo, "commit", "-qm", "heavy blob");
  try {
    await assert.rejects(
      () => doctor({ envRepo: repo, trials: 1, build: false }),
      /failing check/,
      "without the opt-out the 100 MB gate must still fail doctor"
    );

    await doctor({ envRepo: repo, trials: 1, build: false, sourceBundle: false });
    const taskDir = emittedTaskDir(repo);
    assert.ok(
      !fs.existsSync(path.join(taskDir, "environment", "source")),
      "--no-source-bundle must emit no environment/source"
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("CLI: harbor doctor --no-source-bundle reaches the packager (exit 0 on a gate-tripping repo)", async () => {
  const { root, repo } = await makeMiniEnvRepo("mini-doctor-cli");
  fs.writeFileSync(path.join(repo, "blob.bin"), Buffer.alloc(101 * 1024 * 1024));
  g(repo, "add", "-A");
  g(repo, "commit", "-qm", "heavy blob");
  try {
    const out = execFileSync(
      process.execPath,
      [
        CLI_BIN,
        "harbor",
        "doctor",
        "--env-repo",
        repo,
        "--no-build",
        "--no-source-bundle",
      ],
      { encoding: "utf-8" }
    );
    assert.match(out, /--no-source-bundle/, "report should name the opt-out");
    assert.match(out, /release-ready/, "doctor should end green");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
