import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  assertSourceFileSizes,
  emitTaskSources,
} from "../dist/packager/emit/taskSource.js";
import { renderEnvironmentDockerfile } from "../dist/packager/render/environmentDockerfile.js";
import { renderDockerfile } from "../dist/packager/render/dockerfile.js";

function sh(cwd, cmd, args) {
  return execFileSync(cmd, args, { cwd, encoding: "utf-8" }).trim();
}
function g(cwd, ...args) {
  return sh(cwd, "git", args);
}

/**
 * Build a miniature staged context: contextDir/env = a git clone with one
 * initialized submodule (the shape stage.ts produces), plus a rendered
 * per-problem Dockerfile at the context root.
 */
async function makeStagedContext() {
  const root = await mkdtemp(path.join(os.tmpdir(), "source-bundle-test-"));

  // The submodule's source repo.
  const subSrc = path.join(root, "sub-src");
  fs.mkdirSync(subSrc);
  g(subSrc, "init", "-q", "-b", "main");
  g(subSrc, "config", "user.email", "t@t");
  g(subSrc, "config", "user.name", "t");
  fs.writeFileSync(path.join(subSrc, "lib.txt"), "lib v1\n");
  g(subSrc, "add", "-A");
  g(subSrc, "commit", "-qm", "sub v1");

  // The env repo: one extra branch (problem branch) + the submodule.
  const envSrc = path.join(root, "env-src");
  fs.mkdirSync(envSrc);
  g(envSrc, "init", "-q", "-b", "main");
  g(envSrc, "config", "user.email", "t@t");
  g(envSrc, "config", "user.name", "t");
  fs.writeFileSync(path.join(envSrc, "app.txt"), "gold\n");
  g(envSrc, "add", "-A");
  g(envSrc, "commit", "-qm", "gold");
  g(envSrc, "checkout", "-qb", "problem/one");
  fs.writeFileSync(path.join(envSrc, "app.txt"), "broken\n");
  g(envSrc, "commit", "-aqm", "perturb");
  g(envSrc, "checkout", "-q", "main");
  g(
    envSrc,
    "-c",
    "protocol.file.allow=always",
    "submodule",
    "add",
    "-q",
    subSrc,
    "packages/lib"
  );
  g(envSrc, "commit", "-qm", "add submodule");

  // "Staged" clone (what stageBuildContext produces): clone + submodule init.
  const contextDir = path.join(root, "build-context");
  fs.mkdirSync(contextDir);
  const cloneDir = path.join(contextDir, "env");
  execFileSync("git", ["clone", "-q", envSrc, cloneDir]);
  g(
    cloneDir,
    "-c",
    "protocol.file.allow=always",
    "submodule",
    "update",
    "--init",
    "--quiet"
  );
  // Staging materializes source branches as local heads.
  execFileSync("git", [
    "-C",
    cloneDir,
    "fetch",
    "-q",
    envSrc,
    "+refs/heads/problem/one:refs/heads/problem/one",
  ]);
  fs.writeFileSync(path.join(contextDir, "Dockerfile.p1"), "FROM scratch\n");

  return { root, contextDir, cloneDir };
}

test("emitTaskSources emits per-task source, strips git state, bundles refs, reports the graft", async (t) => {
  const { root, contextDir, cloneDir } = await makeStagedContext();
  t.after(() => rm(root, { recursive: true, force: true }));

  const outDir = path.join(root, "out");
  fs.mkdirSync(outDir);
  // Two tasks so the per-task Dockerfile.<problem> filter is exercised.
  const taskA = path.join(outDir, "tasks", "env__main__p1");
  const taskB = path.join(outDir, "tasks", "env__main__p2");
  fs.mkdirSync(path.join(taskA, "environment"), { recursive: true });
  fs.mkdirSync(path.join(taskB, "environment"), { recursive: true });
  // The staged context carries both problems' recipes.
  fs.writeFileSync(path.join(contextDir, "Dockerfile.p2"), "FROM scratch\n");
  const graft = emitTaskSources({
    contextDir,
    cloneDir,
    outDir,
    tasks: [
      { taskDir: taskA, problemId: "p1" },
      { taskDir: taskB, problemId: "p2" },
    ],
    logger: () => {},
  });

  const sourceDir = path.join(taskA, "environment", "source");

  // Per-task source lands under <task>/environment/source (not the output root).
  assert.ok(!fs.existsSync(path.join(outDir, "source")), "no root-level source/");
  assert.ok(!fs.existsSync(path.join(outDir, ".source-stage")), "staging cleaned up");

  // Browsable tree: worktree bytes present, NO .git dirs or pointer files.
  assert.equal(
    fs.readFileSync(path.join(sourceDir, "env", "app.txt"), "utf-8"),
    "gold\n"
  );
  assert.equal(
    fs.readFileSync(path.join(sourceDir, "env", "packages/lib/lib.txt"), "utf-8"),
    "lib v1\n"
  );
  // Each task carries ONLY its own Dockerfile.<problem> audit recipe.
  assert.ok(fs.existsSync(path.join(sourceDir, "Dockerfile.p1")));
  assert.ok(!fs.existsSync(path.join(sourceDir, "Dockerfile.p2")));
  assert.ok(
    fs.existsSync(path.join(taskB, "environment", "source", "Dockerfile.p2"))
  );
  assert.ok(
    !fs.existsSync(path.join(taskB, "environment", "source", "Dockerfile.p1"))
  );
  const gitEntries = execFileSync("find", [sourceDir, "-name", ".git"], {
    encoding: "utf-8",
  }).trim();
  assert.equal(gitEntries, "", ".git dirs/files must be stripped from source/");

  // Superproject bundle carries every branch (problem branches survive).
  const heads = execFileSync(
    "git",
    ["bundle", "list-heads", path.join(sourceDir, "repo.bundle")],
    { encoding: "utf-8" }
  );
  assert.match(heads, /refs\/heads\/main/);
  assert.match(heads, /refs\/heads\/problem\/one/);
  assert.match(heads, /HEAD/);

  // One bundle per initialized submodule, path-encoded.
  assert.ok(
    fs.existsSync(
      path.join(sourceDir, "git-submodules", "packages__lib.bundle")
    )
  );

  // Graft info matches the staged clone's exact positions.
  assert.equal(graft.headBranch, "main");
  assert.equal(graft.headCommit, g(cloneDir, "rev-parse", "HEAD"));
  assert.equal(graft.submodules.length, 1);
  const sub = graft.submodules[0];
  assert.equal(sub.path, "packages/lib");
  assert.equal(sub.name, "packages/lib");
  assert.equal(
    sub.headCommit,
    g(path.join(cloneDir, "packages/lib"), "rev-parse", "HEAD")
  );
});

test("assertSourceFileSizes names files over GitHub's 100 MB limit", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "task-source-size-"));
  t.after(() => rm(dir, { recursive: true, force: true }));

  fs.writeFileSync(path.join(dir, "small.txt"), "ok\n");
  assert.doesNotThrow(() => assertSourceFileSizes(dir));

  // Sparse file over the limit — size is what GitHub judges.
  const big = path.join(dir, "huge.bin");
  fs.writeFileSync(big, "");
  fs.truncateSync(big, 101 * 1024 * 1024);
  assert.throws(
    () => assertSourceFileSizes(dir),
    /huge\.bin \(101 MB\).*--no-source-bundle/s
  );
});

test("renderEnvironmentDockerfile: public base, bundle graft, same provision, zero comments", () => {
  const graft = {
    headBranch: "main",
    headCommit: "a".repeat(40),
    originUrl: "https://github.com/org/env.git",
    submodules: [
      {
        path: "packages/lib",
        parent: "",
        name: "packages/lib",
        url: "https://github.com/org/lib.git",
        headCommit: "b".repeat(40),
        bundleFile: "packages__lib.bundle",
      },
    ],
  };
  const df = renderEnvironmentDockerfile({
    problemId: "p1",
    workspacePath: "workspace",
    graft,
  });

  // Public base only — never the private fleet base image.
  assert.match(df, /FROM amazonlinux:2023/);
  assert.ok(!df.includes("dkr.ecr"), "must not reference a private registry");

  // Zero-comment rule (step 3): no comment lines in the shipped Dockerfile.
  const commentLines = df
    .split("\n")
    .filter((l) => l.trim().startsWith("#"));
  assert.equal(commentLines.length, 0, "shipped Dockerfile must be comment-free");

  // Build context is the task's environment/ dir, so COPY paths are
  // source/-prefixed (harbor builds environment/Dockerfile with environment/
  // as context).
  assert.match(df, /COPY source\/env\/ \/hyperfocal\/env\//);
  assert.match(df, /COPY source\/repo\.bundle/);
  assert.match(df, /COPY source\/git-submodules\//);

  // Graft: all refs fetched, HEADs pinned to the staged positions.
  assert.match(df, /git fetch -q \/hyperfocal\/git-bundles\/repo\.bundle "\+refs\/\*:refs\/\*"/);
  assert.match(df, /git symbolic-ref HEAD refs\/heads\/main/);
  assert.match(df, new RegExp(`update-ref --no-deref HEAD ${"b".repeat(40)}`));
  assert.match(df, /git init -q --separate-git-dir \/hyperfocal\/env\/\.git\/modules\/packages\/lib/);

  // The provision steps are the SAME single-source steps as the real
  // Dockerfile (everything after the repo lands in the image).
  const real = renderDockerfile({ problemId: "p1", workspacePath: "workspace" });
  const provisionMarker = "mkdir -p /hyperfocal/logs";
  const realTail = real.slice(real.indexOf(provisionMarker));
  const selfTail = df.slice(df.indexOf(provisionMarker));
  assert.equal(selfTail, realTail, "provision steps must not drift");

  // No-submodule env: no submodule COPY line.
  const noSub = renderEnvironmentDockerfile({
    problemId: "p1",
    graft: { ...graft, submodules: [] },
  });
  assert.ok(!noSub.includes("COPY source/git-submodules/"));
});
