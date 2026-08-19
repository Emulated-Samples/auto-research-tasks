/**
 * Host-side build + load of the environment module for `harbor package`.
 *
 * The packageProblem() hook must run BEFORE the docker build (its output
 * shapes the Dockerfile), so the packager needs the env's compiled module on
 * the host. Building inside the staged docker context would bloat it with
 * node_modules and slow the secret scan, so the environment is built in a
 * separate hook-build directory: copy environment/ + packages/ (minus
 * env-builder, which is an authoring tool with a secret-bearing history),
 * npm install + build exactly like the generated Dockerfile does, then
 * dynamically import the compiled module.
 *
 * Side benefit: this is an early compile check — a TS error in the env (or
 * mismatched env-base/env-orchestrator submodule pins) fails the package run
 * in minutes instead of failing mid-docker-build.
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { EnvironmentDefinition } from "@hyperfocal/env-base";
import { loadEnvironment } from "../../cli/environment.js";

/** Authoring tool, never a runtime dep; its history carries expired tokens. */
const EXCLUDED_PACKAGE = "env-builder";

/** Skipped when copying source trees; rebuilt fresh by npm install/build. */
const COPY_EXCLUDES = new Set(["node_modules", "dist", ".git"]);

function copyTree(src: string, dest: string): void {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (COPY_EXCLUDES.has(entry.name)) continue;
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyTree(from, to);
    } else if (entry.isSymbolicLink()) {
      fs.symlinkSync(fs.readlinkSync(from), to);
    } else {
      fs.copyFileSync(from, to);
    }
  }
}

function npmBuild(dir: string, label: string, log: (msg: string) => void): void {
  for (const args of [
    ["install", "--no-audit", "--no-fund", "--silent"],
    ["run", "build", "--silent"],
  ]) {
    const rc = spawnSync("npm", args, { cwd: dir, stdio: ["ignore", "pipe", "pipe"], encoding: "utf-8" });
    if (rc.status !== 0) {
      throw new Error(
        `hook build: npm ${args[0]} failed in ${label}. The environment must ` +
          `compile for packaging (the task image runs this same build). ` +
          `Common cause: environment pins an env-orchestrator submodule newer ` +
          `than its env-base submodule (or vice versa) — re-pin them to ` +
          `matching commits.\n${rc.stderr || rc.stdout || ""}`
      );
    }
  }
  log(`  hook build: ${label} compiled`);
}

/**
 * Build the env repo's environment module in `<outDir>/hook-build` and load
 * it. Returns null (with a log line) when the repo has no buildable
 * environment package — minimal/fixture repos — since a missing module just
 * means "no packageProblem hook".
 *
 * The build directory is recreated from scratch each run — predictability
 * over speed; npm's global cache keeps the installs reasonable.
 */
export async function loadPackagingEnvModule(
  envRepo: string,
  outDir: string,
  environmentDistRelPath: string,
  log: (msg: string) => void
): Promise<EnvironmentDefinition | null> {
  const envPackageJson = path.join(envRepo, "environment", "package.json");
  if (!fs.existsSync(envPackageJson)) {
    log("  hook build: no environment/package.json — skipping packageProblem hook discovery");
    return null;
  }

  const buildDir = path.join(outDir, "hook-build");
  fs.rmSync(buildDir, { recursive: true, force: true });

  copyTree(path.join(envRepo, "environment"), path.join(buildDir, "environment"));
  const packagesDir = path.join(envRepo, "packages");
  if (fs.existsSync(packagesDir)) {
    for (const entry of fs.readdirSync(packagesDir, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name === EXCLUDED_PACKAGE) continue;
      copyTree(path.join(packagesDir, entry.name), path.join(buildDir, "packages", entry.name));
    }
  }
  const yamlSrc = path.join(envRepo, "hyperfocal.yaml");
  if (fs.existsSync(yamlSrc)) {
    fs.copyFileSync(yamlSrc, path.join(buildDir, "hyperfocal.yaml"));
  }

  // Same order as the generated Dockerfile: deps first, then the env itself.
  // env-orchestrator is deliberately NOT built here — importing the compiled
  // environment module only needs its own deps (env-base and friends).
  for (const pkg of ["env-base", "mock-mcp-services"]) {
    const dir = path.join(buildDir, "packages", pkg);
    if (fs.existsSync(path.join(dir, "package.json"))) {
      npmBuild(dir, `packages/${pkg}`, log);
    }
  }
  npmBuild(path.join(buildDir, "environment"), "environment", log);

  const distDir = path.join(buildDir, environmentDistRelPath);
  const env = await loadEnvironment(distDir);
  if (!env) {
    throw new Error(
      `hook build: environment compiled but could not be loaded from ${distDir}`
    );
  }
  return env;
}
