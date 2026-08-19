/**
 * Build-context staging for `harbor package`: a TOKENLESS local clone of the
 * packaging commit with submodules (minus env-builder), declared solution
 * refs materialized as local refs, deterministic git state
 * (normalizeGitState below), and an optional live-packages overlay. The
 * fail-closed secret scan lives in secretScan.ts.
 *
 * Under contract v2 the packaging commit IS every problem's starting state
 * (setupProblem() bakes per-problem differences during the docker build), so
 * the staged clone is prepared ONCE and never re-checked-out per problem.
 */

import { execFileSync, spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

// packages/env-builder is an authoring tool, never a runtime dependency of
// any env — and its vendored knowledge base (agent traces with expired ghs_
// tokens) has tripped the secret scan fleet-wide. It must never reach the
// docker build context or the scan, so it is skipped at submodule init and
// purged if anything materialized it anyway.
const EXCLUDED_SUBMODULE = "packages/env-builder";

function log(msg: string): void {
  console.log(`[harbor:package] ${msg}`);
}

/** Run git in `repo`; returns trimmed stdout unless opts.trim === false. */
export function git(
  repo: string,
  args: string[],
  opts: { trim?: boolean } = {}
): string {
  const out = execFileSync("git", ["-C", repo, ...args], {
    encoding: "utf-8",
    maxBuffer: 256 * 1024 * 1024,
  });
  return opts.trim === false ? out : out.trim();
}

/** Resolve `<ref>` or `origin/<ref>`; returns [resolvedRef, commit] or null. */
export function resolveRef(repo: string, ref: string): [string, string] | null {
  for (const candidate of [ref, `origin/${ref}`]) {
    try {
      const sha = git(repo, ["rev-parse", "--verify", "--quiet", candidate]);
      if (sha) return [candidate, sha];
    } catch {
      /* try next */
    }
  }
  return null;
}

export function tokenlessUrl(url: string): string {
  // https://user:token@host/... -> https://host/...
  return url.replace(/^(https?:\/\/)[^@/]+@/, "$1");
}

/**
 * `git submodule update --init --recursive` every submodule EXCEPT
 * packages/env-builder (paths read from the clone's .gitmodules), so the
 * excluded submodule is never fetched, not fetched-then-deleted.
 */
function updateSubmodulesExceptExcluded(
  cloneDir: string,
  extraArgs: string[] = []
): void {
  let pathsOut = "";
  try {
    pathsOut = git(cloneDir, [
      "config",
      "-f",
      ".gitmodules",
      "--get-regexp",
      "^submodule\\..*\\.path$",
    ]);
  } catch {
    return; // no .gitmodules -> no submodules to init
  }
  const submodulePaths = pathsOut
    .split("\n")
    .filter(Boolean)
    .map((line) => line.split(" ").slice(1).join(" "))
    .filter((p) => p !== EXCLUDED_SUBMODULE);
  if (submodulePaths.length === 0) return;
  execFileSync(
    "git",
    [
      "-C",
      cloneDir,
      "submodule",
      "update",
      "--init",
      "--recursive",
      ...extraArgs,
      "--quiet",
      "--",
      ...submodulePaths,
    ],
    { stdio: "inherit" }
  );
}

/** Deinit + delete the excluded submodule if any step materialized it. */
function purgeExcludedSubmodule(cloneDir: string): void {
  const worktree = path.join(cloneDir, EXCLUDED_SUBMODULE);
  // An initialized submodule has a `.git` file in its worktree; an
  // uninitialized gitlink is just an empty dir and needs no deinit.
  if (fs.existsSync(path.join(worktree, ".git"))) {
    try {
      git(cloneDir, ["submodule", "deinit", "-f", EXCLUDED_SUBMODULE]);
    } catch {
      /* deinit is best-effort — the deletes below are what matter */
    }
  }
  fs.rmSync(worktree, { recursive: true, force: true });
  fs.rmSync(path.join(cloneDir, ".git", "modules", EXCLUDED_SUBMODULE), {
    recursive: true,
    force: true,
  });
}

/**
 * Overlay the LIVE shared packages (this orchestrator + its env-base
 * sibling) over the env's pinned submodules — needed for envs pinned to
 * pre-harbor package versions. Runs AFTER submodule init (which would
 * otherwise reset the worktrees). The env's own environment/ code is never
 * touched. This is what remains of the v1 per-problem "restore packages
 * from the packaging commit" dance: with no per-problem checkout, the pins
 * are already the packaging commit's, and only the overlay still applies.
 */
function overlayLivePackages(cloneDir: string): void {
  // This file lives in packages/env-orchestrator/(src|dist)/packager/
  // staging -> orchestrator root is 3 up.
  const orchestratorRoot = path.resolve(
    path.dirname(new URL(import.meta.url).pathname),
    "..",
    "..",
    ".."
  );
  const packagesRoot = path.dirname(orchestratorRoot);
  for (const pkg of ["env-base", "env-orchestrator", "mock-mcp-services"]) {
    const src = path.join(packagesRoot, pkg);
    const dst = path.join(cloneDir, "packages", pkg);
    if (!fs.existsSync(src) || !fs.existsSync(dst)) continue;
    log(`Overlaying live package ${pkg} -> build context`);
    const rc = spawnSync(
      "rsync",
      [
        "-a",
        "--delete",
        "--exclude",
        ".git",
        "--exclude",
        "node_modules",
        `${src}/`,
        `${dst}/`,
      ],
      { stdio: "inherit" }
    );
    if (rc.status !== 0) throw new Error(`rsync overlay failed for ${pkg}`);
  }
}

/**
 * Stage the build context: clone the env repo, init submodules (minus
 * env-builder), materialize every DECLARED solutionRef as a local ref (the
 * in-image solution producer checks the pinned commit out of the clone, so
 * its objects must be present — local clones only carry the source's
 * checked-out branch, fresh platform clones only origin/*; a local ref
 * removes the ambiguity), normalize the remote to a tokenless URL, and
 * overlay the live shared packages when requested. Returns the clone dir.
 */
export function stageBuildContext(
  envRepo: string,
  contextDir: string,
  solutionRefs: string[],
  packagesOverlay: boolean
): string {
  fs.rmSync(contextDir, { recursive: true, force: true });
  fs.mkdirSync(contextDir, { recursive: true });
  const cloneDir = path.join(contextDir, "env");

  log(`Cloning ${envRepo} -> ${cloneDir}`);
  execFileSync("git", ["clone", "--quiet", envRepo, cloneDir], {
    stdio: "inherit",
  });
  // Submodules init'd separately (not `clone --recursive`) so env-builder is
  // never fetched into the context in the first place.
  updateSubmodulesExceptExcluded(cloneDir);

  // Materialize EVERY branch of the source repo as a local branch of the
  // staged clone — not just declared solutionRefs. setupProblem() runs
  // inside the docker build and may check per-problem state branches out
  // (contract v2's answer to problemStateRef), and a `git clone` of a FRESH
  // clone (the builder case) carries only the default branch as refs/heads:
  // local dev checkouts happened to work because their local branches came
  // along, and builders would have failed the same bake loudly. The source's
  // remote-tracking refs cover the fresh-clone case; its local heads cover
  // dev checkouts and scratch branches. Per-branch fetches (not a wildcard
  // refspec) because git refuses to update the staged clone's checked-out
  // branch and a wildcard fetch dies wholesale on that refusal.
  const stagedHead = git(cloneDir, ["rev-parse", "--abbrev-ref", "HEAD"]);
  const sourceBranches = new Set<string>();
  for (const prefix of ["refs/heads", "refs/remotes/origin"]) {
    try {
      for (const line of git(envRepo, [
        "for-each-ref",
        prefix,
        "--format=%(refname:short)",
      ]).split("\n")) {
        const name = line.replace(/^origin\//, "").trim();
        if (name && name !== "HEAD" && name !== stagedHead) {
          sourceBranches.add(name);
        }
      }
    } catch {
      // A repo without that ref namespace is fine.
    }
  }
  for (const name of sourceBranches) {
    for (const source of [`refs/heads/${name}`, `refs/remotes/origin/${name}`]) {
      try {
        execFileSync(
          "git",
          ["-C", cloneDir, "fetch", "--quiet", envRepo, `+${source}:refs/heads/${name}`],
          { stdio: "inherit" }
        );
        break;
      } catch {
        // Try the next source namespace; a branch that exists in neither
        // was a stale listing and is safe to skip.
      }
    }
  }
  for (const ref of solutionRefs) {
    const bare = ref.replace(/^origin\//, "");
    try {
      git(cloneDir, ["rev-parse", "--verify", "--quiet", bare]);
    } catch {
      execFileSync(
        "git",
        ["-C", cloneDir, "fetch", "--quiet", "origin", `${bare}:${bare}`],
        { stdio: "inherit" }
      );
    }
  }

  // Belt and braces: not all envs pin env-builder, and none may ship it.
  purgeExcludedSubmodule(cloneDir);

  // Normalize the remote to a tokenless URL (clone provenance varies: dev
  // container clones embed x-access-token in .git/config).
  let origin = "";
  try {
    origin = git(envRepo, ["remote", "get-url", "origin"]);
  } catch {
    /* repo may have no remote */
  }
  if (origin) {
    git(cloneDir, ["remote", "set-url", "origin", tokenlessUrl(origin)]);
  } else {
    git(cloneDir, ["remote", "remove", "origin"]);
  }
  // FETCH_HEAD can carry the local path used for staging; harmless but noisy.
  fs.rmSync(path.join(cloneDir, ".git", "FETCH_HEAD"), { force: true });

  // Deterministic staging: two clones of the same commit differ only in
  // .git reflogs (timestamps) and .git index files (stat caches of checkout
  // mtimes) — measured 2026-07-17, repo + submodules, everything else
  // byte-identical. Those bytes reach the image via `COPY env/`, so every
  // fresh staging busted the docker COPY layer (and everything after it)
  // even at an identical commit. Drop the reflogs and rebuild each index
  // from HEAD (content-only, no stat cache) so the whole context is
  // byte-reproducible: a warm builder's layer cache then survives across
  // releases of the same commit. In-image git remains fully functional —
  // `checkout HEAD -- <path>` and diffs just re-stat lazily.
  normalizeGitState(cloneDir);

  if (packagesOverlay) {
    overlayLivePackages(cloneDir);
  }

  return cloneDir;
}

/**
 * Make the staged clone's .git state byte-reproducible for a given commit:
 * remove reflogs (timestamped) and rebuild every index from HEAD (the index
 * carries checkout-time stat caches). Covers the superproject and every
 * initialized submodule (worktree .git file -> .git/modules/<path>).
 */
function normalizeGitState(cloneDir: string): void {
  const gitDirs: Array<{ worktree: string; gitDir: string }> = [
    { worktree: cloneDir, gitDir: path.join(cloneDir, ".git") },
  ];
  // Initialized submodules: their real git dirs live under .git/modules/.
  const modulesRoot = path.join(cloneDir, ".git", "modules");
  const walkModules = (dir: string, rel: string): void => {
    if (!fs.existsSync(path.join(dir, "HEAD"))) {
      if (!fs.existsSync(dir)) return;
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
          walkModules(path.join(dir, entry.name), path.join(rel, entry.name));
        }
      }
      return;
    }
    gitDirs.push({ worktree: path.join(cloneDir, rel), gitDir: dir });
  };
  walkModules(modulesRoot, "");
  for (const { worktree, gitDir } of gitDirs) {
    fs.rmSync(path.join(gitDir, "logs"), { recursive: true, force: true });
    fs.rmSync(path.join(gitDir, "index"), { force: true });
    if (fs.existsSync(worktree)) {
      git(worktree, ["read-tree", "HEAD"]);
    }
  }
}
