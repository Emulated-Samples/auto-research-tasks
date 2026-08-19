/**
 * Per-task source emission for `harbor package` (Wave 1 step 1,
 * docs/11-build-finalization-fourth-draft/4-implementation-plan.md §4):
 * every task carries its OWN copy of the exact build context under
 * `environment/source/`, so each task directory is standalone — the shipped
 * environment/Dockerfile builds from it with no env-repo access and no
 * private registry. This replaces the old once-at-output-root `source/`
 * bundle (same git-graft machinery, new destination; per-task duplication
 * of big repos is an accepted owner decision — git packfiles dedupe the
 * blobs in the released repo, but exports inflate fully).
 *
 * Emitted layout, per task (built once into a staging dir, copied per task):
 *
 *   tasks/<task>/environment/source/
 *     env/                     staged, tokenless, secret-scanned build
 *                              context worktree — the same bytes the builder
 *                              built from — with every live .git directory
 *                              AND submodule .git pointer file stripped
 *     Dockerfile.<problem>     the exact per-problem recipe the builder used
 *                              (THIS task's problem only — audit artifact;
 *                              the shipped build recipe is ../Dockerfile)
 *     repo.bundle              `git bundle create --all HEAD` of the staged
 *                              superproject clone: every ref survives
 *                              (setups check problem branches out — e.g.
 *                              nano-vllm's PROBLEM_BRANCH_CHECKOUTS), yet a
 *                              bundle is ONE FILE, so the shipService gate
 *                              against nested .git dirs (control-plane
 *                              PR #106, finding F-7) passes
 *     git-submodules/<p>.bundle  one bundle per INITIALIZED submodule
 *                              (path-encoded, "/" -> "__")
 *
 * WHY per-submodule bundles: staging materializes submodules as real git
 * clones — worktree `.git` FILES pointing into the superproject's
 * .git/modules/<path> object stores (stage.ts updateSubmodulesExceptExcluded
 * + normalizeGitState). A superproject bundle carries gitlink SHAs only,
 * never the submodules' objects, so cloning it and running `git submodule
 * update --init` would have to fetch over the network from possibly-private
 * remotes — not self-contained. Bundling each initialized submodule's
 * git dir keeps the rebuild offline and faithful. The excluded env-builder
 * submodule stays exactly as in the real image: an uninitialized gitlink
 * with no worktree and no objects.
 *
 * The .git stripping never mutates the staged clone itself — the real
 * builds COPY the clone with its git state intact; source/ is a filtered
 * COPY of it. Bundling likewise only reads.
 */

import { execFileSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { git } from "../staging/stage.js";
import { tokenlessUrl } from "../staging/stage.js";

/** GitHub hard-rejects files over 100 MB; shipService (control-plane) fails
 * the ship on them — fail the PACKAGE first, hours earlier, by name. */
const GITHUB_FILE_SIZE_LIMIT_BYTES = 100 * 1024 * 1024;

/** One initialized submodule of the staged clone, as the shipped
 * environment/Dockerfile's git graft needs to reconstruct it. */
export interface StagedSubmoduleInfo {
  /** Worktree path relative to the superproject root (posix). */
  path: string;
  /** Enclosing repo's worktree relative to the superproject ("" = superproject). */
  parent: string;
  /** Logical submodule name in the parent's .gitmodules (fallback: path). */
  name: string;
  /** Configured URL in the parent repo's config, tokenless, if any. */
  url: string | null;
  /** The staged submodule's HEAD — the pinned commit the image must be at. */
  headCommit: string;
  /** File name under source/git-submodules/. */
  bundleFile: string;
}

/** Everything the shipped Dockerfile's git graft must know. */
export interface GitGraftInfo {
  /** Superproject HEAD branch, or null when the packaging HEAD is detached. */
  headBranch: string | null;
  /** Superproject HEAD commit (the packaging commit). */
  headCommit: string;
  /** Tokenless origin URL of the staged clone, if it kept a remote. */
  originUrl: string | null;
  submodules: StagedSubmoduleInfo[];
}

/** Encode a submodule worktree path as a flat bundle file name. */
function bundleFileName(subPath: string): string {
  return `${subPath.replace(/\//g, "__")}.bundle`;
}

/**
 * Enumerate the staged clone's INITIALIZED submodules, recursively.
 * `git submodule status --recursive` prints one line per gitlink —
 * `<flag><sha> <path> (<desc>)`, path relative to the repo the command ran
 * in — where flag "-" marks an uninitialized submodule (env-builder after
 * the staging purge): those are skipped, exactly as the real image ships
 * them.
 */
export function collectGitGraftInfo(cloneDir: string): GitGraftInfo {
  let headBranch: string | null = null;
  try {
    headBranch = git(cloneDir, ["symbolic-ref", "--quiet", "--short", "HEAD"]);
  } catch {
    headBranch = null; // detached HEAD (builder packaging a raw commit)
  }
  const headCommit = git(cloneDir, ["rev-parse", "HEAD"]);
  let originUrl: string | null = null;
  try {
    originUrl = tokenlessUrl(git(cloneDir, ["remote", "get-url", "origin"]));
  } catch {
    originUrl = null; // staging removed the remote (no-remote source repo)
  }

  let statusOut = "";
  try {
    statusOut = git(cloneDir, ["submodule", "status", "--recursive"]);
  } catch {
    statusOut = ""; // no submodules at all
  }
  const initializedPaths: string[] = [];
  for (const line of statusOut.split("\n")) {
    if (!line.trim()) continue;
    const flag = line[0]; // " " ok, "+" pin mismatch, "-" uninitialized
    if (flag === "-") continue;
    // "<flag><sha> <path> (<desc>)" — the path may contain spaces, the
    // trailing "(<desc>)" may be absent; split conservatively.
    const rest = line.slice(1).trim();
    const firstSpace = rest.indexOf(" ");
    if (firstSpace < 0) continue;
    let p = rest.slice(firstSpace + 1);
    const paren = p.lastIndexOf(" (");
    if (paren >= 0 && p.endsWith(")")) p = p.slice(0, paren);
    initializedPaths.push(p);
  }
  initializedPaths.sort(); // deterministic emission + parents before children

  const submodules: StagedSubmoduleInfo[] = initializedPaths.map((subPath) => {
    // Enclosing repo = the longest OTHER submodule path that prefixes this
    // one; the superproject otherwise.
    const parent =
      initializedPaths
        .filter((o) => o !== subPath && subPath.startsWith(`${o}/`))
        .sort((a, b) => b.length - a.length)[0] ?? "";
    const parentDir = parent ? path.join(cloneDir, parent) : cloneDir;
    const relPath = parent ? subPath.slice(parent.length + 1) : subPath;

    // Logical name: the .gitmodules entry whose path equals relPath.
    let name = relPath;
    try {
      const entries = git(parentDir, [
        "config",
        "-f",
        ".gitmodules",
        "--get-regexp",
        "^submodule\\..*\\.path$",
      ]);
      for (const entry of entries.split("\n")) {
        const m = entry.match(/^submodule\.(.+)\.path (.+)$/);
        if (m && m[2] === relPath) {
          name = m[1];
          break;
        }
      }
    } catch {
      /* no .gitmodules in parent — keep the path as name */
    }

    let url: string | null = null;
    try {
      url = tokenlessUrl(git(parentDir, ["config", `submodule.${name}.url`]));
    } catch {
      url = null;
    }

    return {
      path: subPath,
      parent,
      name,
      url,
      headCommit: git(path.join(cloneDir, subPath), ["rev-parse", "HEAD"]),
      bundleFile: bundleFileName(subPath),
    };
  });

  return { headBranch, headCommit, originUrl, submodules };
}

/** Copy `from` -> `to`, skipping every entry named ".git" (live git
 * directories AND submodule pointer files) at any depth. */
function copyTreeWithoutGit(from: string, to: string): void {
  fs.cpSync(from, to, {
    recursive: true,
    filter: (src) => path.basename(src) !== ".git",
  });
}

export interface EmitTaskSourcesOptions {
  /** The staged build-context dir (contains env/ + Dockerfile.<problem>). */
  contextDir: string;
  /** The staged clone inside it (contextDir/env). */
  cloneDir: string;
  /** Package output root; the staging copy lives under it while emitting. */
  outDir: string;
  /** Destination task dirs: source/ lands in <taskDir>/environment/source. */
  tasks: { taskDir: string; problemId: string }[];
  logger?: (msg: string) => void;
}

/**
 * Emit environment/source/ into EVERY task dir. The canonical source tree
 * (context copy + bundles) is assembled once in a staging dir, size-gated
 * once, then copied per task — each task receiving only its OWN
 * Dockerfile.<problem> audit recipe. Returns the graft info the
 * environment/Dockerfile renderers consume.
 */
export function emitTaskSources(opts: EmitTaskSourcesOptions): GitGraftInfo {
  const log =
    opts.logger ?? ((msg: string) => console.log(`[harbor:package] ${msg}`));
  const stageDir = path.join(opts.outDir, ".source-stage");
  fs.rmSync(stageDir, { recursive: true, force: true });

  const graft = collectGitGraftInfo(opts.cloneDir);

  log(`Staging task source tree -> ${stageDir}`);
  copyTreeWithoutGit(opts.contextDir, stageDir);

  // Superproject bundle: --all (every ref — staging materialized every
  // source branch as a local head) + HEAD explicitly (a detached packaging
  // HEAD is not covered by --all).
  execFileSync(
    "git",
    [
      "-C",
      opts.cloneDir,
      "bundle",
      "create",
      path.join(stageDir, "repo.bundle"),
      "--all",
      "HEAD",
    ],
    { stdio: ["ignore", "ignore", "inherit"] }
  );

  if (graft.submodules.length > 0) {
    const subDir = path.join(stageDir, "git-submodules");
    fs.mkdirSync(subDir, { recursive: true });
    for (const sub of graft.submodules) {
      execFileSync(
        "git",
        [
          "-C",
          path.join(opts.cloneDir, sub.path),
          "bundle",
          "create",
          path.join(subDir, sub.bundleFile),
          "--all",
          "HEAD",
        ],
        { stdio: ["ignore", "ignore", "inherit"] }
      );
    }
  }

  // Gate once on the staging copy (every task's copy is a subset of it).
  assertSourceFileSizes(stageDir);

  try {
    for (const task of opts.tasks) {
      const dest = path.join(task.taskDir, "environment", "source");
      fs.rmSync(dest, { recursive: true, force: true });
      // Per-task filter: of the top-level Dockerfile.<problem> audit
      // recipes, only THIS task's problem travels; everything else (env/,
      // bundles) is shared verbatim.
      fs.cpSync(stageDir, dest, {
        recursive: true,
        filter: (src) => {
          const base = path.basename(src);
          if (
            path.dirname(src) === stageDir &&
            base.startsWith("Dockerfile.")
          ) {
            return base === `Dockerfile.${task.problemId}`;
          }
          return true;
        },
      });
    }
  } finally {
    fs.rmSync(stageDir, { recursive: true, force: true });
  }

  log(
    `Per-task source emitted into ${opts.tasks.length} task(s): env tree + ` +
      `repo.bundle + ${graft.submodules.length} submodule bundle(s)`
  );
  return graft;
}

/**
 * Fail the PACKAGE — with the offending files named — before the ship-time
 * gate (shipService assertPublishableFileSizes) would fail it hours later.
 * GitHub hard-rejects any file over 100 MB; LFS/chunking is deliberately
 * deferred until a real env hits this (decision 7.12).
 */
export function assertSourceFileSizes(sourceDir: string): void {
  const oversized: string[] = [];
  const stack = [sourceDir];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(abs);
      } else if (entry.isFile()) {
        const size = fs.lstatSync(abs).size;
        if (size > GITHUB_FILE_SIZE_LIMIT_BYTES) {
          oversized.push(
            `${path.relative(sourceDir, abs)} (${Math.round(size / (1024 * 1024))} MB)`
          );
        }
      }
    }
  }
  if (oversized.length > 0) {
    throw new Error(
      `task source contains ${oversized.length} file(s) over GitHub's 100 MB ` +
        `per-file limit — the release could never ship (shipService enforces the ` +
        `same gate): ${oversized.sort().join(", ")}. Drop them from the repo, or ` +
        `package with --no-source-bundle until LFS/chunking lands (decision 7.12).`
    );
  }
}
