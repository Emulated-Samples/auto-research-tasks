/**
 * Solution production for `harbor package` — contract v2: EVERY reference
 * solution is produced (or at least verified) inside the just-built image.
 *
 * The ladder, first match wins (see env-base Problem):
 *   1. the env module implements solveProblem()  -> run it in the image as
 *      root and diff the workspace ("hook").
 *   2. problem declares solutionRef              -> check the pinned commit
 *      out of the baked clone inside the image and diff ("ref").
 *   3. problem declares solutionPatch            -> copy the committed patch
 *      file and `git apply --check` it inside the image ("patch").
 *   4. nothing                                   -> no solution/ dir; replay
 *      is not applicable ("none").
 * A hook coexisting with a declaration is a package-time hard error: the
 * hook owns solutions when present.
 *
 * Why in-image and not host-side (the v1 split was the bug): git-ref
 * solutions used to be diffed host-side against the source repo and never
 * touched the image, while hook solutions ran in-image. The hook gate found
 * the class of bug host-side diffs hide — harbor runs solution/solve.sh as
 * the task's AGENT user while the image lockdown denies that uid the env
 * internals, and a host-generated patch never notices image drift. Under v2
 * every solution is validated against the exact image trials will run — so
 * solutions always require a build, and --no-build emits no solution/ at all.
 *
 * Why root in an ephemeral container and not trial time (hook-gate E2E
 * finding, 2026-07-15): harbor 0.18 runs solution/solve.sh as the agent user
 * (SolutionConfig has no user field) and the lockdown correctly makes
 * packages/, environment/, and hyperfocal.yaml unreadable to that user.
 * Anything baked into the image that a solver could use, an agent could use;
 * producing the patch here (image itself never modified) means solution
 * content only enters trial containers when harbor uploads solution/
 * per-trial.
 */

import { spawn, spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { Problem } from "@hyperfocal/env-base";
import { git, resolveRef } from "./staging/stage.js";

/**
 * The one solve.sh for every solution kind (hook / ref / patch): apply
 * solution.patch to the live tree. The patch is produced inside the built
 * image at package time (below) — a workspace diff for hook and ref
 * solutions, the committed file (apply-checked in-image) for patch
 * solutions. solve.sh must never exec solve-oracle: harbor 0.18 runs it as
 * the AGENT user and the trust lockdown correctly denies that user
 * packages/ and environment/ (hook-gate E2E, 2026-07-15).
 *
 * TODO(solve-sh-alternatives): the trial-time solution STAYS patch
 * application — owner decision, 2026-07-19. Two alternatives were
 * investigated and rejected (full investigation:
 * docs/11-build-finalization-fourth-draft/2-intermediate-and-end-states.md
 * §3): (a) trial-time hook execution via a custom oracle environment
 * class, and (b) an upstream harbor solution-user field. Neither works
 * today because harbor runs solve.sh as the jailed AGENT user and
 * SolutionConfig has no `user` field at 0.18 or HEAD — the agent-jailed
 * solver cannot reach the env internals a hook needs, and there is no
 * knob to run it as anyone else. Revisit only if harbor grows a
 * solution-user field.
 */
function renderSolveSh(workspacePath: string = "workspace"): string {
  return `#!/bin/bash
# Reference solution (oracle, generated): apply the pinned gold diff to the
# live tree.
#
# MUST run with cwd = repo root: patch paths are ${workspacePath}/...-prefixed and
# \`git apply\` from a subdirectory silently no-ops (rc 0!) on outside paths.
set -euo pipefail

cd /hyperfocal/env
git apply -p1 --whitespace=nowarn /solution/solution.patch
echo "[solve.sh] solution.patch applied"
`;
}

/** Run a command, collecting stdout/stderr (async so productions can overlap). */
function runCollected(
  cmd: string,
  args: string[],
  input?: Buffer
): Promise<{
  status: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ["pipe", "pipe", "pipe"] });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    child.stdout.on("data", (d: Buffer) => out.push(d));
    child.stderr.on("data", (d: Buffer) => err.push(d));
    child.on("error", reject);
    if (input) child.stdin.write(input);
    child.stdin.end();
    child.on("close", (status, signal) =>
      resolve({
        status,
        signal,
        stdout: Buffer.concat(out).toString("utf-8"),
        stderr: Buffer.concat(err).toString("utf-8"),
      })
    );
  });
}

export type SolutionKind = "hook" | "ref" | "patch" | "none";

/** Per-problem solution decision, resolved and pinned before any heavy work. */
export interface SolutionPlan {
  kind: SolutionKind;
  /** Resolved ref name (kind=ref only), e.g. "gold/foo" or "origin/gold/foo". */
  solutionRef: string | null;
  /** Commit the ref resolved to at package time (kind=ref only). */
  solutionCommit: string | null;
  /** Repo-relative committed patch path (kind=patch only). */
  solutionPatch: string | null;
}

function log(msg: string): void {
  console.log(`[harbor:package] ${msg}`);
}

/**
 * Walk the ladder for every selected problem: enforce the hook-vs-declaration
 * conflict, resolve declared refs (pinning their commits), and verify
 * declared patch files are committed at the packaging commit. All failure
 * modes fire here, before staging or docker work.
 */
export function planSolutions(opts: {
  envRepo: string;
  problems: Problem[];
  /** Whether the env module implements solveProblem(). */
  envDefinesSolveHook: boolean;
  packagingCommit: string;
}): Map<string, SolutionPlan> {
  const plans = new Map<string, SolutionPlan>();
  for (const problem of opts.problems) {
    const declared = [
      ...(problem.solutionRef !== undefined ? [`solutionRef: ${problem.solutionRef}`] : []),
      ...(problem.solutionPatch !== undefined ? [`solutionPatch: ${problem.solutionPatch}`] : []),
    ];
    if (opts.envDefinesSolveHook && declared.length > 0) {
      throw new Error(
        `Problem '${problem.id}' declares ${declared.join(" and ")} but the ` +
          `environment module implements solveProblem() — the hook owns ` +
          `solutions when present; remove the declaration from problems.yaml`
      );
    }
    if (opts.envDefinesSolveHook) {
      plans.set(problem.id, {
        kind: "hook",
        solutionRef: null,
        solutionCommit: null,
        solutionPatch: null,
      });
    } else if (problem.solutionRef !== undefined) {
      const resolved = resolveRef(opts.envRepo, problem.solutionRef);
      if (!resolved) {
        throw new Error(
          `solutionRef "${problem.solutionRef}" for problem ${problem.id} ` +
            `does not resolve (tried it and origin/-prefixed); remove the ` +
            `declaration if this problem has no reference solution`
        );
      }
      plans.set(problem.id, {
        kind: "ref",
        solutionRef: resolved[0],
        solutionCommit: resolved[1],
        solutionPatch: null,
      });
    } else if (problem.solutionPatch !== undefined) {
      // The patch must be COMMITTED at the packaging commit — the staged
      // clone (and therefore the image) is built from that commit, and an
      // uncommitted local file would silently not ship.
      try {
        git(opts.envRepo, [
          "cat-file",
          "-e",
          `${opts.packagingCommit}:${problem.solutionPatch}`,
        ]);
      } catch {
        throw new Error(
          `solutionPatch "${problem.solutionPatch}" for problem ${problem.id} ` +
            `is not committed at the packaging commit ` +
            `${opts.packagingCommit.slice(0, 8)} — commit the patch file ` +
            `(or fix the repo-relative path)`
        );
      }
      plans.set(problem.id, {
        kind: "patch",
        solutionRef: null,
        solutionCommit: null,
        solutionPatch: problem.solutionPatch,
      });
    } else {
      plans.set(problem.id, {
        kind: "none",
        solutionRef: null,
        solutionCommit: null,
        solutionPatch: null,
      });
    }
  }
  return plans;
}

/** Single-quote a string for safe splicing into a bash script. */
function shq(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

/**
 * Run a bash script as root in an ephemeral container of the built image
 * (--network none: producing a solution must not need network) and copy
 * /tmp/solution.patch out to destPatch. All container output is replayed
 * into the package log under a [solve:<id>] prefix so failures are loud and
 * attributable. The image itself is never modified.
 */
async function runSolutionScript(
  imageTag: string,
  problemId: string,
  script: string,
  destPatch: string
): Promise<void> {
  const containerName = `hf-solve-${problemId.replace(/[^a-zA-Z0-9_.-]/g, "-")}-${process.pid}-${Date.now()}`;
  try {
    const run = await runCollected("docker", [
      "run",
      "--name",
      containerName,
      "--network",
      "none",
      // The generated Dockerfile has no USER directive so root is already
      // the default; pin it anyway in case packaging.dockerfileExtra ever
      // changes that.
      "--user",
      "0:0",
      imageTag,
      "bash",
      "-c",
      script,
    ]);
    const replay = (label: string, text: string | null | undefined): void => {
      if (!text || !text.trim()) return;
      for (const line of text.replace(/\n$/, "").split("\n")) {
        console.log(`[solve:${problemId}:${label}] ${line}`);
      }
    };
    replay("out", run.stdout);
    replay("err", run.stderr);
    if (run.status !== 0) {
      throw new Error(
        `solution production failed for ${problemId} in ${imageTag} ` +
          `(${run.status === null ? `signal ${run.signal}` : `exit ${run.status}`}) — ` +
          `see [solve:${problemId}] output above`
      );
    }
    // The patch is docker cp'd out of the stopped container rather than read
    // from stdout — robust against solver chatter on stdout.
    const cp = spawnSync(
      "docker",
      ["cp", `${containerName}:/tmp/solution.patch`, destPatch],
      { stdio: "inherit" }
    );
    if (cp.status !== 0) {
      throw new Error(`docker cp of the solution patch failed for ${problemId}`);
    }
  } finally {
    spawnSync("docker", ["rm", "-f", containerName], { stdio: "ignore" });
  }
}

/**
 * Shared script prologue: disposable telemetry dir (env-base freezes its
 * logs dir at module load) and a `pre-solve` snapshot commit of the BAKED
 * post-setup workspace — setupProblem() ran during `docker build` and its
 * workspace changes are NOT in HEAD, so the snapshot is what the solved
 * state gets diffed against.
 */
function snapshotPrologue(ws: string): string[] {
  return [
    "set -euo pipefail",
    'export HYPERFOCAL_LOGS_DIR="$(mktemp -d)"',
    "cd /hyperfocal/env",
    `git add -A -- ${ws}`,
    "git -c user.email=pkg@hyperfocal -c user.name=packager commit -q --allow-empty -m pre-solve",
  ];
}

/**
 * kind=hook: run the env's solveProblem() (via the solve-oracle CLI, which
 * baked images already know how to invoke) in the built image and diff the
 * workspace: post-solve minus post-setup.
 */
async function produceHookSolution(
  imageTag: string,
  problemId: string,
  workspacePath: string,
  destPatch: string
): Promise<void> {
  const ws = shq(workspacePath);
  const script = [
    ...snapshotPrologue(ws),
    `node packages/env-orchestrator/bin/env-orchestrator.js solve-oracle --problem ${shq(problemId)}`,
    `git add -A -- ${ws}`,
    `git diff --binary --cached HEAD -- ${ws} > /tmp/solution.patch`,
  ].join("\n");
  log(
    `  producing hook solution for ${problemId}: solve-oracle as root in ` +
      `${imageTag} (image not modified)...`
  );
  await runSolutionScript(imageTag, problemId, script, destPatch);
}

/**
 * kind=ref: the ref's workspace/ IS the solved state. Inside the image,
 * replace the workspace with the PINNED commit's version (rm + checkout so
 * files the solved state deletes are deleted too — a bare `git checkout
 * <commit> -- <ws>` never removes anything) and diff against the post-setup
 * snapshot. The pinned commit's objects are present because staging
 * materialized every declared solutionRef as a local ref in the clone.
 */
async function produceRefSolution(
  imageTag: string,
  problemId: string,
  solutionCommit: string,
  workspacePath: string,
  destPatch: string
): Promise<void> {
  const ws = shq(workspacePath);
  const script = [
    ...snapshotPrologue(ws),
    `git rm -r -q --ignore-unmatch -- ${ws}`,
    `git checkout ${shq(solutionCommit)} -- ${ws}`,
    `git add -A -- ${ws}`,
    `git diff --binary --cached HEAD -- ${ws} > /tmp/solution.patch`,
  ].join("\n");
  log(
    `  producing ref solution for ${problemId}: git checkout ` +
      `${solutionCommit.slice(0, 8)} -- ${workspacePath} in ${imageTag}...`
  );
  await runSolutionScript(imageTag, problemId, script, destPatch);
}

/**
 * kind=patch: copy the committed patch out of the staged clone, then verify
 * it APPLIES inside the built image (`git apply --check`, same flags and cwd
 * as the emitted solve.sh) — a patch that no longer applies is a broken
 * solution and must fail the package, not the first replay.
 */
async function producePatchSolution(
  imageTag: string,
  problemId: string,
  cloneDir: string,
  patchRelPath: string,
  destPatch: string
): Promise<void> {
  const src = path.join(cloneDir, patchRelPath);
  if (!fs.existsSync(src)) {
    throw new Error(
      `solutionPatch ${patchRelPath} for ${problemId} is missing from the ` +
        `staged clone (${src}) despite being committed — staging bug?`
    );
  }
  fs.copyFileSync(src, destPatch);
  log(
    `  verifying patch solution for ${problemId}: git apply --check of ` +
      `${patchRelPath} in ${imageTag}...`
  );
  // The patch is piped over stdin ("-") so nothing needs to be mounted into
  // or copied into the container.
  const rc = await runCollected(
    "docker",
    [
      "run",
      "--rm",
      "-i",
      "--network",
      "none",
      "--user",
      "0:0",
      imageTag,
      "bash",
      "-c",
      "cd /hyperfocal/env && git apply --check -p1 --whitespace=nowarn -",
    ],
    fs.readFileSync(destPatch)
  );
  if (rc.status !== 0) {
    throw new Error(
      `solutionPatch ${patchRelPath} for ${problemId} does NOT apply to the ` +
        `built image's workspace (git apply --check failed in ${imageTag}) — ` +
        `the committed patch is a broken solution:\n` +
        `${rc.stderr.trim()}`
    );
  }
}

/**
 * Produce the task's solution/ dir (solution.patch + solve.sh) from the
 * built image, per the plan. Requires a build by construction — callers must
 * not invoke this under --no-build. Returns the patch size in bytes.
 *
 * Async on purpose: productions are independent docker runs against
 * distinct images, so the packager runs them in parallel (Promise.all)
 * after the build phase completes.
 */
export async function produceSolution(opts: {
  plan: SolutionPlan;
  problemId: string;
  taskName: string;
  imageTag: string;
  workspacePath: string;
  taskDir: string;
  cloneDir: string;
}): Promise<number> {
  const { plan, problemId, taskName, imageTag, workspacePath, taskDir, cloneDir } =
    opts;
  if (plan.kind === "none") {
    throw new Error(`produceSolution called for ${problemId} with kind=none`);
  }
  const solutionDir = path.join(taskDir, "solution");
  fs.mkdirSync(solutionDir, { recursive: true });
  const destPatch = path.join(solutionDir, "solution.patch");

  if (plan.kind === "hook") {
    await produceHookSolution(imageTag, problemId, workspacePath, destPatch);
  } else if (plan.kind === "ref") {
    await produceRefSolution(
      imageTag,
      problemId,
      plan.solutionCommit!,
      workspacePath,
      destPatch
    );
  } else {
    await producePatchSolution(
      imageTag,
      problemId,
      cloneDir,
      plan.solutionPatch!,
      destPatch
    );
  }

  const patch = fs.readFileSync(destPatch, "utf-8");
  if (!patch.trim()) {
    throw new Error(
      plan.kind === "hook"
        ? `solveProblem() produced no workspace changes for ${problemId} — ` +
            `the hook solution patch is empty (solve-oracle ran clean in ` +
            `${imageTag} but nothing under ${workspacePath} changed vs the ` +
            `baked post-setup state)`
        : plan.kind === "ref"
          ? `solution patch for ${problemId} is EMPTY (${plan.solutionRef} @ ` +
              `${plan.solutionCommit!.slice(0, 8)} matches the baked ` +
              `post-setup ${workspacePath}); check solutionRef or remove the ` +
              `declaration`
          : `solutionPatch ${plan.solutionPatch} for ${problemId} is empty`
    );
  }

  // The one solve.sh for every solution kind: apply solution.patch from the
  // repo root (renderSolveSh).
  const solveSh = path.join(solutionDir, "solve.sh");
  fs.writeFileSync(solveSh, renderSolveSh(workspacePath));
  fs.chmodSync(solveSh, 0o755);

  const bytes = Buffer.byteLength(patch);
  const provenance =
    plan.kind === "hook"
      ? `solveProblem() in ${imageTag}`
      : plan.kind === "ref"
        ? `${plan.solutionRef} @ ${plan.solutionCommit!.slice(0, 8)}, diffed in-image`
        : `${plan.solutionPatch}, verified in-image`;
  log(`  ${taskName}: solution.patch ${bytes} bytes (${provenance})`);
  return bytes;
}
