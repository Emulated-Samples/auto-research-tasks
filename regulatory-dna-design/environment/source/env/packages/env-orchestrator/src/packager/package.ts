/**
 * `env-orchestrator harbor package` — convert a hyperfocal env repo into
 * Harbor task directories + a prebuilt docker image per problem.
 *
 * Pipeline (contract v2 — docs/3-packaging-contract-v2/1-contract-v2.md):
 *   1. Resolve the env repo + packaging commit; load problems.yaml. The
 *      packaging commit IS every problem's starting state — per-problem
 *      differences are baked by the env's own setupProblem() during the
 *      docker build (there is no problemStateRef anymore).
 *   2. Build the env module host-side (hookBuild.ts) to discover the
 *      optional packageProblem() spec hook and solveProblem() solution
 *      hook, then plan solutions per problem (solutions.ts): hook > ref >
 *      patch > none, hook+declaration is a hard error, declared refs are
 *      resolved and pinned and declared patch files verified committed —
 *      all before any heavy work.
 *   3. Stage a TOKENLESS build context (staging.ts): local `git clone` +
 *      submodules EXCEPT packages/env-builder, declared solutionRefs
 *      materialized as local refs, optional --packages-overlay of the live
 *      env-base/env-orchestrator worktrees, then a fail-closed secret scan.
 *   4. Per problem: default TaskSpec from hyperfocal.yaml + optional
 *      packageProblem() amendments (taskSpec.ts), then task-dir emission
 *      into tasks/<env>__<branch-slug>__<problem>/ (task identity is
 *      branch-defined — env x branch x problem): task.toml (ECR pin +
 *      [metadata] branch + min_replay_score for solution-declaring
 *      problems) / instruction.md / tests/test.sh, plus the task's OWN
 *      environment/source/ copy (emit/taskSource.ts) and standalone
 *      environment/Dockerfile (render/environmentDockerfile.ts) that
 *      rebuilds from it — delete the docker_image line from task.toml and
 *      harbor builds from source.
 *   5. Build every missing image in one parallel `docker buildx bake` pass
 *      (build/bake.ts), tagged `<branch-slug>-<problem>-<commit8>-<spec8>`
 *      (+`-r<n>` when the rebuild counter is bumped); published tags are
 *      immutable — a tag that already exists in the registry is pulled
 *      and reused BEFORE any build starts (tag-exists skip), and a release
 *      where every tag exists skips the build phase entirely.
 *   6. Produce solution/ INSIDE the built image (solutions.ts), in parallel
 *      across problems once all builds are done: hook runs solveProblem()
 *      and diffs the workspace; ref checks the pinned commit out and diffs;
 *      patch is copied and `git apply --check`ed. Every solution kind ships
 *      the same git-apply solve.sh. Under --no-build no solution/ dirs are
 *      emitted at all — solutions only exist as products of the exact image
 *      trials will run.
 *   7. manifest.json pins refs, final specs, minReplayScore, and solution
 *      provenance (solutionKind, plus the legacy solutionRef/"hook"
 *      sentinel for current control-plane readers).
 */

import * as fs from "fs";
import * as path from "path";
import { loadProblemsFromDirectory, Problem } from "@hyperfocal/env-base";
import type { TaskSpec } from "@hyperfocal/env-base";
import { loadEnvRepoConfig } from "../config/yaml-config.js";
import { buildImages, type ImageBuildTarget } from "./build/bake.js";
import { emitEnvReadme, type ReadmeTaskRow } from "./emit/readme.js";
import { emitTaskSources } from "./emit/taskSource.js";
import { emitTaskDir } from "./emit/taskDir.js";
import { renderDockerfile } from "./render/dockerfile.js";
import { renderEnvironmentDockerfile } from "./render/environmentDockerfile.js";
import { loadPackagingEnvModule } from "./runtime/hookBuild.js";
import {
  planSolutions,
  produceSolution,
  type SolutionKind,
  type SolutionPlan,
} from "./solutions.js";
import { secretScan } from "./staging/secretScan.js";
import {
  git,
  resolveRef,
  stageBuildContext,
  tokenlessUrl,
} from "./staging/stage.js";
import {
  buildDefaultTaskSpec,
  computeSpecHash,
  validateTaskSpec,
} from "./taskSpec.js";

// Re-exported for checks/doctor.ts (historical import surface).
export { resolveRef };

export interface HarborPackageOptions {
  envRepo: string;
  problems?: string[];
  outDir?: string;
  build: boolean;
  packagesOverlay: boolean;
  imagePrefix?: string;
  /**
   * The task's SOURCE branch on the env repo — the original task branch
   * (e.g. "gqa", "problem/seq-edit"), part of task identity (env x branch
   * x problem; docs/11-build-finalization-fourth-draft/3-release-model §2).
   * When omitted it is derived per problem from the checked-out branch:
   * a `harbor-release/<branch>/<problem>` overlay ref maps back to
   * <branch>; any other branch names itself; a detached HEAD falls back to
   * "main" with a loud warning (builders packaging overlay refs should
   * pass --branch explicitly).
   */
  branch?: string;
  /**
   * Rebuild counter (default 0): a spec-hash input whose bump mints a NEW
   * image name for unchanged content — the platform's Rebuild action
   * (release-model §7). Counter > 0 also appends a visible `-r<n>` tag
   * suffix. Tags are never overwritten.
   */
  rebuildCounter?: number;
  /** Shared fleet base image ref (see templates.ts DockerfileParams.baseImage). */
  baseImage?: string;
  /**
   * Registry prefix to push built images to, e.g.
   * `<account>.dkr.ecr.<region>.amazonaws.com/emulated-tasks`. The image is
   * tagged `<prefix>/<env>:<problem>-<sha>` (so task.toml and the manifest
   * reference the pullable URI, not a local-only tag) and pushed after each
   * build. ECR hosts are logged into automatically via the ambient AWS
   * credentials (instance role on builders). The target repository must
   * already exist — builders deliberately cannot create repositories; the
   * backend does that before dispatch.
   */
  push?: string;
  /**
   * Emit per-task source (default true): environment/source/ inside EVERY
   * task dir — the .git-stripped staged build context plus repo.bundle and
   * per-submodule git bundles — and the standalone environment/Dockerfile
   * that rebuilds from it (see emit/taskSource.ts and
   * render/environmentDockerfile.ts). `--no-source-bundle` opts out: tasks
   * then ship pin-only (no source, no Dockerfile — harbor runs them off
   * the task.toml docker_image pin alone). Deliberately OUTSIDE the spec
   * hash: source documents the build, it never changes what gets baked, so
   * toggling it must not rotate tags.
   *
   * Precedence: an EXPLICIT value here (the CLI only sets one when
   * --no-source-bundle was actually passed) wins over hyperfocal.yaml's
   * packaging.sourceBundle; when this is undefined the yaml key decides;
   * absent both, source is emitted. The yaml key exists because builders
   * invoke the env's pinned `harbor package` with no flags — it is how a
   * heavy-history env (decision 7.12) tells the BUILD path to skip source
   * emission and its 100 MB per-file ship gate.
   */
  sourceBundle?: boolean;
  /** Sink for progress lines; defaults to prefixed console.log. */
  logger?: (msg: string) => void;
}

interface TaskManifestEntry {
  taskDir: string;
  problemId: string;
  /** The task's source branch (task identity is env x branch x problem). */
  branch: string;
  /** branch with "/" flattened to "-" — as used in dir and image names. */
  branchSlug: string;
  /** The rebuild counter the spec hash was computed with (0 = first build). */
  rebuildCounter: number;
  imageTag: string;
  /** Second tag component: content hash of the final spec (see taskSpec.ts). */
  specHash: string;
  /**
   * The final (post-hook) TaskSpec the task was rendered from. Downstream
   * consumers (e.g. the control plane sizing release runs) read compute
   * requirements from here instead of re-parsing task.toml.
   */
  spec: TaskSpec;
  /**
   * The replay reward floor this task must meet (mirror of task.toml
   * [metadata] min_replay_score) — lets the control plane judge platform
   * solution replays without unpacking the bundle. null when the problem
   * declares NO reference solution: there is no replay gate to state
   * (V1-F5 — the old unconditional 1.0 was a misleading dormant gate).
   */
  minReplayScore: number | null;
  /**
   * Under contract v2 the packaging commit is every problem's state, so
   * these always equal the packaging branch/commit. Kept (rather than
   * dropped) so downstream manifest readers don't churn.
   */
  problemStateRef: string;
  problemStateCommit: string;
  /** How the reference solution was produced (contract v2 ladder). */
  solutionKind: SolutionKind;
  /**
   * Legacy provenance field the current control-plane reader keys on: the
   * resolved ref string for kind=ref, the literal "hook" sentinel for
   * kind=hook, null otherwise. New readers should use solutionKind.
   */
  solutionRef: string | null;
  solutionCommit: string | null;
  /** Repo-relative committed patch path (kind=patch only). */
  solutionPatch: string | null;
  solutionPatchBytes: number | null;
}

/** Overlay-ref prefix (release-model §3): the platform packages permanent
 * `harbor-release/<original-branch>/<problem-id>` branches it owns. */
const OVERLAY_REF_PREFIX = "harbor-release/";

/**
 * Branch slug: the branch name with "/" flattened to "-", restricted to
 * docker-tag- and directory-safe characters. Used in the task-dir name
 * (<env>__<branch-slug>__<problem>) and the image tag.
 */
export function slugifyBranch(branch: string): string {
  const slug = branch.replace(/\//g, "-").replace(/[^A-Za-z0-9_.-]/g, "-");
  if (!slug || /^[.-]/.test(slug)) {
    throw new Error(
      `branch ${JSON.stringify(branch)} does not slugify to a usable name ` +
        `(got ${JSON.stringify(slug)})`
    );
  }
  return slug;
}

/**
 * The task's SOURCE branch for one problem. Explicit --branch wins;
 * otherwise the checked-out branch is interpreted: a
 * harbor-release/<branch>/<problem> overlay ref maps back to <branch>
 * (per problem, because the overlay is per (branch, problem) — e.g.
 * harbor-release/problem/base-speedup/base-ranker packaging base-ranker
 * yields problem/base-speedup); any other branch names itself; null
 * (detached HEAD) falls back to "main" — the caller warns loudly, and
 * builders packaging overlay refs pass --branch explicitly.
 */
export function deriveTaskBranch(
  packagingBranch: string | null,
  problemId: string,
  explicit?: string
): string {
  if (explicit) return explicit;
  if (!packagingBranch) return "main";
  if (packagingBranch.startsWith(OVERLAY_REF_PREFIX)) {
    const rest = packagingBranch.slice(OVERLAY_REF_PREFIX.length);
    const suffix = `/${problemId}`;
    return rest.endsWith(suffix) ? rest.slice(0, -suffix.length) : rest;
  }
  return packagingBranch;
}

export function findEnvRepo(explicit?: string): string {
  let dir = path.resolve(explicit || process.cwd());
  if (explicit) {
    if (!fs.existsSync(path.join(dir, "hyperfocal.yaml"))) {
      throw new Error(`--env-repo ${dir} has no hyperfocal.yaml`);
    }
    return dir;
  }
  while (dir !== "/") {
    if (fs.existsSync(path.join(dir, "hyperfocal.yaml"))) return dir;
    dir = path.dirname(dir);
  }
  throw new Error(
    "No hyperfocal.yaml found walking up from cwd; pass --env-repo <path>"
  );
}

export async function packageRelease(
  opts: HarborPackageOptions
): Promise<void> {
  const log =
    opts.logger ?? ((msg: string) => console.log(`[harbor:package] ${msg}`));
  if (opts.push && !opts.build) {
    throw new Error("--push requires building (remove --no-build)");
  }
  const envRepo = findEnvRepo(opts.envRepo);
  // Typed hyperfocal.yaml load with STRICT validation of the packaging-time
  // blocks — a typo'd network mode or compute field fails here, before any
  // heavy work, and can never ship a mis-postured task.
  const config = loadEnvRepoConfig(envRepo);
  const envName: string = config.environment.name || path.basename(envRepo);
  const commit = git(envRepo, ["rev-parse", "HEAD"]);
  const shortSha = commit.slice(0, 8);
  // The manifest's problemStateRef: the packaging branch when there is one,
  // otherwise the commit itself (detached-HEAD builders).
  const packagingBranch = git(envRepo, ["rev-parse", "--abbrev-ref", "HEAD"]);
  const packagingRef = packagingBranch === "HEAD" ? commit : packagingBranch;
  // Task identity is branch-defined (env x branch x problem): the SOURCE
  // branch enters the task-dir name, [metadata], the image tag, and the
  // spec hash. See deriveTaskBranch for the resolution order.
  const headBranch = packagingBranch === "HEAD" ? null : packagingBranch;
  if (!opts.branch && headBranch === null) {
    console.warn(
      `[harbor:package] WARNING: detached HEAD and no --branch given — task ` +
        `branch defaults to "main". If this packaging is for a task branch ` +
        `or a harbor-release/* overlay ref, pass --branch <original-branch>.`
    );
  }
  const rebuildCounter = opts.rebuildCounter ?? 0;
  if (!Number.isInteger(rebuildCounter) || rebuildCounter < 0) {
    throw new Error(
      `--rebuild-counter must be a non-negative integer (got ${opts.rebuildCounter})`
    );
  }

  const allProblems: Problem[] = loadProblemsFromDirectory(
    path.join(envRepo, "environment")
  );
  const selected = opts.problems?.length
    ? allProblems.filter((p) => opts.problems!.includes(p.id))
    : allProblems;
  if (selected.length === 0) {
    throw new Error(
      `No matching problems (have: ${allProblems.map((p) => p.id).join(", ")})`
    );
  }

  const outDir = path.resolve(opts.outDir || path.join(envRepo, "harbor-dist"));
  fs.mkdirSync(outDir, { recursive: true });
  log(`Packaging ${envName}@${shortSha} -> ${outDir}`);

  // The environment's compiled module, built host-side: needed to discover
  // the optional packageProblem() spec hook AND whether solveProblem() owns
  // the solutions, and doubles as an early compile check before any docker
  // work.
  const envModule = await loadPackagingEnvModule(
    envRepo,
    outDir,
    config.paths.environmentDist,
    log
  );
  if (envModule?.packageProblem) {
    log("environment declares a packageProblem() hook — applying it per problem");
  }
  const envDefinesSolveHook = typeof envModule?.solveProblem === "function";
  if (envDefinesSolveHook) {
    log("environment implements solveProblem() — the hook owns every solution");
  }

  // Plan every problem's solution up front (ladder walk, conflict check,
  // ref pinning, patch-committed check) so we fail before staging or builds.
  const plans = planSolutions({
    envRepo,
    problems: selected,
    envDefinesSolveHook,
    packagingCommit: commit,
  });
  if (!opts.build) {
    log(
      "--no-build: solutions are produced in the built image; --no-build " +
        "emits none (no solution/ dirs; replay is not runnable from this output)"
    );
  }

  const contextDir = path.join(outDir, "build-context");
  const cloneDir = stageBuildContext(
    envRepo,
    contextDir,
    [...plans.values()]
      .map((p) => p.solutionRef)
      .filter((r): r is string => r !== null),
    opts.packagesOverlay
  );
  // Once, after staging (overlay included): the context never changes
  // between problems under v2 — there is no per-problem checkout.
  secretScan(contextDir, log);

  // Venue sizing does NOT translate into task.toml: compute.instanceType is
  // the platform's rollout box, and harbor tasks only understand cpus/
  // memoryMb. An env that sized its instance but not its requirements would
  // silently ship with harbor's small defaults (bit sc-gambench: m5.xlarge
  // intent shipped as 2 cpu / 4096 MB) — warn loudly.
  const instanceType = config.compute?.instanceType;
  const yamlCpus = config.compute?.cpus;
  const yamlMemoryMb = config.compute?.memoryMb;
  if (instanceType && (yamlCpus === undefined || yamlMemoryMb === undefined)) {
    const missing = [
      ...(yamlCpus === undefined ? ["compute.cpus"] : []),
      ...(yamlMemoryMb === undefined ? ["compute.memoryMb"] : []),
    ].join(" and ");
    const bar = "!".repeat(76);
    console.warn(
      `\n${bar}\n` +
        `[harbor:package] WARNING: hyperfocal.yaml declares compute.instanceType: ` +
        `${instanceType}\n` +
        `but not ${missing}. instanceType is platform venue sizing and NEVER\n` +
        `reaches task.toml — packaged tasks will fall back to harbor defaults\n` +
        `(cpus: 2, memory_mb: 4096), which is almost certainly not what the\n` +
        `instance choice intended. Declare compute.cpus and compute.memoryMb.\n` +
        `${bar}\n`
    );
  }

  const workspacePath = config.paths.workspace;
  const registryPrefix = opts.push?.replace(/\/+$/, "");

  // packaging.sharedSetupLayer (design docs/7-build-optimization §2): the
  // env owner asserted setupProblem is reset-complete, so every problem's
  // Dockerfile gets the two-RUN layout whose warm layer runs the CANONICAL
  // problem's setup. Canonical = first problem in problems.yaml order over
  // ALL problems — never the packaged subset — so partial packagings share
  // the same warm layer. Toggling this changes each Dockerfile's text and
  // therefore rotates the env's spec hashes (deliberate: the tag names the
  // baked content, which now includes the canonical lower layer).
  const canonicalProblemId = config.packaging?.sharedSetupLayer
    ? allProblems[0].id
    : undefined;
  if (canonicalProblemId) {
    log(
      `packaging.sharedSetupLayer: on — two-RUN Dockerfiles sharing a warm ` +
        `setup layer (canonical problem: ${canonicalProblemId})`
    );
  }

  // Phase 1 — render + emit everything build-independent, per problem:
  // final spec, Dockerfile (written into the shared context), spec hash,
  // task dir. Collected as the build-target list the bake phase consumes.
  interface PlannedTask {
    problem: Problem;
    plan: SolutionPlan;
    branch: string;
    branchSlug: string;
    taskName: string;
    taskDir: string;
    spec: TaskSpec;
    specHash: string;
    imageTag: string;
    minReplayScore: number | null;
    dockerfilePath: string;
    patchBytes: number | null;
  }
  const planned: PlannedTask[] = [];
  for (const problem of selected) {
    const plan = plans.get(problem.id)!;
    const branch = deriveTaskBranch(headBranch, problem.id, opts.branch);
    const branchSlug = slugifyBranch(branch);
    // Branch-aware task name (release-model §2): env x branch x problem is
    // the identity — gqa-decode-m1 alone has seven task branches whose
    // problem id is all "default", so <env>__<problem> cannot name them.
    const taskName = `${envName}__${branchSlug}__${problem.id}`;
    const taskDir = path.join(outDir, "tasks", taskName);

    // Default spec from hyperfocal.yaml, then the env's optional
    // packageProblem() amendments — validated as strictly as hand-written
    // yaml, since a buggy hook must not ship a mis-postured task.
    let spec = buildDefaultTaskSpec(problem.id, config);
    if (envModule?.packageProblem) {
      spec = validateTaskSpec(
        await envModule.packageProblem(problem.id, spec),
        problem.id
      );
      log(`  ${taskName}: packageProblem() amended the spec`);
    }

    const dockerfileContent = renderDockerfile({
      problemId: problem.id,
      imageEnv: spec.imageEnv,
      dockerfileLines: spec.dockerfileLines,
      baseImage: opts.baseImage,
      workspacePath,
      canonicalProblemId,
    });
    // No solution declared => no replay gate at all (V1-F5): the manifest
    // and task.toml [metadata] simply omit the floor instead of stamping a
    // misleading dormant 1.0 on a task nothing can replay.
    const minReplayScore =
      plan.kind === "none" ? null : (problem.minReplayScore ?? 1.0);
    // <branch-slug>-<problem>-<commit8>-<spec8>[-r<n>]: the spec hash makes
    // the tag content-true. The packaging commit alone stopped identifying
    // the image once hooks and spec edits could change what gets baked
    // without a new commit — published images are reused only on a full
    // tag match and never overwritten. The branch enters both the visible
    // name and the hash (branch-defined identity); the rebuild counter
    // enters the hash and, when > 0, the visible -r<n> suffix.
    const specHash = computeSpecHash({
      spec,
      dockerfile: dockerfileContent,
      instruction: problem.prompt,
      // Contract v2: the packaging commit is the problem state.
      problemStateCommit: commit,
      branch,
      rebuildCounter,
      minReplayScore,
      solutionKind: plan.kind,
      workspacePath,
    });

    const rebuildSuffix = rebuildCounter > 0 ? `-r${rebuildCounter}` : "";
    const imageTag = `${registryPrefix || opts.imagePrefix || "hyperfocal"}/${envName}:${branchSlug}-${problem.id}-${shortSha}-${specHash}${rebuildSuffix}`;

    emitTaskDir({
      taskDir,
      envName,
      branch,
      problemId: problem.id,
      prompt: problem.prompt,
      spec,
      imageTag,
      workspacePath,
      minReplayScore,
      buildTimeoutSec: config.packaging?.buildTimeoutSec,
    });

    // The buildable Dockerfile lives with the context (one per problem).
    // Rendered above (its content is part of the spec hash).
    const dockerfilePath = path.join(contextDir, `Dockerfile.${problem.id}`);
    fs.writeFileSync(dockerfilePath, dockerfileContent);

    planned.push({
      problem,
      plan,
      branch,
      branchSlug,
      taskName,
      taskDir,
      spec,
      specHash,
      imageTag,
      minReplayScore,
      dockerfilePath,
      patchBytes: null,
    });
  }

  // Phase 1b — per-task source + the task's standalone environment/
  // Dockerfile (Wave 1 step 1). Each task carries its OWN
  // environment/source/ copy (git-graft machinery unchanged, destination
  // moved from the output root into every task) and the from-source
  // recipe IS environment/Dockerfile now — the retired FROM-prebuilt stub
  // never existed for harbor, which reads the pin from task.toml. Runs
  // after task-dir emission (the staged context already carries every
  // Dockerfile.<problem>) and BEFORE any build, so the >100 MB-per-file
  // guard fails the package in seconds, not after an hours-long bake.
  // Resolution: explicit CLI value (only set when --no-source-bundle was
  // passed) > packaging.sourceBundle in hyperfocal.yaml > default emit —
  // the yaml key is the flagless builder path's only opt-out (see
  // HarborPackageOptions.sourceBundle).
  const emitSourceBundle =
    opts.sourceBundle ?? config.packaging?.sourceBundle ?? true;
  if (!emitSourceBundle && opts.sourceBundle === undefined) {
    log(
      "packaging.sourceBundle: false — pin-only tasks (no environment/" +
        "source/, no environment/Dockerfile; 100 MB source ship gate not " +
        "applicable)"
    );
  }
  if (emitSourceBundle) {
    const graft = emitTaskSources({
      contextDir,
      cloneDir,
      outDir,
      tasks: planned.map((t) => ({
        taskDir: t.taskDir,
        problemId: t.problem.id,
      })),
      logger: log,
    });
    for (const t of planned) {
      fs.writeFileSync(
        path.join(t.taskDir, "environment", "Dockerfile"),
        renderEnvironmentDockerfile({
          problemId: t.problem.id,
          imageEnv: t.spec.imageEnv,
          dockerfileLines: t.spec.dockerfileLines,
          workspacePath,
          graft,
        })
      );
    }
    log(
      `Standalone environment/Dockerfile emitted for ${planned.length} task(s)`
    );
  }

  if (opts.build) {
    // Phase 2 — build every missing image in parallel via buildx bake.
    // Already-published immutable tags are pulled and reused instead
    // (tag-exists skip); an all-present release skips building entirely.
    const targets: ImageBuildTarget[] = planned.map((t) => ({
      problemId: t.problem.id,
      imageTag: t.imageTag,
      dockerfileName: path.basename(t.dockerfilePath),
      dockerfilePath: t.dockerfilePath,
    }));
    buildImages({
      targets,
      contextDir,
      bakeDir: path.join(outDir, "bake"),
      baseImage: opts.baseImage,
      registryPrefix,
      // packaging.bakeNeedsGpu: serial legacy-builder builds so RUN steps
      // see the daemon's nvidia default-runtime (S3-F2 — BuildKit ignores
      // it). The control plane routes such releases to a GPU builder off
      // the same yaml key.
      bakeNeedsGpu: config.packaging?.bakeNeedsGpu,
      log,
    });

    // Phase 3 — solution production, in parallel across problems (each is
    // an independent docker run against its own image). Runs AFTER the
    // build/push phase on purpose: when an immutable ECR tag was reused,
    // the local tag now points at the PUBLISHED image, so the solution
    // comes from exactly what trials will run.
    await Promise.all(
      planned
        .filter((t) => t.plan.kind !== "none")
        .map(async (t) => {
          t.patchBytes = await produceSolution({
            plan: t.plan,
            problemId: t.problem.id,
            taskName: t.taskName,
            imageTag: t.imageTag,
            workspacePath,
            taskDir: t.taskDir,
            cloneDir,
          });
        })
    );
  }

  const manifest: TaskManifestEntry[] = [];
  for (const t of planned) {
    if (t.plan.kind === "none") {
      log(
        `  ${t.taskName}: no reference solution declared — no solution/ dir ` +
          `(replay not applicable; validate via agent runs)`
      );
    }
    manifest.push({
      taskDir: path.relative(outDir, t.taskDir),
      problemId: t.problem.id,
      branch: t.branch,
      branchSlug: t.branchSlug,
      rebuildCounter,
      imageTag: t.imageTag,
      specHash: t.specHash,
      spec: t.spec,
      minReplayScore: t.minReplayScore,
      problemStateRef: packagingRef,
      problemStateCommit: commit,
      solutionKind: t.plan.kind,
      // Legacy sentinel preserved exactly: "hook" for hook solutions (no
      // git ref ever holds the gold state, solutionCommit stays null), the
      // resolved ref for ref solutions, null for patch/none.
      solutionRef: t.plan.kind === "hook" ? "hook" : t.plan.solutionRef,
      solutionCommit: t.plan.solutionCommit,
      solutionPatch: t.plan.solutionPatch,
      solutionPatchBytes: t.patchBytes,
    });
  }

  // Per-env README at the output root (step 7): run instructions for a
  // customer engineer, the delete-the-pin from-source path, the oracle
  // agent for solutions, and the [Internal] task-status table stamped with
  // what the packager knows locally — the control plane's publish path
  // re-renders it with live platform state (renderEnvReadme is exported
  // for exactly that).
  const verifierPolicies = planned
    .map((t) => t.spec.network?.verifier)
    .filter((p) => p?.mode === "allowlist");
  emitEnvReadme(outDir, {
    envName,
    sourceBundle: emitSourceBundle,
    buildTimeoutSec: config.packaging?.buildTimeoutSec ?? 2700,
    tasks: planned.map(
      (t): ReadmeTaskRow => ({
        name: t.taskName,
        gpus: t.spec.compute?.gpus ?? 0,
        hasSolution: t.plan.kind !== "none",
        status: opts.build ? "image built" : "image not yet built",
        checkResults: "—",
      })
    ),
    judgeAllowedHosts: [
      ...new Set(verifierPolicies.flatMap((p) => p?.allowedHosts ?? [])),
    ],
  });

  fs.writeFileSync(
    path.join(outDir, "manifest.json"),
    JSON.stringify(
      {
        environment: envName,
        envRepo: tokenlessUrl(
          (() => {
            try {
              return git(envRepo, ["remote", "get-url", "origin"]);
            } catch {
              return envRepo;
            }
          })()
        ),
        commit,
        generatedAt: new Date().toISOString(),
        packagesOverlay: opts.packagesOverlay,
        baseImage: opts.baseImage ?? null,
        pushedTo: opts.push ?? null,
        tasks: manifest,
      },
      null,
      2
    ) + "\n"
  );

  log(`Done. ${manifest.length} task(s) in ${path.join(outDir, "tasks")}`);
  log(`Manifest: ${path.join(outDir, "manifest.json")}`);
}
