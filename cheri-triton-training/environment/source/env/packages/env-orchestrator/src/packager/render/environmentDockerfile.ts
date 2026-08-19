/**
 * The shipped `environment/Dockerfile` renderer — the task's ONE Dockerfile
 * (owner decision 17, docs/11-build-finalization-fourth-draft/0-overview.md):
 * a standalone recipe that rebuilds the task image from PUBLIC base images
 * using only the task's own `environment/source/` directory — no env-repo
 * access, no private registry.
 *
 * How it is used: task.toml KEEPS the ECR `docker_image` pin from the first
 * publish, and harbor ignores environment/Dockerfile whenever that pin is
 * set (`should_use_prebuilt_docker_image`, harbor
 * src/harbor/environments/definition.py). A customer without registry
 * access deletes the one `docker_image` line from task.toml and reruns —
 * harbor then builds THIS file with the task's environment/ directory as
 * build context (harbor's docker provider passes environment_dir as the
 * compose build context), which is why every COPY below is
 * `source/`-prefixed. The old 2-line FROM-prebuilt Dockerfile stub is
 * retired: harbor reads the prebuilt pin from task.toml, never from a
 * Dockerfile.
 *
 * The recipe is assembled from the SAME single sources as the real build:
 * the public-base fallback toolchain (taskBase.ts) and the provision RUN
 * steps (dockerfile.ts provisionRunSteps) — only the "get the repo into the
 * image" step differs. The real build COPYs the staged clone WITH its live
 * git state; the released repo cannot carry live .git dirs (they'd publish
 * as empty gitlinks — shipService fails the ship on them, finding F-7), so
 * this build reconstructs the git state from bundle files:
 *
 *   - worktree bytes:  COPY source/env/  (the .git-stripped staged context)
 *   - superproject:    fetched from source/repo.bundle (`git bundle create
 *                      --all HEAD` of the staged clone) into a fresh
 *                      `git init` — every branch survives, which setups
 *                      rely on (e.g. nano-vllm checks problem branches out
 *                      during setup)
 *   - submodules:      staging materializes each submodule as a real clone
 *                      whose objects live under the superproject's
 *                      .git/modules/<path>; the superproject bundle carries
 *                      only gitlink SHAs, so each initialized submodule
 *                      ships its own bundle (source/git-submodules/) and is
 *                      grafted back via `git init --separate-git-dir` +
 *                      fetch + pinned HEAD — reproducing the exact layout
 *                      (worktree .git pointer file, objects under
 *                      .git/modules, detached HEAD at the pin)
 *
 * Indexes are rebuilt with `git read-tree HEAD`, mirroring the packager's
 * deterministic staging (stage.ts normalizeGitState), and the graft shares
 * one RUN with the provision steps so the trust-lockdown chmod over .git
 * never duplicates layer content (decision D1 lineage).
 *
 * Fidelity bar — auditable and runnable, NOT hermetic: this build produces
 * a FUNCTIONALLY EQUIVALENT image (same repo bytes, same refs, same setup
 * steps), but base image digests, distro packages, and npm-resolved
 * dependencies move with time, so it is not byte-identical to the pinned
 * prebuilt image the release's checks ran against. The README carries this
 * caveat now that the emitted Dockerfile itself is comment-free
 * (zero-comment rule, step 3 — stripDockerfileComments at the exit).
 */

import * as path from "path";
import type { StagedSubmoduleInfo, GitGraftInfo } from "../emit/taskSource.js";
import {
  IMAGE_ENV_ROOT,
  provisionRunSteps,
  renderEnvSpecificBlock,
  renderRun,
  stripDockerfileComments,
} from "./dockerfile.js";
import { TASK_BASE_FALLBACK_RECIPE } from "./taskBase.js";

/** Where the bundles land in the image; removed at the end of the RUN (the
 * bytes stay in the COPY layer — cosmetic, the final filesystem is clean). */
const BUNDLE_DIR = "/hyperfocal/git-bundles";

/** Scratch ref the bundle's HEAD is fetched into so a detached staged HEAD's
 * objects always transfer; deleted once HEAD is pinned. */
const HEAD_SCRATCH_REF = "refs/hyperfocal/bundle-head";

/**
 * The build-context-relative dir the task's source tree lives in: the
 * shipped Dockerfile sits at environment/Dockerfile and harbor builds with
 * environment/ as context, so everything it COPYs lives under ./source/.
 */
const SOURCE_DIR = "source";

export interface EnvironmentDockerfileParams {
  problemId: string;
  imageEnv?: Record<string, string>;
  dockerfileLines?: string[];
  workspacePath?: string;
  graft: GitGraftInfo;
}

function q(value: string): string {
  return `"${value.replace(/(["\\$`])/g, "\\$1")}"`;
}

/** The superproject git graft, as shell steps for the single RUN. */
function superprojectGraftSteps(graft: GitGraftInfo): string[] {
  const steps = [
    "git init -q .",
    `git fetch -q ${BUNDLE_DIR}/repo.bundle "+refs/*:refs/*" "+HEAD:${HEAD_SCRATCH_REF}" --update-head-ok`,
    graft.headBranch
      ? `git symbolic-ref HEAD refs/heads/${graft.headBranch}`
      : `git update-ref --no-deref HEAD ${graft.headCommit}`,
    "git read-tree HEAD",
    `git update-ref -d ${HEAD_SCRATCH_REF}`,
  ];
  if (graft.originUrl) {
    steps.push(`git remote add origin ${q(graft.originUrl)}`);
  }
  return steps;
}

/** One submodule's graft: reconstruct .git/modules/<path> + the worktree
 * .git pointer, fetch its bundle, pin HEAD, register it in the parent. */
function submoduleGraftSteps(sub: StagedSubmoduleInfo): string[] {
  // Real nested layout: the child's git dir lives under the ENCLOSING
  // repo's git dir (.git/modules/<parent>/modules/<rel> for nested subs).
  const parentGitDir = sub.parent
    ? `${IMAGE_ENV_ROOT}/.git/modules/${sub.parent}`
    : `${IMAGE_ENV_ROOT}/.git`;
  const relPath = sub.parent
    ? sub.path.slice(sub.parent.length + 1)
    : sub.path;
  const gitDir = `${parentGitDir}/modules/${relPath}`;
  const worktree = `${IMAGE_ENV_ROOT}/${sub.path}`;
  const parentWorktree = sub.parent
    ? `${IMAGE_ENV_ROOT}/${sub.parent}`
    : IMAGE_ENV_ROOT;
  const bundle = `${BUNDLE_DIR}/git-submodules/${sub.bundleFile}`;
  return [
    `mkdir -p ${path.posix.dirname(gitDir)}`,
    `git init -q --separate-git-dir ${gitDir} ${worktree}`,
    `git -C ${worktree} fetch -q ${bundle} "+refs/*:refs/*" "+HEAD:${HEAD_SCRATCH_REF}" --update-head-ok`,
    `git -C ${worktree} update-ref --no-deref HEAD ${sub.headCommit}`,
    `git -C ${worktree} read-tree HEAD`,
    `git -C ${worktree} update-ref -d ${HEAD_SCRATCH_REF}`,
    ...(sub.url
      ? [`git -C ${parentWorktree} config submodule.${sub.name}.url ${q(sub.url)}`]
      : []),
    `git -C ${parentWorktree} config submodule.${sub.name}.active true`,
  ];
}

export function renderEnvironmentDockerfile(
  p: EnvironmentDockerfileParams
): string {
  const graftSteps = [
    ...superprojectGraftSteps(p.graft),
    ...p.graft.submodules.flatMap(submoduleGraftSteps),
    `rm -rf ${BUNDLE_DIR}`,
  ];
  const runSteps = [
    ...graftSteps,
    ...provisionRunSteps(p.problemId, p.workspacePath),
  ];
  const copySubBundles =
    p.graft.submodules.length > 0
      ? `COPY ${SOURCE_DIR}/git-submodules/ ${BUNDLE_DIR}/git-submodules/\n`
      : "";

  return stripDockerfileComments(`${TASK_BASE_FALLBACK_RECIPE}${renderEnvSpecificBlock(p)}
COPY ${SOURCE_DIR}/env/ /hyperfocal/env/
COPY ${SOURCE_DIR}/repo.bundle ${BUNDLE_DIR}/repo.bundle
${copySubBundles}WORKDIR /hyperfocal/env

${renderRun(runSteps)}

CMD ["sleep", "infinity"]
`);
}
