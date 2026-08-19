/**
 * Per-env README emission (Wave 1 step 7,
 * docs/11-build-finalization-fourth-draft/4-implementation-plan.md §4).
 *
 * The reader is an engineer at a customer lab who just cloned the released
 * repo, knows harbor, and knows nothing about our internals — every
 * instruction says what to do and when it applies, in their vocabulary.
 * The body wording follows the owner-reviewed skeleton in the plan doc
 * (the delete-the-pin sentence itself is still marked owner-pending there;
 * this renderer is the one place to touch when the final sentence lands).
 *
 * Emitted at the package output root by package.ts; the control plane's
 * publish path (Wave 2) re-renders it with live platform state via the
 * exported renderEnvReadme — the packager fills the [Internal] task-status
 * table with what it can know locally (built vs not-yet-built, solution
 * presence) and publish overwrites those rows with real check results.
 */

import * as fs from "fs";
import * as path from "path";

export interface ReadmeTaskRow {
  /** Task directory name: <env>__<branch-slug>__<problem>. */
  name: string;
  /** task.toml gpus value — GPU tasks get called out in the notes. */
  gpus: number;
  /** Whether the task ships a solution/ directory. */
  hasSolution: boolean;
  /**
   * Status cell for the [Internal] table. The packager stamps what it
   * knows ("image built" / "image not yet built"); the publish path
   * replaces it with platform state.
   */
  status: string;
  /** Check-results cell; "—" until platform checks have run. */
  checkResults: string;
}

export interface RenderEnvReadmeParams {
  envName: string;
  /**
   * Whether this packaging emitted the per-task source tree (default true).
   * When false (packaging.sourceBundle: false or --no-source-bundle — the
   * heavy-git-history envs of decision 7.12), the tasks ship pin-only and
   * the from-source section is replaced by an honest note: there is no
   * environment/source/ or environment/Dockerfile to build, so the
   * delete-the-pin instructions would point the reader at nothing.
   */
  sourceBundle?: boolean;
  /** task.toml [environment].build_timeout_sec, echoed as guidance. */
  buildTimeoutSec: number;
  tasks: ReadmeTaskRow[];
  /**
   * Verifier allowlist hosts when this env's verifier ends with an
   * LLM-judge step (from packaging.network.verifier allowlist) — drives
   * the judge-env note. Empty/undefined = no note.
   */
  judgeAllowedHosts?: string[];
}

function formatTimeout(sec: number): string {
  if (sec % 3600 === 0) {
    const h = sec / 3600;
    return `${h} hour${h === 1 ? "" : "s"}`;
  }
  const min = Math.round(sec / 60);
  return `${min} minutes`;
}

export function renderEnvReadme(p: RenderEnvReadmeParams): string {
  const gpuTasks = p.tasks.filter((t) => t.gpus > 0).map((t) => t.name);
  const statusRows = p.tasks
    .map((t) => `| ${t.name} | ${t.status} | ${t.checkResults} |`)
    .join("\n");

  const notes: string[] = [];
  if (gpuTasks.length > 0) {
    notes.push(
      `- GPU tasks (${gpuTasks.join(", ")}): these declare GPUs in ` +
        `task.toml. Harbor's plain Docker runner refuses GPU tasks but ` +
        `supports backends such as Modal.`
    );
  }
  if (p.judgeAllowedHosts && p.judgeAllowedHosts.length > 0) {
    notes.push(
      `- This environment's verifier ends with an LLM-judge step. Export ` +
        `OPENROUTER_API_KEY in the environment where you run harbor and ` +
        `allow network access to ${p.judgeAllowedHosts.join(", ")}.`
    );
  }
  const notesSection =
    notes.length > 0 ? `\n## Notes for specific setups\n\n${notes.join("\n")}\n` : "";

  // From-source section — only when this packaging actually shipped source.
  // Pin-only packagings (sourceBundle false) carry no environment/source/ or
  // environment/Dockerfile, so delete-the-pin instructions would point the
  // reader at nothing; say why honestly instead.
  const fromSourceSection =
    p.sourceBundle !== false
      ? `If the pull fails with an authorization error, your team does not have registry access but you can build it from source (environment/source/). Remove the \`docker_image\` line from the task's task.toml and run the same command — harbor then builds environment/Dockerfile locally (first build downloads public base images and can take up to ${formatTimeout(p.buildTimeoutSec)}; cached after that):

    sed -i '/^docker_image/d' tasks/<task-name>/task.toml
    harbor run tasks/<task-name> --agent claude-code --model <model>`
      : `Source omitted: this repository's git history exceeds public-hosting file limits, so these tasks ship without environment/source/ or environment/Dockerfile and cannot be built from source — use the prebuilt image the \`docker_image\` line names. If the pull fails with an authorization error, ask us for registry access.`;
  const notYetBuiltHint =
    p.sourceBundle !== false
      ? "build from source, or check back"
      : "check back";

  return `# ${p.envName} — harbor tasks

Each directory under tasks/ is a self-contained harbor task.

Each task.toml names a ready-made image in our registry (the \`docker_image\` line). If your team has been given registry access, harbor pulls it automatically and there is nothing to set up:

    harbor run tasks/<task-name> --agent claude-code --model <model>

${fromSourceSection}

## Running the solution

Tasks that include a solution/ directory ship the reference fix as solution/solution.patch. To apply and grade it, run the task with harbor's built-in oracle agent (it executes solution/solve.sh instead of an AI agent):

    harbor run tasks/<task-name> --agent oracle

## [Internal] Task status

| task | status | check results |
|---|---|---|
${statusRows}

("image not yet built" means the ready-made image is not in the registry yet — ${notYetBuiltHint}.)
${notesSection}`;
}

/** Write README.md at the package output root. */
export function emitEnvReadme(outDir: string, p: RenderEnvReadmeParams): void {
  fs.writeFileSync(path.join(outDir, "README.md"), renderEnvReadme(p));
}
