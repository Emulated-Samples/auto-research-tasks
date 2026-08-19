/**
 * `env-orchestrator harbor <package|doctor|validate|run|grade>` — the THIN
 * CLI face of the packager library (src/packager/): parse args, call the
 * packager's public API, exit. No packaging logic lives here, and this file
 * imports nothing of the packager but its index.
 */

import { doctor, grade, packageRelease, run, validate } from "../../packager/index.js";

const HELP = `env-orchestrator harbor — package hyperfocal envs as Harbor tasks

Usage:
  env-orchestrator harbor package [--env-repo <path>] [--problem <id>]...
      [--out <dir>] [--no-build] [--packages-overlay] [--image-prefix <pfx>]
      [--base-image <ref>] [--push <registry-prefix>] [--no-source-bundle]
      [--branch <task-branch>] [--rebuild-counter <n>]
    Convert an env repo into harbor task dir(s) + docker image(s). The
    packaging commit is every problem's starting state (setupProblem()
    bakes per-problem differences during the docker build). Reference
    solutions are ALL produced inside the built image, per the ladder:
    solveProblem() hook (run as root, workspace diffed) > declared
    solutionRef (pinned commit checked out of the baked clone, diffed) >
    declared solutionPatch (committed patch copied + git apply --check'd)
    > none. A hook coexisting with a declaration is a hard error. Because
    solutions come from the image, --no-build emits NO solution/ dirs.
    --packages-overlay bakes THIS orchestrator/env-base over the env's
    pinned submodules (for envs pinned to pre-harbor package versions).
    --base-image FROMs the shared fleet base (hyperfocal-task-base in ECR)
    instead of the public amazonlinux:2023 image + inline dnf toolchain.
    Task identity is branch-defined (env x branch x problem): task dirs are
    named <env>__<branch-slug>__<problem> and the branch lands in
    [metadata]. --branch names the task's SOURCE branch explicitly; when
    omitted it is derived from the checked-out branch (a
    harbor-release/<branch>/<problem> overlay ref maps back to <branch>;
    detached HEAD falls back to "main" with a warning).
    --push tags images
    <registry-prefix>/<env>:<branch-slug>-<problem>-<commit8>-<spec8> and
    pushes them after building (ECR login is automatic; repo must already
    exist). spec8 hashes the final task spec (branch and rebuild counter
    included), so spec changes mint new tags and published images are
    never overwritten. --rebuild-counter <n> (default 0) is the Rebuild
    action's input: bumping it mints a NEW image name (-r<n> suffix) for
    unchanged content instead of ever overwriting a tag.
    An env may define packageProblem(id, spec) in its EnvironmentDefinition
    to amend the default spec built from hyperfocal.yaml (per-problem image
    env vars, extra Dockerfile lines, network/compute overrides).
    By default the output also carries the audit-facing source bundle:
    source/ (the .git-stripped staged build context + repo.bundle + per-
    submodule git bundles) and environment/Dockerfile.selfcontained per
    task dir — a public-bases-only rebuild recipe (auditable, not
    hermetic; the prebuilt spec-hash image stays the sole graded
    artifact). --no-source-bundle opts out and reproduces the pre-W4
    output exactly. hyperfocal.yaml can opt out declaratively with
    packaging.sourceBundle: false (heavy-git-history envs, decision 7.12 —
    the flagless builder path honors it); precedence is CLI flag (when
    explicitly passed) > yaml key > default emit.

  env-orchestrator harbor doctor [--env-repo <path>] [--problem <id>]...
      [--trials N] [--no-build] [--base-image <ref>] [--no-source-bundle]
    Author-loop preflight: strict hyperfocal.yaml parse, problems.yaml +
    declared-solutionRef resolution, a full local package (docker build =
    in-image setup smoke + packageProblem() hook), solution-replay trials
    (default 1), and a gold-score-vs-floor report so minReplayScore can be
    set from measured numbers. Everything the release pipeline would
    catch, before a builder sees the env. --no-build stops after task-dir
    emission (no solution/ dirs — solutions are produced in the built
    image). --no-source-bundle mirrors package's flag: skip the audit
    source tree (and with it the 100 MB per-file ship gate) — use it to
    doctor envs that ship pin-only because their git history exceeds the
    gate (decision 7.12); default is to emit and gate the source, exactly
    like package — including honoring packaging.sourceBundle: false from
    hyperfocal.yaml when no flag is passed.

  env-orchestrator harbor validate --task <taskDir> [--trials N]
      [--jobs-dir <dir>] [--skip-secret-scan]
    Release gate: image secret-scan + N solution-replay trials (default 3)
    through the PINNED harbor CLI; every trial must score >= the problem's
    minReplayScore (default 1.0). Tasks with no runnable replay here (no
    reference solution, or gpus > 0) pass with an explicit evidence record
    — their validation venue is platform runs.

  env-orchestrator harbor run --task <taskDir> [--agent <name>]
      [--jobs-dir <dir>] [-- <extra harbor args...>]
    Thin wrapper over pinned \`harbor run\` (default agent: oracle —
    harbor's name for the no-LLM solution-replay agent).

  env-orchestrator harbor grade [--problem <id>]
    In-container verifier entrypoint (invoked by the generated tests/
    test.sh shim): runs the env's native tests and writes reward.json +
    test telemetry to /logs/verifier (override: HARBOR_VERIFIER_DIR).
    Harness crash => NO reward file + non-zero exit, so harbor records a
    trial exception (infra) instead of a fake agent score.

Environment:
  HARBOR_SOURCE   pip source override for the pinned harbor CLI (local path
                  or URL) when the pinned version is not on PyPI.
`;

function flagValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

function flagValues(args: string[], flag: string): string[] {
  const values: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === flag && args[i + 1]) values.push(args[i + 1]);
  }
  return values;
}

export async function handleHarborCommand(): Promise<void> {
  // argv: node env-orchestrator harbor <sub> [args...]
  const argv = process.argv.slice(2); // ["harbor", sub, ...]
  const sub = argv[1];
  const rest = argv.slice(2);
  const passthroughIdx = rest.indexOf("--");
  const args = passthroughIdx >= 0 ? rest.slice(0, passthroughIdx) : rest;
  const passthrough = passthroughIdx >= 0 ? rest.slice(passthroughIdx + 1) : [];

  switch (sub) {
    case "package":
      await packageRelease({
        envRepo: flagValue(args, "--env-repo") ?? "",
        problems: flagValues(args, "--problem"),
        outDir: flagValue(args, "--out"),
        build: !args.includes("--no-build"),
        packagesOverlay: args.includes("--packages-overlay"),
        imagePrefix: flagValue(args, "--image-prefix"),
        baseImage: flagValue(args, "--base-image"),
        push: flagValue(args, "--push"),
        // Only an EXPLICIT --no-source-bundle reaches the packager; when
        // absent, hyperfocal.yaml's packaging.sourceBundle decides (default
        // emit). Passing `true` unconditionally here would silently override
        // the yaml key for every flagless invocation (builders).
        ...(args.includes("--no-source-bundle") && { sourceBundle: false }),
        branch: flagValue(args, "--branch"),
        ...(flagValue(args, "--rebuild-counter") !== undefined && {
          rebuildCounter: Number(flagValue(args, "--rebuild-counter")),
        }),
      });
      break;

    case "doctor":
      await doctor({
        envRepo: flagValue(args, "--env-repo") ?? "",
        problems: flagValues(args, "--problem"),
        trials: Number(flagValue(args, "--trials") ?? 1),
        build: !args.includes("--no-build"),
        baseImage: flagValue(args, "--base-image"),
        // Same explicit-only contract as package: absent flag => the env's
        // packaging.sourceBundle key decides (doctor mirrors what a builder
        // would do).
        ...(args.includes("--no-source-bundle") && { sourceBundle: false }),
      });
      break;

    case "validate":
      await validate({
        taskDir:
          flagValue(args, "--task") ??
          (() => {
            throw new Error("--task <taskDir> is required");
          })(),
        trials: Number(flagValue(args, "--trials") ?? 3),
        jobsDir: flagValue(args, "--jobs-dir"),
        skipSecretScan: args.includes("--skip-secret-scan"),
      });
      break;

    case "grade":
      await grade({ problemId: flagValue(args, "--problem") });
      // Exit 0 explicitly regardless of test scores (agent failures are
      // real rewards, not verifier errors) — and don't let stray env-module
      // handles keep the verifier container alive.
      process.exit(0);
      break;

    case "run":
      await run({
        taskDir:
          flagValue(args, "--task") ??
          (() => {
            throw new Error("--task <taskDir> is required");
          })(),
        agent: flagValue(args, "--agent") ?? "oracle",
        jobsDir: flagValue(args, "--jobs-dir"),
        passthrough,
      });
      break;

    case undefined:
    case "help":
    case "--help":
    case "-h":
      console.log(HELP);
      break;

    default:
      console.error(`Unknown harbor subcommand: ${sub}\n`);
      console.log(HELP);
      process.exit(1);
  }
}
