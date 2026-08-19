/**
 * `env-orchestrator harbor doctor` — the author-loop preflight: run the
 * whole release pipeline's checks LOCALLY, before a Package-release click
 * ever reaches a builder.
 *
 * Checks, in order (each reported ✓/✗ and summarized at the end):
 *   1. hyperfocal.yaml strict parse — the rollout runtime is deliberately
 *      lenient, so packaging typos are invisible in the authoring loop
 *      until this (or a real package run) parses them strictly.
 *   2. problems.yaml load + git ref resolution of DECLARED solutionRefs —
 *      the same resolution `harbor package` does up front. (There is no
 *      problemStateRef under contract v2: the packaging commit is every
 *      problem's state.)
 *   3. docker buildx presence — the parallel bake build path (build/bake.ts)
 *      needs it; builders whose AMI predates the buildx rollout get a
 *      pinned static self-install at package time.
 *   4. Package + in-image setup smoke: a full local `harbor package`
 *      (docker build runs the env's own setupProblem() inside the image,
 *      compiles the env, and applies the packageProblem() hook if any).
 *      Skipped with --no-build (then only task-dir emission is checked —
 *      solutions are produced in the built image, so --no-build emits no
 *      solution/ dirs at all).
 *   5. In-image solution-replay trials per problem-with-solution (default 1
 *      trial — doctor is a fast loop, the release gate runs more).
 *   6. Gold-score-vs-floor report: the measured replay rewards next to the
 *      problem's minReplayScore, so authors can set the floor from
 *      observed numbers instead of guessing ("gold scored 0.845; your
 *      floor is 0.8").
 *
 * GPU tasks skip 5-6 locally (their replay needs GPU compute; the platform
 * runs those as replay release-runs) — everything else still runs.
 */

import * as fs from "fs";
import * as path from "path";
import { loadProblemsFromDirectory, Problem } from "@hyperfocal/env-base";
import { loadEnvRepoConfig } from "../../config/yaml-config.js";
import { buildxVersion } from "../build/buildx.js";
import { dockerHasNvidiaRuntime } from "../build/dockerInfo.js";
import { findEnvRepo, packageRelease, resolveRef } from "../package.js";
import {
  runReplayTrials,
  prewarmEgressControlSidecar,
} from "../runtime/replay.js";
import {
  readGpus,
  readMinReplayScore,
  readNetworkModes,
} from "../runtime/taskInfo.js";
import { ensureHarborCli } from "../runtime/venv.js";

export interface HarborDoctorOptions {
  envRepo: string;
  problems?: string[];
  trials: number;
  build: boolean;
  baseImage?: string;
  /**
   * Mirrors `harbor package`'s --no-source-bundle (PackageOptions.sourceBundle,
   * default true). Without the opt-out, doctor ALWAYS emitted the audit source
   * tree and hit its 100 MB per-file ship gate, so envs whose repo.bundle
   * exceeds the gate (heavy git histories — decision 7.12) could never doctor
   * green even though they package and ship fine pin-only via
   * `package --no-source-bundle`. Doctor's verdict should be reachable for the
   * path the env actually ships on. Left undefined (no flag), packageRelease
   * falls back to hyperfocal.yaml's packaging.sourceBundle — so an env that
   * declares `sourceBundle: false` doctors exactly like it builds, no flag
   * needed. Explicit false (the flag) wins over the yaml key.
   */
  sourceBundle?: boolean;
}

interface CheckResult {
  name: string;
  ok: boolean;
  detail?: string;
}

function log(msg: string): void {
  console.log(`[harbor:doctor] ${msg}`);
}

export async function doctor(opts: HarborDoctorOptions): Promise<void> {
  const results: CheckResult[] = [];
  const record = (name: string, ok: boolean, detail?: string) => {
    results.push({ name, ok, detail });
    log(`${ok ? "✓" : "✗"} ${name}${detail ? ` — ${detail}` : ""}`);
  };

  const envRepo = findEnvRepo(opts.envRepo);

  // 1. Strict yaml parse. The parsed config is kept: check 3b below needs
  // packaging.bakeNeedsGpu.
  let config: ReturnType<typeof loadEnvRepoConfig> | undefined;
  try {
    config = loadEnvRepoConfig(envRepo);
    record("hyperfocal.yaml strict parse", true);
  } catch (err) {
    record(
      "hyperfocal.yaml strict parse",
      false,
      err instanceof Error ? err.message : String(err)
    );
  }

  // 2. problems.yaml + declared-solutionRef resolution. The full ladder
  // (hook-vs-declaration conflict, solutionPatch committed check) needs the
  // compiled env module and runs inside `harbor package` below.
  try {
    const problems: Problem[] = loadProblemsFromDirectory(
      path.join(envRepo, "environment")
    );
    const selected = opts.problems?.length
      ? problems.filter((p) => opts.problems!.includes(p.id))
      : problems;
    if (selected.length === 0) {
      throw new Error(
        `no matching problems (have: ${problems.map((p) => p.id).join(", ")})`
      );
    }
    for (const problem of selected) {
      if (problem.solutionRef !== undefined) {
        if (!resolveRef(envRepo, problem.solutionRef)) {
          throw new Error(
            `problem ${problem.id}: solutionRef '${problem.solutionRef}' does ` +
              `not resolve (remove the declaration if there is no git-ref solution)`
          );
        }
      }
    }
    record(
      "problems.yaml + solutionRef resolution",
      true,
      `${selected.length} problem(s)`
    );
  } catch (err) {
    record(
      "problems.yaml + solutionRef resolution",
      false,
      err instanceof Error ? err.message : String(err)
    );
  }

  // 3. docker buildx presence — the parallel bake build path needs it.
  // Advisory when no build is requested (and on Linux builders `harbor
  // package` auto-installs the pinned static binary), hard fail when this
  // box would have to build without it.
  {
    const version = buildxVersion();
    if (version) {
      record("docker buildx (parallel bake builds)", true, version);
    } else {
      const autoInstallable =
        process.platform === "linux" &&
        (process.arch === "x64" || process.arch === "arm64");
      record(
        "docker buildx (parallel bake builds)",
        autoInstallable || !opts.build,
        "not found — `harbor package` self-installs a pinned static buildx " +
          "on Linux builders; elsewhere install the buildx CLI plugin " +
          "(Docker Desktop bundles it)"
      );
    }
  }

  // 3b. packaging.bakeNeedsGpu: the env's setup executes CUDA work during
  // docker build, which needs docker's nvidia runtime registered (the
  // legacy-builder path applies the daemon's default-runtime — S3-F2).
  // ADVISORY, never a failure: local dev boxes legitimately lack the
  // runtime (GPU builders bake it via nvidia-container-toolkit, #110);
  // this warns so a local `doctor` full-build attempt isn't a mystery.
  if (config?.packaging?.bakeNeedsGpu) {
    const nvidia = dockerHasNvidiaRuntime();
    record(
      "nvidia container runtime (packaging.bakeNeedsGpu)",
      true,
      nvidia === true
        ? "registered"
        : nvidia === false
          ? "WARNING: not registered on this box — the serial legacy-builder " +
            "CUDA bake would fail here; fine for local dev (GPU builders " +
            "register it at bake — nvidia-container-toolkit AMI)"
          : "WARNING: docker info unavailable — cannot check; the CUDA bake " +
            "needs a box with the nvidia runtime registered"
    );
  }

  // 4. Package (+ in-image setup smoke when building). A parse or ref
  // failure above would just fail again louder here — skip to the summary.
  const fatalSoFar = results.some((r) => !r.ok);
  const outDir = path.join(envRepo, "harbor-doctor");
  if (!fatalSoFar) {
    try {
      await packageRelease({
        envRepo,
        problems: opts.problems,
        outDir,
        build: opts.build,
        packagesOverlay: false,
        baseImage: opts.baseImage,
        sourceBundle: opts.sourceBundle,
      });
      // Name whichever opt-out skipped the source stage: the explicit CLI
      // flag wins; otherwise the env's own packaging.sourceBundle: false
      // (the flagless builder-path opt-out) applies inside packageRelease.
      const sourceNote =
        opts.sourceBundle === false
          ? ", --no-source-bundle"
          : opts.sourceBundle === undefined &&
              config?.packaging?.sourceBundle === false
            ? ", packaging.sourceBundle: false"
            : "";
      record(
        opts.build
          ? `package + in-image setup smoke (docker build ran setupProblem${sourceNote})`
          : `package (task-dir emission only, --no-build${sourceNote})`,
        true
      );
    } catch (err) {
      record(
        "package",
        false,
        err instanceof Error ? err.message : String(err)
      );
    }
  }

  // 5 + 6. Replay trials + score-vs-floor report, per packaged task.
  const tasksDir = path.join(outDir, "tasks");
  const packaged =
    !fatalSoFar && results.every((r) => r.ok) && fs.existsSync(tasksDir);
  if (packaged && opts.build) {
    const harborBin = ensureHarborCli(log);
    for (const taskName of fs.readdirSync(tasksDir).sort()) {
      const taskDir = path.join(tasksDir, taskName);
      if (!fs.existsSync(path.join(taskDir, "task.toml"))) continue;

      if (!fs.existsSync(path.join(taskDir, "solution"))) {
        record(
          `replay ${taskName}`,
          true,
          "no reference solution — replay not applicable, skipped"
        );
        continue;
      }
      if (readGpus(taskDir) > 0) {
        record(
          `replay ${taskName}`,
          true,
          "gpus > 0 — deferred to platform replay runs"
        );
        continue;
      }

      try {
        if (readNetworkModes(taskDir).some((m) => m !== "public")) {
          prewarmEgressControlSidecar(harborBin, log);
        }
        const minScore = readMinReplayScore(taskDir);
        const { summaries } = runReplayTrials({
          taskDir,
          trials: opts.trials,
          jobsDir: path.join(outDir, "doctor-jobs"),
          minScore,
          harborBin,
          log,
        });
        // The floor report: measured rewards vs the declared floor, so the
        // author can set minReplayScore from evidence.
        const rewards = summaries.flatMap((s) =>
          s.rewards ? Object.values(s.rewards) : []
        );
        const worst = rewards.length ? Math.min(...rewards) : NaN;
        const exceptions = summaries.filter((s) => s.exception);
        if (exceptions.length > 0) {
          record(
            `replay ${taskName}`,
            false,
            `${exceptions.length}/${summaries.length} trial(s) raised ` +
              `(${exceptions.map((s) => s.exception).join(", ")})`
          );
        } else {
          const ok = summaries.every((s) => s.ok);
          record(
            `replay ${taskName}`,
            ok,
            `gold scored ${worst.toFixed(3)} (worst of ${rewards.length} ` +
              `reward(s) over ${summaries.length} trial(s)); your floor is ${minScore}` +
              (ok ? "" : " — BELOW FLOOR")
          );
        }
      } catch (err) {
        record(
          `replay ${taskName}`,
          false,
          err instanceof Error ? err.message : String(err)
        );
      }
    }
  } else if (packaged && !opts.build) {
    log(
      "replay trials skipped (--no-build emits no solution dirs — solutions " +
        "are produced in the built image)"
    );
  }

  // Summary.
  const failed = results.filter((r) => !r.ok);
  console.log("");
  log("—— report ——");
  for (const r of results) {
    log(`${r.ok ? "✓" : "✗"} ${r.name}${r.detail ? ` — ${r.detail}` : ""}`);
  }
  if (failed.length > 0) {
    throw new Error(
      `doctor found ${failed.length} failing check(s) — fix these before packaging a release`
    );
  }
  log("environment is release-ready as far as local checks can tell");
}
