/**
 * Solution-replay trials for packaged harbor tasks: N runs of the task's
 * reference solution (solution/solve.sh) through the PINNED harbor CLI,
 * judged against the task's minReplayScore. Harbor names its no-LLM replay
 * agent "oracle" — our vocabulary is "solution replay"; the `harbor run
 * --agent oracle` invocation below keeps harbor's name.
 *
 * Shared by `harbor validate` (the release gate) and `harbor doctor` (the
 * author-loop score report); the platform's replay release-runs execute the
 * same `harbor run --agent oracle` shape on run boxes.
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

export interface TrialSummary {
  trialName: string;
  rewards: Record<string, number> | null;
  exception: string | null;
  /** No exception AND at least one reward AND every reward >= minScore. */
  ok: boolean;
}

export interface ReplayTrialsResult {
  jobName: string;
  jobDir: string;
  summaries: TrialSummary[];
}

/**
 * Pre-build harbor's egress-control sidecar image ONCE before launching
 * concurrent replay trials. Harbor 0.18 builds that image lazily per trial
 * behind a filesystem lock that is NOT reentrant, so N concurrent trials on
 * a COLD image cache crash the run; on a warm cache the build is skipped
 * entirely (E2E-1 findings, DESIGN.md FAQ "How does harbor 0.18 network
 * enforcement actually behave?"). Warming the cache up front makes
 * `-k N --n-concurrent > 1` safe for tasks that declare any non-public
 * network_mode. Uses harbor's OWN builder (same content-addressed tag and
 * lock), via the pinned venv's python. Failure = warn and continue: trials
 * may still pass on a warm cache.
 */
export function prewarmEgressControlSidecar(
  harborBin: string,
  log: (msg: string) => void
): void {
  // Mirrors DockerEnvironment._ensure_egress_control_sidecar_image_built
  // (harbor 0.18), minus the env-var bookkeeping — the point is only that
  // the image exists and the build lock is settled before trials start.
  const snippet = `
import asyncio

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.utils import (
    default_docker_platform,
    ensure_docker_image_built,
)


async def main() -> None:
    image = await ensure_docker_image_built(
        docker_name=DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
        docker_build_context=DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
        dockerfile_path=DockerEnvironment._egress_control_sidecar_dockerfile_path(),
        build_args={},
        platform=await default_docker_platform(),
        logger=None,
    )
    print(image)


asyncio.run(main())
`;
  const python = path.join(path.dirname(harborBin), "python");
  log(
    "Task declares a non-public network_mode — pre-warming harbor's " +
      "egress-control sidecar image (its lazy per-trial build lock is not " +
      "reentrant; concurrent trials crash on a cold cache)..."
  );
  const rc = spawnSync(python, ["-c", snippet], {
    encoding: "utf-8",
    timeout: 10 * 60 * 1000,
  });
  const image = (rc.stdout || "").trim().split("\n").pop();
  if (rc.status === 0 && image) {
    log(`Egress-control sidecar image ready: ${image}`);
  } else {
    console.warn(
      `WARNING: sidecar pre-warm failed (exit ${rc.status}); ` +
        `continuing — trials may still pass on a warm image cache.\n` +
        `${(rc.stderr || "").slice(-2000)}`
    );
  }
}

/**
 * Run `trials` replays of the task's reference solution and judge each
 * per-trial result.json against `minScore`. Throws when harbor itself fails
 * or the trial count comes back short; per-trial pass/fail is the CALLER's
 * policy (the gate throws on any failure, doctor just reports).
 */
export function runReplayTrials(opts: {
  taskDir: string;
  trials: number;
  jobsDir: string;
  minScore: number;
  harborBin: string;
  log: (msg: string) => void;
}): ReplayTrialsResult {
  const { taskDir, trials, jobsDir, minScore, harborBin, log } = opts;
  const jobName = `validate-${path.basename(taskDir)}-${Date.now()}`;

  log(`Solution replay x${trials} via pinned harbor CLI (job ${jobName})...`);
  const rc = spawnSync(
    harborBin,
    [
      "run",
      "-p",
      taskDir,
      "--agent",
      "oracle",
      "-k",
      String(trials),
      "--n-concurrent",
      String(Math.min(trials, 3)),
      "--job-name",
      jobName,
      "-o",
      jobsDir,
    ],
    { stdio: "inherit" }
  );
  if (rc.status !== 0) {
    throw new Error(`harbor run exited ${rc.status}`);
  }

  // Judge from per-trial result.json files (the authoritative artifacts).
  const jobDir = path.join(jobsDir, jobName);
  const summaries: TrialSummary[] = [];
  for (const entry of fs.readdirSync(jobDir)) {
    const resultPath = path.join(jobDir, entry, "result.json");
    if (!fs.existsSync(resultPath)) continue;
    const result = JSON.parse(fs.readFileSync(resultPath, "utf-8"));
    const rewards: Record<string, number> | null =
      result?.verifier_result?.rewards ?? null;
    const exception: string | null =
      result?.exception_info?.exception_type ?? null;
    const values = rewards ? Object.values(rewards) : [];
    const ok =
      !exception &&
      values.length > 0 &&
      values.every((v) => Number(v) >= minScore);
    summaries.push({ trialName: entry, rewards, exception, ok });
  }

  if (summaries.length !== trials) {
    throw new Error(
      `Expected ${trials} trials, found ${summaries.length} in ${jobDir}`
    );
  }
  return { jobName, jobDir, summaries };
}
