/**
 * Task-dir emission for `harbor package`: everything that lands in the task
 * directory BEFORE any docker work — task.toml, instruction.md, and the
 * tests/test.sh verifier shim. solution/ is deliberately NOT emitted here:
 * under contract v2 solutions are produced from the built image
 * (solutions.ts), so they only exist when the package run builds. The
 * task's environment/Dockerfile (the standalone from-source recipe) and
 * environment/source/ are emitted by package.ts via
 * render/environmentDockerfile.ts + emit/taskSource.ts; the old 2-line
 * FROM-prebuilt Dockerfile stub is retired — harbor reads the prebuilt pin
 * from task.toml [environment].docker_image, never from a Dockerfile.
 *
 * The old hyperfocal-validate.json sidecar is retired too: harbor 0.18.0
 * verifiably tolerates unknown task.toml content (junk-table experiment,
 * 2026-07-19 — a full trial ran to reward with an unknown top-level table
 * present), and TaskConfig.metadata is dict[str, Any], so the replay gate
 * now rides in [metadata] min_replay_score. Emitted ONLY for problems that
 * declare a reference solution — a solutionless problem has no replay gate
 * (V1-F5: the old unconditional 1.0 floor made those tasks look like they
 * carried an unmeetable gate).
 *
 * The templates these artifacts render from live in this module too:
 *  - tests/test.sh runs as root and is a THIN SHIM into `env-orchestrator
 *    harbor grade` (runtime/grade.ts) — which runs the native tests, writes
 *    /logs/verifier/reward.json (flat per-test dict) on harness success,
 *    and on harness failure writes NO reward file and exits non-zero so
 *    harbor raises RewardFileNotFoundError (trial exception = visibly
 *    infra). Never the old fail-closed {"reward": 0} shape — reward 0 is
 *    reserved for agent-attributable outcomes.
 */

import * as fs from "fs";
import * as path from "path";
import { stringify as stringifyToml } from "smol-toml";
import type { TaskSpec } from "@hyperfocal/env-base";
import type {
  TaskNetworkConfig,
  TaskNetworkPolicy,
} from "../../config/yaml-config.js";
import { IMAGE_ENV_ROOT } from "../render/dockerfile.js";

export interface TaskTomlParams {
  envName: string;
  /**
   * The task's source branch on the env repo (the ORIGINAL task branch,
   * e.g. "gqa" or "problem/seq-edit" — not the harbor-release/* overlay
   * ref). Part of task identity (env x branch x problem) and recorded in
   * [metadata] so the directory name is not the only carrier.
   */
  branch: string;
  problemId: string;
  imageTag: string;
  /**
   * Replay reward floor for [metadata] min_replay_score. null when the
   * problem declares NO reference solution — then no floor is emitted at
   * all (there is no replay to gate; V1-F5).
   */
  minReplayScore: number | null;
  difficulty?: string;
  tags?: string[];
  /**
   * task.toml [environment].build_timeout_sec — the budget harbor gives a
   * customer-side `docker build` of environment/Dockerfile on the
   * documented from-source path (harbor's own default 600s is too tight
   * for gambench-class provisions). From hyperfocal.yaml
   * packaging.buildTimeoutSec; defaults to 2700.
   */
  buildTimeoutSec?: number;
  agentTimeoutSec?: number;
  verifierTimeoutSec?: number;
  cpus?: number;
  memoryMb?: number;
  storageMb?: number;
  /**
   * From hyperfocal.yaml `compute:` — requirement fields only. The venue
   * (compute.cloud) deliberately never reaches the artifact: harbor tasks
   * declare what they need, the runner chooses where (-e docker/modal/...).
   */
  gpus?: number;
  /** GPU allowlist ("any" already expanded by the packager); the runner picks. */
  gpuTypes?: string[];
  /** From hyperfocal.yaml `packaging.network`; defaults to public baseline. */
  network?: TaskNetworkConfig;
  /**
   * Repo-relative workspace dir from hyperfocal.yaml paths.workspace
   * (default "workspace") — becomes the agent's workdir under the in-image
   * repo root.
   */
  workspacePath?: string;
}

/** `network_mode` (+ `allowed_hosts`) entries for one task.toml section. */
function networkFields(
  policy: TaskNetworkPolicy | undefined
): Record<string, unknown> {
  if (!policy) return {};
  return {
    network_mode: policy.mode,
    ...(policy.mode === "allowlist" && {
      allowed_hosts: policy.allowedHosts ?? [],
    }),
  };
}

/**
 * task.toml is built as a plain object and serialized with smol-toml —
 * escaping, section ordering, and whitespace are the serializer's job, not
 * template-literal splicing. Numeric timeouts serialize as TOML integers;
 * harbor's pydantic models coerce int -> float for those fields.
 */
export function renderTaskToml(p: TaskTomlParams): string {
  // Baseline defaults to public: API-driven agents (claude-code/codex) are
  // installed by harbor at agent-setup (under the BASELINE policy) and call
  // their APIs from inside the container, and rubric-graded envs call their
  // LLM judge from the verifier. Envs that want stricter posture declare it
  // via hyperfocal.yaml packaging.network; labs can also override per run.
  const doc = {
    version: "1.0",
    metadata: {
      source: "hyperfocal",
      environment: p.envName,
      branch: p.branch,
      problem_id: p.problemId,
      difficulty: p.difficulty ?? "unknown",
      tags: ["hyperfocal", ...(p.tags ?? [])],
      // The replay gate, folded from the retired hyperfocal-validate.json
      // sidecar (harbor tolerates extra [metadata] keys — see header).
      // Absent entirely for solutionless problems: no solution, no gate.
      ...(p.minReplayScore !== null && { min_replay_score: p.minReplayScore }),
    },
    agent: {
      timeout_sec: p.agentTimeoutSec ?? 7200,
      user: "agent",
      ...networkFields(p.network?.agent),
    },
    // TODO(separate-verifier): harbor supports a dedicated verifier
    // container ([verifier] environment_mode = "separate" + an optional
    // [verifier.environment] table — VerifierConfig in harbor
    // src/harbor/models/task/config.py); we deliberately do not emit it
    // this round (owner decision). When it lands it must be an OPTION PER
    // ENV, never a global default: some envs need the agent and the
    // verifier in the SAME image because the verifier probes state the
    // agent built inside it — the dind-cluster envs are the concrete case
    // (the verifier inspects the docker-in-docker cluster the agent stood
    // up; a separate verifier container would be probing an empty world).
    // The emission seam is exactly here: a per-env packaging knob adding
    // environment_mode/environment to the verifier table below.
    verifier: {
      timeout_sec: p.verifierTimeoutSec ?? 900,
      user: "root",
      ...networkFields(p.network?.verifier),
      env: {},
    },
    solution: {
      env: {},
    },
    environment: {
      docker_image: p.imageTag,
      build_timeout_sec: p.buildTimeoutSec ?? 2700,
      cpus: p.cpus ?? 2,
      memory_mb: p.memoryMb ?? 4096,
      storage_mb: p.storageMb ?? 20480,
      gpus: p.gpus ?? 0,
      ...(p.gpus && p.gpuTypes?.length ? { gpu_types: p.gpuTypes } : {}),
      ...networkFields(p.network?.environment ?? { mode: "public" }),
      workdir: path.posix.join(IMAGE_ENV_ROOT, p.workspacePath ?? "workspace"),
      mcp_servers: [],
    },
  };
  return stringifyToml(doc) + "\n";
}

/**
 * Thin shim only: every piece of verifier logic (native test run, reward
 * emission, telemetry copy, harness-failure semantics) lives in
 * `env-orchestrator harbor grade` (runtime/grade.ts) so it is versioned TS,
 * not generated bash. The one thing the shim still owns is redirecting
 * telemetry to a fresh root-owned dir (mktemp -d => mode 700) BEFORE node
 * starts — env-base freezes the logs dir at module load, so it cannot be
 * retargeted from inside the process — which keeps agent-planted files in
 * the world-writable default logs dir from ever being read as results.
 */
export function renderTestSh(problemId: string): string {
  return `#!/bin/bash
# Harbor verifier entrypoint (generated) — shim into \`harbor grade\`.
export HYPERFOCAL_LOGS_DIR="$(mktemp -d)"
cd /hyperfocal/env
exec node packages/env-orchestrator/bin/env-orchestrator.js harbor grade --problem ${problemId}
`;
}

export interface EmitTaskDirParams {
  taskDir: string;
  envName: string;
  /** The task's source branch (see TaskTomlParams.branch). */
  branch: string;
  problemId: string;
  prompt: string;
  spec: TaskSpec;
  imageTag: string;
  workspacePath: string;
  /** null = no reference solution declared => no replay floor emitted. */
  minReplayScore: number | null;
  /** hyperfocal.yaml packaging.buildTimeoutSec (default applied in render). */
  buildTimeoutSec?: number;
}

/** Create the task dir and write every build-independent artifact into it. */
export function emitTaskDir(p: EmitTaskDirParams): void {
  fs.rmSync(p.taskDir, { recursive: true, force: true });
  fs.mkdirSync(path.join(p.taskDir, "tests"), { recursive: true });
  fs.mkdirSync(path.join(p.taskDir, "environment"), { recursive: true });

  fs.writeFileSync(
    path.join(p.taskDir, "task.toml"),
    renderTaskToml({
      envName: p.envName,
      branch: p.branch,
      problemId: p.problemId,
      imageTag: p.imageTag,
      minReplayScore: p.minReplayScore,
      buildTimeoutSec: p.buildTimeoutSec,
      // Final-spec timeouts; renderTaskToml applies the 7200/900 defaults
      // when unset.
      agentTimeoutSec: p.spec.agentTimeoutSec,
      verifierTimeoutSec: p.spec.verifierTimeoutSec,
      gpus: p.spec.compute?.gpus ?? 0,
      gpuTypes: p.spec.compute?.gpuTypes,
      cpus: p.spec.compute?.cpus,
      memoryMb: p.spec.compute?.memoryMb,
      storageMb: p.spec.compute?.storageMb,
      network: p.spec.network,
      workspacePath: p.workspacePath,
    })
  );
  fs.writeFileSync(path.join(p.taskDir, "instruction.md"), p.prompt);

  const testSh = path.join(p.taskDir, "tests", "test.sh");
  fs.writeFileSync(testSh, renderTestSh(p.problemId));
  fs.chmodSync(testSh, 0o755);

  // environment/ stays created (empty here): package.ts fills it with the
  // standalone Dockerfile + source/ (render/environmentDockerfile.ts +
  // emit/taskSource.ts) unless the source bundle was opted out, in which
  // case the docker_image pin in task.toml alone satisfies harbor
  // (has_agent_environment_definition passes on the pin).
}
