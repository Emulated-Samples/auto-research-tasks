/**
 * TaskSpec assembly for `harbor package`.
 *
 * The packager builds one default TaskSpec per problem from hyperfocal.yaml
 * (packaging.image / packaging.network / timeouts / compute), hands it to
 * the environment's optional packageProblem() hook for surgical amendments,
 * validates whatever comes back, and renders the final spec into the task's
 * Dockerfile and task.toml. The final spec also feeds the image tag's spec
 * hash (see computeSpecHash), so any spec change produces a NEW immutable
 * image instead of silently overwriting a published one.
 */

import { createHash } from "crypto";
import type { TaskSpec, TaskSpecNetworkPolicy } from "@hyperfocal/env-base";
import type { HyperfocalConfig } from "../config/yaml-config.js";
import {
  GPU_TYPES_ANY,
  KNOWN_GPU_TYPES,
} from "../config/yaml-config.js";

const NETWORK_MODES = ["public", "no-network", "allowlist"] as const;
const NETWORK_PHASES = ["environment", "agent", "verifier"] as const;
const ENV_VAR_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

// TODO(rubric-packaging): rubric-graded tests call an LLM judge from the
// verifier, which packaged tasks must support. The emitter side already
// works: packaging.network.verifier { mode: allowlist, allowedHosts:
// [openrouter.ai] } flows through to task.toml [verifier] network_mode/
// allowed_hosts (the key-passthrough half landed — control-plane #115 plus
// the strict network parsing here). Still open: a host-injected
// OPENROUTER_API_KEY reaching task.toml verifier.env via harbor's ${VAR}
// resolution, and investigating harbor-native rubric support before
// building more. See docs/3-packaging-contract-v2/6-future-directions.md.
// TODO(modal-packaging): compute.volumes (Modal named volumes for large
// read-only assets during iteration) has no packaging translation — at
// package time such assets must be baked into the image instead
// (packaging.image.dockerfileLines + a prebuilt base image). The control
// plane's TODO(modal-volumes-not-plumbed) cross-references this marker.

/**
 * The default spec for one problem, straight from hyperfocal.yaml. GPU
 * allowlists are expanded here ("any" -> every type the platform knows) so
 * the hook sees the concrete list that will ship.
 */
export function buildDefaultTaskSpec(
  problemId: string,
  config: HyperfocalConfig
): TaskSpec {
  const packaging = config.packaging;
  const compute = config.compute;

  const rawGpuTypes = compute?.gpuTypes;
  const gpuTypes = rawGpuTypes?.some((t) => t.toLowerCase() === GPU_TYPES_ANY)
    ? [...KNOWN_GPU_TYPES]
    : rawGpuTypes;

  const computeSpec = {
    ...(compute?.cpus !== undefined && { cpus: compute.cpus }),
    ...(compute?.memoryMb !== undefined && { memoryMb: compute.memoryMb }),
    ...(compute?.gpus !== undefined && { gpus: compute.gpus }),
    ...(gpuTypes !== undefined && { gpuTypes }),
  };

  return {
    problemId,
    dockerfileLines: [...(packaging?.image?.dockerfileLines ?? [])],
    imageEnv: { ...(packaging?.image?.env ?? {}) },
    ...(packaging?.network && { network: structuredClone(packaging.network) }),
    ...(Object.keys(computeSpec).length > 0 && { compute: computeSpec }),
    ...(packaging?.agentTimeoutSec !== undefined && {
      agentTimeoutSec: packaging.agentTimeoutSec,
    }),
    ...(packaging?.verifierTimeoutSec !== undefined && {
      verifierTimeoutSec: packaging.verifierTimeoutSec,
    }),
  };
}

function fail(problemId: string, msg: string): never {
  throw new Error(`packageProblem(${problemId}) returned an invalid spec: ${msg}`);
}

function validateNetworkPolicy(
  problemId: string,
  phase: string,
  policy: TaskSpecNetworkPolicy
): void {
  if (!(NETWORK_MODES as readonly string[]).includes(policy.mode)) {
    fail(
      problemId,
      `network.${phase}.mode must be one of ${NETWORK_MODES.join(" | ")} (got ${JSON.stringify(policy.mode)})`
    );
  }
  if (policy.mode === "allowlist") {
    if (
      !Array.isArray(policy.allowedHosts) ||
      policy.allowedHosts.length === 0 ||
      policy.allowedHosts.some((h) => typeof h !== "string" || !h.trim())
    ) {
      fail(
        problemId,
        `network.${phase} mode allowlist requires a non-empty allowedHosts list of strings`
      );
    }
  } else if (policy.allowedHosts !== undefined) {
    fail(problemId, `network.${phase}.allowedHosts is only valid with mode: allowlist`);
  }
}

/**
 * Validate a hook-amended spec. Hooks are env-author code running on the
 * builder — treat their output with the same strictness as hand-written
 * yaml, so a buggy hook fails the build loudly instead of shipping a
 * mis-postured task.
 */
export function validateTaskSpec(spec: unknown, problemId: string): TaskSpec {
  if (typeof spec !== "object" || spec === null || Array.isArray(spec)) {
    fail(problemId, "expected the (amended) spec object to be returned");
  }
  const s = spec as TaskSpec;
  if (s.problemId !== problemId) {
    fail(problemId, `problemId must stay ${JSON.stringify(problemId)} (got ${JSON.stringify(s.problemId)})`);
  }
  if (
    !Array.isArray(s.dockerfileLines) ||
    s.dockerfileLines.some((l) => typeof l !== "string" || !l.trim())
  ) {
    fail(problemId, "dockerfileLines must be a list of non-empty strings");
  }
  if (
    typeof s.imageEnv !== "object" ||
    s.imageEnv === null ||
    Array.isArray(s.imageEnv)
  ) {
    fail(problemId, "imageEnv must be a mapping of NAME -> string value");
  }
  for (const [name, value] of Object.entries(s.imageEnv)) {
    if (!ENV_VAR_NAME.test(name)) {
      fail(problemId, `imageEnv name ${JSON.stringify(name)} is not a valid environment variable name`);
    }
    if (typeof value !== "string") {
      fail(problemId, `imageEnv.${name} must be a string (got ${JSON.stringify(value)})`);
    }
  }
  if (s.network !== undefined) {
    if (typeof s.network !== "object" || s.network === null || Array.isArray(s.network)) {
      fail(problemId, "network must be a mapping of phase -> policy");
    }
    for (const phase of Object.keys(s.network)) {
      if (!(NETWORK_PHASES as readonly string[]).includes(phase)) {
        fail(problemId, `network.${phase} is not a phase (expected ${NETWORK_PHASES.join(", ")})`);
      }
    }
    for (const phase of NETWORK_PHASES) {
      const policy = s.network[phase];
      if (policy !== undefined) validateNetworkPolicy(problemId, phase, policy);
    }
  }
  if (s.compute !== undefined) {
    const c = s.compute;
    if (
      c.cpus !== undefined &&
      (typeof c.cpus !== "number" || !Number.isFinite(c.cpus) || c.cpus <= 0)
    ) {
      fail(problemId, "compute.cpus must be a positive number");
    }
    for (const field of ["memoryMb", "storageMb"] as const) {
      const v = c[field];
      if (v !== undefined && (typeof v !== "number" || !Number.isInteger(v) || v <= 0)) {
        fail(problemId, `compute.${field} must be a positive integer`);
      }
    }
    if (
      c.gpus !== undefined &&
      (typeof c.gpus !== "number" || !Number.isInteger(c.gpus) || c.gpus < 0)
    ) {
      fail(problemId, "compute.gpus must be a non-negative integer");
    }
    if (
      c.gpuTypes !== undefined &&
      (!Array.isArray(c.gpuTypes) || c.gpuTypes.some((t) => typeof t !== "string" || !t.trim()))
    ) {
      fail(problemId, "compute.gpuTypes must be a list of non-empty strings");
    }
  }
  for (const field of ["agentTimeoutSec", "verifierTimeoutSec"] as const) {
    const v = s[field];
    if (v !== undefined && (typeof v !== "number" || !Number.isInteger(v) || v <= 0)) {
      fail(problemId, `${field} must be a positive integer number of seconds`);
    }
  }
  return s;
}

/** JSON with recursively sorted object keys, so hashing is order-stable. */
export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([k, v]) => [k, sortKeys(v)])
    );
  }
  return value;
}

/**
 * Content hash for the image tag: `<branch-slug>-<problem>-<commit8>-<spec8>`.
 *
 * Covers everything that shapes the built image and its task rendering
 * beyond the packaging commit itself: the final (post-hook) TaskSpec, the
 * rendered build Dockerfile, the instruction, the task's source BRANCH
 * (tasks are branch-defined — env x branch x problem is the identity, and
 * the same problem id can ship from several branches), and the REBUILD
 * COUNTER (bumping it is the platform's "build this again under a new
 * name" action: same content, new hash, new immutable tag — tags are never
 * overwritten; see docs/11-build-finalization-fourth-draft/3-release-model
 * §7). Any change here mints a new immutable tag — published images are
 * never overwritten, and "same commit" alone no longer implies "same
 * image". Under contract v2 the problemStateCommit input is always the
 * packaging commit (there is no per-problem state ref anymore); the field
 * is kept so the hash's input shape stays explicit.
 *
 * minReplayScore is null (and hashes as null) when the problem declares no
 * reference solution — there is no replay gate to cover (V1-F5).
 */
export function computeSpecHash(inputs: {
  spec: TaskSpec;
  dockerfile: string;
  instruction: string;
  problemStateCommit: string;
  branch: string;
  rebuildCounter: number;
  minReplayScore: number | null;
  solutionKind: string;
  workspacePath: string;
}): string {
  const digest = createHash("sha256")
    .update(canonicalJson({ ...inputs, spec: JSON.parse(canonicalJson(inputs.spec)) }))
    .digest("hex");
  return digest.slice(0, 8);
}
