/**
 * YAML Configuration Manager
 *
 * Loads and validates hyperfocal.yaml from the environment repo root.
 * Replaces the old JSON-based config-manager.ts.
 */

import * as fs from "fs";
import * as path from "path";
import YAML from "yaml";

import type {
  PermissionsMode,
  TaskSpecNetwork,
  TaskSpecNetworkPolicy,
} from "@hyperfocal/env-base";

export type AgentType = "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent";

/**
 * Network policies emitted into task.toml (harbor >= 0.18 schema; the old
 * boolean `allow_internet` is deprecated and cannot express allowlists).
 *
 * `environment` is the BASELINE: it applies at container start, during
 * agent setup/install (harbor installs CLI agents like claude-code inside
 * the container at setup time, which needs egress), and to any phase
 * without an override. `agent`/`verifier` override the baseline during
 * agent.run() / verify() only.
 *
 * The shapes are aliases of env-base's TaskSpec network types — the yaml
 * schema and the packageProblem() hook contract describe the same thing,
 * and aliasing keeps them from drifting apart.
 */
export type TaskNetworkPolicy = TaskSpecNetworkPolicy;
export type TaskNetworkConfig = TaskSpecNetwork;

/**
 * hyperfocal.yaml `packaging:` block — knobs that only matter when the env
 * is packaged into harbor tasks (never read at rollout time).
 */
/**
 * hyperfocal.yaml `packaging.image:` block — declarative edits to the task
 * image. The structured successor to the old `dockerfileExtra` string
 * keyhole.
 */
export interface PackagingImageConfig {
  /**
   * Environment variables baked into the image as Docker ENV — visible in
   * every phase (setup, agent, verifier, oracle).
   */
  env?: Record<string, string>;
  /**
   * Verbatim Dockerfile lines inserted after the fleet toolchain, before
   * the repo COPY — the escape hatch for system deps the env's setup
   * cannot express (e.g. a rust toolchain).
   */
  dockerfileLines?: string[];
}

export interface PackagingConfig {
  /** Per-phase network policies for task.toml; omitted => public baseline. */
  network?: TaskNetworkConfig;
  /** Declarative image edits (ENV vars, extra Dockerfile lines). */
  image?: PackagingImageConfig;
  /**
   * task.toml [agent].timeout_sec — how long harbor lets the agent work.
   * Positive integer seconds; omitted => renderTaskToml's default (7200).
   */
  agentTimeoutSec?: number;
  /**
   * task.toml [verifier].timeout_sec — grading budget. Positive integer
   * seconds; omitted => renderTaskToml's default (900). Envs with heavy
   * graders (e.g. gambench's ~100min dataset sweep) must raise this or
   * harbor kills the verifier mid-grade.
   */
  verifierTimeoutSec?: number;
  /**
   * task.toml [environment].build_timeout_sec — how long harbor lets a
   * customer-side `docker build` of the shipped environment/Dockerfile run
   * before killing it (harbor's own default is 600s). Only exercised on the
   * documented from-source path (the customer deletes the docker_image pin);
   * envs whose provision exceeds the emitted default (gambench-class 12min+
   * bakes) must raise this from their MEASURED provision time with generous
   * margin (2x is fine — a too-long timeout costs nothing). Positive integer
   * seconds; omitted => renderTaskToml's default (2700).
   */
  buildTimeoutSec?: number;
  /**
   * Opt-in shared setup layer for packaged image builds (docs/
   * 7-build-optimization design §2). Setting this to true means: the env
   * owner asserts that setupProblem fully overwrites any prior problem's
   * state (reset-complete); the packager may then build one shared layer
   * running the first problem's setup and build every problem's thin layer
   * on top of it. Default off; absent means off; no auto-detection, no
   * probe — an explicit human assertion, validated by E2E for the envs we
   * flip ourselves. Do NOT set this without reading the env's setupProblem:
   * a non-reset-complete setup builds green images with the wrong problem's
   * files inside (it works and lies).
   */
  sharedSetupLayer?: boolean;
  /**
   * The env's setupProblem executes CUDA work during docker build; the
   * packager builds serially with the legacy docker builder so the daemon's
   * nvidia default-runtime applies (BuildKit ignores default-runtime —
   * S3-F2); also routes the release to a GPU builder. Default off; absent
   * means off. Distinct from compute.gpus on purpose: most GPU envs bake
   * fine on CPU builders (their setup only downloads toolchains/wheels) and
   * must not pay the GPU-builder tax — set this ONLY when the bake itself
   * dies without a GPU device (docs/7-build-optimization 5-findings.md
   * S3-F2/V1-F3).
   */
  bakeNeedsGpu?: boolean;
  /**
   * Emit the per-task audit source tree (environment/source/ + the
   * standalone environment/Dockerfile) when packaging. Default true (absent
   * means emit). Set to false for envs whose git history exceeds the 100 MB
   * per-file public-hosting gate (decision 7.12: repo.bundle for
   * sc-silentbench/sc-svpgsbench/sc-imputebench/sc-manifold-bench class
   * histories cannot ship) — such envs ship pin-only: tasks carry no
   * environment/source/ and no environment/Dockerfile, harbor runs them off
   * the task.toml docker_image pin alone, and the per-env README says so
   * instead of offering a from-source path that cannot work. Builders run
   * the env's pinned `harbor package` with no flags, so this key is the ONLY
   * way an env can tell the BUILD path to skip source emission (and with it
   * the 100 MB ship gate). Precedence: an explicit CLI --no-source-bundle
   * always wins; otherwise this key decides; otherwise the default (emit).
   * Like the CLI flag, deliberately OUTSIDE the spec hash — source documents
   * the build and never changes what gets baked, so toggling it must not
   * rotate image tags.
   */
  sourceBundle?: boolean;
}

/**
 * hyperfocal.yaml `compute:` block. The packager maps only the REQUIREMENT
 * fields (gpus/gpuTypes/cpus/memoryMb) into task.toml; `cloud` is the
 * platform's rollout venue and deliberately never reaches shipped
 * artifacts. The control plane validates this block independently (with
 * stricter platform rules) at run creation — see the control-plane repo's
 * domain/compute.ts.
 */
export interface ComputeConfig {
  cloud?: "aws" | "modal";
  /**
   * EC2 instance type for platform rollouts (venue sizing). NEVER reaches
   * shipped artifacts — harbor tasks declare requirements (cpus/memoryMb),
   * not venues. The packager warns loudly when this is set without both
   * cpus and memoryMb, because the intent behind the instance choice would
   * otherwise silently degrade to harbor's small defaults (bit sc-gambench:
   * m5.xlarge intent shipped as 2 cpu / 4 GB).
   */
  instanceType?: string;
  gpus?: number;
  /**
   * Allowlist of acceptable GPU hardware, normalized: the yaml accepts
   * `gpuTypes: [T4, A10]`, the scalar `gpuTypes: any`, or the singular
   * sugar `gpuType: T4` — all land here as a list (`any` stays the
   * GPU_TYPES_ANY sentinel; consumers expand or resolve it).
   */
  gpuTypes?: string[];
  cpus?: number;
  memoryMb?: number;
  /**
   * Named volumes mounted into the rollout sandbox: absolute mount path ->
   * volume name. Modal-only today (the control plane rejects it for
   * cloud: aws) — used for large read-only assets like model checkpoints
   * that shouldn't be baked into images during iteration. Never reaches
   * shipped artifacts: at packaging time the asset must be baked into the
   * image instead (packaging.dockerfileExtra + a prebuilt docker_image).
   */
  volumes?: Record<string, string>;
}

/** Sentinel allowlist value meaning "every GPU type the platform knows". */
export const GPU_TYPES_ANY = "any";

/**
 * Every GPU type the platform knows. Mirrors the control plane's
 * KNOWN_GPU_TYPES (domain/compute.ts) — the expansion of `gpuTypes: any`
 * must not promise hardware the platform would reject.
 */
export const KNOWN_GPU_TYPES = [
  "T4",
  "L4",
  "A10",
  "L40S",
  "A100",
  "A100-80GB",
  "H100",
  "H200",
  "B200",
] as const;

/**
 * Normalize the accepted GPU-allowlist spellings into one list. Singular
 * `gpuType: X` is sugar for `gpuTypes: [X]`; both together is an error —
 * refuse rather than guess which one the author meant. Mirrors the control
 * plane's parseComputeRequest (domain/compute.ts).
 */
export function normalizeGpuTypes(
  gpuType: unknown,
  gpuTypes: unknown
): string[] | undefined {
  if (gpuType !== undefined && typeof gpuType !== "string") {
    throw new Error(
      `compute.gpuType must be a string (got ${JSON.stringify(gpuType)})`
    );
  }
  let list = gpuTypes;
  if (typeof list === "string") {
    // Accept `gpuTypes: any` (yaml scalar) as ["any"].
    list = [list];
  }
  if (
    list !== undefined &&
    (!Array.isArray(list) || list.some((t) => typeof t !== "string"))
  ) {
    throw new Error(
      `compute.gpuTypes must be a list of GPU type strings or "${GPU_TYPES_ANY}" (got ${JSON.stringify(gpuTypes)})`
    );
  }
  if (gpuType !== undefined && list !== undefined) {
    throw new Error(
      "compute.gpuType and compute.gpuTypes are mutually exclusive — use gpuTypes"
    );
  }
  return (list as string[] | undefined) ?? (gpuType !== undefined ? [gpuType] : undefined);
}

export interface HyperfocalConfig {
  version: string;
  environment: {
    name: string;
    description: string;
  };
  paths: {
    root: string;
    environmentDist: string;
    workspace: string;
  };
  agent: {
    awsAccess: boolean;
    defaultModel: string;
    /**
     * Agent type to use:
     * - "claude-code": Uses Claude Code CLI with built-in tools (default)
     * - "anthropic-coding": Uses Anthropic API directly with custom tools
     * - "opencode": Uses OpenCode CLI for multi-provider support
     * - "codex": Uses OpenAI Codex CLI via `codex exec`
     * - "mini-swe-agent": Uses mini-swe-agent CLI via LiteLLM
     */
    type?: AgentType;
    /**
     * Permission mode for the agent:
     * - "linux-user" (default): unprivileged hyperfocal-agent, kernel-enforced
     *   filesystem isolation of grader/gold/env state
     * - "claude-permissions": agent runs as root; only the Claude CLI's
     *   --allowedTools/--disallowedTools string patterns restrain it (opt-in)
     */
    permissionsMode?: PermissionsMode;
    /**
     * Patterns appended to DEFAULT_DISALLOWED_TOOLS (env-base) for this
     * environment. Use Claude Code patterns, e.g. "Bash(docker exec*)",
     * "Write(/path/*)", "Bash(slack *)".
     *
     * Additive only — to weaken the defaults, change the defaults in
     * env-base/ClaudeCodeAgent.ts. yaml can never make the sandbox laxer.
     */
    disallowedTools?: string[];
    /**
     * Tools appended to DEFAULT_ALLOWED_TOOLS (env-base) for this env.
     * Use for envs that need an extra Claude Code tool (e.g. "Grep").
     * Note: "deny > allow" — entries here cannot override disallow rules.
     */
    allowedTools?: string[];
    /**
     * Environment variables for the agent child process, declared by the env
     * repo. Restricted at injection (agent-runner) to the HYPERFOCAL_CLAUDE_
     * namespace — agent-behavior knobs like the background-task reaper — so
     * yaml can tune agent behavior but can never inject credentials or
     * arbitrary process config. Process-level env (run-request envVars)
     * overrides these, so a run can still tighten/loosen a knob per launch.
     */
    env?: Record<string, string>;
  };
  output?: {
    schemaFile: string;
    type?: string;
  };
  /** Packaging-time knobs; only validated/consumed by `harbor package`. */
  packaging?: PackagingConfig;
  /** Compute declaration; requirement fields flow into task.toml. */
  compute?: ComputeConfig;
}

const DEFAULT_CONFIG: HyperfocalConfig = {
  version: "1.0",
  environment: { name: "unnamed-environment", description: "" },
  paths: {
    root: "/hyperfocal/env",
    environmentDist: "environment/dist",
    workspace: "workspace",
  },
  agent: {
    awsAccess: false,
    defaultModel: "opus",
  },
};

/**
 * Load configuration from hyperfocal.yaml
 * Falls back to defaults if not found
 */
export function loadConfig(): HyperfocalConfig {
  const yamlPath = findYamlPath();

  if (!yamlPath || !fs.existsSync(yamlPath)) {
    console.log("No hyperfocal.yaml found, using defaults");
    return DEFAULT_CONFIG;
  }

  const content = fs.readFileSync(yamlPath, "utf-8");
  const parsed = YAML.parse(content) as Partial<HyperfocalConfig>;

  // Merge with defaults for missing fields
  return {
    ...DEFAULT_CONFIG,
    ...parsed,
    environment: { ...DEFAULT_CONFIG.environment, ...parsed.environment },
    paths: { ...DEFAULT_CONFIG.paths, ...parsed.paths },
    agent: { ...DEFAULT_CONFIG.agent, ...parsed.agent },
    output: parsed.output, // Optional - only include if defined
  };
}

/**
 * Resolve all paths to absolute paths
 */
export function getResolvedPaths(config: HyperfocalConfig) {
  return {
    root: config.paths.root,
    environmentDist: path.join(config.paths.root, config.paths.environmentDist),
    workspace: path.join(config.paths.root, config.paths.workspace),
  };
}

/**
 * Find the hyperfocal.yaml file
 * Searches: current directory, HYPERFOCAL_ENV_ROOT, /hyperfocal/env, /hyperfocal/env/environment
 */
function findYamlPath(): string | null {
  // Try current directory first
  if (fs.existsSync("hyperfocal.yaml")) {
    return path.resolve("hyperfocal.yaml");
  }

  // Try environment root from env var
  const envRoot = process.env.HYPERFOCAL_ENV_ROOT;
  if (envRoot) {
    const envRootPath = path.join(envRoot, "hyperfocal.yaml");
    if (fs.existsSync(envRootPath)) {
      return envRootPath;
    }
  }

  // Try default EC2 path (repo root)
  const defaultPath = "/hyperfocal/env/hyperfocal.yaml";
  if (fs.existsSync(defaultPath)) {
    return defaultPath;
  }

  // Try environment subdirectory (alternative location)
  const envSubdirPath = "/hyperfocal/env/environment/hyperfocal.yaml";
  if (fs.existsSync(envSubdirPath)) {
    return envSubdirPath;
  }

  return null;
}

/**
 * Get the path where hyperfocal.yaml was found
 */
export function getConfigPath(): string | null {
  return findYamlPath();
}

/**
 * Check if running in local development (no RUN_ID means local)
 */
export function isLocalDev(): boolean {
  return !process.env.RUN_ID;
}

/**
 * Get default configuration object
 */
export function getDefaultConfig(): HyperfocalConfig {
  return { ...DEFAULT_CONFIG };
}

// ===========================================================================
// Strict parsing for packaging-time blocks
//
// The runtime loadConfig() above is deliberately lenient (rollouts should
// not die on a typo in a block they never read). Packaging is the opposite:
// a typo'd network mode must fail the package command loudly, never ship a
// task with the wrong network posture.
// ===========================================================================

const NETWORK_MODES = ["public", "no-network", "allowlist"] as const;

function isNetworkMode(value: unknown): value is TaskNetworkPolicy["mode"] {
  return (
    typeof value === "string" &&
    (NETWORK_MODES as readonly string[]).includes(value)
  );
}

function parseNetworkPolicy(
  value: unknown,
  where: string
): TaskNetworkPolicy | undefined {
  if (value === undefined || value === null) return undefined;
  const policy = value as { mode?: unknown; allowedHosts?: unknown };
  if (!isNetworkMode(policy.mode)) {
    throw new Error(
      `${where}: mode must be one of ${NETWORK_MODES.join(" | ")} (got ${JSON.stringify(policy.mode)})`
    );
  }
  const hosts = policy.allowedHosts;
  if (hosts !== undefined) {
    if (
      !Array.isArray(hosts) ||
      hosts.some((h) => typeof h !== "string" || !h.trim())
    ) {
      throw new Error(
        `${where}: allowedHosts must be a list of non-empty strings`
      );
    }
    if (policy.mode !== "allowlist") {
      throw new Error(
        `${where}: allowedHosts is only valid with mode: allowlist`
      );
    }
  }
  if (
    policy.mode === "allowlist" &&
    (!Array.isArray(hosts) || hosts.length === 0)
  ) {
    throw new Error(
      `${where}: mode allowlist requires a non-empty allowedHosts list`
    );
  }
  return {
    mode: policy.mode,
    ...(hosts !== undefined && { allowedHosts: hosts as string[] }),
  };
}

/**
 * Parse + validate the `packaging.network` shape (all fields optional;
 * omitted => public baseline):
 *
 *   packaging:
 *     network:
 *       mode: public | no-network | allowlist   # [environment] baseline
 *       allowedHosts: [api.example.com, ...]    # allowlist mode only
 *       agent:    { mode: ..., allowedHosts: [...] }   # agent.run() override
 *       verifier: { mode: ..., allowedHosts: [...] }   # verify() override
 */
function parseNetworkConfig(raw: unknown): TaskNetworkConfig | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("packaging.network must be a mapping");
  }
  const doc = raw as Record<string, unknown>;

  // Baseline is declared inline (mode/allowedHosts at the top level).
  const environment =
    doc.mode !== undefined
      ? parseNetworkPolicy(
          { mode: doc.mode, allowedHosts: doc.allowedHosts },
          "packaging.network"
        )
      : undefined;
  const agent = parseNetworkPolicy(doc.agent, "packaging.network.agent");
  const verifier = parseNetworkPolicy(
    doc.verifier,
    "packaging.network.verifier"
  );

  if (!environment && !agent && !verifier) return undefined;
  return {
    ...(environment && { environment }),
    ...(agent && { agent }),
    ...(verifier && { verifier }),
  };
}

/** Positive-integer-seconds validator for packaging timeout knobs. */
function parseTimeoutSec(value: unknown, where: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value <= 0
  ) {
    throw new Error(
      `${where} must be a positive integer number of seconds (got ${JSON.stringify(value)})`
    );
  }
  return value;
}

/** Docker ENV variable names: letters, digits, underscore; no leading digit. */
const ENV_VAR_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

function parseImageConfig(raw: unknown): PackagingImageConfig | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("packaging.image must be a mapping");
  }
  const doc = raw as Record<string, unknown>;
  for (const key of Object.keys(doc)) {
    if (key !== "env" && key !== "dockerfileLines") {
      throw new Error(
        `packaging.image.${key} is not a recognized key (expected env, dockerfileLines)`
      );
    }
  }

  let env: Record<string, string> | undefined;
  if (doc.env !== undefined) {
    if (
      typeof doc.env !== "object" ||
      doc.env === null ||
      Array.isArray(doc.env)
    ) {
      throw new Error("packaging.image.env must be a mapping of NAME: value");
    }
    env = {};
    for (const [name, value] of Object.entries(
      doc.env as Record<string, unknown>
    )) {
      if (!ENV_VAR_NAME.test(name)) {
        throw new Error(
          `packaging.image.env: ${JSON.stringify(name)} is not a valid environment variable name`
        );
      }
      if (
        typeof value !== "string" &&
        typeof value !== "number" &&
        typeof value !== "boolean"
      ) {
        throw new Error(
          `packaging.image.env.${name} must be a scalar (got ${JSON.stringify(value)})`
        );
      }
      env[name] = String(value);
    }
  }

  let dockerfileLines: string[] | undefined;
  if (doc.dockerfileLines !== undefined) {
    if (
      !Array.isArray(doc.dockerfileLines) ||
      doc.dockerfileLines.some((l) => typeof l !== "string" || !l.trim())
    ) {
      throw new Error(
        "packaging.image.dockerfileLines must be a list of non-empty strings"
      );
    }
    dockerfileLines = doc.dockerfileLines as string[];
  }

  if (!env && !dockerfileLines) return undefined;
  return {
    ...(env && { env }),
    ...(dockerfileLines && { dockerfileLines }),
  };
}

const KNOWN_PACKAGING_KEYS = new Set([
  "network",
  "image",
  "agentTimeoutSec",
  "verifierTimeoutSec",
  "buildTimeoutSec",
  "sharedSetupLayer",
  "bakeNeedsGpu",
  "sourceBundle",
]);

export function parsePackagingConfig(
  raw: unknown
): PackagingConfig | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("packaging must be a mapping");
  }
  const doc = raw as Record<string, unknown>;
  if (doc.dockerfileExtra !== undefined) {
    throw new Error(
      "packaging.dockerfileExtra was replaced by packaging.image.dockerfileLines " +
        "(a YAML list of Dockerfile lines) — move each line there. " +
        "Plain ENV lines are better expressed as packaging.image.env."
    );
  }
  for (const key of Object.keys(doc)) {
    if (!KNOWN_PACKAGING_KEYS.has(key)) {
      throw new Error(
        `packaging.${key} is not a recognized key (expected ${[...KNOWN_PACKAGING_KEYS].join(", ")}). ` +
          "The rollout runtime ignores typos here silently; packaging does not."
      );
    }
  }
  const agentTimeoutSec = parseTimeoutSec(
    doc.agentTimeoutSec,
    "packaging.agentTimeoutSec"
  );
  const verifierTimeoutSec = parseTimeoutSec(
    doc.verifierTimeoutSec,
    "packaging.verifierTimeoutSec"
  );
  const buildTimeoutSec = parseTimeoutSec(
    doc.buildTimeoutSec,
    "packaging.buildTimeoutSec"
  );
  const network = parseNetworkConfig(doc.network);
  const image = parseImageConfig(doc.image);
  let sharedSetupLayer: boolean | undefined;
  if (doc.sharedSetupLayer !== undefined && doc.sharedSetupLayer !== null) {
    if (typeof doc.sharedSetupLayer !== "boolean") {
      throw new Error(
        `packaging.sharedSetupLayer must be a boolean (got ${JSON.stringify(doc.sharedSetupLayer)})`
      );
    }
    sharedSetupLayer = doc.sharedSetupLayer;
  }
  let bakeNeedsGpu: boolean | undefined;
  if (doc.bakeNeedsGpu !== undefined && doc.bakeNeedsGpu !== null) {
    if (typeof doc.bakeNeedsGpu !== "boolean") {
      throw new Error(
        `packaging.bakeNeedsGpu must be a boolean (got ${JSON.stringify(doc.bakeNeedsGpu)})`
      );
    }
    bakeNeedsGpu = doc.bakeNeedsGpu;
  }
  let sourceBundle: boolean | undefined;
  if (doc.sourceBundle !== undefined && doc.sourceBundle !== null) {
    if (typeof doc.sourceBundle !== "boolean") {
      throw new Error(
        `packaging.sourceBundle must be a boolean (got ${JSON.stringify(doc.sourceBundle)})`
      );
    }
    sourceBundle = doc.sourceBundle;
  }
  return {
    ...(network && { network }),
    ...(image && { image }),
    ...(agentTimeoutSec !== undefined && { agentTimeoutSec }),
    ...(verifierTimeoutSec !== undefined && { verifierTimeoutSec }),
    ...(buildTimeoutSec !== undefined && { buildTimeoutSec }),
    ...(sharedSetupLayer !== undefined && { sharedSetupLayer }),
    ...(bakeNeedsGpu !== undefined && { bakeNeedsGpu }),
    ...(sourceBundle !== undefined && { sourceBundle }),
  };
}

export function parseComputeConfig(raw: unknown): ComputeConfig | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("compute must be a mapping");
  }
  const c = raw as Record<string, unknown>;
  if (c.cloud !== undefined && c.cloud !== "aws" && c.cloud !== "modal") {
    throw new Error(
      `compute.cloud must be "aws" or "modal" (got ${JSON.stringify(c.cloud)})`
    );
  }
  if (
    c.instanceType !== undefined &&
    (typeof c.instanceType !== "string" || c.instanceType.trim() === "")
  ) {
    throw new Error(
      `compute.instanceType must be a non-empty string (got ${JSON.stringify(c.instanceType)})`
    );
  }
  if (
    c.gpus !== undefined &&
    (typeof c.gpus !== "number" || !Number.isInteger(c.gpus) || c.gpus < 0)
  ) {
    throw new Error(
      `compute.gpus must be a non-negative integer (got ${JSON.stringify(c.gpus)})`
    );
  }
  const gpuTypes = normalizeGpuTypes(c.gpuType, c.gpuTypes);
  if (c.cpus !== undefined && (typeof c.cpus !== "number" || c.cpus <= 0)) {
    throw new Error(
      `compute.cpus must be a positive number (got ${JSON.stringify(c.cpus)})`
    );
  }
  if (
    c.memoryMb !== undefined &&
    (typeof c.memoryMb !== "number" ||
      !Number.isInteger(c.memoryMb) ||
      c.memoryMb <= 0)
  ) {
    throw new Error(
      `compute.memoryMb must be a positive integer (got ${JSON.stringify(c.memoryMb)})`
    );
  }
  if (c.volumes !== undefined) {
    if (
      typeof c.volumes !== "object" ||
      c.volumes === null ||
      Array.isArray(c.volumes)
    ) {
      throw new Error(
        `compute.volumes must be a mapping of absolute mount path -> volume name (got ${JSON.stringify(c.volumes)})`
      );
    }
    for (const [mount, name] of Object.entries(
      c.volumes as Record<string, unknown>
    )) {
      if (!mount.startsWith("/")) {
        throw new Error(
          `compute.volumes mount paths must be absolute (got ${JSON.stringify(mount)})`
        );
      }
      if (typeof name !== "string" || name.trim() === "") {
        throw new Error(
          `compute.volumes["${mount}"] must be a non-empty volume name (got ${JSON.stringify(name)})`
        );
      }
    }
    if (c.cloud === "aws") {
      throw new Error(
        "compute.volumes is Modal-only — remove it or set compute.cloud: modal " +
          "(EC2 equivalent is future work; see the control plane's domain/compute.ts)"
      );
    }
  }
  return {
    ...(c.cloud !== undefined && { cloud: c.cloud as ComputeConfig["cloud"] }),
    ...(c.instanceType !== undefined && {
      instanceType: c.instanceType as string,
    }),
    ...(c.gpus !== undefined && { gpus: c.gpus as number }),
    ...(gpuTypes !== undefined && { gpuTypes }),
    ...(c.cpus !== undefined && { cpus: c.cpus as number }),
    ...(c.memoryMb !== undefined && { memoryMb: c.memoryMb as number }),
    ...(c.volumes !== undefined && {
      volumes: c.volumes as Record<string, string>,
    }),
  };
}

/**
 * Load hyperfocal.yaml from an explicit env repo root with STRICT parsing
 * of the packaging-time blocks (packaging, compute). This is the loader
 * `harbor package` uses; the discovery-based loadConfig() above stays
 * lenient for rollout runtime.
 *
 * Unlike loadConfig(), a missing environment.name is left empty (callers
 * fall back to the repo directory name) rather than defaulted.
 */
export function loadEnvRepoConfig(envRepoDir: string): HyperfocalConfig {
  const yamlPath = path.join(envRepoDir, "hyperfocal.yaml");
  const content = fs.readFileSync(yamlPath, "utf-8");
  const parsed = (YAML.parse(content) ?? {}) as Partial<HyperfocalConfig> &
    Record<string, unknown>;

  return {
    ...DEFAULT_CONFIG,
    ...parsed,
    environment: { name: "", description: "", ...parsed.environment },
    paths: { ...DEFAULT_CONFIG.paths, ...parsed.paths },
    agent: { ...DEFAULT_CONFIG.agent, ...parsed.agent },
    output: parsed.output,
    packaging: parsePackagingConfig(parsed.packaging),
    compute: parseComputeConfig(parsed.compute),
  };
}
