/**
 * Core types for Hyperfocal environments
 */

/**
 * Per-service MCP configuration attached to a Problem. Each sub-shape maps
 * directly to the options consumed by the matching mock service's
 * `setup<Service>Problem` helper. `dataDir` is optional on all three —
 * environments default it to `mock-data/<problemId>/<service>` when omitted.
 *
 * The mock services write their listening URLs into the canonical
 * `<workspace>/.hyperfocal/mcp-servers.json` (see `mcp/file.ts`); each
 * agent class translates that neutral file to whatever its CLI accepts.
 */
export interface ProblemMcpSentry {
  dataDir?: string;
  org?: { id?: string; slug?: string; name?: string };
  project?: {
    id: string;
    slug: string;
    name: string;
    platform: string;
  };
  team?: { id: string; slug: string; name?: string };
  user?: { id: string; name: string; email: string };
  dsn?: { id?: string; dsn?: string; name?: string };
}

export interface ProblemMcpLinear {
  dataDir?: string;
}

export interface ProblemMcpGithub {
  dataDir?: string;
  repoSource?: string;
  owner?: string;
  name?: string;
  defaultBranch?: string;
  readOnly?: boolean;
}

export interface ProblemMcpNotion {
  dataDir?: string;
}

export interface ProblemMcp {
  sentry?: ProblemMcpSentry;
  linear?: ProblemMcpLinear;
  github?: ProblemMcpGithub;
  notion?: ProblemMcpNotion;
}

/**
 * A problem definition
 */
export interface Problem {
  id: string;
  prompt: string;
  default?: boolean;
  /**
   * Optional mock MCP services to spin up for this problem. When present,
   * the environment starts the listed services in-process during
   * `setupProblem` and writes their URLs into
   * `workspace/.hyperfocal/mcp-servers.json` (canonical neutral schema)
   * so the solver CLI picks them up via its agent class translator.
   */
  mcp?: ProblemMcp;
  /**
   * Git ref whose `workspace/` IS this problem's solved state (gold state).
   * The packager materializes the reference solution patch by checking the
   * ref out inside the built image (not a host-side diff), pinned to a
   * commit at package time. There is NO default: a problem that declares
 * neither solutionRef nor solutionPatch (and whose environment has no
 * solveProblem() hook) packages with no reference solution at all — the
 * replay check is then not applicable. Mutually exclusive with
   * `solutionPatch`. When the environment also implements `solveProblem()`,
   * the hook takes priority — the packager errors if both a hook and a
   * declaration exist for the same problem.
   */
  solutionRef?: string;
  /**
   * Repo-relative path to a committed .patch file that IS the reference
   * solution — for imported datasets (e.g. deepswe) where the solution is
   * a static patch, not a git ref. Applied and verified against the built
   * image at package time (`git apply --check`); the packager copies it as
   * the task's solution.patch. Mutually exclusive with `solutionRef`.
   */
  solutionPatch?: string;
  /**
   * Minimum score (0 < x <= 1) a solution-replay run must clear for this
   * problem's reference solution. Defaults to 1.0 (every test must fully
   * pass). Replay of the reference solution should ideally score 1.0 —
   * the floor is the concession for calibrated/rubric environments whose
   * gold honestly scores below 1.0.
   *
   * Calibrated continuous-scoring environments anchor the reward on an
   * external reference, so even the gold solution can honestly score below
   * 1.0 (gambench's gold graded 0.845 in-container) — for those, the
   * default floor can never be cleared. Set this safely BELOW the gold
   * solution's observed score floor (its score has run-to-run variance).
   * Rollout grading never reads this; it only tunes the replay gate.
   */
  minReplayScore?: number;
}

/**
 * Network policy for one harbor phase of a packaged task.
 */
export interface TaskSpecNetworkPolicy {
  mode: "public" | "no-network" | "allowlist";
  /** Hostnames reachable when mode is "allowlist". */
  allowedHosts?: string[];
}

/**
 * Per-phase network posture of a packaged task. `environment` is the
 * baseline (including setup); `agent` and `verifier` override it.
 */
export interface TaskSpecNetwork {
  environment?: TaskSpecNetworkPolicy;
  agent?: TaskSpecNetworkPolicy;
  verifier?: TaskSpecNetworkPolicy;
}

/**
 * Compute sizing of a packaged task, rendered into task.toml and used by
 * the platform to pick run hardware.
 */
export interface TaskSpecCompute {
  cpus?: number;
  memoryMb?: number;
  storageMb?: number;
  gpus?: number;
  gpuTypes?: string[];
}

/**
 * The mutable packaging spec for one problem's harbor task.
 *
 * The packager builds this from hyperfocal.yaml (packaging/compute blocks)
 * and hands it to the environment's optional `packageProblem()` hook, which
 * amends it — the hook edits the spec, it never does the packaging itself.
 * The final spec is rendered into the task's Dockerfile and task.toml, and
 * its content participates in the image tag, so any edit produces a new
 * immutable image instead of overwriting a published one.
 */
export interface TaskSpec {
  problemId: string;
  /**
   * Extra Dockerfile lines spliced between the base image and the repo
   * COPY. Seeded from `packaging.image.dockerfileLines`.
   */
  dockerfileLines: string[];
  /**
   * Environment variables baked into the image as Docker ENV — visible in
   * every phase (setup, agent, verifier, oracle). Seeded from
   * `packaging.image.env`.
   */
  imageEnv: Record<string, string>;
  network?: TaskSpecNetwork;
  compute?: TaskSpecCompute;
  agentTimeoutSec?: number;
  verifierTimeoutSec?: number;
}

/**
 * Per-criterion evaluation result from rubric grading
 */
export interface CriterionResult {
  requirement: string;
  weight: number;
  verdict: "MET" | "UNMET";
  reason: string;
}

/**
 * Test result from running environment tests
 *
 * Status meanings:
 * - "passed": Test ran and assertions passed (typically score >= 1.0)
 * - "failed": Test ran but assertions failed (score <= 0). Negative scores
 *   (penalty tests) currently map here; magnitude lives in the score field.
 * - "errored": Test couldn't run due to environment/infrastructure issue
 * - "partially_passed": Test ran with partial success (0 < score < 1.0)
 * - "skipped": Test was not evaluated (missing API key, no trace, etc.).
 *   Skipped tests appear in results for visibility but are excluded
 *   from the rollout score aggregation.
 *
 * Scores are signed real numbers, conventionally in [-1, 1]. Positive values
 * reward; negative values penalize. The orchestrator computes the rollout
 * score as `sum(weight * score) / sum(weight)` via aggregateTestResults.
 */
export interface TestResult {
  id: string;
  name: string;
  description?: string;
  status: "passed" | "failed" | "errored" | "partially_passed" | "skipped";
  duration: number;
  error?: string;
  output?: string;
  /**
   * Universal score. Conventionally in [-1, 1]: positive = reward,
   * negative = penalty. Aggregated as a weighted mean over non-skipped,
   * non-errored tests by aggregateTestResults.
   */
  score?: number;
  /**
   * Optional importance multiplier (>= 0). Defaults to 1 when absent.
   * Used by aggregateTestResults to compute the weighted rollout score.
   * Set higher to make this test contribute more, lower to dampen it.
   * weight: 0 is explicit "this test does not count toward the score."
   */
  weight?: number;
  /** LLM judge explanation (rubric tests only) */
  rationale?: string;
  /** Per-criterion breakdown (rubric tests only) */
  criteriaResults?: CriterionResult[];
  /** Model used for rubric evaluation, e.g. "openai/gpt-5.3-chat" */
  rubricModel?: string;
}

/**
 * Simple test result (used by test runners)
 *
 * For environment errors (infrastructure issues), set errored: true
 * This distinguishes "test failed" from "test couldn't run"
 *
 * When score is present, it overrides success for status mapping:
 * - score >= 1.0 -> "passed"
 * - 0 < score < 1.0 -> "partially_passed"
 * - score <= 0    -> "failed" (negative scores represent penalties; the
 *   numerical magnitude is preserved on the resulting TestResult.score)
 *
 * Score is a signed real number, conventionally in [-1, 1]. Positive values
 * reward; negative values penalize. Weight is declared on the SimpleTest
 * itself (not here) because it's a property of the test, not the outcome.
 */
export interface SimpleTestResult {
  success: boolean;
  error?: string;
  /** Set to true if the failure was due to environment/infrastructure issue */
  errored?: boolean;
  /** Set to true if the test was not evaluated (missing dependency, no input data, etc.) */
  skipped?: boolean;
  /**
   * Universal score. Conventionally in [-1, 1]: positive = reward,
   * negative = penalty. When present, overrides success for status mapping.
   */
  score?: number;
  /**
   * Optional runtime weight override. When present, takes precedence over
   * the SimpleTest.weight declared at registration time. Use this for
   * severity-escalated tests where a failure's importance scales with how
   * badly the threshold was missed — eg. a downtime test might use its
   * default weight for a slightly-over fail, but return a higher weight
   * when downtime exceeds 2× threshold. Subject to the same defaulting
   * rules: undefined → uses SimpleTest.weight (which itself defaults to 1).
   */
  weight?: number;
  /** LLM judge explanation (rubric tests only) */
  rationale?: string;
  /** Per-criterion breakdown (rubric tests only) */
  criteriaResults?: CriterionResult[];
}

/**
 * A simple test definition
 */
export interface SimpleTest {
  id: string;
  name: string;
  description: string;
  /**
   * Importance multiplier for this test in the rollout score. Defaults to 1
   * when absent. Set higher to make this test contribute more, lower to
   * dampen it. weight: 0 is explicit "this test does not count toward the
   * score." Weight is propagated onto the resulting TestResult by
   * runSimpleTests.
   */
  weight?: number;
  run: (logger: Logger) => Promise<SimpleTestResult>;
}

/**
 * A batch test that returns multiple TestResults from a single execution.
 * Use for subprocess-based test suites (pytest, jest, go test, etc.)
 * where individual test results come from parsing structured output.
 *
 * Note on weighting: BatchTest does not carry a batch-level `weight` because
 * batch authors construct TestResults at runtime — set `weight` on each
 * emitted TestResult directly inside `runBatch` for per-case importance.
 */
export interface BatchTest {
  id: string;
  name: string;
  description: string;
  runBatch: (logger: Logger) => Promise<TestResult[]>;
}

/**
 * Logger interface for test output
 */
export interface Logger {
  info(message: string): void;
  error(message: string): void;
  warn(message: string): void;
  debug(message: string): void;
}

/**
 * Environment definition interface.
 * 
 * This contains the logic for a runnable environment that exposes
 * problems, sets up a workspace, executes tests, and details cleanup. 
 */
export interface EnvironmentDefinition {
  /**
   * List all problems supported by this environment.
   *
   * @returns A promise resolving to the available problems.
   */
  listProblems(): Promise<Problem[]>;

  /**
   * Prepare the environment for a specific problem.
   *
   * This is where we perform tasks like provisioning resources,
   * configuring state, installing dependencies, etc. Most times
   * problems share a setup, so the setup between problems is the
   * same.
   *
   * @param problemId Optional identifier of the problem to set up.
   * @param logger Optional logger for emitting setup diagnostics.
   */
  setupProblem(problemId?: string, logger?: Logger): Promise<void>;

  /**
   * Execute tests for a given problem. Implementations should assume the
   * problem has already been set up and solved (see implementation for
   * `env-orchestrator solve`)
   *
   * @param problemId Identifier of the problem to test.
   * @param logger Logger for test output and diagnostics.
   * @returns A promise resolving to the test results.
   */
  // TODO: Optionally widen return type to
  //   `TestResult[] | { tests: TestResult[]; rolloutScore: number }`
  // so environments can compute their own rollout score (e.g. gated /
  // conditional aggregations where downstream-test outcomes depend on
  // upstream-test outcomes). Not implemented; for now the orchestrator
  // always aggregates via aggregateTestResults.
  runTests(problemId: string, logger: Logger): Promise<TestResult[]>;

  /**
   * Produce the reference solution for a problem programmatically. Runs
   * against a workspace that `setupProblem` has already prepared — the
   * mirror image of setup: where setup perturbs, this restores.
   *
   * Consulted whenever defined: it takes priority over a problem's
   * `solutionRef`/`solutionPatch` declaration, and the packager errors if
   * both a hook and a declaration exist for the same problem (that check
   * lives in the packager, not here). Problems that declare neither and
   * whose environment has no hook simply ship no reference solution.
   *
   * @param problemId Identifier of the problem to solve.
   * @param logger Optional logger for diagnostics.
   */
  solveProblem?(problemId: string, logger?: Logger): Promise<void>;

  /**
   * Amend the packaging spec for a problem before its harbor task is
   * rendered. The packager builds the default spec from hyperfocal.yaml;
   * this hook makes surgical edits for cases the yaml schema can't express
   * (per-problem image env vars, extra Dockerfile lines, network or compute
   * overrides). Return the amended spec — mutating the argument and
   * returning it is fine.
   *
   * Runs at package time on the builder host (before the docker build), so
   * it must not depend on problem setup having run. Most environments never
   * need this; prefer the declarative `packaging.` blocks in
   * hyperfocal.yaml when they suffice.
   *
   * @param problemId Identifier of the problem being packaged.
   * @param spec The default spec derived from hyperfocal.yaml.
   * @returns The amended spec.
   */
  packageProblem?(problemId: string, spec: TaskSpec): TaskSpec | Promise<TaskSpec>;

  /**
   * Clean up any resources created during setup or testing.
   *
   * This method is optional. Implementations should be idempotent and
   * tolerate partial or failed setups.
   *
   * @param logger Optional logger for cleanup diagnostics.
   */
  cleanup?(logger?: Logger): Promise<void>;
}

/**
 * Permission modes for the Claude Code agent.
 * 
 * - "claude-permissions": Uses --allowedTools/--disallowedTools for granular control.
 *   Runs as root. CWD isolation + tool restrictions replace Linux user isolation.
 *   This is the default and recommended mode.
 * 
 * - "linux-user": Uses --permission-mode bypassPermissions with a dedicated
 *   hyperfocal-agent Linux user. Legacy mode - requires setupClaudeCredentials()
 *   and workspace permission setup.
 */
export type PermissionsMode = "claude-permissions" | "linux-user";

/**
 * Agent configuration for Claude Code CLI
 * 
 * Uses the official Claude Code CLI with built-in tools (Bash, Read, Write, Edit).
 * Authentication is via OAuth (~/.claude/.credentials.json), not API key.
 */
export interface ClaudeCodeConfiguration {
  type: "claude-code";
  model: string;
  options?: {
    maxTurns?: number;
    // TODO: Add maxBudgetUsd when we want to enforce cost caps per session
  };
  /** Permission mode: "claude-permissions" (default) or "linux-user" (legacy) */
  permissionsMode?: PermissionsMode;
  /** Tools the agent is allowed to use (claude-permissions mode only) */
  allowedTools?: string[];
  /** Tool patterns to block, e.g. "Bash(git:*)" (claude-permissions mode only) */
  disallowedTools?: string[];
  /**
   * Path to the canonical neutral MCP servers file (the
   * `mcp-servers.json` written by setup). When set (and the file
   * parses), the agent translates it to Claude's `--mcp-config` schema,
   * writes that to a per-pid temp file, and appends
   * `--mcp-config <temp> --strict-mcp-config` plus `mcp__*` prefixes in
   * `--tools` / `--allowedTools` for each server in the file.
   *
   * Use `mcpServersFilePath(workspace)` from the `mcp` module to build
   * this path. Leave unset (or point at a missing file) to run the agent
   * without MCP servers.
   */
  mcpConfigPath?: string;
  /** Mirror raw CLI stdout/stderr to the session debug log for protocol debugging. */
  codingAgentDebugLogs?: boolean;
}

/**
 * Agent configuration for Anthropic Coding Agent
 * 
 * Uses Anthropic API directly with custom tools (bash, str_replace_editor).
 * Requires an API key for authentication.
 */
export interface AnthropicCodingConfiguration {
  type: "anthropic-coding";
  model: string;
  credentials: {
    anthropic: string;
  };
  options?: {
    maxTurns?: number;
    temperature?: number;
  };
}

/**
 * Auth entry types for OpenCode's auth.json.
 * Matches the Zod schema in OpenCode's packages/opencode/src/auth/index.ts.
 */
export type AuthEntry =
  | { type: "api"; key: string }
  | { type: "oauth"; access: string; refresh: string; expires: number; accountId?: string };

/**
 * Agent configuration for OpenCode CLI
 *
 * Uses anomalyco/opencode as a provider-agnostic coding agent.
 * Supports 75+ LLM providers via auth.json credential routing.
 * Authentication is via ~/.local/share/opencode/auth.json (written by orchestrator).
 *
 * Model format: "provider/model" (e.g. "anthropic/claude-opus-4-6", "openai/gpt-5.3-codex")
 */
export interface OpenCodeConfiguration {
  type: "opencode";
  /** Model in provider/model format, e.g. "anthropic/claude-opus-4-6" */
  model: string;
  /** Provider credentials — keyed by provider name (anthropic, openai, etc.) */
  credentials: Record<string, AuthEntry>;
  /** Permission mode: "claude-permissions" (default) or "linux-user" (legacy) */
  permissionsMode?: PermissionsMode;
  /**
   * Path to the canonical neutral MCP servers file (the
   * `mcp-servers.json` written by setup). When set (and the file
   * parses), the agent translates it to OpenCode's `opencode.json` shape,
   * writes that to a per-pid temp file, and points the CLI at it via
   * the `OPENCODE_CONFIG` env var. The translated config also raises
   * `agent.build.steps` (OpenCode's per-agent iteration ceiling).
   *
   * Use `mcpServersFilePath(workspace)` from the `mcp` module to build
   * this path. Leave unset (or point at a missing file) to run the agent
   * without MCP servers; the higher step ceiling is still applied.
   */
  mcpConfigPath?: string;
  /**
   * When true, append `--print-logs --log-level DEBUG` to the OpenCode
   * CLI invocation and mirror the full stderr stream to
   * `~/.local/share/opencode/runtime/<pid>/stderr.log`. Use when
   * investigating silent terminations (e.g. permission auto-rejects,
   * provider drops). The orchestrator surfaces this flag via the
   * `OPENCODE_DEBUG=1` env var.
   */
  debug?: boolean;
  /** Mirror raw CLI stdout/stderr to the session debug log for protocol debugging. */
  codingAgentDebugLogs?: boolean;
}

/**
 * Agent configuration for OpenAI Codex CLI.
 *
 * Uses `codex exec` in non-interactive JSONL mode. Authentication is either
 * via a per-run CODEX_HOME containing Codex auth.json or a scoped CODEX_API_KEY
 * environment variable supplied by env-orchestrator.
 */
export interface CodexConfiguration {
  type: "codex";
  model: string;
  /** Codex sandbox mode for model-generated shell commands. */
  sandbox?: "read-only" | "workspace-write" | "danger-full-access";
  /** Approval policy for non-interactive exec runs. Defaults to "never". */
  approvalPolicy?: "untrusted" | "on-failure" | "on-request" | "never";
  /** Path to the canonical neutral MCP servers file written by setup. */
  mcpConfigPath?: string;
  /** Mirror raw CLI stdout/stderr to the session debug log for protocol debugging. */
  codingAgentDebugLogs?: boolean;
}

/**
 * Agent configuration for mini-swe-agent.
 *
 * Uses the Python mini-swe-agent CLI in local yolo mode. Authentication is via
 * LiteLLM-compatible provider API keys supplied through the orchestrator's
 * allowlisted child environment.
 */
export interface MiniSweAgentConfiguration {
  type: "mini-swe-agent";
  /** LiteLLM provider/model string, e.g. "anthropic/claude-opus-4-6". */
  model: string;
  /** Optional mini-swe-agent YAML config name/path. Defaults to mini.yaml. */
  configPath?: string;
  /** Optional per-run mini-swe-agent cost limit. */
  costLimit?: number;
  /** Optional per-run mini-swe-agent step limit. */
  stepLimit?: number;
  /** Path to neutral MCP servers file; currently logged as unsupported. */
  mcpConfigPath?: string;
  /** Mirror raw CLI stdout/stderr to the session debug log for protocol debugging. */
  codingAgentDebugLogs?: boolean;
}

/**
 * Union type for all agent configurations
 */
export type AgentConfiguration =
  | ClaudeCodeConfiguration
  | AnthropicCodingConfiguration
  | OpenCodeConfiguration
  | CodexConfiguration
  | MiniSweAgentConfiguration;

/**
 * Output manifest configuration
 * Defines the schema that agents must conform to when submitting outputs
 */
export interface OutputConfig {
  schemaFile: string; // Path to JSON Schema file (relative to hyperfocal.yaml)
  type?: string; // Optional type identifier for documentation
}

/**
 * Result of validating a manifest against a schema
 */
export interface ManifestValidationResult {
  valid: boolean;
  errors?: ManifestValidationError[];
}

/**
 * Individual validation error
 */
export interface ManifestValidationError {
  path: string; // JSON pointer to error location
  message: string; // Human-readable error
  received?: unknown; // What was actually provided
}

/**
 * Options for running the agent
 */
export interface AgentRunOptions {
  schemaPath?: string; // Path to output schema (if manifest system is enabled)
}
