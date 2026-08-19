/**
 * @hyperfocal/env-base
 *
 * Base package for Hyperfocal RL environments
 */

// Export types
export type {
  Problem,
  TestResult,
  SimpleTestResult,
  SimpleTest,
  BatchTest,
  Logger,
  EnvironmentDefinition,
  PermissionsMode,
  ClaudeCodeConfiguration,
  AnthropicCodingConfiguration,
  OpenCodeConfiguration,
  CodexConfiguration,
  MiniSweAgentConfiguration,
  AuthEntry,
  AgentConfiguration,
  OutputConfig,
  ManifestValidationResult,
  ManifestValidationError,
  AgentRunOptions,
  CriterionResult,
  TaskSpec,
  TaskSpecNetwork,
  TaskSpecNetworkPolicy,
  TaskSpecCompute,
} from "./types.js";

// Export execute utilities
export { execute, executeWithExitCode } from "./execute.js";
export type { ExecuteOptions, ExecutionResult } from "./execute.js";

// Export loggers
export { ConsoleLogger, SilentLogger } from "./logger.js";

// Export testing utilities
export {
  runSimpleTests,
  isEnvironmentError,
  aggregateTestResults,
} from "./testing.js";
export type {
  AggregatedScore,
  AggregatedScoreContribution,
} from "./testing.js";

// Export subprocess testing utilities
export { runSubprocessTests, parseJunitXml } from "./subprocess-testing.js";
export type { SubprocessTestOptions } from "./subprocess-testing.js";

// Export agents
export {
  ClaudeCodeAgent,
  AnthropicCodingAgent,
  OpenCodeAgent,
  CodexAgent,
  MiniSweAgent,
  BLOCKED_GIT_SUBCOMMANDS,
  DEFAULT_ALLOWED_TOOLS,
  DEFAULT_DISALLOWED_TOOLS,
  ClaudeCodeRateLimitError,
  UNKNOWN_CLAUDE_RATE_LIMIT_TYPE,
  normalizeClaudeApiRateLimitResult,
  normalizeClaudeRateLimitEvent,
} from "./agents/index.js";

// Export problem loading utilities
export {
  loadProblems,
  findProblemsFile,
  loadProblemsFromDirectory,
} from "./problems.js";
export type { PromptTemplateValues } from "./problems.js";

// Export manifest utilities
export {
  validateManifest,
  writeManifest,
  readManifest,
  getManifestPath,
  loadSchema,
  generateSchemaDescription,
  generateSchemaExample,
} from "./manifest.js";

// Export telemetry utilities
export {
  TelemetrySession,
  createSession,
  getLogsDir,
  listProblemsWithLogs,
  getProblemMetadata,
  getCombinedLogPath,
  readAgentTrace,
  // Rollout lifecycle functions
  updateProblemPhase,
  recordForcedAgentSessionEnd,
  finalizeRollout,
  // S3 sync for uploading logs to cloud
  S3TelemetrySync,
  createS3SyncFromEnv,
} from "./telemetry/index.js";
export type { ProviderRateLimitObservation } from "./telemetry/index.js";

export type {
  LogEvent,
  LogCategory,
  LogEventType,
  SessionMetadata,
  ProblemMetadata,
  SessionSummary,
  RolloutPhase,
  RolloutStatus,
  S3TelemetrySyncConfig,
} from "./telemetry/index.js";

// Export MCP servers module
export {
  MCP_SERVERS_FILENAME,
  mcpServersFilePath,
  readMcpServersFile,
  readMcpServersFromPath,
  upsertMcpServer,
  removeMcpServer,
} from "./mcp/index.js";

export type { McpServerSpec, McpServersFile } from "./mcp/index.js";

// Export rubric module
export {
  gradeWithRubric,
  createRubricTest,
  createOpenRouterGenerateFn,
  preprocessTrace,
  collectWorkspaceCode,
  JUDGE_SYSTEM_PROMPT,
  buildUserPrompt,
} from "./rubric/index.js";

export type {
  Criterion,
  CriterionContextType,
  RubricContext,
  GenerateFn,
  EvaluationReport,
  RubricTestConfig,
  TraceMode,
  TracePreprocessOptions,
  CollectWorkspaceCodeOptions,
} from "./rubric/index.js";

// Harbor packaging support (all harbor-facing code lives under src/harbor/)
export {
  toHarborRewards,
  writeHarborRewards,
} from "./harbor/rewards.js";
export type {
  HarborRewards,
  HarborRewardsMeta,
  HarborRewardsResult,
} from "./harbor/rewards.js";
