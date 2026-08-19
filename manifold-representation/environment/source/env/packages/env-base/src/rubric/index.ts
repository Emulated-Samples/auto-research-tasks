/**
 * Rubric grading module for Hyperfocal environments.
 *
 * Provides LLM-as-judge evaluation of agent outputs against weighted criteria.
 *
 * Key features:
 * - Per-criterion parallel evaluation with weighted scoring
 * - Trace preprocessing to reduce false positives from raw JSONL traces
 * - OpenRouter integration with gpt-5.3-chat as default judge
 * - Automatic API key detection (skips gracefully when unavailable)
 *
 * @example
 * ```typescript
 * import { createRubricTest, preprocessTrace, type Criterion } from "@hyperfocal/env-base";
 *
 * const test = createRubricTest({
 *   id: "rubric-code-quality",
 *   name: "Code Quality",
 *   description: "Evaluates code quality via LLM judge",
 *   criteria: [
 *     { weight: 10, requirement: "Uses proper error handling", context: ["code"] },
 *   ],
 *   getContext: async () => ({ code: readFileSync("./src/main.rs", "utf-8") }),
 * });
 * ```
 */

// Types
export type {
  Criterion,
  CriterionContextType,
  RubricContext,
  GenerateFn,
  EvaluationReport,
} from "./types.js";

// Re-export CriterionResult from main types for convenience
export type { CriterionResult } from "../types.js";

// Grader
export { gradeWithRubric } from "./grader.js";

// Helpers
export { createRubricTest } from "./helpers.js";
export type { RubricTestConfig } from "./helpers.js";

// Trace preprocessing
export { preprocessTrace } from "./trace.js";
export type { TraceMode, TracePreprocessOptions } from "./trace.js";

// Generate function
export { createOpenRouterGenerateFn } from "./generate.js";

// Prompts (exported for testing/customization)
export { JUDGE_SYSTEM_PROMPT, buildUserPrompt } from "./prompts.js";

// Context assembly
export { collectWorkspaceCode } from "./context.js";
export type { CollectWorkspaceCodeOptions } from "./context.js";
