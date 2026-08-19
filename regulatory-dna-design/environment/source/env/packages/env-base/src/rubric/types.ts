/**
 * Core types for the rubric grading system.
 *
 * Modeled after Hud's rubric package (types.py) but adapted for
 * Hyperfocal's enriched context model where criteria can specify
 * which context types they need.
 */

import type { CriterionResult } from "../types.js";

// Re-export for convenience
export type { CriterionResult };

/**
 * A single evaluation criterion with a weight and requirement description.
 *
 * Positive weights reward the criterion being MET.
 * Negative weights penalize when bad behavior is detected (UNMET).
 */
export interface Criterion {
  /** How important this criterion is. Positive = reward for MET, Negative = penalty for UNMET. */
  weight: number;
  /** The requirement text that the judge evaluates against the context. */
  requirement: string;
  /** Which context types this criterion needs. Defaults to all available context. */
  context?: CriterionContextType[];
}

/**
 * Available context types the grading engine can assemble into judge prompts.
 *
 * - "code": Git diff, file contents, or code summary from the workspace
 * - "trace": Agent execution trace (JSONL events or summarized)
 * - "testResults": Functional test results summary
 */
export type CriterionContextType = "code" | "trace" | "testResults";

/**
 * Structured context provided by the environment for grading.
 *
 * The grading engine assembles per-criterion prompts by selecting
 * only the context fields each criterion requests via its `context` property.
 * If a criterion omits `context`, all available fields are included.
 */
export interface RubricContext {
  /** Git diff, file contents, or code summary */
  code?: string;
  /** Agent execution trace (JSONL events or summarized) */
  trace?: string;
  /** Functional test results summary */
  testResults?: string;
}

/**
 * Async function that calls an LLM and returns the response text.
 *
 * This is the injection point for LLM providers. The default implementation
 * uses OpenRouter via the openai npm package, but any function matching
 * this signature works (Anthropic, local models, etc.).
 */
export type GenerateFn = (systemPrompt: string, userPrompt: string) => Promise<string>;

/**
 * Full evaluation report from the grading engine.
 */
export interface EvaluationReport {
  /** Normalized score from 0.0 to 1.0 */
  score: number;
  /** Per-criterion evaluation results */
  reports: CriterionResult[];
}
