/**
 * Convenience helpers for creating rubric-based SimpleTests.
 */

import type { Logger, SimpleTest, SimpleTestResult } from "../types.js";
import type { Criterion, GenerateFn, RubricContext } from "./types.js";
import type { TracePreprocessOptions } from "./trace.js";
import { gradeWithRubric } from "./grader.js";
import { createOpenRouterGenerateFn } from "./generate.js";
import { preprocessTrace } from "./trace.js";

/**
 * Configuration for creating a rubric test.
 */
export interface RubricTestConfig {
  /** Unique test identifier */
  id: string;
  /** Human-readable test name */
  name: string;
  /** Description of what this rubric evaluates */
  description: string;
  /** The evaluation criteria */
  criteria: Criterion[];
  /**
   * Lazy context assembly function. Called when the test runs,
   * not at definition time. This ensures traces/test results
   * are available (they don't exist until the test phase).
   */
  getContext: (logger: Logger) => Promise<RubricContext>;
  /**
   * LLM generate function for the judge. If omitted, one is created
   * automatically using OpenRouter with the default model (or the
   * model specified by `modelName`). If OPENROUTER_API_KEY is not
   * set, the test logs a warning and skips.
   */
  generateFn?: GenerateFn;
  /**
   * OpenRouter model name for the judge (e.g. "openai/gpt-5.3-chat").
   * Only used when `generateFn` is not provided. If both `modelName`
   * and `generateFn` are set, `generateFn` takes precedence and a
   * warning is logged.
   */
  modelName?: string;
  /**
   * Score threshold for "passed" status. Default: 1.0 (only perfect = passed).
   * Scores >= threshold -> passed, 0 < score < threshold -> partially_passed, 0 -> failed.
   */
  passThreshold?: number;
  /**
   * If set, any `trace` field returned by `getContext` is automatically
   * preprocessed via `preprocessTrace()` before being sent to the judge.
   * Pass `{}` for defaults (summary mode, 800 char results).
   *
   * Use either this OR call `preprocessTrace()` manually in `getContext`
   * -- not both, as that would double-preprocess the trace.
   */
  tracePreprocessOptions?: TracePreprocessOptions;
}

/**
 * Create a rubric-based SimpleTest that can be mixed with binary tests
 * in runSimpleTests().
 *
 * The test evaluates criteria against context assembled at runtime,
 * using an LLM judge to determine MET/UNMET for each criterion.
 *
 * If no `generateFn` is provided, one is created automatically using
 * OpenRouter with the default model. If OPENROUTER_API_KEY is not set,
 * the test skips gracefully with a warning.
 *
 * @example
 * ```typescript
 * const test = createRubricTest({
 *   id: "rubric-code-quality",
 *   name: "Code Quality",
 *   description: "Evaluates Rust code quality and idioms",
 *   criteria: [
 *     { weight: 10, requirement: "Uses idiomatic Rust patterns", context: ["code"] },
 *     { weight: -20, requirement: "Uses banned crates", context: ["code"] },
 *   ],
 *   getContext: async (logger) => ({
 *     code: fs.readFileSync("/path/to/code", "utf-8"),
 *   }),
 *   passThreshold: 0.7,
 * });
 * ```
 */
export function createRubricTest(config: RubricTestConfig): SimpleTest {
  const threshold = config.passThreshold ?? 1.0;

  return {
    id: config.id,
    name: config.name,
    description: config.description,
    run: async (logger: Logger): Promise<SimpleTestResult> => {
      let generateFn = config.generateFn;

      if (generateFn && config.modelName) {
        logger.warn(
          `  modelName "${config.modelName}" is ignored when generateFn is provided`,
        );
      }

      if (!generateFn) {
        if (!process.env.OPENROUTER_API_KEY) {
          logger.warn(
            "  OPENROUTER_API_KEY not set -- skipping rubric evaluation",
          );
          return { success: true, skipped: true, error: "OPENROUTER_API_KEY not set" };
        }
        generateFn = config.modelName
          ? createOpenRouterGenerateFn({ model: config.modelName })
          : createOpenRouterGenerateFn();
      }

      try {
        logger.info(
          `  Evaluating ${config.criteria.length} criteria via LLM judge...`,
        );

        const context = await config.getContext(logger);

        if (config.tracePreprocessOptions && context.trace) {
          context.trace = preprocessTrace(
            context.trace,
            config.tracePreprocessOptions,
          );
        }

        const report = await gradeWithRubric(
          config.criteria,
          context,
          generateFn,
        );

        for (const cr of report.reports) {
          const symbol = cr.verdict === "MET" ? "\u2713" : "\u2717";
          const weightStr =
            cr.weight >= 0 ? `+${cr.weight}` : `${cr.weight}`;
          logger.info(
            `  ${symbol} [${weightStr}] ${cr.requirement}: ${cr.reason}`,
          );
        }

        const pct = Math.round(report.score * 100);
        logger.info(
          `  Score: ${pct}% (threshold: ${Math.round(threshold * 100)}%)`,
        );

        const rationale = report.reports
          .map((r) => `[${r.verdict}] ${r.requirement}: ${r.reason}`)
          .join("\n");

        return {
          success: report.score >= threshold,
          score: report.score,
          rationale,
          criteriaResults: report.reports,
        };
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        logger.error(`  Rubric evaluation failed: ${errorMessage}`);
        return {
          success: false,
          error: errorMessage,
          errored: true,
        };
      }
    },
  };
}
