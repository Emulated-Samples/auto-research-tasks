/**
 * PerCriterionGrader — evaluates each criterion independently in parallel.
 *
 * TypeScript port of Hud's per_criterion_grader.py with a critical fix
 * to negative weight handling. See the FAQ in the plan for details.
 */

import type { CriterionResult } from "../types.js";
import type {
  Criterion,
  CriterionContextType,
  EvaluationReport,
  GenerateFn,
  RubricContext,
} from "./types.js";
import { JUDGE_SYSTEM_PROMPT, buildUserPrompt } from "./prompts.js";

/**
 * Parse a JSON response from the judge LLM.
 *
 * Handles common LLM quirks: markdown fences, stray text before {, etc.
 */
function parseJsonResponse(response: string): Record<string, unknown> {
  // Strip markdown code fences if present
  let cleaned = response.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }

  // Find the first { and last }
  const firstBrace = cleaned.indexOf("{");
  const lastBrace = cleaned.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1) {
    throw new Error(`No JSON object found in response: ${cleaned.slice(0, 200)}`);
  }

  return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1));
}

/**
 * Select context fields for a criterion based on its context requirements.
 *
 * If the criterion specifies context types, only those fields are included.
 * If context is omitted, all available fields are included.
 */
function selectContext(
  fullContext: RubricContext,
  criterionContextTypes?: CriterionContextType[]
): { code?: string; trace?: string; testResults?: string } {
  if (!criterionContextTypes || criterionContextTypes.length === 0) {
    // Default: include all available context
    return { ...fullContext };
  }

  const selected: { code?: string; trace?: string; testResults?: string } = {};
  for (const type of criterionContextTypes) {
    if (type === "code" && fullContext.code !== undefined) {
      selected.code = fullContext.code;
    } else if (type === "trace" && fullContext.trace !== undefined) {
      selected.trace = fullContext.trace;
    } else if (type === "testResults" && fullContext.testResults !== undefined) {
      selected.testResults = fullContext.testResults;
    }
  }
  return selected;
}

/**
 * Evaluate a single criterion against the provided context using the judge LLM.
 */
async function judgeSingleCriterion(
  criterion: Criterion,
  context: RubricContext,
  generateFn: GenerateFn
): Promise<CriterionResult> {
  const selectedContext = selectContext(context, criterion.context);
  const userPrompt = buildUserPrompt(criterion, selectedContext);

  try {
    const response = await generateFn(JUDGE_SYSTEM_PROMPT, userPrompt);
    const result = parseJsonResponse(response);

    const criterionType = criterion.weight < 0 ? "negative" : "positive";
    let passed: boolean;
    let explanation: string;

    if (criterionType === "negative") {
      passed = !(result.issue_present as boolean ?? true);
      explanation = (result.explanation as string) ?? "No explanation provided";
    } else {
      passed = (result.criteria_met as boolean) ?? false;
      explanation = (result.explanation as string) ?? "No explanation provided";
    }

    return {
      requirement: criterion.requirement,
      weight: criterion.weight,
      verdict: passed ? "MET" : "UNMET",
      reason: explanation,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      requirement: criterion.requirement,
      weight: criterion.weight,
      verdict: "UNMET",
      reason: `Error during evaluation: ${errorMessage}`,
    };
  }
}

/**
 * Aggregate criterion reports into a final score.
 *
 * IMPORTANT: This fixes Hud's negative weight bug.
 *
 * Hud's implementation: MET * weight for ALL criteria, which means
 * negative criteria that are MET (issue absent, good) SUBTRACT from score.
 *
 * Our fix:
 * - Positive criteria: MET → +weight, UNMET → 0
 * - Negative criteria: MET (issue absent, good) → 0, UNMET (issue present, bad) → +weight (negative)
 */
function aggregate(reports: CriterionResult[]): EvaluationReport {
  const totalPositiveWeight = reports
    .filter((r) => r.weight > 0)
    .reduce((sum, r) => sum + r.weight, 0);

  let weightedSum = 0;
  for (const report of reports) {
    if (report.weight >= 0) {
      // Positive criterion: reward when MET
      weightedSum += report.verdict === "MET" ? report.weight : 0;
    } else {
      // Negative criterion: penalty when UNMET (bad behavior IS present)
      weightedSum += report.verdict === "UNMET" ? report.weight : 0;
    }
  }

  const rawScore =
    totalPositiveWeight > 0 ? weightedSum / totalPositiveWeight : 0;
  const score = Math.max(0, Math.min(1, rawScore));

  return { score, reports };
}

/**
 * Grade content against a rubric using per-criterion parallel evaluation.
 *
 * Each criterion is evaluated independently by the judge LLM, then
 * results are aggregated using weighted scoring.
 *
 * @param criteria - The rubric criteria to evaluate against
 * @param context - Structured context (code, trace, testResults)
 * @param generateFn - Async LLM call function
 * @returns Evaluation report with score (0-1) and per-criterion results
 */
export async function gradeWithRubric(
  criteria: Criterion[],
  context: RubricContext,
  generateFn: GenerateFn
): Promise<EvaluationReport> {
  // Evaluate all criteria in parallel
  const reports = await Promise.all(
    criteria.map((criterion) =>
      judgeSingleCriterion(criterion, context, generateFn)
    )
  );

  return aggregate(reports);
}
