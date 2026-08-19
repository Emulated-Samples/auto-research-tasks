/**
 * Judge system prompts for rubric evaluation.
 *
 * Adapted from Hud's PerCriterionGrader DEFAULT_SYSTEM_PROMPT
 * (per_criterion_grader.py lines 10-55). Extended with support
 * for enriched context types (code, trace, testResults).
 */

/**
 * System prompt for the per-criterion judge LLM.
 *
 * Instructs the model to evaluate a single criterion against provided context,
 * returning structured JSON with a verdict and explanation.
 */
export const JUDGE_SYSTEM_PROMPT = `You are evaluating whether a specific criterion is true of the \
provided output. Each criterion describes either a desired trait (positive) or an \
undesired trait (negative). You will be told which via the <criterion_type> field.

Your task is to determine whether the criterion, as written, is satisfied (for positive \
criteria) or whether the issue described is present (for negative criteria) based on the \
full context provided. Do not make any value judgements about the quality or usefulness \
of the output — only evaluate literal truth.

Evaluation rules:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- For length requirements: Carefully measure the number of words, characters, items, etc.
- For code quality criteria: Evaluate based on the actual code provided, not assumptions.
- For process criteria: Evaluate based on the trace of actions provided, not assumptions.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.

Your response must be valid JSON:

If <criterion_type> is "positive":
- Return "criteria_met": true if the requirement is fully satisfied.
- Return "criteria_met": false if the requirement is not satisfied.

If <criterion_type> is "negative":
- Return "issue_present": true if the issue is present in the output.
- Return "issue_present": false if the issue is absent.

Always include a concise explanation justifying your answer.

Examples:

Positive criterion:
{
"criteria_met": true,
"explanation": "The implementation uses Result types for error handling throughout, with no unwrap() calls."
}

Negative criterion:
{
"issue_present": true,
"explanation": "The code imports the ndarray crate in Cargo.toml, which is a banned external tensor library."
}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix.`;

/**
 * Build the user prompt for a single criterion evaluation.
 *
 * Assembles the criterion, its type (positive/negative), and only the
 * context fields that the criterion requests.
 */
export function buildUserPrompt(
  criterion: { weight: number; requirement: string; context?: string[] },
  contextParts: { code?: string; trace?: string; testResults?: string }
): string {
  const criterionType = criterion.weight < 0 ? "negative" : "positive";

  let contextSections = "";

  if (contextParts.code !== undefined) {
    contextSections += `\n<code>\n${contextParts.code}\n</code>`;
  }

  if (contextParts.trace !== undefined) {
    contextSections += `\n<trace>\n${contextParts.trace}\n</trace>`;
  }

  if (contextParts.testResults !== undefined) {
    contextSections += `\n<test_results>\n${contextParts.testResults}\n</test_results>`;
  }

  return `<criterion_type>${criterionType}</criterion_type>

<criterion>
${criterion.requirement}
</criterion>
${contextSections}`;
}
