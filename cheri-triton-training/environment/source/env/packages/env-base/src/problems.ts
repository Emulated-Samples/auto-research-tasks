/**
 * Problem loading utilities
 *
 * Supports loading problem definitions from YAML or JSON files.
 * YAML is preferred for readability of multiline prompts.
 */

import * as fs from "fs";
import * as path from "path";
import { parse as parseYaml } from "yaml";
import type { Problem } from "./types.js";

/**
 * Mustache-style placeholder substitution map for problem prompts.
 * Keys are referenced in YAML as `{{key}}` and replaced at load time.
 */
export interface PromptTemplateValues {
  [key: string]: string;
}

const TEMPLATE_TOKEN = /\{\{\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\}\}/g;

/**
 * Substitute every `{{key}}` token in `prompt` with `values[key]`.
 *
 * Throws on unknown keys so a typo in problems.yaml fails the load
 * loudly rather than leaking a literal `{{foo}}` to the agent. This
 * also doubles as a check that the template-value map covers everything
 * the YAML references.
 */
function applyTemplate(
  prompt: string,
  values: PromptTemplateValues,
  problemId: string,
): string {
  return prompt.replace(TEMPLATE_TOKEN, (match, key: string) => {
    if (!Object.prototype.hasOwnProperty.call(values, key)) {
      throw new Error(
        `Problem '${problemId}' references unknown template key '{{${key}}}'. ` +
          `Add it to the templateValues map passed to loadProblems(), or remove ` +
          `the reference from problems.yaml.`,
      );
    }
    return values[key];
  });
}

/**
 * Load problems from a YAML or JSON file.
 *
 * Supports both formats for backwards compatibility.
 * YAML is preferred for new environments as it handles multiline prompts better.
 *
 * @param filePath - Path to the problems file (.yaml, .yml, or .json)
 * @param templateValues - Optional Mustache-style substitution map. When
 *                         provided, every `{{key}}` token in `prompt` is
 *                         replaced with the corresponding value. Unknown
 *                         keys throw.
 * @returns Array of Problem definitions
 * @throws Error if file not found, invalid format, or validation fails
 */
export function loadProblems(
  filePath: string,
  templateValues?: PromptTemplateValues,
): Problem[] {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Problems file not found: ${filePath}`);
  }

  const content = fs.readFileSync(filePath, "utf-8");
  const ext = path.extname(filePath).toLowerCase();

  let parsed: unknown;

  if (ext === ".yaml" || ext === ".yml") {
    parsed = parseYaml(content);
  } else if (ext === ".json") {
    parsed = JSON.parse(content);
  } else {
    throw new Error(
      `Unsupported file format: ${ext}. Use .yaml, .yml, or .json`
    );
  }

  // Validate the structure
  if (!Array.isArray(parsed)) {
    throw new Error("Problems file must contain an array of problem definitions");
  }

  for (const problem of parsed) {
    if (typeof problem !== "object" || problem === null) {
      throw new Error("Each problem must be an object");
    }

    if (typeof problem.id !== "string" || !problem.id.trim()) {
      throw new Error(`Problem missing or invalid 'id' field`);
    }

    if (typeof problem.prompt !== "string" || !problem.prompt.trim()) {
      throw new Error(`Problem '${problem.id}' missing or invalid 'prompt' field`);
    }

    if (problem.default !== undefined && typeof problem.default !== "boolean") {
      throw new Error(
        `Problem '${problem.id}' has invalid 'default' field (must be boolean)`
      );
    }

    // Contract v2 removed problemStateRef. Reject it by name (rather than
    // silently ignoring it) so envs pinned to an older contract fail loudly
    // at their desk when they re-pin — silent-typo yaml is how bugs hide.
    if (problem.problemStateRef !== undefined) {
      throw new Error(
        `Problem '${problem.id}' declares 'problemStateRef', which was removed ` +
          `(contract v2): the packaging commit is the problem state — express ` +
          `per-problem starting states inside setupProblem(problemId), which runs ` +
          `during the image build and can check out any branch`
      );
    }

    // Contract v2 removed the solution mode field; the mode is now derived
    // from what the environment/problem actually provides.
    if (problem.solution !== undefined) {
      throw new Error(
        `Problem '${problem.id}' declares 'solution', which was removed ` +
          `(contract v2): solution mode is now derived — implement solveProblem() ` +
          `for programmatic solutions, declare solutionRef or solutionPatch for ` +
          `declarative ones, or declare nothing for no reference solution`
      );
    }

    if (
      problem.solutionRef !== undefined &&
      (typeof problem.solutionRef !== "string" || !problem.solutionRef.trim())
    ) {
      throw new Error(
        `Problem '${problem.id}' has invalid 'solutionRef' field (must be a non-empty string)`
      );
    }

    if (
      problem.solutionPatch !== undefined &&
      (typeof problem.solutionPatch !== "string" || !problem.solutionPatch.trim())
    ) {
      throw new Error(
        `Problem '${problem.id}' has invalid 'solutionPatch' field (must be a non-empty string)`
      );
    }

    // A problem's reference solution is either a git ref or a committed
    // patch file, never both — refuse the ambiguity at load.
    if (problem.solutionRef !== undefined && problem.solutionPatch !== undefined) {
      throw new Error(
        `Problem '${problem.id}' declares both 'solutionRef' and 'solutionPatch'; ` +
          `they are mutually exclusive — declare exactly one`
      );
    }

    // Contract v2 renamed oracleMinScore. Reject the old name so the value
    // is carried over instead of silently ignored.
    if (problem.oracleMinScore !== undefined) {
      throw new Error(
        `Problem '${problem.id}' declares 'oracleMinScore', which was renamed to ` +
          `minReplayScore (the floor a solution-replay run must clear; replay of ` +
          `the reference solution should ideally score 1.0 — the floor is the ` +
          `concession for calibrated/rubric envs whose gold honestly scores below 1.0)`
      );
    }

    // minReplayScore tunes the solution-replay gate for calibrated
    // continuous-scoring problems (gold < 1.0 by design). It is a proportion
    // of full reward: 0 or negative would gate nothing, > 1 is unsatisfiable.
    if (problem.minReplayScore !== undefined) {
      const v: unknown = problem.minReplayScore;
      if (typeof v !== "number" || !Number.isFinite(v) || v <= 0 || v > 1) {
        throw new Error(
          `Problem '${problem.id}' has invalid 'minReplayScore' field ` +
            `(must be a number in (0, 1]; got ${JSON.stringify(v)})`
        );
      }
    }

    if (problem.mcp !== undefined) {
      if (typeof problem.mcp !== "object" || problem.mcp === null || Array.isArray(problem.mcp)) {
        throw new Error(
          `Problem '${problem.id}' has invalid 'mcp' field (must be an object)`
        );
      }
      for (const [service, config] of Object.entries(problem.mcp)) {
        if (config === undefined) continue;
        if (typeof config !== "object" || config === null || Array.isArray(config)) {
          throw new Error(
            `Problem '${problem.id}' has invalid 'mcp.${service}' field (must be an object)`
          );
        }
      }
    }
  }

  const problems = parsed as Problem[];

  // Template substitution runs after validation so we know `prompt` is
  // present and a string. When no values are passed we leave prompts
  // untouched (preserves pre-templating callers).
  if (templateValues) {
    for (const problem of problems) {
      problem.prompt = applyTemplate(problem.prompt, templateValues, problem.id);
    }
  }

  return problems;
}

/**
 * Find the problems file in a directory.
 *
 * Searches for problems.yaml, problems.yml, then problems.json (in that order).
 * Returns the first file found, preferring YAML over JSON.
 *
 * @param directory - Directory to search in
 * @returns Full path to the problems file, or null if not found
 */
export function findProblemsFile(directory: string): string | null {
  const candidates = ["problems.yaml", "problems.yml", "problems.json"];

  for (const candidate of candidates) {
    const fullPath = path.join(directory, candidate);
    if (fs.existsSync(fullPath)) {
      return fullPath;
    }
  }

  return null;
}

/**
 * Load problems from a directory, auto-detecting the file format.
 *
 * Convenience function that combines findProblemsFile and loadProblems.
 *
 * @param directory - Directory containing the problems file
 * @param templateValues - Optional Mustache substitution map; see loadProblems()
 * @returns Array of Problem definitions
 * @throws Error if no problems file found or validation fails
 */
export function loadProblemsFromDirectory(
  directory: string,
  templateValues?: PromptTemplateValues,
): Problem[] {
  const problemsFile = findProblemsFile(directory);

  if (!problemsFile) {
    throw new Error(
      `No problems file found in ${directory}. ` +
        `Expected one of: problems.yaml, problems.yml, or problems.json`
    );
  }

  return loadProblems(problemsFile, templateValues);
}
