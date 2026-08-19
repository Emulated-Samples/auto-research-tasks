/**
 * `env-orchestrator prompt` — show the (interpolated) prompt for a problem.
 * Split out of the old environments.ts god-file with no behavior change.
 */

import type { EnvironmentDefinition } from "@hyperfocal/env-base";
import { parseProblem } from "../args.js";
import { loadConfig } from "../../config/yaml-config.js";
import { interpolatePrompt } from "./shared/promptTemplates.js";

/**
 * Handle 'prompt' command - show prompt for a problem
 */
export async function handlePromptCommand(
  env: EnvironmentDefinition
): Promise<void> {
  const problems = await env.listProblems();
  const problemId =
    parseProblem() || problems.find((p) => p.default)?.id || problems[0]?.id;
  const problem = problems.find((p) => p.id === problemId);

  if (!problem) {
    throw new Error(`Problem not found: ${problemId}`);
  }

  const config = loadConfig();
  const interpolatedPrompt = interpolatePrompt(problem.prompt, config);

  console.log(`Prompt for problem: ${problem.id}`);
  console.log("=".repeat(60));
  console.log(interpolatedPrompt);
}
