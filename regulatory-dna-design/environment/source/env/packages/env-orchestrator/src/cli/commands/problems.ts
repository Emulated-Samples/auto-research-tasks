/**
 * `env-orchestrator problems` — list available problems. Split out of the
 * old environments.ts god-file with no behavior change.
 */

import type { EnvironmentDefinition } from "@hyperfocal/env-base";

/**
 * Handle 'problems' command - list available problems
 */
export async function handleProblemsCommand(
  env: EnvironmentDefinition
): Promise<void> {
  const problemsList = await env.listProblems();
  const jsonFlag = process.argv.includes("--json");

  if (jsonFlag) {
    process.stdout.write(JSON.stringify(problemsList, null, 2));
  } else {
    console.log("Available problems:\n");
    problemsList.forEach((problem) => {
      const defaultLabel = problem.default ? " (default)" : "";
      console.log(`${problem.id}${defaultLabel}`);
    });
    console.log();
  }
}
