/**
 * Environment loading utilities
 */

import type { EnvironmentDefinition } from "@hyperfocal/env-base";
import { pathToFileURL } from "url";

/**
 * Load environment dynamically from the specified path
 *
 * @param environmentPath - Path to the environment dist directory
 */
export async function loadEnvironment(
  environmentPath: string
): Promise<EnvironmentDefinition | null> {
  if (!environmentPath) {
    return null;
  }

  try {
    // Use dynamic import for ESM modules
    const moduleUrl = pathToFileURL(environmentPath + "/index.js").href;
    const module = await import(moduleUrl);

    // Get default export
    const env = module.default || module;

    // Verify required methods exist
    if (!env.listProblems || !env.setupProblem || !env.runTests) {
      throw new Error(
        "Environment must export EnvironmentDefinition with: listProblems, setupProblem, runTests"
      );
    }

    // CAUTION: this object is rebuilt from an explicit method list, so any
    // method added to EnvironmentDefinition MUST also be added here or it is
    // silently dropped from every loaded environment (this ate solveProblem
    // once — see docs 2-harbor-fleet-conversion FAQ).
    return {
      listProblems: env.listProblems.bind(env),
      setupProblem: env.setupProblem.bind(env),
      runTests: env.runTests.bind(env),
      cleanup: env.cleanup?.bind(env),
      // Optional programmatic reference solution (solution: "hook" problems);
      // consumed by `solve-oracle`.
      solveProblem: env.solveProblem?.bind(env),
      // Optional packaging-spec amendment hook; consumed by `harbor package`.
      packageProblem: env.packageProblem?.bind(env),
    };
  } catch (error) {
    console.error(
      `\n⚠️  Failed to load environment from ${environmentPath}\n` +
        `Error: ${error instanceof Error ? error.message : String(error)}\n`
    );
    return null;
  }
}

/**
 * Show environment not configured error
 */
export function showEnvironmentNotConfiguredError(): void {
  console.error("\n❌ No environment configured.\n");
  console.log("To set up an environment:");
  console.log("  1. Create a hyperfocal.yaml in your environment repo");
  console.log("  2. Set paths.environmentDist to point to your environment module\n");
  console.log("Run: env-orchestrator env init --name my-environment\n");
}
