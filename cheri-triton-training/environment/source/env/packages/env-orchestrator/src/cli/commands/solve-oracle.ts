/**
 * `env-orchestrator solve-oracle [--problem <id>]` — run the environment's
 * programmatic reference solution (the solveProblem() hook, decision D2).
 *
 * When the environment module implements solveProblem(), the hook owns the
 * gold state (contract v2 ladder) — envs whose setup perturbs the workspace
 * at runtime have no git ref that ever holds the solved tree. Two callers:
 *   - `harbor package`: runs this as root in the just-built image to
 *     produce solution/solution.patch (harbor/solutions.ts). It must NOT be
 *     exec'd from a packaged solve.sh: harbor runs solve.sh as the agent
 *     user, and the image lockdown correctly denies that user packages/ and
 *     environment/;
 *   - dev boxes: run it natively to check a hook before packaging.
 *
 * The subcommand keeps its historical "solve-oracle" name on purpose: baked
 * task images invoke it by name, so renaming it is a separate decision.
 */

import { ConsoleLogger, type EnvironmentDefinition } from "@hyperfocal/env-base";
import { parseProblem } from "../args.js";

export async function handleSolveOracleCommand(
  env: EnvironmentDefinition
): Promise<void> {
  const problems = await env.listProblems();
  const problemId =
    parseProblem() || problems.find((p) => p.default)?.id || problems[0]?.id;

  if (!problemId) {
    throw new Error("No problem specified and no default problem found");
  }
  if (!problems.some((p) => p.id === problemId)) {
    throw new Error(`Problem not found: ${problemId}`);
  }

  if (!env.solveProblem) {
    throw new Error(
      "This environment module does not implement solveProblem() — " +
        "`solve-oracle` only works for environments with a programmatic " +
        "reference solution. Implement solveProblem(problemId, logger) in " +
        "the environment module, or declare solutionRef/solutionPatch in " +
        "problems.yaml instead."
    );
  }

  const logger = new ConsoleLogger();
  logger.info(`[solve-oracle] Running solveProblem("${problemId}")...`);
  await env.solveProblem(problemId, logger);
  logger.info(`[solve-oracle] solveProblem("${problemId}") completed`);
}
