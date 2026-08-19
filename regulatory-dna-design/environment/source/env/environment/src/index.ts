/**
 * varseq VEP Environment — grader entry point.
 *
 * Host-side pytest grader (kafka-streams mold): runTests spawns the hidden
 * behavioral-tests/ suite against the agent's workspace varseq, weights each
 * test via the registry, then applies multiplicative catastrophe gates.
 * See /hyperfocal/verifier_design.md.
 */
import * as path from "path";
import { fileURLToPath } from "url";
import type {
  EnvironmentDefinition,
  Logger,
  Problem,
  TestResult,
} from "@hyperfocal/env-base";
import {
  ConsoleLogger,
  loadProblemsFromDirectory,
  runSimpleTests,
} from "@hyperfocal/env-base";
import { getTestsForProblem } from "./grader/index.js";
import { applyGraderScoring } from "./grader/registry.js";
import { applyCatastropheMultiplier } from "./grader/catastrophe.js";
import { setupProblem as runSetup } from "./setup/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Problems loaded from environment/problems.yaml (".." because this runs from dist/).
const ENV_DIR = path.resolve(__dirname, "..");
const problems: Problem[] = loadProblemsFromDirectory(ENV_DIR);

class Environment implements EnvironmentDefinition {
  async listProblems(): Promise<Problem[]> {
    return problems;
  }

  async setupProblem(problemId?: string, logger?: Logger): Promise<void> {
    await runSetup(problemId, logger ?? new ConsoleLogger());
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    const tests = getTestsForProblem(problemId);
    logger.info(`Running ${tests.length} grader batch(es) for '${problemId}'`);
    const raw = await runSimpleTests(tests, logger);
    const scored = applyGraderScoring(raw);
    return applyCatastropheMultiplier(scored, logger);
  }
}

export default new Environment();
