/**
 * `env-orchestrator test` — run a problem's native tests. Also the engine
 * behind `harbor grade` (the in-container verifier reuses this handler so a
 * packaged image can never grade differently from a dev box). Split out of
 * the old environments.ts god-file with no behavior change.
 */

import type {
  EnvironmentDefinition,
  TestResult,
  TelemetrySession,
} from "@hyperfocal/env-base";
import {
  createSession,
  ConsoleLogger,
  writeHarborRewards,
} from "@hyperfocal/env-base";
import { parseProblem } from "../args.js";
import { createTelemetryLogger } from "./shared/agentPrereqs.js";

/**
 * Handle 'test' command - run tests for a problem
 */
export async function handleTestCommand(
  env: EnvironmentDefinition,
  skipExit = false,
  existingSession?: TelemetrySession
): Promise<TestResult[]> {
  const problems = await env.listProblems();
  const problemId =
    parseProblem() || problems.find((p) => p.default)?.id || problems[0]?.id;

  if (!problemId) {
    throw new Error("No problem specified and no default problem found");
  }

  // Run cleanup before tests if --cleanup flag is passed
  const cleanupFlag = process.argv.includes("--cleanup");
  if (cleanupFlag && env.cleanup) {
    console.log("Running pre-test cleanup...");
    await env.cleanup(new ConsoleLogger());
  }

  const xmlFlag = process.argv.includes("--xml");
  const jsonFlag = process.argv.includes("--json");
  const format: "human" | "xml" | "json" = xmlFlag
    ? "xml"
    : jsonFlag
      ? "json"
      : "human";

  // Create or reuse telemetry session
  const session = existingSession || createSession(problemId, "tests", "test");

  // Create logger that writes to session
  const telemetryLogger = createTelemetryLogger(session);

  // For non-human formats, use silent logger for actual test output
  // but still log to session
  const testLogger =
    format === "human" ? telemetryLogger : createTelemetryLogger(session);

  session.log("test_start", `Running tests for problem: ${problemId}`);
  session.log("info", "=".repeat(60));

  const results = await env.runTests(problemId, testLogger);

  // Log each result as structured event
  for (const result of results) {
    const icon = result.status === "passed" ? "[OK]"
      : result.status === "partially_passed" ? "[PARTIAL]"
      : result.status === "errored" ? "[ERROR]" : "[FAIL]";
    const scoreStr = result.score !== undefined && result.status === "partially_passed"
      ? `: ${Math.round(result.score * 100)}%` : "";
    session.log(
      "test_result",
      `${icon} ${result.name}${scoreStr} (${result.duration}ms)`,
      {
        testId: result.id,
        name: result.name,
        status: result.status,
        duration: result.duration,
        score: result.score,
        rationale: result.rationale,
        criteriaResults: result.criteriaResults,
        rubricModel: result.rubricModel,
        error: result.error,
        output: result.output,
      }
    );
  }

  // Harbor dual-write (always on): the flat per-test rewards dict under
  // logs/<problem>/harbor/. This is the ONLY reward source harbor task
  // images consume (tests/test.sh copies it to /logs/verifier/reward.json),
  // and always-on means every dev run exercises the exact path that ships.
  // Non-fatal: emission failure must never break platform test runs.
  try {
    const harbor = writeHarborRewards(problemId, results);
    session.log("info", `Harbor rewards written: ${harbor.rewardPath}`);
  } catch (harborError) {
    session.log(
      "info",
      `Harbor reward emission failed (non-fatal): ${
        harborError instanceof Error ? harborError.message : String(harborError)
      }`
    );
  }

  // Summary - count each status type
  const passed = results.filter((r) => r.status === "passed").length;
  const partiallyPassed = results.filter((r) => r.status === "partially_passed").length;
  const failed = results.filter((r) => r.status === "failed").length;
  const errored = results.filter((r) => r.status === "errored").length;

  const summaryParts = [`[OK] ${passed} passed`];
  if (partiallyPassed > 0) {
    summaryParts.push(`[PARTIAL] ${partiallyPassed} partially passed`);
  }
  summaryParts.push(`[FAIL] ${failed} failed`);
  if (errored > 0) {
    summaryParts.push(`[ERROR] ${errored} errored`);
  }
  session.log("info", `\n${summaryParts.join(", ")}`);

  // Format output for non-human formats
  if (format === "json") {
    const problem = problems.find((p) => p.id === problemId);
    const output = JSON.stringify(
      {
        problem,
        summary: {
          total: results.length,
          passed,
          partiallyPassed,
          failed,
          errored,
          duration: results.reduce((sum, r) => sum + r.duration, 0),
        },
        tests: results,
      },
      null,
      2
    );
    process.stdout.write(output);
  }

  // Determine session end status:
  // - errored tests = environment issue = session "errored"
  // - failed or partially_passed tests = agent didn't fully solve = session "failed"
  // - all passed = session "completed"
  const notPassedCount = failed + partiallyPassed;
  if (!existingSession) {
    if (errored > 0) {
      session.end("errored", `${errored} test(s) errored`);
    } else if (notPassedCount > 0) {
      session.end("failed", `${notPassedCount} test(s) failed or partially passed`);
    } else {
      session.end("completed");
    }
  }

  // Exit with error if any tests failed, partially passed, or errored
  const hasFailures = notPassedCount > 0 || errored > 0;

  if (!skipExit) {
    process.exit(hasFailures ? 1 : 0);
  }

  return results;
}
