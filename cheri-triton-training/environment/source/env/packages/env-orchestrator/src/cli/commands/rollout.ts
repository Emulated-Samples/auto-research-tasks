/**
 * `env-orchestrator rollout` — full cycle: setup -> solve -> test. Split out
 * of the old environments.ts god-file with no behavior change; the
 * agent-exit-to-grading decision now lives in shared/gradingDecision.ts.
 */

import type {
  EnvironmentDefinition,
  TelemetrySession,
  S3TelemetrySync,
  RolloutPhase,
} from "@hyperfocal/env-base";
import {
  createSession,
  createS3SyncFromEnv,
  ConsoleLogger,
  updateProblemPhase,
  recordForcedAgentSessionEnd,
  finalizeRollout,
  isEnvironmentError,
  aggregateTestResults,
} from "@hyperfocal/env-base";
import { parseProblem, parseModel, defaultPermissionsMode } from "../args.js";
import { loadConfig, getResolvedPaths } from "../../config/yaml-config.js";
import { runAgent } from "../../config/agent-runner.js";
import {
  AgentExecutionError,
  agentExecutionStatus,
} from "../../internal/agent-failure.js";
import { createBaseline, collectDiff } from "../diff-collector.js";
import { loadCredentials } from "../../config/credentials.js";
import {
  type CommandOptions,
  createTelemetryLogger,
  validateAgentCredentials,
} from "./shared/agentPrereqs.js";
import { decideGradingAfterAgentExit } from "./shared/gradingDecision.js";
import {
  getSchemaPath,
  interpolatePrompt,
  warnIfCredentialRefreshUnmentioned,
} from "./shared/promptTemplates.js";
import { ensureSetupSolverIdentity } from "./setup.js";

/**
 * Handle 'rollout' command - full cycle: setup → solve → test
 */
export async function handleRolloutCommand(
  env: EnvironmentDefinition,
  options: CommandOptions = {}
): Promise<void> {
  const problems = await env.listProblems();
  const problemId =
    parseProblem() || problems.find((p) => p.default)?.id || problems[0]?.id;
  const problem = problems.find((p) => p.id === problemId);

  if (!problem) {
    throw new Error(`Problem not found: ${problemId}`);
  }

  const config = loadConfig();
  const paths = getResolvedPaths(config);
  const model = parseModel() || config.agent.defaultModel;
  const schemaPath = getSchemaPath(config);

  // Interpolate prompt with schema info
  const interpolatedPrompt = interpolatePrompt(problem.prompt, config);

  // Warn if awsAccess is enabled but prompt doesn't mention credentials refresh
  // This helps prompt authors remember to include instructions for long-running tasks
  warnIfCredentialRefreshUnmentioned(config, problem.prompt);

  // Determine agent type early (need it for credential validation)
  const agentType = options.agentType || config.agent.type || "claude-code";
  const permissionsMode = options.permissionsMode || config.agent.permissionsMode || defaultPermissionsMode(agentType);
  // Load credentials from .env file (or process.env fallback)
  // OpenCode and Codex use their own auth paths; Anthropic agents need ANTHROPIC_API_KEY.
  const creds = loadCredentials(config);
  const apiKey = creds.anthropicApiKey || "";
  validateAgentCredentials(agentType, creds, config);

  // Start S3 telemetry sync if configured (non-blocking)
  // This uploads logs to S3 periodically during the rollout
  let telemetrySync: S3TelemetrySync | null = null;
  try {
    telemetrySync = createS3SyncFromEnv();
    if (telemetrySync) {
      await telemetrySync.start();
    }
  } catch (error) {
    console.warn("[rollout] Failed to start telemetry sync:", error);
    // Continue without telemetry - non-fatal
  }

  // Create setup telemetry immediately. Tests telemetry is intentionally lazy:
  // setup and agent failures must not manufacture a tests session or artifact for
  // a phase the rollout never reached.
  //
  // Pass the interpolated prompt to setupSession so it's captured in metadata.json.
  // This allows the control plane to display exactly what prompt the agent received.
  const setupSession = createSession(problemId, "setup", "rollout", undefined, interpolatedPrompt);
  let testSession: TelemetrySession | undefined;
  let rolloutPhase: RolloutPhase = "setup";

  const setRolloutPhase = (phase: RolloutPhase): void => {
    rolloutPhase = phase;
    updateProblemPhase(problemId, phase);
  };

  // createSession() has now materialized metadata.json, so explicitly record the
  // initial phase before solver provisioning, pre-cleanup, or setupProblem runs.
  setRolloutPhase("setup");

  // Log rollout start
  setupSession.log("session_start", `Starting rollout`);
  setupSession.log("info", `   Problem: ${problemId}`);
  setupSession.log("info", `   Model: ${model}`);
  if (schemaPath) {
    setupSession.log("info", `   Schema: ${schemaPath}`);
  }
  setupSession.log("info", "=".repeat(60));

  const startTime = Date.now();
  const safeCollectAgentDiff = async (): Promise<void> => {
    try {
      await collectDiff(paths.workspace, problemId);
    } catch (diffError) {
      console.warn("[rollout] Failed to collect agent diff:", diffError);
    }
  };

  try {
    // linux-user environments may audit solver permissions during setup. The
    // identity must therefore exist before setupProblem, not first in runAgent.
    ensureSetupSolverIdentity(permissionsMode);

    // Run cleanup before rollout if --cleanup flag is passed
    const cleanupFlag = process.argv.includes("--cleanup");
    if (cleanupFlag && env.cleanup) {
      setupSession.log("info", "Running pre-rollout cleanup...");
      await env.cleanup(createTelemetryLogger(setupSession));
    }

    // Step 1: Setup (in-process, trusted code)
    setupSession.log("info", "\nStep 1/3: Running setup...");
    await env.setupProblem(problemId, createTelemetryLogger(setupSession));
    setupSession.log("setup_end", "Setup completed");
    setupSession.end("completed");

    // Update phase: setup complete, moving to agent
    setRolloutPhase("agent");

    // Create baseline tag before agent runs (for diff collection)
    try {
      createBaseline(paths.workspace);
    } catch (baselineError) {
      console.warn("[rollout] Failed to create baseline:", baselineError);
      // Non-fatal - continue with rollout
    }

    // Step 2: Solve
    console.log(`\n Step 2/3: Running solve (agent: ${agentType}, permissions: ${permissionsMode})...`);

    let agentResult: Awaited<ReturnType<typeof runAgent>> | undefined;
    let agentError: Error | undefined;
    try {
      agentResult = await runAgent({
        prompt: interpolatedPrompt,
        workspacePath: paths.workspace,
        model,
        config,
        apiKey,
        problemId,
        schemaPath,
        agentType,
        permissionsMode,
      });
    } catch (error) {
      agentError = error instanceof Error ? error : new Error(String(error));
    } finally {
      // Capture what the agent changed before tests or cleanup can mutate the
      // workspace. This is intentionally non-fatal and runs even when solve
      // fails before writing a manifest.
      await safeCollectAgentDiff();
    }

    if (agentError) {
      throw agentError;
    }
    if (!agentResult) {
      throw new Error("Agent did not return an execution result");
    }

    // The agent-exit-to-grading decision (shared/gradingDecision.ts): grade
    // on clean exit, grade-with-warning on timeout, throw otherwise.
    const gradingDecision = decideGradingAfterAgentExit(agentResult);
    if (!gradingDecision.runTests) {
      throw new AgentExecutionError(agentResult);
    }
    const timeoutContinuation = gradingDecision.timeoutContinuation;
    if (timeoutContinuation) {
      console.warn(timeoutContinuation.message);
      recordForcedAgentSessionEnd(
        problemId,
        "failed",
        timeoutContinuation.message,
        timeoutContinuation.data
      );
    } else {
      console.log("Agent completed");
    }

    // Update phase: agent complete, moving to tests
    setRolloutPhase("tests");

    // Step 3: Test (in-process, trusted code)
    testSession = createSession(problemId, "tests", "rollout");
    if (timeoutContinuation) {
      testSession.log("warn", timeoutContinuation.message, timeoutContinuation.data);
    }
    testSession.log("info", "\nStep 3/3: Running tests...");

    const testLogger = createTelemetryLogger(testSession);
    const testResults = await env.runTests(problemId, testLogger);

    // Log each result as structured event
    for (const result of testResults) {
      const icon = result.status === "passed" ? "[OK]"
        : result.status === "partially_passed" ? "[PARTIAL]"
        : result.status === "errored" ? "[ERROR]" : "[FAIL]";
      const scoreStr = result.score !== undefined && result.status === "partially_passed"
        ? `: ${Math.round(result.score * 100)}%` : "";
      testSession.log(
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

    const passedCount = testResults.filter((r) => r.status === "passed").length;
    const partiallyPassedCount = testResults.filter((r) => r.status === "partially_passed").length;
    const failedCount = testResults.filter((r) => r.status === "failed").length;
    const erroredCount = testResults.filter((r) => r.status === "errored").length;
    const skippedCount = testResults.filter((r) => r.status === "skipped").length;
    const duration = Date.now() - startTime;

    // Weighted aggregate rollout score. aggregateTestResults excludes
    // skipped/errored from both numerator and denominator and respects
    // per-test weights declared on each TestResult (defaulting to 1).
    const { rolloutScore } = aggregateTestResults(testResults);

    // Summary with all status types
    const summaryParts = [`${passedCount} passed`];
    if (partiallyPassedCount > 0) {
      summaryParts.push(`${partiallyPassedCount} partially passed`);
    }
    summaryParts.push(`${failedCount} failed`);
    if (erroredCount > 0) {
      summaryParts.push(`${erroredCount} errored`);
    }
    if (skippedCount > 0) {
      summaryParts.push(`${skippedCount} skipped`);
    }
    testSession.log("info", `\nTest results: ${summaryParts.join(", ")} (score: ${Math.round(rolloutScore * 100)}%)`);

    // Handle errors (environment issues) - highest priority
    if (erroredCount > 0) {
      const errorMsg = `${erroredCount} test(s) errored`;
      testSession.log(
        "error",
        `\nRollout errored (${errorMsg})`
      );
      testSession.log("info", `   Duration: ${Math.round(duration / 1000)}s`);
      testSession.end("errored", errorMsg);

      // Finalize rollout status before sync
      finalizeRollout(problemId, "errored", errorMsg, rolloutScore);

      // Final sync before exit
      if (telemetrySync) {
        try {
          await telemetrySync.stop();
        } catch (syncError) {
          console.warn("[rollout] Failed to stop telemetry sync:", syncError);
        }
      }

      // Run cleanup to tear down sandbox resources
      if (env.cleanup) {
        try {
          console.log("\nRunning post-rollout cleanup...");
          await env.cleanup(new ConsoleLogger());
          console.log("[OK] Cleanup completed");
        } catch (cleanupError) {
          console.warn(`[WARN] Cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : cleanupError}`);
        }
      }

      process.exit(1);
    }

    // Handle failures (agent didn't solve) — partially_passed counts as failure
    const notPassedCount = failedCount + partiallyPassedCount;
    if (notPassedCount > 0) {
      const errorMsg = `${notPassedCount} test(s) failed or partially passed`;
      testSession.log(
        "error",
        `\nRollout failed (${errorMsg})`
      );
      testSession.log("info", `   Duration: ${Math.round(duration / 1000)}s`);
      testSession.end("failed", errorMsg);

      // Finalize rollout status before sync
      finalizeRollout(problemId, "failed", errorMsg, rolloutScore);

      // Final sync before exit
      if (telemetrySync) {
        try {
          await telemetrySync.stop();
        } catch (syncError) {
          console.warn("[rollout] Failed to stop telemetry sync:", syncError);
        }
      }

      // Run cleanup to tear down sandbox resources
      if (env.cleanup) {
        try {
          console.log("\nRunning post-rollout cleanup...");
          await env.cleanup(new ConsoleLogger());
          console.log("[OK] Cleanup completed");
        } catch (cleanupError) {
          console.warn(`[WARN] Cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : cleanupError}`);
        }
      }

      process.exit(1);
    }

    testSession.log("session_end", `\nRollout completed successfully`);
    testSession.log("info", `   Duration: ${Math.round(duration / 1000)}s`);
    testSession.end("completed");

    // Finalize rollout status before sync
    finalizeRollout(problemId, "completed", undefined, rolloutScore);

    // Final sync before exit
    if (telemetrySync) {
      try {
        await telemetrySync.stop();
      } catch (syncError) {
        console.warn("[rollout] Failed to stop telemetry sync:", syncError);
      }
    }

    // Run cleanup to tear down sandbox resources after successful rollout
    if (env.cleanup) {
      try {
        console.log("\nRunning post-rollout cleanup...");
        await env.cleanup(new ConsoleLogger());
        console.log("[OK] Cleanup completed");
      } catch (cleanupError) {
        console.warn(`[WARN] Cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : cleanupError}`);
      }
    }
  } catch (error) {
    const duration = Date.now() - startTime;
    const errorMsg = error instanceof Error ? error.message : String(error);

    // Setup executes trusted environment provisioning. Any exception before the
    // agent phase is therefore an infrastructure failure regardless of wording;
    // classifying it through message patterns would turn arbitrary ImportError,
    // permission, or pre-cleanup text into a failed solver attempt.
    //
    // Once the agent starts, preserve its typed failure status. Test-phase
    // exceptions retain the existing environment-error message classification.
    const propagatedAgentStatus = agentExecutionStatus(error);
    const isEnvError = isEnvironmentError(errorMsg);
    const status = rolloutPhase === "setup"
      ? "errored"
      : propagatedAgentStatus || (isEnvError ? "errored" : "failed");

    // End all sessions with appropriate status
    if (!setupSession.hasEnded()) {
      setupSession.end(status, errorMsg);
    }
    if (testSession && !testSession.hasEnded()) {
      testSession.log("error", `\nRollout ${status}`);
      testSession.log("info", `   Duration: ${Math.round(duration / 1000)}s`);
      testSession.log("error", `   Error: ${errorMsg}`);
      testSession.end(status, errorMsg);
    }

    // Finalize rollout status before sync
    finalizeRollout(problemId, status, errorMsg);

    // Final sync before exit
    if (telemetrySync) {
      try {
        await telemetrySync.stop();
      } catch (syncError) {
        console.warn("[rollout] Failed to stop telemetry sync:", syncError);
      }
    }

    // Skip cleanup on setup/agent errors — resources may be needed for debugging.
    // Use `env-orchestrator cleanup` manually after investigation.
    console.log("\n[WARN] Skipping cleanup (resources preserved for debugging).");
    console.log("   Run `env-orchestrator cleanup` manually when done investigating.");

    process.exit(1);
  }
}
