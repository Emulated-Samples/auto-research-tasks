/**
 * `env-orchestrator solve` — have an agent solve a problem. Split out of
 * the old environments.ts god-file with no behavior change.
 */

import type {
  EnvironmentDefinition,
  TelemetrySession,
} from "@hyperfocal/env-base";
import { createSession } from "@hyperfocal/env-base";
import { parseProblem, parseModel, defaultPermissionsMode } from "../args.js";
import { loadConfig, getResolvedPaths } from "../../config/yaml-config.js";
import { runAgent } from "../../config/agent-runner.js";
import { AgentExecutionError, agentExecutionStatus } from "../../internal/agent-failure.js";
import { loadCredentials } from "../../config/credentials.js";
import {
  type CommandOptions,
  validateAgentCredentials,
} from "./shared/agentPrereqs.js";
import {
  getSchemaPath,
  interpolatePrompt,
  warnIfCredentialRefreshUnmentioned,
} from "./shared/promptTemplates.js";

export type { CommandOptions };

/**
 * Handle 'solve' command - have an agent solve a problem
 */
export async function handleSolveCommand(
  env: EnvironmentDefinition,
  existingSession?: TelemetrySession,
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
  const workspacePath = paths.workspace;
  const schemaPath = getSchemaPath(config);

  // Interpolate prompt with schema info
  const interpolatedPrompt = interpolatePrompt(problem.prompt, config);

  // Warn if awsAccess is enabled but prompt doesn't mention credentials refresh
  // This helps prompt authors remember to include instructions for long-running tasks
  warnIfCredentialRefreshUnmentioned(config, problem.prompt);

  // Determine agent type: CLI flag > config > default
  const agentType = options.agentType || config.agent.type || "claude-code";
  const permissionsMode = options.permissionsMode || config.agent.permissionsMode || defaultPermissionsMode(agentType);
  // Load credentials from .env file (or process.env fallback)
  // OpenCode and Codex use their own auth paths; Anthropic agents need ANTHROPIC_API_KEY.
  const creds = loadCredentials(config);
  const apiKey = creds.anthropicApiKey || "";
  validateAgentCredentials(agentType, creds, config);

  // Create or reuse telemetry session.
  // Pass model + agentType so they end up on the per-problem metadata.json
  // SessionSummary (the rollout path goes through agent-entry.ts which does
  // the same thing; this branch handles `solve` invoked locally).
  const session =
    existingSession ||
    createSession(problemId, "agent", "solve", model, undefined, agentType);

  session.log("session_start", `Solving problem: ${problemId}`);
  session.log("info", `Using model: ${model}`);
  session.log("info", `Workspace: ${workspacePath}`);
  if (schemaPath) {
    session.log("info", `Output schema: ${schemaPath}`);
  }
  session.log("info", "=".repeat(60));
  session.log("info", `Agent type: ${agentType}, permissions: ${permissionsMode}`);

  try {
    const agentResult = await runAgent({
      prompt: interpolatedPrompt,
      workspacePath,
      model,
      config,
      apiKey,
      problemId,
      schemaPath,
      agentType,
      permissionsMode,
    });

    if (agentResult.exitCode !== 0) {
      throw new AgentExecutionError(agentResult);
    }

    // Only end session if we created it
    if (!existingSession) {
      session.end("completed");
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    session.log("error", `Solve failed: ${errorMsg}`);

    if (!existingSession) {
      session.end(agentExecutionStatus(error) || "failed", errorMsg);
    }
    throw error;
  }
}
