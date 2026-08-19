/**
 * `env-orchestrator setup` — run a problem's setup. This is the handler
 * that runs IN-IMAGE during every packaged docker build (render/dockerfile
 * invokes `setup --problem <id>`), which is why it gets its own reviewable
 * module. Split out of the old environments.ts god-file with no behavior
 * change.
 */

import type {
  EnvironmentDefinition,
  PermissionsMode,
  TelemetrySession,
} from "@hyperfocal/env-base";
import { createSession } from "@hyperfocal/env-base";
import { parseProblem, defaultPermissionsMode } from "../args.js";
import { loadConfig } from "../../config/yaml-config.js";
import { ensureAgentUser } from "../../config/workspace-isolation.js";

/**
 * Provision the OS identity before trusted setup when linux-user isolation is
 * selected. Environments may perform setup-time permission audits; deferring
 * user creation until runAgent() makes those audits impossible on fresh
 * workers and leaves a gap between setup and the declared security model.
 *
 * The injectable callback keeps the policy testable without mutating the test
 * host's user database.
 */
export function ensureSetupSolverIdentity(
  permissionsMode: PermissionsMode,
  provision: () => void = ensureAgentUser
): void {
  if (permissionsMode === "linux-user") provision();
}

/**
 * Handle 'setup' command - run setup for a problem
 */
export async function handleSetupCommand(
  env: EnvironmentDefinition,
  existingSession?: TelemetrySession
): Promise<void> {
  const problems = await env.listProblems();
  const problemId =
    parseProblem() || problems.find((p) => p.default)?.id || problems[0]?.id;

  if (!problemId) {
    throw new Error("No problem specified and no default problem found");
  }

  // Create or reuse telemetry session
  const session = existingSession || createSession(problemId, "setup", "setup");

  const config = loadConfig();
  const agentType = config.agent.type || "claude-code";
  const permissionsMode =
    config.agent.permissionsMode || defaultPermissionsMode(agentType);

  session.log("setup_start", `Running setup for problem: ${problemId}`);
  session.log("info", "=".repeat(60));

  try {
    ensureSetupSolverIdentity(permissionsMode);
    await env.setupProblem(problemId);
    session.log("setup_end", "\nSetup completed successfully");

    // Only end session if we created it
    if (!existingSession) {
      session.end("completed");
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    session.log("error", `\nSetup failed: ${errorMsg}`);

    if (!existingSession) {
      session.end("failed", errorMsg);
    }
    process.exit(1);
  }
}
