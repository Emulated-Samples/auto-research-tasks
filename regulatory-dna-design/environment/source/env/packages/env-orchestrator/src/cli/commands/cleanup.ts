/**
 * `env-orchestrator cleanup` — tear down test infrastructure. Split out of
 * the old environments.ts god-file with no behavior change.
 */

import type {
  EnvironmentDefinition,
  TelemetrySession,
} from "@hyperfocal/env-base";
import { createSession } from "@hyperfocal/env-base";
import { createTelemetryLogger } from "./shared/agentPrereqs.js";

/**
 * Handle 'cleanup' command - clean up test infrastructure
 */
export async function handleCleanupCommand(
  env: EnvironmentDefinition,
  existingSession?: TelemetrySession
): Promise<void> {
  if (!env.cleanup) {
    console.log("ℹ️  This environment does not implement cleanup.");
    console.log("   No action taken.");
    return;
  }

  // Create or reuse telemetry session
  const session =
    existingSession || createSession("cleanup", "cleanup", "cleanup");

  // Create logger for telemetry
  const logger = createTelemetryLogger(session);

  session.log("cleanup_start", "Running cleanup...");

  try {
    await env.cleanup(logger);
    session.log("cleanup_end", "Cleanup completed successfully");

    if (!existingSession) {
      session.end("completed");
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    session.log("error", `Cleanup failed: ${msg}`);

    if (!existingSession) {
      session.end("failed", msg);
    }

    console.error(`[ERROR] Cleanup failed: ${msg}`);
    process.exit(1);
  }
}
