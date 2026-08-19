/**
 * Main CLI orchestrator
 */

import { parseCommand, parseAgentType, parsePermissionsMode, defaultPermissionsMode } from "./args.js";
import { handleEnvCommand } from "./commands/env.js";
import { runAgentEntry } from "../internal/agent-entry.js";
import { showEnvironmentNotConfiguredError } from "./environment.js";
import { showHelp } from "./help.js";
import { handleProblemsCommand } from "./commands/problems.js";
import { handlePromptCommand } from "./commands/prompt.js";
import { handleSetupCommand } from "./commands/setup.js";
import { handleTestCommand } from "./commands/test.js";
import { handleSolveCommand } from "./commands/solve.js";
import { handleRolloutCommand } from "./commands/rollout.js";
import { handleValidateManifestCommand } from "./commands/validate-manifest.js";
import { handleCheckCommand } from "./commands/check.js";
import { handleCleanupCommand } from "./commands/cleanup.js";
import { handleUploadTelemetryCommand } from "./commands/upload-telemetry.js";
import { loadConfig } from "../config/yaml-config.js";
import { handleHarborCommand } from "./commands/harbor.js";
import { handleSolveOracleCommand } from "./commands/solve-oracle.js";
import { loadEnvCommandContext } from "./env-context.js";

/**
 * Main CLI handler
 */
export async function main(): Promise<void> {
  try {
    const { command } = parseCommand();

    // Parse --agent-type flag
    const agentType = parseAgentType();
    switch (command) {
      case "env":
        await handleEnvCommand();
        break;

      case "_internal-run-agent":
        // Internal command for spawning agent in child process
        await runAgentEntry();
        break;

      case "validate-manifest":
        await handleValidateManifestCommand();
        break;

      case "check":
        await handleCheckCommand();
        break;

      case "upload-telemetry":
        // One-shot upload of a post-rollout artifact (e.g. QA report) under the
        // rollout's telemetry prefix. No environment load required.
        await handleUploadTelemetryCommand();
        break;

      case "harbor":
        // Harbor packaging family (package | validate | run). Operates on an
        // env repo from the OUTSIDE — no environment load required.
        await handleHarborCommand();
        break;

      case "problems":
      case "prompt":
      case "setup":
      case "test":
      case "solve":
      case "solve-oracle":
      case "rollout":
      case "cleanup": {
        // `--cloud modal` replays this same command inside an ephemeral Modal
        // sandbox against the current working tree (GPU dev loop) instead of
        // executing locally. Handled before config/env load: the local
        // machine may lack the GPU the env needs.
        const cloudIdx = process.argv.indexOf("--cloud");
        if (cloudIdx !== -1) {
          const venue = process.argv[cloudIdx + 1];
          if (venue !== "modal") {
            throw new Error(`--cloud only supports "modal" (got "${venue ?? ""}")`);
          }
          const { handleCloudCommand } = await import("./commands/cloud.js");
          await handleCloudCommand();
          break;
        }
        // Config load, env-var exports, credential injection, and the
        // environment module import — shared with `harbor grade` (the
        // in-container verifier), see env-context.ts.
        const ctx = await loadEnvCommandContext();
        if (!ctx) {
          showEnvironmentNotConfiguredError();
          process.exit(1);
        }
        const env = ctx.env;

        switch (command) {
          case "problems":
            await handleProblemsCommand(env);
            break;
          case "prompt":
            await handlePromptCommand(env);
            break;
          case "setup":
            await handleSetupCommand(env);
            break;
          case "test":
            await handleTestCommand(env);
            break;
          case "solve": {
            const solveConfig = loadConfig();
            const solveAgentType = agentType || solveConfig.agent.type || "claude-code";
            const solvePermsMode =
              parsePermissionsMode() ||
              solveConfig.agent.permissionsMode ||
              defaultPermissionsMode(solveAgentType);
            await handleSolveCommand(env, undefined, {
              permissionsMode: solvePermsMode,
              agentType,
            });
            break;
          }
          case "solve-oracle":
            // Programmatic reference solution (solution: "hook") — trusted
            // env code, no agent involved. Runs as root inside packaged
            // containers (via solution/solve.sh) and natively on dev boxes.
            await handleSolveOracleCommand(env);
            break;
          case "rollout": {
            const rolloutConfig = loadConfig();
            const rolloutAgentType = agentType || rolloutConfig.agent.type || "claude-code";
            const rolloutPermsMode =
              parsePermissionsMode() ||
              rolloutConfig.agent.permissionsMode ||
              defaultPermissionsMode(rolloutAgentType);
            console.log(`🔒 Permissions mode: ${rolloutPermsMode}`);
            if (agentType) {
              console.log(`🤖 Using agent type: ${agentType}`);
            }
            await handleRolloutCommand(env, {
              permissionsMode: rolloutPermsMode,
              agentType,
            });
            break;
          }
          case "cleanup":
            await handleCleanupCommand(env);
            break;
        }
        break;
      }

      case "help":
      case "--help":
      case "-h":
      case undefined:
        showHelp();
        break;

      default:
        console.error(`❌ Unknown command: ${command}`);
        console.log();
        showHelp();
        process.exit(1);
    }
  } catch (error) {
    console.error("❌ Error:", error instanceof Error ? error.message : error);
    process.exit(1);
  }
}
