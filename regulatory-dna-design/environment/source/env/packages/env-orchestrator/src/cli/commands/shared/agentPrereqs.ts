/**
 * Shared prerequisites for agent-running commands (solve, rollout):
 * per-command options and agent credential validation. Split out of the old
 * environments.ts god-file with no behavior change.
 */

import type { PermissionsMode, TelemetrySession, Logger } from "@hyperfocal/env-base";
import type { AgentType } from "../../args.js";
import type { HyperfocalConfig } from "../../../config/yaml-config.js";
import {
  LITELLM_PROVIDER_ENV_KEYS,
  hasCodexAuth,
  loadCredentials,
  loadLiteLlmProviderEnv,
} from "../../../config/credentials.js";

/**
 * Options for commands that support permission mode configuration
 */
export interface CommandOptions {
  /** Permission mode: "linux-user" (default) or "claude-permissions" (root, opt-in) */
  permissionsMode?: PermissionsMode;
  agentType?: AgentType;
}

export function validateAgentCredentials(
  agentType: AgentType,
  creds: ReturnType<typeof loadCredentials>,
  config: HyperfocalConfig
): void {
  if (agentType === "opencode") {
    return;
  }
  if (agentType === "mini-swe-agent") {
    const providerEnv = loadLiteLlmProviderEnv(config);
    if (Object.keys(providerEnv).length > 0) {
      return;
    }
    throw new Error(
      "mini-swe-agent provider credentials not found in .env file or process.env.\n" +
      `  Expected one of: ${LITELLM_PROVIDER_ENV_KEYS.join(", ")}\n` +
      "  Expected location: /hyperfocal/env/environment/.env\n" +
      "  Run: env-orchestrator env show   (to check current config)"
    );
  }
  if (agentType === "codex") {
    if (creds.codexApiKey || hasCodexAuth()) {
      return;
    }
    throw new Error(
      "Codex credentials not found. Run `codex login` to create ~/.codex/auth.json " +
      "or set CODEX_API_KEY in /hyperfocal/env/environment/.env."
    );
  }
  if (!creds.anthropicApiKey) {
    throw new Error(
      "ANTHROPIC_API_KEY not found in .env file or process.env.\n" +
      "  Expected location: /hyperfocal/env/environment/.env\n" +
      "  Run: env-orchestrator env show   (to check current config)"
    );
  }
}

/**
 * Create a logger that writes to both console and telemetry session
 */
export function createTelemetryLogger(session: TelemetrySession): Logger {
  return {
    info: (msg: string) => session.log("info", msg),
    error: (msg: string) => session.log("error", msg),
    warn: (msg: string) => session.log("warn", msg),
    debug: (msg: string) => session.log("info", msg), // Map debug to info
  };
}
