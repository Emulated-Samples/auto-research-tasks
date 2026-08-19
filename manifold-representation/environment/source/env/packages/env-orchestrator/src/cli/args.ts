/**
 * CLI argument parsing
 */

import type { PermissionsMode } from "@hyperfocal/env-base";

export type AgentType = "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent";

/**
 * Parse command and subcommand from argv
 */
export function parseCommand(): { command: string | undefined; subcommand: string | undefined } {
  const args = process.argv.slice(2);
  return {
    command: args[0],
    subcommand: args[1],
  };
}

/**
 * Parse --problem flag
 */
export function parseProblem(): string | undefined {
  const args = process.argv.slice(2);
  const problemIndex = args.findIndex((a) => a === "--problem" || a === "-p");
  if (problemIndex !== -1 && args[problemIndex + 1]) {
    return args[problemIndex + 1];
  }
  return undefined;
}

/**
 * Parse --model flag
 */
export function parseModel(): string | undefined {
  const args = process.argv.slice(2);
  const modelIndex = args.findIndex((a) => a === "--model" || a === "-m");
  if (modelIndex !== -1 && args[modelIndex + 1]) {
    return args[modelIndex + 1];
  }
  return undefined;
}

/**
 * Parse --agent-type flag
 * @returns "claude-code" | "anthropic-coding" | "opencode" | "codex" | "mini-swe-agent" | undefined
 */
export function parseAgentType(): AgentType | undefined {
  const args = process.argv.slice(2);
  const typeIndex = args.findIndex((a) => a === "--agent-type" || a === "-a");
  if (typeIndex !== -1 && args[typeIndex + 1]) {
    const value = args[typeIndex + 1];
    if (
      value === "claude-code" ||
      value === "anthropic-coding" ||
      value === "opencode" ||
      value === "codex" ||
      value === "mini-swe-agent"
    ) {
      return value;
    }
    console.warn(`Warning: Invalid agent type "${value}", must be "claude-code", "anthropic-coding", "opencode", "codex", or "mini-swe-agent"`);
  }
  return undefined;
}

/**
 * Parse --permissions-mode flag
 * @returns "claude-permissions" | "linux-user" | undefined
 */
export function parsePermissionsMode(): PermissionsMode | undefined {
  const args = process.argv.slice(2);
  const idx = args.findIndex((a) => a === "--permissions-mode");
  if (idx !== -1 && args[idx + 1]) {
    const mode = args[idx + 1];
    if (mode === "claude-permissions" || mode === "linux-user") {
      return mode as PermissionsMode;
    }
    console.error(
      `[ERROR] Invalid --permissions-mode: "${mode}". Must be "claude-permissions" or "linux-user".`
    );
    process.exit(1);
  }
  return undefined;
}

/**
 * The platform-wide default permissions mode, used when neither a
 * --permissions-mode flag nor hyperfocal.yaml `agent.permissionsMode` is set.
 *
 * Kernel-enforced unprivileged isolation (linux-user) is the default for ALL
 * agent types: the agent runs as the unprivileged hyperfocal-agent and the
 * KERNEL — not string-matched disallowedTools — protects grader state, the gold
 * solution, and env internals from a reward-hacking agent. `claude-permissions`
 * (agent as root) remains an explicit opt-in via the flag or hyperfocal.yaml.
 *
 * Single source of truth: both cli/main.ts and cli/commands/environments.ts
 * import this. It previously existed as two copies, and the linux-user flip was
 * applied to only one — so rollouts silently stayed on claude-permissions.
 * Consolidated here so the two can never drift again.
 *
 * `agentType` is accepted for call-site compatibility but intentionally unused
 * (the default no longer varies by agent type).
 */
export function defaultPermissionsMode(_agentType?: string): PermissionsMode {
  return "linux-user";
}

/**
 * Get value for a flag
 */
export function getFlag(flag: string): string | undefined {
  const args = process.argv.slice(2);
  const index = args.findIndex((a) => a === flag || a === `--${flag}`);
  if (index !== -1 && args[index + 1]) {
    return args[index + 1];
  }
  return undefined;
}
