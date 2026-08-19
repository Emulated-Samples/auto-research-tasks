/**
 * Neutral, agent-agnostic schema for MCP servers attached to a rollout.
 *
 * Setup-phase code in environments writes one of these files to
 * `<workspace>/.hyperfocal/mcp-servers.json`. Each agent class translates
 * to whatever shape its CLI wants at spawn time (Claude's `--mcp-config`
 * JSON, OpenCode's `opencode.json`, etc.).
 *
 * The transport vocabulary intentionally avoids `type: "http"` (Claude's
 * key) or `type: "remote"` (OpenCode's key) — neither leaks into the
 * canonical schema.
 */

export type McpServerSpec =
  | { transport: "http"; url: string }
  | {
      transport: "stdio";
      command: string[];
      env?: Record<string, string>;
    };

/**
 * On-disk representation of `mcp-servers.json`. The `version` field is
 * present so future schema changes can be migrated without breaking
 * older artifacts. Bump when the shape of `servers` changes.
 */
export interface McpServersFile {
  version: 1;
  servers: Record<string, McpServerSpec>;
}
