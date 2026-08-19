/**
 * On-disk helpers for the canonical MCP servers file.
 *
 * The file lives at `<workspace>/.hyperfocal/mcp-servers.json` and is the
 * single contract between an environment's setup phase (which spins up
 * MCP servers and writes their addresses) and the agent's solve phase
 * (which translates the file to whatever its CLI accepts).
 *
 * All helpers tolerate missing/corrupted files — callers that need
 * "no MCP for this rollout" semantics simply observe a missing file.
 */

import * as fs from "fs";
import * as path from "path";
import type { McpServerSpec, McpServersFile } from "./types.js";

/** Filename relative to the workspace root. Constant so callers don't reinvent it. */
export const MCP_SERVERS_FILENAME = ".hyperfocal/mcp-servers.json";

/** Absolute path to the canonical mcp-servers.json for a given workspace. */
export function mcpServersFilePath(workspace: string): string {
  return path.join(workspace, MCP_SERVERS_FILENAME);
}

/**
 * Read and parse the canonical file. Returns `null` when the file is
 * missing, malformed, or references a future schema version this build
 * doesn't understand.
 */
export function readMcpServersFile(workspace: string): McpServersFile | null {
  return readMcpServersFromPath(mcpServersFilePath(workspace));
}

/**
 * Path-taking variant of `readMcpServersFile`. Use this when the caller
 * already has an absolute path (e.g., agent classes that receive a
 * `mcpConfigPath` field on their configuration object).
 */
export function readMcpServersFromPath(filePath: string): McpServersFile | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(raw) as Partial<McpServersFile>;
    if (parsed.version !== 1 || typeof parsed.servers !== "object" || parsed.servers === null) {
      return null;
    }
    return { version: 1, servers: parsed.servers as Record<string, McpServerSpec> };
  } catch {
    return null;
  }
}

/**
 * Add or replace one server entry, creating the file if necessary.
 * Returns the absolute path so callers can log it.
 */
export function upsertMcpServer(
  workspace: string,
  name: string,
  spec: McpServerSpec,
): string {
  const file = readMcpServersFile(workspace) ?? { version: 1 as const, servers: {} };
  file.servers[name] = spec;
  return writeMcpServersFile(workspace, file);
}

/**
 * Remove a server entry. Removes the whole file when the last entry is
 * deleted so a stale empty `mcp-servers.json` doesn't surprise the next
 * rollout. No-op when the file or named entry is absent.
 */
export function removeMcpServer(workspace: string, name: string): void {
  const file = readMcpServersFile(workspace);
  if (!file || !(name in file.servers)) return;
  delete file.servers[name];
  if (Object.keys(file.servers).length === 0) {
    const configPath = mcpServersFilePath(workspace);
    try {
      fs.unlinkSync(configPath);
    } catch {
      writeMcpServersFile(workspace, file);
    }
    return;
  }
  writeMcpServersFile(workspace, file);
}

function writeMcpServersFile(workspace: string, file: McpServersFile): string {
  const configPath = mcpServersFilePath(workspace);
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(file, null, 2));
  return configPath;
}
