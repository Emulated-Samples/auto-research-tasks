/**
 * Configuration manager for env-orchestrator
 */

import * as fs from "fs";
import * as path from "path";

const CONFIG_DIR = path.join(process.env.HOME || "/root", ".hyperfocal");
const CONFIG_FILE = path.join(CONFIG_DIR, "env-orchestrator.json");

interface Config {
  environmentPath?: string;
  workspacePath?: string;
}

/**
 * Ensure config directory exists
 */
function ensureConfigDir(): void {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

/**
 * Load config from disk
 */
function loadConfig(): Config {
  ensureConfigDir();
  if (fs.existsSync(CONFIG_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(CONFIG_FILE, "utf-8"));
    } catch {
      return {};
    }
  }
  return {};
}

/**
 * Save config to disk
 */
function saveConfig(config: Config): void {
  ensureConfigDir();
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
}

/**
 * Get environment path from config
 */
export function getEnvironmentPath(): string | undefined {
  const config = loadConfig();
  return config.environmentPath;
}

/**
 * Set environment path in config
 */
export function setEnvironmentPath(envPath: string): void {
  const config = loadConfig();
  config.environmentPath = path.resolve(envPath);
  saveConfig(config);
}

/**
 * Get workspace path from config
 */
export function getWorkspacePath(): string | undefined {
  const config = loadConfig();
  return config.workspacePath;
}

/**
 * Set workspace path in config
 */
export function setWorkspacePath(wsPath: string): void {
  const config = loadConfig();
  config.workspacePath = path.resolve(wsPath);
  saveConfig(config);
}

/**
 * Get credentials from environment variables
 */
export function getCredentials() {
  return {
    anthropic: process.env.ANTHROPIC_API_KEY,
    awsAccessKeyId: process.env.AWS_ACCESS_KEY_ID,
    awsSecretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    awsRegion: process.env.AWS_REGION || "us-west-2",
  };
}
