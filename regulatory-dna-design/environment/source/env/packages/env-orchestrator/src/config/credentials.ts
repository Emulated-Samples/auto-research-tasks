/**
 * Credential Manager
 *
 * Loads credentials from .env file, with fresh reads on every call.
 * This enables the control plane to refresh credentials via SSM without
 * restarting the orchestrator process.
 *
 * Search order for .env file:
 * 1. {config.paths.root}/.env (primary: /hyperfocal/env/.env)
 * 2. ./environment/.env (when running from env repo root)
 * 3. ./.env (current directory)
 * 4. ~/.hyperfocal/.env (user home fallback)
 *
 * Falls back to process.env if no .env file found (backward compatible).
 */

import * as fs from "fs";
import * as path from "path";
import type { HyperfocalConfig } from "./yaml-config.js";
import type { AuthEntry } from "@hyperfocal/env-base";

export interface AwsCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
  region: string;
  accountId?: string;
  expiration?: Date;
}

export interface AllCredentials {
  aws?: AwsCredentials;
  anthropicApiKey?: string;
  openaiApiKey?: string;
  codexApiKey?: string;
  openrouterApiKey?: string;
  googleApiKey?: string;
  xaiApiKey?: string;
  telemetryBucket?: string;
  runId?: string;
  rolloutId?: string;
}

export const LITELLM_PROVIDER_ENV_KEYS = [
  "ANTHROPIC_API_KEY",
  "OPENAI_API_KEY",
  "OPENROUTER_API_KEY",
  "GOOGLE_API_KEY",
  "GEMINI_API_KEY",
  "XAI_API_KEY",
  "PORTKEY_API_KEY",
  "REQUESTY_API_KEY",
] as const;

/**
 * Find the .env file path, checking multiple locations
 */
export function findEnvFilePath(config?: HyperfocalConfig): string | null {
  const searchPaths: string[] = [];

  // 1. Primary: config.paths.root/environment/.env (e.g., /hyperfocal/env/environment/.env)
  // This is where userdata writes credentials, alongside hyperfocal.yaml
  if (config?.paths?.root) {
    searchPaths.push(path.join(config.paths.root, "environment", ".env"));
  }

  // 2. Environment subdir (when running from repo root)
  searchPaths.push(path.resolve("environment", ".env"));

  // 3. Current directory (for local dev running from environment/)
  searchPaths.push(path.resolve(".env"));

  // 4. Legacy: root of repo (backward compat)
  if (config?.paths?.root) {
    searchPaths.push(path.join(config.paths.root, ".env"));
  }

  // 5. User home fallback
  const home = process.env.HOME || "/root";
  searchPaths.push(path.join(home, ".hyperfocal", ".env"));

  for (const envPath of searchPaths) {
    if (fs.existsSync(envPath)) {
      return envPath;
    }
  }

  return null;
}

/**
 * Parse .env file content into key-value pairs
 * (simplified dotenv parsing without mutating process.env)
 */
function parseEnvFile(content: string): Record<string, string> {
  const result: Record<string, string> = {};

  for (const line of content.split("\n")) {
    const trimmed = line.trim();

    // Skip empty lines and comments
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, eqIndex).trim();
    let value = trimmed.slice(eqIndex + 1).trim();

    // Handle quoted values
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    result[key] = value;
  }

  return result;
}

function parseJwtExpiryMillis(token?: string): number {
  if (!token) return 0;
  try {
    const [, payload] = token.split(".");
    if (!payload) return 0;
    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
    const decoded = JSON.parse(Buffer.from(padded, "base64").toString("utf-8"));
    const exp = typeof decoded.exp === "number" ? decoded.exp : 0;
    return exp > 0 ? exp * 1000 : 0;
  } catch {
    return 0;
  }
}

function loadCodexOpenAIOAuth(): AuthEntry | null {
  const authPath = getCodexAuthPath();
  if (!fs.existsSync(authPath)) {
    return null;
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(authPath, "utf-8")) as {
      tokens?: {
        access_token?: string;
        refresh_token?: string;
        account_id?: string;
      };
    };
    const access = parsed.tokens?.access_token;
    const refresh = parsed.tokens?.refresh_token;
    if (!access || !refresh) {
      return null;
    }
    const accountId = parsed.tokens?.account_id;
    return {
      type: "oauth",
      access,
      refresh,
      expires: parseJwtExpiryMillis(access),
      ...(accountId ? { accountId } : {}),
    };
  } catch {
    return null;
  }
}

export function getCodexAuthPath(homeDir = process.env.HOME || "/root"): string {
  return path.join(homeDir, ".codex", "auth.json");
}

export function hasCodexAuth(homeDir = process.env.HOME || "/root"): boolean {
  return fs.existsSync(getCodexAuthPath(homeDir));
}

/**
 * Load credentials from .env file (fresh read every call)
 *
 * This function re-parses the .env file each time it's called,
 * allowing the control plane to update credentials via SSM
 * without restarting the orchestrator.
 */
export function loadCredentials(config?: HyperfocalConfig): AllCredentials {
  const envPath = findEnvFilePath(config);

  if (envPath) {
    // Parse .env file directly (don't mutate process.env)
    const envFileContent = fs.readFileSync(envPath, "utf-8");
    const parsed = parseEnvFile(envFileContent);

    // Build credentials from parsed .env
    const result: AllCredentials = {};

    if (parsed.AWS_ACCESS_KEY_ID && parsed.AWS_SECRET_ACCESS_KEY) {
      result.aws = {
        accessKeyId: parsed.AWS_ACCESS_KEY_ID,
        secretAccessKey: parsed.AWS_SECRET_ACCESS_KEY,
        sessionToken: parsed.AWS_SESSION_TOKEN,
        region: parsed.AWS_REGION || "us-west-2",
        accountId: parsed.AWS_ACCOUNT_ID,
        expiration: parsed.AWS_CREDENTIALS_EXPIRATION
          ? new Date(parseInt(parsed.AWS_CREDENTIALS_EXPIRATION) * 1000)
          : undefined,
      };
    }

    result.anthropicApiKey = parsed.ANTHROPIC_API_KEY;
    result.openaiApiKey = parsed.OPENAI_API_KEY;
    result.codexApiKey = parsed.CODEX_API_KEY;
    result.openrouterApiKey = parsed.OPENROUTER_API_KEY;
    result.googleApiKey = parsed.GOOGLE_API_KEY || parsed.GEMINI_API_KEY;
    result.xaiApiKey = parsed.XAI_API_KEY;
    result.telemetryBucket = parsed.TELEMETRY_S3_BUCKET;
    result.runId = parsed.RUN_ID;
    result.rolloutId = parsed.ROLLOUT_ID;

    return result;
  }

  // Fallback to process.env (backward compatible)
  const result: AllCredentials = {};

  if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY) {
    result.aws = {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
      sessionToken: process.env.AWS_SESSION_TOKEN,
      region: process.env.AWS_REGION || "us-west-2",
      accountId: process.env.AWS_ACCOUNT_ID,
    };
  }

  result.anthropicApiKey = process.env.ANTHROPIC_API_KEY;
  result.openaiApiKey = process.env.OPENAI_API_KEY;
  result.codexApiKey = process.env.CODEX_API_KEY;
  result.openrouterApiKey = process.env.OPENROUTER_API_KEY;
  result.googleApiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  result.xaiApiKey = process.env.XAI_API_KEY;
  result.telemetryBucket = process.env.TELEMETRY_S3_BUCKET;
  result.runId = process.env.RUN_ID;
  result.rolloutId = process.env.ROLLOUT_ID;

  return result;
}

/**
 * Load the explicit LiteLLM provider environment allowlist used by
 * mini-swe-agent. This intentionally returns only provider credentials, not
 * arbitrary process.env, so the child-process boundary remains narrow.
 */
export function loadLiteLlmProviderEnv(config?: HyperfocalConfig): Record<string, string> {
  const envPath = findEnvFilePath(config);
  const fileSource = envPath
    ? parseEnvFile(fs.readFileSync(envPath, "utf-8"))
    : {};
  const result: Record<string, string> = {};

  for (const key of LITELLM_PROVIDER_ENV_KEYS) {
    const value = fileSource[key] || process.env[key];
    if (value) {
      result[key] = value;
    }
  }

  return result;
}

/**
 * Get just the AWS credentials (convenience function)
 */
export function loadAwsCredentials(
  config?: HyperfocalConfig
): AwsCredentials | null {
  return loadCredentials(config).aws || null;
}

/**
 * Check if credentials file exists
 */
export function hasCredentialsFile(config?: HyperfocalConfig): boolean {
  return findEnvFilePath(config) !== null;
}

/**
 * Get the path where credentials would be written
 */
export function getCredentialsWritePath(config?: HyperfocalConfig): string {
  if (config?.paths?.root) {
    return path.join(config.paths.root, "environment", ".env");
  }
  return "/hyperfocal/env/environment/.env";
}

/**
 * Load provider credentials for OpenCode from .env file.
 *
 * Reads the .env file and produces a Record<string, AuthEntry> suitable for
 * writing to OpenCode's auth.json. For each provider:
 *   1. If PROVIDER_OAUTH_ACCESS is present → build an oauth entry
 *   2. Else if PROVIDER_API_KEY is present → build an api entry
 *   3. Otherwise → skip (provider not configured)
 *
 * OAuth takes precedence because if you've configured OAuth, you want the
 * Bearer auth flow, not the API key flow.
 */
export function loadProviderCredentials(config?: HyperfocalConfig): Record<string, AuthEntry> {
  const envPath = findEnvFilePath(config);
  const result: Record<string, AuthEntry> = {};
  if (!envPath) {
    const codexOpenAI = loadCodexOpenAIOAuth();
    if (codexOpenAI) {
      result.openai = codexOpenAI;
      console.log("  Loaded OpenCode OpenAI OAuth from Codex auth fallback");
    }
    return result;
  }

  const envFileContent = fs.readFileSync(envPath, "utf-8");
  const parsed = parseEnvFile(envFileContent);

  // Provider mapping: { envPrefix: providerKey }
  const providers: Record<string, string> = {
    ANTHROPIC: "anthropic",
    OPENAI: "openai",
    OPENROUTER: "openrouter",
    XAI: "xai",
    GOOGLE: "google",
  };

  for (const [envPrefix, providerKey] of Object.entries(providers)) {
    // Check for OAuth first (takes precedence)
    const oauthAccess = parsed[`${envPrefix}_OAUTH_ACCESS`];
    if (oauthAccess) {
      const accountId = parsed[`${envPrefix}_OAUTH_ACCOUNT_ID`];
      const entry: AuthEntry = {
        type: "oauth",
        access: oauthAccess,
        refresh: parsed[`${envPrefix}_OAUTH_REFRESH`] || "",
        expires: parseInt(parsed[`${envPrefix}_OAUTH_EXPIRES`] || "0", 10),
        ...(accountId ? { accountId } : {}),
      };
      result[providerKey] = entry;

      // Warn if token is expiring soon (within 1 hour)
      if (entry.expires > 0) {
        const expiresInMs = entry.expires - Date.now();
        if (expiresInMs < 3600_000) {
          console.warn(`  Warning: ${providerKey} OAuth token expires in ${Math.round(expiresInMs / 60_000)} minutes`);
        }
      }
    }
    // Fall back to API key
    else {
      const apiKey = parsed[`${envPrefix}_API_KEY`];
      if (apiKey) {
        result[providerKey] = { type: "api", key: apiKey };
      }
    }
  }

  const codexOpenAI = loadCodexOpenAIOAuth();
  const openAI = result.openai;
  if (
    codexOpenAI &&
    (!openAI ||
      (openAI.type === "oauth" &&
        (!openAI.refresh || openAI.expires <= 0)))
  ) {
    result.openai = codexOpenAI;
    console.log("  Loaded OpenCode OpenAI OAuth from Codex auth fallback");
  }

  return result;
}
