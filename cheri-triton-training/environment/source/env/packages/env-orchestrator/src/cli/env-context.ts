/**
 * Shared preamble for every command that operates on the configured
 * environment module (setup/test/solve/rollout/solve-oracle and, in
 * packaged containers, `harbor grade`): load hyperfocal.yaml, resolve
 * paths, export the env vars the environment expects, inject credentials,
 * then dynamically import the environment module.
 *
 * Extracted from cli/main.ts so the harbor verifier entrypoint runs the
 * EXACT same environment-loading path as `env-orchestrator test` instead
 * of a second, drift-prone copy.
 */

import type { EnvironmentDefinition } from "@hyperfocal/env-base";
import {
  loadConfig,
  getResolvedPaths,
  type HyperfocalConfig,
} from "../config/yaml-config.js";
import { loadCredentials } from "../config/credentials.js";
import { loadEnvironment } from "./environment.js";
import { getSchemaPath } from "./commands/shared/promptTemplates.js";

export interface EnvCommandContext {
  config: HyperfocalConfig;
  paths: ReturnType<typeof getResolvedPaths>;
  env: EnvironmentDefinition;
}

/**
 * Returns null when no environment module could be loaded (the loader has
 * already printed the failure detail; callers decide whether that is a
 * user-facing config error or a harness crash).
 */
export async function loadEnvCommandContext(): Promise<EnvCommandContext | null> {
  // Load config from YAML instead of JSON
  const config = loadConfig();
  const paths = getResolvedPaths(config);

  // Set WORKSPACE_PATH env var for the environment to use
  process.env.WORKSPACE_PATH = paths.workspace;

  // Set HYPERFOCAL_OUTPUT_SCHEMA env var if schema is configured
  // This allows the environment to validate manifests without hardcoding paths
  const schemaPath = getSchemaPath(config);
  if (schemaPath) {
    process.env.HYPERFOCAL_OUTPUT_SCHEMA = schemaPath;
  }

  // Inject AWS credentials from .env file into process.env BEFORE loading
  // the environment module. This is required because the environment's AWS
  // SDK clients are initialized at module load time and read from process.env.
  //
  // Security note: This injects credentials into the orchestrator process
  // (trusted infrastructure code). The agent runs in a separate child process
  // with filtered credentials controlled by agent-runner.ts and awsAccess config.
  //
  // TODO: Handle credential refresh for long-running processes. Current AWS
  // session credentials expire after 1 hour. For now, this works because
  // orchestrator commands complete well within that window.
  const creds = loadCredentials(config);
  if (creds.aws) {
    process.env.AWS_ACCESS_KEY_ID = creds.aws.accessKeyId;
    process.env.AWS_SECRET_ACCESS_KEY = creds.aws.secretAccessKey;
    if (creds.aws.sessionToken) {
      process.env.AWS_SESSION_TOKEN = creds.aws.sessionToken;
    }
    process.env.AWS_REGION = creds.aws.region;
    if (creds.aws.accountId) {
      process.env.AWS_ACCOUNT_ID = creds.aws.accountId;
    }
  }

  if (creds.openrouterApiKey) {
    process.env.OPENROUTER_API_KEY = creds.openrouterApiKey;
  }

  const env = await loadEnvironment(paths.environmentDist);
  if (!env) {
    return null;
  }

  return { config, paths, env };
}
