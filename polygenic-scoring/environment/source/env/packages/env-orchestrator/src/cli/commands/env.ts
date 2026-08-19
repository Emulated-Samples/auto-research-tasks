/**
 * Env Command Handlers
 *
 * New CLI subcommands for environment configuration:
 * - env show [scope]
 * - env validate
 * - env setup-isolation [--dry-run]
 * - env init [--name <name>]
 */

import * as fs from "fs";
import { execSync } from "child_process";
import {
  loadConfig,
  getResolvedPaths,
  isLocalDev,
} from "../../config/yaml-config.js";
import {
  ensureAgentUser,
  setupWorkspacePermissions,
  lockdownSensitiveDirectories,
  getAgentUser,
  setupClaudeCredentials,
  isClaudeSetUp,
} from "../../config/workspace-isolation.js";
import {
  LITELLM_PROVIDER_ENV_KEYS,
  findEnvFilePath,
  hasCodexAuth,
  loadCredentials,
  loadLiteLlmProviderEnv,
} from "../../config/credentials.js";

/**
 * Handle the `env` command and its subcommands
 */
export async function handleEnvCommand(): Promise<void> {
  const args = process.argv.slice(3);
  const subcommand = args[0];

  switch (subcommand) {
    case "show":
      handleEnvShow(args[1]);
      break;
    case "validate":
      handleEnvValidate();
      break;
    case "setup-isolation":
      handleEnvSetupIsolation(args.includes("--dry-run"));
      break;
    case "init":
      handleEnvInit(args);
      break;
    default:
      console.log("Usage:");
      console.log("  env-orchestrator env show [scope]");
      console.log("  env-orchestrator env validate");
      console.log("  env-orchestrator env setup-isolation [--dry-run]");
      console.log("  env-orchestrator env init [--name <name>]");
  }
}

/**
 * Show current configuration and credentials
 */
function handleEnvShow(_scope?: string): void {
  const config = loadConfig();
  const paths = getResolvedPaths(config);

  // TODO(logging): Replace whitespace-aligned console output with a proper
  // CLI logging/formatting layer so status rendering is consistent.
  console.log("\nConfiguration\n");
  console.log(`Environment: ${config.environment.name}`);
  console.log(`Description: ${config.environment.description || "(none)"}`);

  console.log("\nPaths:");
  console.log(`  root:            ${paths.root}`);
  console.log(`  environmentDist: ${paths.environmentDist}`);
  console.log(`  workspace:       ${paths.workspace}`);

  console.log("\nAgent:");
  console.log(`  awsAccess:     ${config.agent.awsAccess}`);
  console.log(`  defaultModel:  ${config.agent.defaultModel}`);
  console.log(`  type:          ${config.agent.type || "claude-code"}`);

  // Show .env file status and credentials from file
  console.log("\nCredentials (.env file):");
  const envPath = findEnvFilePath(config);
  if (envPath) {
    console.log(`  .env file: ${envPath}`);
    const creds = loadCredentials(config);
    
    if (creds.aws) {
      console.log(`  AWS_ACCESS_KEY_ID: ${creds.aws.accessKeyId.slice(0, 12)}...`);
      console.log(`  AWS_SECRET_ACCESS_KEY: ****`);
      console.log(`  AWS_SESSION_TOKEN: ${creds.aws.sessionToken ? "set" : "not set"}`);
      console.log(`  AWS_REGION: ${creds.aws.region}`);
      if (creds.aws.accountId) {
        console.log(`  AWS_ACCOUNT_ID: ${creds.aws.accountId}`);
      }
      if (creds.aws.expiration) {
        const expiresIn = Math.round((creds.aws.expiration.getTime() - Date.now()) / 1000 / 60);
        if (expiresIn > 0) {
          console.log(`  Expires in: ${expiresIn} minutes`);
        } else {
          console.log(`  [WARN] EXPIRED ${Math.abs(expiresIn)} minutes ago`);
        }
      }
    } else {
      console.log("  AWS credentials: not found in .env");
    }
    
    console.log(`  ANTHROPIC_API_KEY: ${creds.anthropicApiKey ? creds.anthropicApiKey.slice(0, 12) + "..." : "not set"}`);
    console.log(`  OPENAI_API_KEY: ${creds.openaiApiKey ? creds.openaiApiKey.slice(0, 12) + "..." : "not set"}`);
    console.log(`  CODEX_API_KEY: ${creds.codexApiKey ? creds.codexApiKey.slice(0, 12) + "..." : "not set"}`);
    console.log(`  Codex auth.json: ${hasCodexAuth() ? "present" : "not found"}`);
    console.log(`  OPENROUTER_API_KEY: ${creds.openrouterApiKey ? creds.openrouterApiKey.slice(0, 12) + "..." : "not set"}`);
    console.log(`  GOOGLE_API_KEY/GEMINI_API_KEY: ${creds.googleApiKey ? creds.googleApiKey.slice(0, 12) + "..." : "not set"}`);
    console.log(`  XAI_API_KEY: ${creds.xaiApiKey ? creds.xaiApiKey.slice(0, 12) + "..." : "not set"}`);
    if (creds.telemetryBucket) {
      console.log(`  TELEMETRY_S3_BUCKET: ${creds.telemetryBucket}`);
    }
    if (creds.runId) {
      console.log(`  RUN_ID: ${creds.runId}`);
    }
    if (creds.rolloutId) {
      console.log(`  ROLLOUT_ID: ${creds.rolloutId}`);
    }
  } else {
    console.log("  .env file: not found (using process.env fallback)");
    console.log("\nCredentials (from process.env):");
    printCredentialStatus("ANTHROPIC_API_KEY");
    printCredentialStatus("OPENAI_API_KEY");
    printCredentialStatus("CODEX_API_KEY");
    console.log(`  Codex auth.json: ${hasCodexAuth() ? "present" : "not found"}`);
    printCredentialStatus("OPENROUTER_API_KEY");
    printCredentialStatus("GOOGLE_API_KEY");
    printCredentialStatus("GEMINI_API_KEY");
    printCredentialStatus("XAI_API_KEY");
    printCredentialStatus("AWS_ACCESS_KEY_ID");
    printCredentialStatus("AWS_SECRET_ACCESS_KEY");
    printCredentialStatus("AWS_SESSION_TOKEN");
    printCredentialStatus("TELEMETRY_S3_BUCKET");
    printCredentialStatus("RUN_ID");
    printCredentialStatus("ROLLOUT_ID");
  }
  console.log();
}

/**
 * Print credential status with masking for sensitive values
 */
function printCredentialStatus(key: string): void {
  const value = process.env[key];
  if (value) {
    const masked =
      key.includes("SECRET") || key.includes("KEY") || key.includes("TOKEN")
        ? value.slice(0, 8) + "..."
        : value;
    console.log(`  ${key}: ${masked}`);
  } else {
    console.log(`  ${key}: not set`);
  }
}

/**
 * Validate configuration and check for common issues
 */
function handleEnvValidate(): void {
  const config = loadConfig();
  const paths = getResolvedPaths(config);
  let errors = 0;
  let warnings = 0;

  console.log("\nValidating configuration...\n");

  // Detect local dev environment
  const localDev = isLocalDev();
  if (localDev && paths.root === "/hyperfocal/env") {
    console.log("[WARN] Running locally with default EC2 paths.");
    console.log(
      "   Create a local hyperfocal.yaml with paths.root pointing to your local environment,"
    );
    console.log("   or set HYPERFOCAL_ENV_ROOT environment variable.\n");
    warnings++;
  }

  // Check paths
  console.log("Paths:");
  errors += checkPath(paths.root, "root");
  errors += checkPath(paths.environmentDist, "environmentDist");

  // Workspace may not exist yet (created by setup), so just warn
  if (!fs.existsSync(paths.workspace)) {
    console.log(`  [WARN] workspace: ${paths.workspace} (not found, will be created)`);
    warnings++;
  } else {
    console.log(`  [OK] workspace: ${paths.workspace}`);
  }

  // Check agent user (skip on local dev)
  console.log("\nWorkspace isolation:");
  if (localDev) {
    console.log(`  [WARN] Skipping user check (local dev - use --permissions-mode claude-permissions)`);
    warnings++;
  } else {
    try {
      execSync(`id ${getAgentUser()}`, { stdio: "ignore" });
      console.log(`  [OK] ${getAgentUser()} user exists`);
    } catch {
      console.log(
        `  [ERROR] ${getAgentUser()} user does not exist (run: env setup-isolation)`
      );
      errors++;
    }
  }

  // Check credentials (prefer .env file, fallback to process.env)
  console.log("\nCredentials:");
  const envFilePath = findEnvFilePath(config);
  const creds = loadCredentials(config);

  const agentType = config.agent.type || "claude-code";
  if (agentType === "codex") {
    if (hasCodexAuth()) {
      console.log(`  [OK] Codex auth.json: present`);
    } else if (envFilePath && creds.codexApiKey) {
      console.log(`  [OK] CODEX_API_KEY: set (from .env file)`);
    } else if (process.env.CODEX_API_KEY) {
      console.log(`  [WARN] CODEX_API_KEY: set (from process.env, not .env file)`);
      warnings++;
    } else {
      console.log(`  [ERROR] Codex credentials: missing ~/.codex/auth.json and CODEX_API_KEY`);
      errors++;
    }
  } else if (agentType === "opencode") {
    console.log(`  [INFO] OpenCode uses provider auth.json; skipping ANTHROPIC_API_KEY requirement`);
  } else if (agentType === "mini-swe-agent") {
    const providerEnv = loadLiteLlmProviderEnv(config);
    const configuredKeys = LITELLM_PROVIDER_ENV_KEYS.filter((key) => providerEnv[key]);
    if (configuredKeys.length > 0) {
      console.log(`  [OK] mini-swe-agent LiteLLM provider keys: ${configuredKeys.join(", ")}`);
    } else {
      console.log(
        `  [ERROR] mini-swe-agent provider credentials: none of ${LITELLM_PROVIDER_ENV_KEYS.join(", ")} are set`
      );
      errors++;
    }
  } else {
    // Check ANTHROPIC_API_KEY - determine source based on whether .env file exists
    if (envFilePath && creds.anthropicApiKey) {
      console.log(`  [OK] ANTHROPIC_API_KEY: set (from .env file)`);
    } else if (process.env.ANTHROPIC_API_KEY) {
      console.log(`  [WARN] ANTHROPIC_API_KEY: set (from process.env, not .env file)`);
      warnings++;
    } else {
      console.log(`  [ERROR] ANTHROPIC_API_KEY: not set`);
      errors++;
    }
  }

  // Check AWS credentials if needed
  if (config.agent.awsAccess) {
    if (envFilePath && creds.aws) {
      console.log(`  [OK] AWS_ACCESS_KEY_ID: set (from .env file)`);
      console.log(`  [OK] AWS_SECRET_ACCESS_KEY: set (from .env file)`);
    } else if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY) {
      console.log(`  [WARN] AWS credentials: set (from process.env, not .env file)`);
      warnings++;
    } else {
      console.log(`  [ERROR] AWS_ACCESS_KEY_ID: not set`);
      console.log(`  [ERROR] AWS_SECRET_ACCESS_KEY: not set`);
      errors += 2;
    }
  }

  console.log();
  if (errors > 0) {
    console.log(`[ERROR] Validation failed (${errors} errors, ${warnings} warnings)\n`);
    process.exit(1);
  } else if (warnings > 0) {
    console.log(`[WARN] Configuration valid with ${warnings} warning(s)\n`);
  } else {
    console.log("[OK] Configuration valid\n");
  }
}

/**
 * Check if a path exists
 */
function checkPath(p: string, name: string): number {
  if (fs.existsSync(p)) {
    console.log(`  [OK] ${name}: ${p}`);
    return 0;
  } else {
    console.log(`  [ERROR] ${name}: ${p} (not found)`);
    return 1;
  }
}

/**
 * Check if an environment variable is set
 */
function checkEnvVar(key: string): number {
  if (process.env[key]) {
    console.log(`  [OK] ${key}: set`);
    return 0;
  } else {
    console.log(`  [ERROR] ${key}: not set`);
    return 1;
  }
}

/**
 * Set up workspace isolation (create user, set permissions)
 */
function handleEnvSetupIsolation(dryRun: boolean): void {
  const config = loadConfig();
  const paths = getResolvedPaths(config);

  if (dryRun) {
    console.log("\n[DRY RUN] Would set up workspace isolation...\n");
  } else {
    console.log("\nSetting up workspace isolation...\n");
  }

  if (!dryRun) {
    ensureAgentUser();
  } else {
    console.log(`[dry-run] Would create user: ${getAgentUser()}`);
  }

  setupWorkspacePermissions(paths.workspace, dryRun);

  // Lock down sensitive directories so agent can't read them
  lockdownSensitiveDirectories(paths.root, dryRun);

  // Set up Claude Code CLI credentials (for claude-code agent type)
  // This adds the agent user to root group and sets up /root/.claude permissions
  if (!dryRun) {
    console.log("\nSetting up Claude Code CLI access...");
    setupClaudeCredentials();
  } else {
    console.log(`[dry-run] Would set up Claude Code CLI credentials`);
    console.log(`[dry-run]   - Add ${getAgentUser()} to root group`);
    console.log(`[dry-run]   - Set permissions on /root/.claude`);
    console.log(`[dry-run]   - Copy CLI to /usr/local/bin/claude`);
  }

  // Show Claude setup status
  if (!dryRun) {
    const claudeReady = isClaudeSetUp();
    if (claudeReady) {
      console.log("[OK] Claude Code CLI: Ready");
    } else {
      console.log("[WARN] Claude Code CLI: Not fully configured");
      console.log("   Run 'claude login' as root to authenticate");
    }
  }

  if (dryRun) {
    console.log("\n[OK] Dry run complete (no changes made)\n");
  } else {
    console.log("\n[OK] Isolation setup complete\n");
  }
}

/**
 * Initialize a new hyperfocal.yaml file
 */
function handleEnvInit(args: string[]): void {
  const nameIdx = args.indexOf("--name");
  const name = nameIdx !== -1 ? args[nameIdx + 1] : "my-environment";

  const content = `version: "1.0"

environment:
  name: ${name}
  description: ""

paths:
  root: /hyperfocal/env
  environmentDist: environment/dist
  workspace: workspace

agent:
  awsAccess: false
  defaultModel: opus
`;

  if (fs.existsSync("hyperfocal.yaml")) {
    console.log("[WARN] hyperfocal.yaml already exists. Delete it first to reinitialize.");
    process.exit(1);
  }

  fs.writeFileSync("hyperfocal.yaml", content);
  console.log("[OK] Created hyperfocal.yaml");
}

