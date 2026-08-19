/**
 * Prompt templating shared by the prompt/solve/rollout commands: schema-file
 * resolution, {{ template_var }} interpolation, and the awsAccess
 * credential-refresh prompt lint. Split out of the old environments.ts
 * god-file (TODO(maintainability)) with no behavior change.
 */

import * as path from "path";
import {
  loadSchema,
  generateSchemaDescription,
  generateSchemaExample,
} from "@hyperfocal/env-base";
import {
  getConfigPath,
  type HyperfocalConfig,
} from "../../../config/yaml-config.js";

/**
 * Get the absolute path to the output schema file, if configured.
 *
 * @param config - The hyperfocal configuration
 * @returns Absolute path to schema file, or undefined if not configured
 */
export function getSchemaPath(config: HyperfocalConfig): string | undefined {
  if (!config.output?.schemaFile) {
    return undefined;
  }

  const configPath = getConfigPath();
  if (!configPath) {
    return undefined;
  }

  // Resolve schema path relative to the config file location
  return path.resolve(path.dirname(configPath), config.output.schemaFile);
}

/**
 * Interpolate template variables in the prompt.
 *
 * Supports:
 * - {{ aws_credentials_path }} - Path to agent-readable AWS credentials file
 * - {{ output_schema }} - Full JSON schema
 * - {{ output_schema_description }} - Human-readable description
 * - {{ output_schema_example }} - Example valid manifest
 *
 * @param prompt - The raw prompt with template variables
 * @param config - The hyperfocal configuration
 * @returns Interpolated prompt
 */
export function interpolatePrompt(
  prompt: string,
  config: HyperfocalConfig
): string {
  // AWS credentials path interpolation
  // Always replace, regardless of awsAccess setting (makes prompts portable)
  const awsCredentialsPath = ".hyperfocal/credentials.env";
  prompt = prompt.replace(
    /\{\{\s*aws_credentials_path\s*\}\}/g,
    awsCredentialsPath
  );

  const schemaPath = getSchemaPath(config);

  if (!schemaPath) {
    // No output config, strip any template vars
    return prompt
      .replace(/\{\{\s*output_schema\s*\}\}/g, "")
      .replace(/\{\{\s*output_schema_description\s*\}\}/g, "")
      .replace(/\{\{\s*output_schema_example\s*\}\}/g, "");
  }

  try {
    const schema = loadSchema(schemaPath);

    return prompt
      .replace(
        /\{\{\s*output_schema\s*\}\}/g,
        JSON.stringify(schema, null, 2)
      )
      .replace(
        /\{\{\s*output_schema_description\s*\}\}/g,
        generateSchemaDescription(schema)
      )
      .replace(
        /\{\{\s*output_schema_example\s*\}\}/g,
        JSON.stringify(generateSchemaExample(schema), null, 2)
      );
  } catch (error) {
    // If schema loading fails, leave template vars as-is (will be visible to agent)
    console.warn(`Warning: Failed to load output schema: ${error}`);
    return prompt;
  }
}

/**
 * Warn if awsAccess is enabled but the prompt doesn't mention credentials
 * refresh — helps prompt authors remember to include instructions for
 * long-running tasks. Shared verbatim by solve and rollout.
 */
export function warnIfCredentialRefreshUnmentioned(
  config: HyperfocalConfig,
  rawPrompt: string
): void {
  if (!config.agent.awsAccess) return;
  const hasCredentialInterpolation =
    rawPrompt.includes("{{ aws_credentials_path }}") ||
    rawPrompt.includes("{{aws_credentials_path}}");
  const mentionsCredentialsFile =
    rawPrompt.toLowerCase().includes("credentials.env") ||
    rawPrompt.toLowerCase().includes(".hyperfocal/credentials");

  if (!hasCredentialInterpolation && !mentionsCredentialsFile) {
    console.warn("");
    console.warn(
      "[WARN] awsAccess is enabled but the problem prompt doesn't reference credentials refresh."
    );
    console.warn(
      "   For long-running tasks (>1 hour), the agent should source fresh credentials from:"
    );
    console.warn("   .hyperfocal/credentials.env");
    console.warn("");
    console.warn(
      "   Add to your prompt: 'Before AWS operations, source {{ aws_credentials_path }} for fresh credentials.'"
    );
    console.warn("");
  }
}
