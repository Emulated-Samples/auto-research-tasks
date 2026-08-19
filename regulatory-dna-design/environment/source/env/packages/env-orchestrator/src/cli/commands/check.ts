/**
 * `env-orchestrator check` — pre-flight check for workspace readiness.
 * Split out of the old environments.ts god-file with no behavior change.
 */

import { loadSchema } from "@hyperfocal/env-base";
import { loadConfig, getResolvedPaths } from "../../config/yaml-config.js";
import { getSchemaPath } from "./shared/promptTemplates.js";

/**
 * Handle 'check' command - pre-flight check for workspace readiness
 *
 * Validates that the workspace is ready for testing:
 * 1. Manifest exists at .hyperfocal/manifest.json
 * 2. Manifest is valid JSON
 * 3. Manifest passes schema validation (if schema configured)
 */
export async function handleCheckCommand(): Promise<void> {
  const config = loadConfig();
  const paths = getResolvedPaths(config);
  const schemaPath = getSchemaPath(config);

  console.log("Pre-flight check for workspace readiness\n");

  const fs = await import("fs");
  const { getManifestPath, readManifest, validateManifest } = await import(
    "@hyperfocal/env-base"
  );

  const manifestPath = getManifestPath(paths.workspace);
  let hasErrors = false;

  // Step 1: Check manifest exists
  console.log("1. Checking manifest exists...");
  if (!fs.existsSync(manifestPath)) {
    console.error(`   [ERROR] Manifest not found: ${manifestPath}`);
    console.error("");
    console.error("   To fix this, either:");
    console.error("   • Run 'env-orchestrator solve' to have an agent create the manifest");
    console.error("   • Manually create .hyperfocal/manifest.json in the workspace");
    console.error("");
    process.exit(1);
  }
  console.log(`   [OK] Found: ${manifestPath}`);

  // Step 2: Check manifest is valid JSON
  console.log("2. Checking manifest is valid JSON...");
  const manifest = readManifest(paths.workspace);
  if (!manifest) {
    console.error("   [ERROR] Failed to parse manifest as JSON");
    console.error("");
    console.error("   Check the file for syntax errors.");
    process.exit(1);
  }
  console.log("   [OK] Valid JSON");

  // Step 3: Check manifest passes schema validation (if schema configured)
  if (schemaPath) {
    console.log("3. Validating manifest against schema...");
    try {
      const schema = loadSchema(schemaPath);
      const result = validateManifest(manifest, schema);

      if (!result.valid) {
        console.error("   [ERROR] Manifest validation failed:");
        console.error("");
        for (const error of result.errors || []) {
          console.error(`      ${error.path}: ${error.message}`);
          if (error.received !== undefined) {
            console.error(`        received: ${JSON.stringify(error.received)}`);
          }
        }
        console.error("");
        console.error(`   Schema: ${schemaPath}`);
        console.error("   Fix the manifest to match the schema requirements.");
        hasErrors = true;
      } else {
        console.log("   [OK] Passes schema validation");
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error(`   [ERROR] Schema validation error: ${msg}`);
      hasErrors = true;
    }
  } else {
    console.log("3. Schema validation... (skipped - no schema configured)");
  }

  // Summary
  console.log("");
  if (hasErrors) {
    console.error("[ERROR] Pre-flight check failed. Fix the issues above before running tests.");
    process.exit(1);
  } else {
    console.log("[OK] Workspace is ready for testing!");
    console.log("");
    console.log("   Run 'env-orchestrator test' to execute tests.");
  }
}
