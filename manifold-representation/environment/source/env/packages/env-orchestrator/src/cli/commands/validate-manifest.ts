/**
 * `env-orchestrator validate-manifest` — validate a manifest file against
 * the configured output schema. Split out of the old environments.ts
 * god-file with no behavior change.
 */

import * as path from "path";
import { loadSchema } from "@hyperfocal/env-base";
import { loadConfig, getResolvedPaths } from "../../config/yaml-config.js";
import { getSchemaPath } from "./shared/promptTemplates.js";

/**
 * Handle 'validate-manifest' command - validate a manifest file against the schema
 */
export async function handleValidateManifestCommand(): Promise<void> {
  const config = loadConfig();
  const paths = getResolvedPaths(config);
  const schemaPath = getSchemaPath(config);

  if (!schemaPath) {
    console.error("[ERROR] No output schema configured in hyperfocal.yaml");
    console.error("   Add an 'output' section with 'schemaFile' to enable manifest validation.");
    process.exit(1);
  }

  // Check for --manifest flag to specify custom path
  const manifestFlagIndex = process.argv.indexOf("--manifest");
  let manifestPath: string;

  if (manifestFlagIndex !== -1 && process.argv[manifestFlagIndex + 1]) {
    manifestPath = path.resolve(process.argv[manifestFlagIndex + 1]);
  } else {
    // Default to .hyperfocal/manifest.json in workspace
    const { getManifestPath } = await import("@hyperfocal/env-base");
    manifestPath = getManifestPath(paths.workspace);
  }

  // Check if manifest exists
  const fs = await import("fs");
  if (!fs.existsSync(manifestPath)) {
    console.error(`[ERROR] Manifest not found: ${manifestPath}`);
    process.exit(1);
  }

  // Load and validate
  try {
    const manifestContent = fs.readFileSync(manifestPath, "utf-8");
    const manifest = JSON.parse(manifestContent);
    const schema = loadSchema(schemaPath);

    const { validateManifest } = await import("@hyperfocal/env-base");
    const result = validateManifest(manifest, schema);

    if (result.valid) {
      console.log(`[OK] Manifest is valid: ${manifestPath}`);
    } else {
      console.error(`[ERROR] Manifest validation failed: ${manifestPath}`);
      console.error("");
      for (const error of result.errors || []) {
        console.error(`   ${error.path}: ${error.message}`);
        if (error.received !== undefined) {
          console.error(`     received: ${JSON.stringify(error.received)}`);
        }
      }
      process.exit(1);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[ERROR] Failed to validate manifest: ${msg}`);
    process.exit(1);
  }
}
