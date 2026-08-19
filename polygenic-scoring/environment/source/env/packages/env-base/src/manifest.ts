/**
 * Manifest Validation and Persistence
 *
 * Provides JSON Schema validation using AJV and standardized file I/O
 * for output manifests. This module enables the Output Manifest System
 * where agents declare how their solutions should be run and tested.
 */

import AjvModule, { ErrorObject } from "ajv";
import * as fs from "fs";
import * as path from "path";
import type {
  ManifestValidationResult,
  ManifestValidationError,
} from "./types.js";

// Handle both ESM and CJS imports of AJV
const Ajv = AjvModule.default || AjvModule;

/** Directory for hyperfocal artifacts in workspace */
const MANIFEST_DIR = ".hyperfocal";

/** Standard manifest filename */
const MANIFEST_FILENAME = "manifest.json";

/**
 * Validate a manifest against a JSON Schema.
 *
 * Uses AJV for validation with allErrors mode to report all issues at once.
 *
 * @param manifest - The manifest object to validate
 * @param schema - The JSON Schema to validate against
 * @returns Validation result with errors if invalid
 */
export function validateManifest(
  manifest: Record<string, unknown>,
  schema: Record<string, unknown>
): ManifestValidationResult {
  const ajv = new Ajv({ allErrors: true });
  const validate = ajv.compile(schema);

  if (validate(manifest)) {
    return { valid: true };
  }

  const errors: ManifestValidationError[] = (validate.errors || []).map(
    (e: ErrorObject) => ({
      path: e.instancePath || "/",
      message: e.message || "Unknown error",
      received: e.data,
    })
  );

  return { valid: false, errors };
}

/**
 * Write manifest to the standard location in workspace.
 *
 * Creates the .hyperfocal directory if it doesn't exist.
 *
 * @param workspacePath - Path to the workspace directory
 * @param manifest - The manifest object to write
 * @returns Absolute path to the written manifest file
 */
export function writeManifest(
  workspacePath: string,
  manifest: Record<string, unknown>
): string {
  const manifestDir = path.join(workspacePath, MANIFEST_DIR);
  const manifestPath = path.join(manifestDir, MANIFEST_FILENAME);

  if (!fs.existsSync(manifestDir)) {
    fs.mkdirSync(manifestDir, { recursive: true });
  }

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifestPath;
}

/**
 * Read manifest from the standard location in workspace.
 *
 * @param workspacePath - Path to the workspace directory
 * @returns The manifest object, or null if not found
 */
export function readManifest(
  workspacePath: string
): Record<string, unknown> | null {
  const manifestPath = path.join(workspacePath, MANIFEST_DIR, MANIFEST_FILENAME);

  if (!fs.existsSync(manifestPath)) {
    return null;
  }

  const content = fs.readFileSync(manifestPath, "utf-8");
  return JSON.parse(content);
}

/**
 * Get the standard manifest path for a workspace.
 *
 * @param workspacePath - Path to the workspace directory
 * @returns Absolute path to where the manifest should be
 */
export function getManifestPath(workspacePath: string): string {
  return path.join(workspacePath, MANIFEST_DIR, MANIFEST_FILENAME);
}

/**
 * Load a JSON Schema from a file.
 *
 * @param schemaPath - Absolute path to the schema file
 * @returns The parsed JSON Schema object
 * @throws Error if file not found or invalid JSON
 */
export function loadSchema(schemaPath: string): Record<string, unknown> {
  if (!fs.existsSync(schemaPath)) {
    throw new Error(`Schema file not found: ${schemaPath}`);
  }

  const content = fs.readFileSync(schemaPath, "utf-8");
  return JSON.parse(content);
}

/**
 * Generate a human-readable description from a JSON Schema.
 *
 * Extracts title, description, and documents required/optional fields
 * with their descriptions.
 *
 * @param schema - The JSON Schema object
 * @returns Markdown-formatted description
 */
export function generateSchemaDescription(
  schema: Record<string, unknown>
): string {
  const lines: string[] = [];

  // Add title and description if present
  if (schema.title) {
    lines.push(`## ${schema.title}`);
    lines.push("");
  }
  if (schema.description) {
    lines.push(schema.description as string);
    lines.push("");
  }

  const required = (schema.required as string[]) || [];
  const properties = (schema.properties as Record<string, unknown>) || {};

  // Document required fields
  if (required.length > 0) {
    lines.push("### Required Fields");
    lines.push("");
    for (const key of required) {
      const prop = properties[key] as Record<string, unknown> | undefined;
      if (prop) {
        const typeInfo = formatPropertyType(prop);
        const desc = prop.description || "";
        lines.push(`- **${key}** (${typeInfo}): ${desc}`);
      } else {
        lines.push(`- **${key}**: (required)`);
      }
    }
    lines.push("");
  }

  // Document optional fields
  const optional = Object.keys(properties).filter((k) => !required.includes(k));
  if (optional.length > 0) {
    lines.push("### Optional Fields");
    lines.push("");
    for (const key of optional) {
      const prop = properties[key] as Record<string, unknown>;
      const typeInfo = formatPropertyType(prop);
      const desc = prop.description || "";
      const defaultVal =
        prop.default !== undefined ? ` (default: ${JSON.stringify(prop.default)})` : "";
      lines.push(`- **${key}** (${typeInfo}): ${desc}${defaultVal}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format the type of a property for documentation.
 */
function formatPropertyType(prop: Record<string, unknown>): string {
  const type = prop.type;

  if (Array.isArray(type)) {
    return type.join(" | ");
  }

  if (type === "string" && prop.enum) {
    return (prop.enum as string[]).map((e) => `"${e}"`).join(" | ");
  }

  if (type === "integer") {
    let info = "integer";
    if (prop.minimum !== undefined) info += `, min: ${prop.minimum}`;
    if (prop.maximum !== undefined) info += `, max: ${prop.maximum}`;
    return info;
  }

  return (type as string) || "any";
}

/**
 * Generate an example manifest from a JSON Schema.
 *
 * Uses default values and examples from the schema when available,
 * otherwise generates placeholder values based on type.
 *
 * @param schema - The JSON Schema object
 * @returns An example manifest object
 */
export function generateSchemaExample(
  schema: Record<string, unknown>
): Record<string, unknown> {
  // If schema has examples array, use the first one
  if (Array.isArray(schema.examples) && schema.examples.length > 0) {
    return schema.examples[0] as Record<string, unknown>;
  }

  // Otherwise, generate from properties
  const properties = (schema.properties as Record<string, unknown>) || {};
  const required = (schema.required as string[]) || [];
  const result: Record<string, unknown> = {};

  // Include all required properties and optional ones with defaults/examples
  for (const [key, prop] of Object.entries(properties)) {
    const propObj = prop as Record<string, unknown>;
    const isRequired = required.includes(key);

    // Determine value
    let value: unknown;
    if (propObj.default !== undefined) {
      value = propObj.default;
    } else if (Array.isArray(propObj.examples) && propObj.examples.length > 0) {
      value = propObj.examples[0];
    } else if (propObj.enum && Array.isArray(propObj.enum)) {
      value = propObj.enum[0];
    } else {
      value = generatePlaceholderValue(propObj);
    }

    // Include if required or has a meaningful value
    if (isRequired || propObj.default !== undefined) {
      result[key] = value;
    }
  }

  return result;
}

/**
 * Generate a placeholder value for a property type.
 */
function generatePlaceholderValue(prop: Record<string, unknown>): unknown {
  const type = Array.isArray(prop.type) ? prop.type[0] : prop.type;

  switch (type) {
    case "string":
      return "";
    case "integer":
    case "number":
      return prop.minimum !== undefined ? prop.minimum : 0;
    case "boolean":
      return false;
    case "object":
      return {};
    case "array":
      return [];
    case "null":
      return null;
    default:
      return null;
  }
}
