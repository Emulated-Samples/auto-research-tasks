/**
 * Workspace code collection utilities for rubric context assembly.
 *
 * Provides a reusable utility for gathering source files from the workspace
 * into a string suitable for LLM judge evaluation.
 */

import * as fs from "fs";
import * as path from "path";
import { globSync } from "glob";

/**
 * Options for collecting workspace code.
 */
export interface CollectWorkspaceCodeOptions {
  /**
   * Absolute path to workspace root.
   */
  root: string;

  /**
   * Glob patterns for files to include (relative to root).
   * @example ["src/**\/*.rs", "Cargo.toml", "*.py"]
   */
  patterns: string[];

  /**
   * Glob patterns for files/dirs to exclude.
   * @default ["**\/node_modules/**", "**\/target/**", "**\/.git/**"]
   */
  exclude?: string[];

  /**
   * Max characters per file before truncation.
   * @default 3000
   */
  maxPerFile?: number;

  /**
   * Max total characters in output.
   * @default 24000
   */
  maxTotal?: number;

  // TODO: Add includeTree option to prepend directory structure
  // includeTree?: boolean;
  // treeRoot?: string;
}

const DEFAULT_EXCLUDE = ["**/node_modules/**", "**/target/**", "**/.git/**"];

/**
 * Collect workspace source files into a string suitable for LLM judge context.
 *
 * Globs for matching files, reads their contents, truncates large files,
 * and concatenates everything into a formatted string with file headers.
 *
 * @example
 * ```typescript
 * const code = collectWorkspaceCode({
 *   root: "/hyperfocal/env/workspace/my-crate",
 *   patterns: ["Cargo.toml", "src/**\/*.rs"],
 * });
 * // Returns:
 * // # File: Cargo.toml
 * // [package]
 * // name = "my-crate"
 * // ...
 * //
 * // # File: src/lib.rs
 * // pub mod foo;
 * // ...
 * ```
 */
export function collectWorkspaceCode(
  options: CollectWorkspaceCodeOptions
): string {
  const {
    root,
    patterns,
    exclude = DEFAULT_EXCLUDE,
    maxPerFile = 3_000,
    maxTotal = 24_000,
  } = options;

  let output = "";

  // Glob for matching files
  let files: string[];
  try {
    files = globSync(patterns, {
      cwd: root,
      ignore: exclude,
      nodir: true,
    });
  } catch {
    return "# error: failed to glob workspace";
  }

  // Sort for deterministic output
  files.sort();

  // Read and concatenate
  for (const rel of files) {
    if (output.length >= maxTotal) break;

    const fullPath = path.join(root, rel);

    let content: string;
    try {
      content = fs.readFileSync(fullPath, "utf-8");
    } catch {
      continue;
    }

    if (content.length > maxPerFile) {
      content = content.slice(0, maxPerFile) + "\n// ... truncated ...";
    }

    output += `# File: ${rel}\n${content}\n\n`;
  }

  return output || "# no matching files found";
}
