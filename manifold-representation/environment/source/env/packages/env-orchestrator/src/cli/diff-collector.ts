/**
 * DiffCollector - Captures workspace file changes after agent runs
 *
 * Uses a tag-based baseline approach (Option C) for robustness:
 * 1. Before the agent runs, we create a git tag "hyperfocal-baseline" at HEAD
 * 2. After the agent runs, we diff against this tag
 * 3. This works whether or not the agent makes commits
 *
 * The workspace is freshly cloned from GitHub before each rollout,
 * so HEAD represents the clean state. The tag preserves this reference
 * even if the agent makes commits.
 *
 * Output structure:
 *   /hyperfocal/logs/{problem-id}/diffs/
 *     ├── changes.json          ← List of changed files with stats
 *     └── contents/             ← Individual file contents
 *         └── {path-encoded}.json   ← { baseContent, currentContent }
 *
 * Edge cases handled:
 *   - Large files (>1MB): Truncated with marker
 *   - Binary files: Skipped with marker
 *   - Deleted files: baseContent present, currentContent null
 *   - Added files: baseContent null, currentContent present
 *   - No changes: Empty changes array
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import { getLogsDir } from "@hyperfocal/env-base";

const BASELINE_TAG = "hyperfocal-baseline";
const MAX_FILE_SIZE = 1024 * 1024; // 1MB
const MAX_GIT_OUTPUT = 10 * 1024 * 1024; // 10MB buffer for git commands (large repos)

// The orchestrator runs as root while the workspace is chowned to the agent
// user (linux-user isolation). git >= 2.35 refuses to operate on a repo owned
// by a different uid ("detected dubious ownership") — and because every diff
// command used to end in `2>/dev/null || echo ''`, that refusal silently read
// as an EMPTY DIFF: every rollout on an isolated workspace reported
// "Captured 0 file changes" and the submission was unrecoverable after
// teardown. Trust is not weakened by the exemption: the orchestrator only
// ever points these commands at the workspace it manages.
const GIT_ENV = {
  ...process.env,
  GIT_CONFIG_COUNT: "1",
  GIT_CONFIG_KEY_0: "safe.directory",
  GIT_CONFIG_VALUE_0: "*",
};

export interface FileChange {
  path: string;
  status: "added" | "modified" | "deleted";
  additions?: number;
  deletions?: number;
}

export interface FileContent {
  path: string;
  baseContent: string | null; // null if file was added
  currentContent: string | null; // null if file was deleted
}

export interface DiffResult {
  changes: FileChange[];
  timestamp: string;
  workspacePath: string;
  baselineTag: string;
  /**
   * Set when the git diff itself failed. An empty `changes` with this set
   * means "collection broke", NOT "the agent changed nothing" — consumers
   * (run-diff, package, conversion services) must distinguish the two. This
   * exists because a silent empty diff made every submission in a batch
   * unrecoverable after teardown (sc-silentbench, 2026-07-16/17).
   */
  collectionError?: string;
}

/**
 * Create a baseline tag before the agent runs
 * This preserves the clean state reference even if agent makes commits
 */
export function createBaseline(workspacePath: string): void {
  try {
    // Check if we're in a git repository
    execSync("git rev-parse --git-dir", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });

    // Delete existing baseline tag if present (from previous run)
    execSync(`git tag -d ${BASELINE_TAG} 2>/dev/null || true`, {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });

    // Create a synthetic commit representing the post-setup working tree.
    // Tagging HEAD is not enough because setup can leave tracked files dirty
    // and create untracked files before the agent starts. `commit-tree`
    // creates an object without moving HEAD, so the agent still sees the
    // original branch while diffs compare against the true solve baseline.
    execSync("git add -A .", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });
    const tree = execSync("git write-tree", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    }).trim();
    const parent = execSync("git rev-parse HEAD", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    }).trim();
    const baselineCommit = execSync(
      `git commit-tree ${tree} -p ${parent} -m "hyperfocal baseline"`,
      {
        cwd: workspacePath,
        env: GIT_ENV,
        encoding: "utf-8",
        stdio: "pipe",
      }
    ).trim();
    execSync(`git tag ${BASELINE_TAG} ${baselineCommit}`, {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });
    execSync("git reset --mixed HEAD", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });

    console.log(`[diff-collector] Created baseline tag at post-setup tree`);
  } catch (error) {
    console.warn(
      `[diff-collector] Failed to create baseline tag:`,
      error instanceof Error ? error.message : error
    );
    // Non-fatal - we can still try to diff against HEAD later
  }
}

/**
 * Collect file changes from workspace after agent runs
 */
export async function collectDiff(
  workspacePath: string,
  problemId: string
): Promise<DiffResult> {
  const timestamp = new Date().toISOString();

  // Get changed files
  const { changes, error: collectionError } = getFileChanges(workspacePath);

  // Save to logs directory
  const logsDir = getLogsDir();
  const diffDir = path.join(logsDir, problemId, "diffs");
  fs.mkdirSync(diffDir, { recursive: true });

  // Save changes.json
  const result: DiffResult = {
    changes,
    timestamp,
    workspacePath,
    baselineTag: BASELINE_TAG,
    ...(collectionError ? { collectionError } : {}),
  };
  fs.writeFileSync(
    path.join(diffDir, "changes.json"),
    JSON.stringify(result, null, 2)
  );

  // Save file contents for changed files
  const contentsDir = path.join(diffDir, "contents");
  fs.mkdirSync(contentsDir, { recursive: true });

  for (const change of changes) {
    const content = getFileContent(workspacePath, change);
    // Encode path: replace / with -- to create flat filenames
    const encodedPath = change.path.replace(/\//g, "--");
    fs.writeFileSync(
      path.join(contentsDir, `${encodedPath}.json`),
      JSON.stringify(content, null, 2)
    );
  }

  if (collectionError) {
    // Loud and machine-readable, never a quiet zero: the submission is about
    // to become unrecoverable if this is wrong and the box is torn down.
    console.error(`[diff-collector] FAILED to capture diff: ${collectionError}`);
  } else {
    console.log(`[diff-collector] Captured ${changes.length} file changes`);
  }
  return result;
}

/**
 * Get list of changed files with status and stats
 * 
 * This captures:
 * 1. Modified/deleted tracked files (via git diff)
 * 2. New untracked files (via git ls-files --others)
 * 
 * IMPORTANT: Uses "-- ." to scope diffs to the workspace directory only.
 * The git repo root is typically the environment dir, but we only want
 * changes in the workspace/ subdirectory where the agent works.
 * 
 * Git outputs paths relative to repo root (e.g., "workspace/file.txt"),
 * so we strip the prefix to get workspace-relative paths (e.g., "file.txt").
 */
function getFileChanges(
  workspacePath: string
): { changes: FileChange[]; error?: string } {
  try {
    // Determine what to diff against
    // Priority: baseline tag > HEAD
    const diffTarget = getBaselineRef(workspacePath);
    
    // Get git prefix to strip from paths (e.g., "workspace/")
    const gitPrefix = getGitPrefix(workspacePath);

    // Get name-status (file list with change type)
    // Include both staged and unstaged changes for tracked files
    // "-- ." scopes to files within the current directory (workspace)
    // maxBuffer needed for large repos (e.g., tinygrad has 900+ files)
    const nameStatus = execSync(
      `git -c core.fileMode=false diff --name-status ${diffTarget} -- .`,
      { cwd: workspacePath, env: GIT_ENV, encoding: "utf-8", maxBuffer: MAX_GIT_OUTPUT }
    );

    // Get untracked files (new files created by agent)
    // --exclude-standard respects .gitignore
    // Running from workspace dir automatically scopes to workspace
    const untrackedFiles = listUntrackedFiles(workspacePath);

    // Get numstat (additions/deletions)
    // "-- ." scopes to files within the current directory (workspace)
    const numstat = execSync(
      `git -c core.fileMode=false diff --numstat ${diffTarget} -- .`,
      { cwd: workspacePath, env: GIT_ENV, encoding: "utf-8", maxBuffer: MAX_GIT_OUTPUT }
    );

    // Helper to strip git prefix from paths (e.g., "workspace/file.txt" -> "file.txt")
    const stripPrefix = (filepath: string): string => {
      if (gitPrefix && filepath.startsWith(gitPrefix)) {
        return filepath.slice(gitPrefix.length);
      }
      return filepath;
    };

    // Parse numstat into a map
    const statsMap = new Map<
      string,
      { additions: number; deletions: number }
    >();
    numstat
      .split("\n")
      .filter((line) => line.trim())
      .forEach((line) => {
        const parts = line.trim().split("\t");
        if (parts.length >= 3) {
          const additions = parseInt(parts[0], 10) || 0;
          const deletions = parseInt(parts[1], 10) || 0;
          // Strip prefix to get workspace-relative path
          const filepath = stripPrefix(parts.slice(2).join("\t"));
          // Skip binary files (show as - -)
          if (parts[0] !== "-" && parts[1] !== "-") {
            statsMap.set(filepath, { additions, deletions });
          }
        }
      });

    // Parse name-status
    const changes: FileChange[] = nameStatus
      .split("\n")
      .filter((line) => line.trim())
      .map((line): FileChange | null => {
        const parts = line.trim().split("\t");
        if (parts.length >= 2) {
          const statusCode = parts[0][0];
          let filepath: string;

          // For renamed/copied files (R100, C100), use the new path
          if (statusCode === "R" || statusCode === "C") {
            filepath = parts.length >= 3 ? parts[2] : parts[1];
          } else {
            filepath = parts[1];
          }
          
          // Strip prefix to get workspace-relative path
          filepath = stripPrefix(filepath);

          let status: FileChange["status"];
          if (statusCode === "A") status = "added";
          else if (statusCode === "D") status = "deleted";
          else status = "modified"; // M, R, C, T, etc.

          const stats = statsMap.get(filepath);
          const change: FileChange = {
            path: filepath,
            status,
          };
          if (stats?.additions !== undefined) change.additions = stats.additions;
          if (stats?.deletions !== undefined) change.deletions = stats.deletions;
          return change;
        }
        return null;
      })
      .filter((c): c is FileChange => c !== null);

    // Add untracked files as "added"
    const untrackedChanges: FileChange[] = untrackedFiles
      .split("\n")
      .filter((line) => line.trim())
      .map((filepath) => {
        // Count lines in the new file for stats
        const fullPath = path.join(workspacePath, filepath);
        let additions = 0;
        try {
          if (fs.existsSync(fullPath)) {
            const content = fs.readFileSync(fullPath, "utf-8");
            additions = content.split("\n").length;
          }
        } catch {
          // Ignore errors (binary files, etc.)
        }

        const change: FileChange = {
          path: filepath,
          status: "added",
        };
        if (additions > 0) change.additions = additions;
        return change;
      });

    return { changes: [...changes, ...untrackedChanges] };
  } catch (error) {
    // Do NOT collapse a failure into an empty diff: with the old
    // `2>/dev/null || echo ''` swallow, a git refusal (e.g. dubious-ownership
    // on the agent-owned workspace) read as "agent changed nothing" and the
    // real submission died with the instance. Surface it instead.
    return {
      changes: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function listUntrackedFiles(workspacePath: string): string {
  return execSync(
    `git ls-files --others --exclude-standard`,
    { cwd: workspacePath, env: GIT_ENV, encoding: "utf-8", maxBuffer: MAX_GIT_OUTPUT }
  );
}

/**
 * Get the baseline reference to diff against
 * Returns the baseline tag if it exists, otherwise HEAD
 */
function getBaselineRef(workspacePath: string): string {
  try {
    // Check if baseline tag exists
    execSync(`git rev-parse ${BASELINE_TAG}`, {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });
    return BASELINE_TAG;
  } catch {
    // Tag doesn't exist, fall back to HEAD
    console.log(`[diff-collector] No baseline tag found, diffing against HEAD`);
    return "HEAD";
  }
}

/**
 * Get the git prefix (relative path from repo root to cwd)
 * Used for constructing paths for git show commands
 */
function getGitPrefix(workspacePath: string): string {
  try {
    const prefix = execSync("git rev-parse --show-prefix", {
      cwd: workspacePath,
      env: GIT_ENV,
      encoding: "utf-8",
      stdio: "pipe",
    });
    return prefix.trim();
  } catch {
    return "";
  }
}

/**
 * Get file content for a changed file
 * Handles edge cases: large files, binary files, deleted files
 */
function getFileContent(
  workspacePath: string,
  change: FileChange
): FileContent {
  const filePath = path.join(workspacePath, change.path);
  const diffTarget = getBaselineRef(workspacePath);
  // Git show requires path relative to repo root, not workspace
  const gitPrefix = getGitPrefix(workspacePath);
  const gitPath = gitPrefix + change.path;

  let baseContent: string | null = null;
  let currentContent: string | null = null;

  // Get base content (from baseline/HEAD)
  if (change.status !== "added") {
    try {
      // Check if file is binary (use gitPath for git commands)
      const isBinary = isFileBinary(workspacePath, gitPath, diffTarget);
      if (isBinary) {
        baseContent = "<< Binary file >>";
      } else {
        const content = execSync(
          `git show "${diffTarget}:${gitPath}" 2>/dev/null || echo ''`,
          {
            cwd: workspacePath,
            env: GIT_ENV,
            encoding: "utf-8",
            maxBuffer: MAX_FILE_SIZE * 2,
          }
        );

        if (content.length > MAX_FILE_SIZE) {
          baseContent = `<< File too large (${Math.round(content.length / 1024)}KB, truncated) >>\n${content.substring(0, MAX_FILE_SIZE)}`;
        } else {
          baseContent = content;
        }
      }
    } catch (error) {
      baseContent = `<< Error reading base content: ${error instanceof Error ? error.message : error} >>`;
    }
  }

  // Get current content (from working directory)
  if (change.status !== "deleted") {
    try {
      // Check if file exists and is readable
      if (!fs.existsSync(filePath)) {
        currentContent = null;
      } else {
        const stats = fs.statSync(filePath);

        // Directory entries can land here when the agent commits to an
        // embedded git repo nested in the workspace (e.g. a deploy-repo
        // clone). git diff treats those as a single subproject change,
        // and reading the path with fs.readFileSync would throw EISDIR.
        // Note: paths .gitignored at the workspace level (e.g.
        // workspace/rust-pooler in pgbouncer-pgcat) never reach here at
        // all because git ls-files --others --exclude-standard skips
        // them. Envs that need those captured should snapshot via a
        // filesystem-walk hook instead (TODO: add an extraSnapshotPaths
        // config option that runs alongside the git-based detector).
        if (stats.isDirectory()) {
          currentContent = "<< Directory (likely nested git repo or subproject) >>";
        } else if (stats.size > MAX_FILE_SIZE) {
          const partial = fs.readFileSync(filePath, {
            encoding: "utf-8",
            flag: "r",
          });
          currentContent = `<< File too large (${Math.round(stats.size / 1024)}KB, truncated) >>\n${partial.substring(0, MAX_FILE_SIZE)}`;
        } else {
          // Check if binary
          const buffer = fs.readFileSync(filePath);
          if (isBinaryBuffer(buffer)) {
            currentContent = "<< Binary file >>";
          } else {
            currentContent = buffer.toString("utf-8");
          }
        }
      }
    } catch (error) {
      currentContent = `<< Error reading current content: ${error instanceof Error ? error.message : error} >>`;
    }
  }

  return { path: change.path, baseContent, currentContent };
}

/**
 * Check if a file in git is binary
 */
function isFileBinary(
  workspacePath: string,
  filepath: string,
  ref: string
): boolean {
  try {
    // Use git diff to check if file is binary
    const output = execSync(
      `git -c core.fileMode=false diff --numstat ${ref} -- "${filepath}" 2>/dev/null || echo '- - '`,
      { cwd: workspacePath, env: GIT_ENV, encoding: "utf-8", maxBuffer: MAX_GIT_OUTPUT }
    );
    // Binary files show as "- -" in numstat
    return output.trim().startsWith("- -") || output.trim().startsWith("-\t-");
  } catch {
    return false;
  }
}

/**
 * Check if a buffer contains binary data
 * Looks for null bytes in the first 8000 bytes
 */
function isBinaryBuffer(buffer: Buffer): boolean {
  const checkLength = Math.min(buffer.length, 8000);
  for (let i = 0; i < checkLength; i++) {
    if (buffer[i] === 0) {
      return true;
    }
  }
  return false;
}

