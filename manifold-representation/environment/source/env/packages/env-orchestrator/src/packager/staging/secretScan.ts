/**
 * Fail-closed token scan of a staged build context. Runs once per package,
 * after staging (overlay included) — the context never changes between
 * problems under contract v2, so one scan covers every problem's build.
 */

import { spawnSync } from "child_process";

// Assembled via concatenation so the scanner never matches its own source
// (a literal "x-access" + "-token:" in one string would self-match when the
// live packages are overlaid into the build context).
const SECRET_PATTERNS = [
  "ghs_[A-Za-z0-9]{16,}",
  "ghp_[A-Za-z0-9]{16,}",
  "github_pat" + "_[A-Za-z0-9_]{16,}",
  "sk-" + "ant-[A-Za-z0-9-]{8,}",
  "x-access" + "-token:",
].join("|");

/** Fail-closed token scan of the staged context. */
export function secretScan(
  dir: string,
  log: (msg: string) => void = (msg) => console.log(`[harbor:package] ${msg}`)
): void {
  const rc = spawnSync(
    "grep",
    ["-rIlE", SECRET_PATTERNS, dir, "--exclude-dir=node_modules"],
    { encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 }
  );
  // grep: 0 = matches found (BAD), 1 = clean, >1 = error
  if (rc.status === 0) {
    throw new Error(
      `Secret scan FAILED — token patterns found in build context:\n${rc.stdout}`
    );
  }
  if (rc.status !== 1) {
    throw new Error(`Secret scan errored: ${rc.stderr}`);
  }
  log("Secret scan clean");
}
