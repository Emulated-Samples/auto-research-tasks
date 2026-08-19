/**
 * Readers for a packaged harbor task dir (task.toml + the gate config that
 * rides beside it). Raw-toml greps on purpose — consumers only need single
 * fields, not the full parsed schema.
 */

import * as fs from "fs";
import * as path from "path";

export function readImageTag(taskDir: string): string | null {
  const toml = fs.readFileSync(path.join(taskDir, "task.toml"), "utf-8");
  return toml.match(/^docker_image\s*=\s*"([^"]+)"/m)?.[1] ?? null;
}

export function readGpus(taskDir: string): number {
  const toml = fs.readFileSync(path.join(taskDir, "task.toml"), "utf-8");
  return Number(toml.match(/^gpus\s*=\s*(\d+)/m)?.[1] ?? 0);
}

/**
 * Every `network_mode = "..."` declared anywhere in task.toml (agent /
 * verifier / environment sections). Consumers only need "is any phase
 * non-public", not the full parsed policy.
 */
export function readNetworkModes(taskDir: string): string[] {
  const toml = fs.readFileSync(path.join(taskDir, "task.toml"), "utf-8");
  return [...toml.matchAll(/^network_mode\s*=\s*"([^"]+)"/gm)].map((m) => m[1]);
}

/**
 * Per-problem solution-replay reward floor. Written by `harbor package`
 * into task.toml [metadata] min_replay_score since the 2026-07-19 fold
 * (harbor 0.18 verifiably tolerates extra metadata keys — junk-table
 * experiment; the old claim that it strict-rejects unknown fields was
 * wrong). Only solution-declaring problems carry the key at all — no
 * solution, no replay gate (V1-F5). Reading order:
 *   1. task.toml [metadata] min_replay_score (current emission)
 *   2. hyperfocal-validate.json beside task.toml (pre-fold task dirs in
 *      old S3 bundles, including the pre-v2 oracleMinScore key)
 *   3. absent everywhere = the historical all-tests-1.0 gate.
 * Sourced from problems.yaml `minReplayScore` (see env-base Problem).
 */
export function readMinReplayScore(taskDir: string): number {
  const tomlPath = path.join(taskDir, "task.toml");
  const toml = fs.existsSync(tomlPath) ? fs.readFileSync(tomlPath, "utf-8") : "";
  const tomlMatch = toml.match(/^min_replay_score\s*=\s*([0-9.]+)/m);
  if (tomlMatch) {
    const v = Number(tomlMatch[1]);
    if (!Number.isFinite(v) || v <= 0 || v > 1) {
      throw new Error(
        `task.toml [metadata] min_replay_score ${tomlMatch[1]} is invalid ` +
          `(must be a number in (0, 1])`
      );
    }
    return v;
  }
  const file = path.join(taskDir, "hyperfocal-validate.json");
  if (!fs.existsSync(file)) return 1.0;
  const parsed = JSON.parse(fs.readFileSync(file, "utf-8"));
  // Fallback to the pre-v2 key: task dirs packaged before the rename still
  // exist in S3 bundles.
  const v: unknown = parsed?.minReplayScore ?? parsed?.oracleMinScore;
  if (v === undefined || v === null) return 1.0;
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0 || v > 1) {
    throw new Error(
      `${file} has invalid minReplayScore ${JSON.stringify(v)} ` +
        `(must be a number in (0, 1])`
    );
  }
  return v;
}
