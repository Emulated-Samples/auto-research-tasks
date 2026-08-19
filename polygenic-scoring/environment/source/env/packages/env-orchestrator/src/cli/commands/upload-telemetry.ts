/**
 * `env-orchestrator upload-telemetry <file> [--key=<key>] [--dry-run]`
 *
 * One-shot upload of a single artifact under the current rollout's telemetry
 * prefix, reusing env-base's S3 plumbing (bucket, credentials, key convention).
 * Intended for files produced AFTER the rollout — and after S3TelemetrySync has
 * stopped — most notably the post-rollout QA analysis report. Lands the file at
 * `runs/<RUN_ID>/<ROLLOUT_ID>/<problemId>/<name>` so it sits beside the
 * rollout's other telemetry and is served by the same prefix readers.
 *
 * No-ops (exit 0) when telemetry is not configured (e.g. dev containers), so it
 * is safe to wire unconditionally into the rollout tail.
 *
 * Split out of the old environments.ts god-file with no behavior change.
 */

import * as path from "path";
import { createS3SyncFromEnv } from "@hyperfocal/env-base";

export async function handleUploadTelemetryCommand(): Promise<void> {
  const args = process.argv.slice(2);
  const positionals = args.filter((a) => !a.startsWith("--"));
  const file = positionals[1]; // positionals[0] === "upload-telemetry"
  const dryRun = args.includes("--dry-run");
  const keyFlag = args.find((a) => a.startsWith("--key="));
  const keyUnderPrefix = keyFlag ? keyFlag.slice("--key=".length) : undefined;

  if (!file) {
    console.error(
      "Usage: env-orchestrator upload-telemetry <file> [--key=<key>] [--dry-run]"
    );
    process.exit(1);
  }

  const absPath = path.isAbsolute(file)
    ? file
    : path.resolve(process.cwd(), file);

  const sync = createS3SyncFromEnv();
  if (!sync) {
    console.log(
      "[upload-telemetry] Telemetry disabled (TELEMETRY_S3_BUCKET / RUN_ID / ROLLOUT_ID not set); skipping upload."
    );
    return;
  }

  try {
    const uri = await sync.uploadArtifact(absPath, keyUnderPrefix, dryRun);
    console.log(
      `[upload-telemetry] ${dryRun ? "(dry-run) would upload" : "uploaded"} ${absPath} -> ${uri}`
    );
  } catch (err) {
    console.error(
      `[upload-telemetry] Failed: ${err instanceof Error ? err.message : String(err)}`
    );
    process.exit(1);
  }
}
