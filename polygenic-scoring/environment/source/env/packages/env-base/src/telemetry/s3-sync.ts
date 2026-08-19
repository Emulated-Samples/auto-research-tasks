/**
 * S3 Telemetry Sync
 *
 * Periodically syncs telemetry logs from local filesystem to S3 with adaptive backoff:
 *   - First 10 minutes: every 5 seconds (high visibility during active work)
 *   - 10-30 minutes:    every 30 seconds
 *   - After 30 minutes: every 2 minutes
 *
 * Uploads product telemetry and opted-in debugging sidecars:
 *   - metadata.json (session status/timing)
 *   - {category}/{timestamp}.jsonl (structured logs)
 *   - {category}/{timestamp}.debug.log (raw coding-agent CLI stream)
 *
 * Does NOT upload (local dev convenience files):
 *   - combined.log
 *   - {category}/{timestamp}.log (human-readable mirror)
 *
 * TODO: Incremental upload optimization
 * Currently, we re-upload entire .jsonl files on each sync. For long rollouts,
 * this becomes wasteful (re-uploading 10MB+ of unchanged data). Proposed approach:
 *   1. Track last-uploaded byte offset per file (store in memory or .sync-state file)
 *   2. On sync, read only new bytes appended since last upload
 *   3. Use S3 multipart upload or append to a "live" object
 *   4. On session end, finalize/rename to permanent object
 * This would reduce S3 PUT costs and improve sync latency for large logs.
 * For now, the current approach is simple and correct, just not optimal.
 *
 * Usage:
 *   const sync = new S3TelemetrySync({
 *     runId: process.env.RUN_ID!,
 *     rolloutId: process.env.ROLLOUT_ID!,
 *     bucketName: process.env.TELEMETRY_S3_BUCKET!,
 *   });
 *   await sync.start();
 *   // ... run rollout ...
 *   await sync.stop();
 *
 * Environment Variables:
 *   - TELEMETRY_S3_BUCKET: S3 bucket name (required)
 *   - RUN_ID: Run identifier (required)
 *   - ROLLOUT_ID: Rollout identifier (required)
 *   - TELEMETRY_AWS_ACCESS_KEY_ID: AWS access key for telemetry bucket (management account)
 *   - TELEMETRY_AWS_SECRET_ACCESS_KEY: AWS secret key for telemetry bucket
 *   - TELEMETRY_AWS_SESSION_TOKEN: AWS session token (for assumed roles)
 *   - AWS_REGION: AWS region (optional, defaults to us-west-2)
 *
 * Note: The telemetry S3 bucket is in the management account, not the sandbox.
 * We use separate TELEMETRY_AWS_* credentials to access it.
 */

import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { fromInstanceMetadata } from "@aws-sdk/credential-providers";
import * as fs from "fs";
import * as path from "path";
import { getLogsDir } from "./index.js";

export interface S3TelemetrySyncConfig {
  /** Run ID (groups multiple rollouts) */
  runId: string;
  /** Unique rollout ID */
  rolloutId: string;
  /** S3 bucket name */
  bucketName: string;
  /** AWS region (defaults to us-west-2) */
  region?: string;
}

interface BackoffSchedule {
  /** Duration in ms from start when this interval applies */
  until: number;
  /** Sync interval in ms */
  interval: number;
}

const BACKOFF_SCHEDULE: BackoffSchedule[] = [
  { until: 10 * 60 * 1000, interval: 5 * 1000 }, // First 10 min: every 5s
  { until: 30 * 60 * 1000, interval: 30 * 1000 }, // 10-30 min: every 30s
  { until: Infinity, interval: 2 * 60 * 1000 }, // After 30 min: every 2 min
];

/** Categories to upload (directories within problem folder) */
const UPLOAD_CATEGORIES = ["agent", "tests", "setup"];

/** Special directories with different upload patterns */
const DIFFS_CATEGORY = "diffs";

export class S3TelemetrySync {
  private client: S3Client;
  private config: S3TelemetrySyncConfig;
  private logsDir: string;
  private startTime: number = 0;
  private timer: NodeJS.Timeout | null = null;
  private syncCount = 0;
  private isRunning = false;

  constructor(config: S3TelemetrySyncConfig) {
    if (!config.bucketName) {
      throw new Error("S3TelemetrySync: bucketName is required");
    }
    if (!config.runId) {
      throw new Error("S3TelemetrySync: runId is required");
    }
    if (!config.rolloutId) {
      throw new Error("S3TelemetrySync: rolloutId is required");
    }

    this.config = config;

    // Use dedicated telemetry credentials (management account)
    // If TELEMETRY_AWS_* is set, use those (injected by control plane)
    // Otherwise, use EC2 instance metadata (bypasses AWS_* env vars from sandbox)
    let credentials;
    if (process.env.TELEMETRY_AWS_ACCESS_KEY_ID) {
      credentials = {
        accessKeyId: process.env.TELEMETRY_AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.TELEMETRY_AWS_SECRET_ACCESS_KEY!,
        sessionToken: process.env.TELEMETRY_AWS_SESSION_TOKEN,
      };
    } else if (process.env.AWS_ACCESS_KEY_ID) {
      // Sandbox AWS creds are set - use instance metadata to access management account S3
      // This bypasses the sandbox credentials in the environment
      console.log("[s3-telemetry] Using EC2 instance metadata for credentials (bypassing sandbox creds)");
      credentials = fromInstanceMetadata({ timeout: 1000, maxRetries: 3 });
    }
    // If neither is set, let SDK use default credential chain

    this.client = new S3Client({
      region: config.region || "us-west-2",
      credentials,
    });
    this.logsDir = getLogsDir();
  }

  /**
   * Start the sync service
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.warn("[s3-telemetry] Already running");
      return;
    }

    this.isRunning = true;
    this.startTime = Date.now();

    console.log(
      `[s3-telemetry] Starting sync to s3://${this.config.bucketName}/${this.getS3Prefix()}`
    );
    console.log(
      `[s3-telemetry] Schedule: 5s (0-10min), 30s (10-30min), 2min (30min+)`
    );

    // Initial sync
    await this.sync();

    // Schedule next sync with backoff
    this.scheduleNextSync();
  }

  /**
   * Stop the sync service and perform final sync
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    this.isRunning = false;

    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    // Final sync
    console.log(`[s3-telemetry] Performing final sync...`);
    await this.sync();
    console.log(`[s3-telemetry] Stopped (${this.syncCount} total syncs)`);
  }

  /**
   * Get the S3 key prefix for this rollout
   */
  getS3Prefix(): string {
    return `runs/${this.config.runId}/${this.config.rolloutId}`;
  }

  /**
   * Upload a single artifact under this rollout's telemetry prefix, reusing the
   * same bucket, credentials, and key convention as the periodic sync. Used for
   * one-shot files produced AFTER the rollout — and after the sync service has
   * already stopped — e.g. the post-rollout QA analysis report.
   *
   * @param localPath  Absolute path to the file to upload.
   * @param keyUnderPrefix  Optional key relative to the rollout prefix. If
   *   omitted and the file lives under the logs dir, the key mirrors its
   *   on-disk layout (`<problemId>/<...>`) so the artifact lands beside the
   *   rollout's other telemetry; otherwise the basename is used.
   * @param dryRun  If true, compute and return the S3 URI without uploading.
   * @returns the full `s3://bucket/key` URI.
   */
  async uploadArtifact(
    localPath: string,
    keyUnderPrefix?: string,
    dryRun = false
  ): Promise<string> {
    if (!fs.existsSync(localPath)) {
      throw new Error(`uploadArtifact: file not found: ${localPath}`);
    }
    let rel = keyUnderPrefix;
    if (!rel) {
      const fromLogs = path.relative(this.logsDir, localPath);
      rel =
        fromLogs && !fromLogs.startsWith("..") && !path.isAbsolute(fromLogs)
          ? fromLogs
          : path.basename(localPath);
    }
    const key = `${this.getS3Prefix()}/${rel}`;
    const uri = `s3://${this.config.bucketName}/${key}`;
    if (!dryRun) {
      await this.uploadFile(localPath, key);
    }
    return uri;
  }

  /**
   * Schedule the next sync based on backoff schedule
   */
  private scheduleNextSync(): void {
    if (!this.isRunning) return;

    const elapsed = Date.now() - this.startTime;
    const schedule =
      BACKOFF_SCHEDULE.find((s) => elapsed < s.until) ||
      BACKOFF_SCHEDULE[BACKOFF_SCHEDULE.length - 1];

    this.timer = setTimeout(async () => {
      if (!this.isRunning) return;

      try {
        await this.sync();
      } catch (error) {
        console.error("[s3-telemetry] Sync failed:", error);
      }

      this.scheduleNextSync();
    }, schedule.interval);

    // Don't keep process alive for sync
    this.timer.unref();
  }

  /**
   * Perform a sync - upload telemetry files and metadata to S3
   */
  private async sync(): Promise<void> {
    if (!fs.existsSync(this.logsDir)) {
      return; // No logs yet
    }

    const prefix = this.getS3Prefix();
    const startTime = Date.now();
    let filesUploaded = 0;

    // Find all problem directories
    const problemDirs = fs.readdirSync(this.logsDir).filter((name) => {
      try {
        return fs.statSync(path.join(this.logsDir, name)).isDirectory();
      } catch {
        return false;
      }
    });

    for (const problemId of problemDirs) {
      const problemDir = path.join(this.logsDir, problemId);

      // Upload metadata.json
      const metadataPath = path.join(problemDir, "metadata.json");
      if (fs.existsSync(metadataPath)) {
        await this.uploadFile(
          metadataPath,
          `${prefix}/${problemId}/metadata.json`
        );
        filesUploaded++;
      }

      // Upload product JSONL plus raw coding-agent debug logs. Plain .log
      // files remain local because they duplicate combined.log and can be
      // very noisy; debug logs are intentionally uploaded because they are the
      // only byte-faithful provider protocol record when enabled.
      for (const category of UPLOAD_CATEGORIES) {
        const categoryDir = path.join(problemDir, category);
        if (!fs.existsSync(categoryDir)) continue;

        try {
          const files = fs.readdirSync(categoryDir);
          for (const file of files) {
            if (!this.shouldUploadCategoryFile(file)) continue;

            const filePath = path.join(categoryDir, file);
            if (fs.statSync(filePath).isFile()) {
              await this.uploadFile(
                filePath,
                `${prefix}/${problemId}/${category}/${file}`
              );
              filesUploaded++;
            }
          }
        } catch {
          // Directory might have changed during iteration
        }
      }

      // Upload diffs directory (special handling for .json files)
      const diffsDir = path.join(problemDir, DIFFS_CATEGORY);
      if (fs.existsSync(diffsDir)) {
        try {
          // Upload changes.json
          const changesPath = path.join(diffsDir, "changes.json");
          if (fs.existsSync(changesPath)) {
            await this.uploadFile(
              changesPath,
              `${prefix}/${problemId}/${DIFFS_CATEGORY}/changes.json`
            );
            filesUploaded++;
          }

          // Upload contents/*.json files
          const contentsDir = path.join(diffsDir, "contents");
          if (fs.existsSync(contentsDir)) {
            const contentFiles = fs.readdirSync(contentsDir);
            for (const file of contentFiles) {
              if (!file.endsWith(".json")) continue;

              const filePath = path.join(contentsDir, file);
              if (fs.statSync(filePath).isFile()) {
                await this.uploadFile(
                  filePath,
                  `${prefix}/${problemId}/${DIFFS_CATEGORY}/contents/${file}`
                );
                filesUploaded++;
              }
            }
          }
        } catch {
          // Directory might have changed during iteration
        }
      }
    }

    this.syncCount++;
    const duration = Date.now() - startTime;

    // Log every 10th sync or first sync
    if (this.syncCount === 1 || this.syncCount % 10 === 0) {
      console.log(
        `[s3-telemetry] Sync #${this.syncCount}: ${filesUploaded} files in ${duration}ms`
      );
    }
  }

  /**
   * Upload a single file to S3
   */
  private async uploadFile(localPath: string, s3Key: string): Promise<void> {
    const content = fs.readFileSync(localPath);

    await this.client.send(
      new PutObjectCommand({
        Bucket: this.config.bucketName,
        Key: s3Key,
        Body: content,
        ContentType: this.getContentType(localPath),
      })
    );
  }

  /**
   * Get MIME type for a file
   */
  private getContentType(filePath: string): string {
    if (filePath.endsWith(".json")) return "application/json";
    if (filePath.endsWith(".jsonl")) return "application/x-ndjson";
    if (filePath.endsWith(".md")) return "text/markdown";
    if (filePath.endsWith(".debug.log")) return "text/plain";
    return "application/octet-stream";
  }

  private shouldUploadCategoryFile(fileName: string): boolean {
    return fileName.endsWith(".jsonl") || fileName.endsWith(".debug.log");
  }
}

/**
 * Create S3 telemetry sync from environment variables.
 * Returns null if not configured (graceful degradation for local dev).
 */
export function createS3SyncFromEnv(): S3TelemetrySync | null {
  const bucketName = process.env.TELEMETRY_S3_BUCKET;
  const runId = process.env.RUN_ID;
  const rolloutId = process.env.ROLLOUT_ID;

  if (!bucketName) {
    console.log("[s3-telemetry] Disabled (TELEMETRY_S3_BUCKET not set)");
    return null;
  }

  if (!runId || !rolloutId) {
    console.log("[s3-telemetry] Disabled (RUN_ID or ROLLOUT_ID not set)");
    return null;
  }

  // Info about which credentials are being used
  if (process.env.TELEMETRY_AWS_ACCESS_KEY_ID) {
    console.log("[s3-telemetry] Using explicit telemetry credentials");
  } else if (process.env.AWS_ACCESS_KEY_ID) {
    console.log("[s3-telemetry] Using EC2 instance role (bypassing sandbox AWS_* creds)");
  } else {
    console.log("[s3-telemetry] Using default credential chain");
  }

  return new S3TelemetrySync({
    bucketName,
    runId,
    rolloutId,
    region: process.env.AWS_REGION,
  });
}

