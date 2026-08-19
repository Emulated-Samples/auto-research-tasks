/**
 * `env-orchestrator harbor validate` — release gate for a packaged task
 * (decision D7): N solution-replay trials through the PINNED harbor CLI
 * (every trial must score >= the problem's minReplayScore on every test,
 * default 1.0 — see taskInfo.ts readMinReplayScore) + an image secret-scan.
 *
 * The unsolved-state check was deliberately rejected (D5) — pre-ship
 * confidence beyond this gate comes from kicking off real agent runs.
 *
 * Tasks whose replay cannot run HERE still pass the gate with an explicit
 * evidence record saying why (no solution/ — no reference solution exists;
 * gpus > 0 — the replay needs GPU compute this box doesn't have). The
 * platform's replay release-runs are the real validation venue for those; a
 * built image is never refused just because this box can't exercise it.
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import {
  prewarmEgressControlSidecar,
  runReplayTrials,
} from "../runtime/replay.js";
import {
  readGpus,
  readImageTag,
  readMinReplayScore,
  readNetworkModes,
} from "../runtime/taskInfo.js";
import { ensureHarborCli } from "../runtime/venv.js";

// Re-exported for existing consumers (tests import these from validate.js).
export { readMinReplayScore, readNetworkModes } from "../runtime/taskInfo.js";

// Concatenated so the scanner never matches its own source when these
// files ship inside the image (see staging.ts).
const IMAGE_SECRET_PATTERNS = [
  "ghs_[A-Za-z0-9]{16,}",
  "ghp_[A-Za-z0-9]{16,}",
  "github_pat" + "_[A-Za-z0-9_]{16,}",
  "sk-" + "ant-[A-Za-z0-9-]{8,}",
  "x-access" + "-token:",
].join("|");

function log(msg: string): void {
  console.log(`[harbor:validate] ${msg}`);
}

export interface HarborValidateOptions {
  taskDir: string;
  trials: number;
  jobsDir?: string;
  skipSecretScan: boolean;
}

function imageSecretScan(imageTag: string): void {
  log(`Secret-scanning image ${imageTag}...`);
  const rc = spawnSync(
    "docker",
    [
      "run",
      "--rm",
      "--entrypoint",
      "/bin/sh",
      imageTag,
      "-c",
      // Scan the areas our pipeline writes; /proc & friends excluded by scope.
      `grep -rIlE '${IMAGE_SECRET_PATTERNS}' /hyperfocal /root /home /etc 2>/dev/null | head -20`,
    ],
    { encoding: "utf-8", maxBuffer: 16 * 1024 * 1024 }
  );
  const hits = (rc.stdout || "").trim();
  if (hits) {
    throw new Error(`Image secret scan FAILED — hits:\n${hits}`);
  }
  log("Image secret scan clean");
}

/**
 * The machine-readable gate record. The release pipeline treats "no
 * validation.json anywhere" as "gate did not run" and fails the build, so
 * even skipped-replay outcomes write one, with `mode` saying why.
 */
function writeValidationRecord(
  dir: string,
  record: Record<string, unknown>
): void {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, "validation.json"),
    JSON.stringify({ ...record, completedAt: new Date().toISOString() }, null, 2) +
      "\n"
  );
}

export async function validate(opts: HarborValidateOptions): Promise<void> {
  const taskDir = path.resolve(opts.taskDir);
  if (!fs.existsSync(path.join(taskDir, "task.toml"))) {
    throw new Error(`${taskDir} is not a harbor task dir (no task.toml)`);
  }

  const imageTag = readImageTag(taskDir);
  if (!opts.skipSecretScan) {
    if (!imageTag) {
      throw new Error("task.toml has no docker_image; cannot secret-scan");
    }
    imageSecretScan(imageTag);
  }

  const jobsDir = path.resolve(
    opts.jobsDir || path.join(taskDir, "..", "..", "validation-jobs")
  );
  const skippedEvidenceDir = () =>
    path.join(jobsDir, `validate-${path.basename(taskDir)}-${Date.now()}`);

  const hasSolution = fs.existsSync(path.join(taskDir, "solution"));
  if (!hasSolution) {
    log(
      "Task has no solution/ (no reference solution) — skipping replay " +
        "trials. Release confidence must come from real agent runs (D4/D5)."
    );
    writeValidationRecord(skippedEvidenceDir(), {
      taskDir,
      imageTag,
      trials: 0,
      passed: true,
      mode: "no-solution — replay not applicable; validate via agent runs",
    });
    return;
  }

  if (readGpus(taskDir) > 0) {
    log(
      "Task declares gpus > 0 — this box cannot run its replay; deferring " +
        "replay validation to platform replay runs on GPU compute. The " +
        "image itself is still built, scanned, and shippable."
    );
    writeValidationRecord(skippedEvidenceDir(), {
      taskDir,
      imageTag,
      trials: 0,
      passed: true,
      mode: "gpu-deferred — replay validation happens via platform replay runs",
    });
    return;
  }

  const harborBin = ensureHarborCli(log);

  // Any non-public phase (deny-all / allowlist / no-network) makes harbor
  // attach its egress-control sidecar — settle that image's build before the
  // concurrent trials race its non-reentrant build lock.
  const nonPublicModes = readNetworkModes(taskDir).filter(
    (mode) => mode !== "public"
  );
  if (nonPublicModes.length > 0) {
    prewarmEgressControlSidecar(harborBin, log);
  }

  // Per-problem replay threshold (default 1.0 = historical all-tests-pass
  // gate). Read BEFORE the trials so a malformed file fails fast.
  const minScore = readMinReplayScore(taskDir);
  log(`Replay reward threshold: every reward >= ${minScore}`);

  const { jobName, jobDir, summaries } = runReplayTrials({
    taskDir,
    trials: opts.trials,
    jobsDir,
    minScore,
    harborBin,
    log,
  });

  for (const s of summaries) {
    const detail = s.exception
      ? `exception=${s.exception}`
      : `rewards=${JSON.stringify(s.rewards)}`;
    log(
      `  ${s.ok ? "PASS" : "FAIL"} ${s.trialName} ${detail} (threshold ${minScore})`
    );
  }

  const failures = summaries.filter((s) => !s.ok);
  if (failures.length > 0) {
    throw new Error(
      `Replay gate FAILED: ${failures.length}/${summaries.length} trials below ` +
        `the reward threshold ${minScore}`
    );
  }

  log(
    `Replay gate PASSED (${opts.trials}/${opts.trials} trials, every reward >= ${minScore})`
  );

  writeValidationRecord(jobDir, {
    taskDir,
    imageTag,
    trials: opts.trials,
    minReplayScore: minScore,
    passed: true,
    harborJob: jobName,
  });
}
