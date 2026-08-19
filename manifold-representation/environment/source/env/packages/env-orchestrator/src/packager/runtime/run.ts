/**
 * `env-orchestrator harbor run` — thin wrapper over the PINNED stock
 * `harbor run` for poking at packaged tasks (dev surface, decision D9).
 * Adds nothing but the version pin and defaults; everything after `--` is
 * passed to harbor verbatim. Engineers can always use stock harbor directly.
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { ensureHarborCli } from "./venv.js";

function log(msg: string): void {
  console.log(`[harbor:run] ${msg}`);
}

export interface HarborRunOptions {
  taskDir: string;
  agent: string;
  jobsDir?: string;
  passthrough: string[];
}

export async function run(opts: HarborRunOptions): Promise<void> {
  const taskDir = path.resolve(opts.taskDir);
  if (!fs.existsSync(path.join(taskDir, "task.toml"))) {
    throw new Error(`${taskDir} is not a harbor task dir (no task.toml)`);
  }
  const harborBin = ensureHarborCli(log);

  const args = [
    "run",
    "-p",
    taskDir,
    "--agent",
    opts.agent,
    "-o",
    path.resolve(opts.jobsDir || "harbor-jobs"),
    ...opts.passthrough,
  ];
  log(`${harborBin} ${args.join(" ")}`);
  const rc = spawnSync(harborBin, args, { stdio: "inherit" });
  process.exit(rc.status ?? 1);
}
