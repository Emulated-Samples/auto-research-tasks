/**
 * `env-orchestrator harbor grade --problem <id>` — the in-container harbor
 * verifier entrypoint (invoked as root by the generated tests/test.sh shim).
 *
 * Runs the env's NATIVE test path (the same code `env-orchestrator test`
 * uses), then relocates the dual-written harbor reward file plus the native
 * test telemetry to harbor's contract dir (/logs/verifier by default;
 * HARBOR_VERIFIER_DIR overrides for tests/dev).
 *
 * FAILURE SEMANTICS (decision A3, verified against harbor source): after
 * test.sh, harbor requires <verifier>/reward.json — a missing file raises
 * RewardFileNotFoundError and records a trial EXCEPTION. So:
 *   - harness runs to completion  -> write reward.json (real per-test
 *     scores from env-base's harbor rewards emitter, PLUS the explicit
 *     aggregate "reward" key — the platform's rollout score) + telemetry,
 *     exit 0. Agent failures flow through naturally as real 0 scores.
 *   - harness itself crashes (env module load failure, setup invariant
 *     broken, node error, missing reward emission) -> write NOTHING to
 *     reward.json and exit non-zero. The trial records an exception, which
 *     keeps infra failures distinguishable from agent failures.
 * NEVER write the old fail-closed {"reward": 0} shape — its literal
 * `reward` key doesn't match the per-testId dict and pollutes downstream
 * consumers; reward 0 is reserved for agent-attributable outcomes.
 *
 * Trust model (unchanged from the old generated test.sh):
 *   - The agent controls workspace/ only; .git, environment/ (verifiers)
 *     and packages/ are root-owned o-rwx (baked at image build).
 *   - The shim redirects HYPERFOCAL_LOGS_DIR to a fresh root-owned mktemp
 *     dir before node starts, so agent-planted files in the world-writable
 *     default logs dir can never be read as results.
 */

import * as fs from "fs";
import * as path from "path";
import { aggregateTestResults, getLogsDir } from "@hyperfocal/env-base";
// Deliberate packager -> cli seam: grade IS the in-container verifier
// entrypoint and reuses the exact test pipeline `env-orchestrator test`
// ships (same env-context load, same reward dual-write), so a packaged
// image can never grade differently from a dev box.
import { loadEnvCommandContext } from "../../cli/env-context.js";
import { handleTestCommand } from "../../cli/commands/test.js";

/** Harbor's reward-file contract dir; override via HARBOR_VERIFIER_DIR. */
const DEFAULT_VERIFIER_DIR = "/logs/verifier";

function log(msg: string): void {
  console.log(`[harbor:grade] ${msg}`);
}

function copyIfExists(src: string, dstDir: string): void {
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(dstDir, path.basename(src)));
  }
}

export interface HarborGradeOptions {
  /** Problem to grade; defaults to the env's default problem. */
  problemId?: string;
}

export async function grade(opts: HarborGradeOptions = {}): Promise<void> {
  const verifierDir = process.env.HARBOR_VERIFIER_DIR || DEFAULT_VERIFIER_DIR;

  // Prepare a clean destination up front — but deliberately write no
  // reward.json here (see failure semantics above): if anything below
  // throws, harbor must find NO reward file and raise its exception.
  fs.mkdirSync(verifierDir, { recursive: true });
  try {
    fs.chmodSync(verifierDir, 0o700);
  } catch {
    /* best-effort, matches the old test.sh */
  }
  for (const entry of fs.readdirSync(verifierDir)) {
    fs.rmSync(path.join(verifierDir, entry), { recursive: true, force: true });
  }

  const ctx = await loadEnvCommandContext();
  if (!ctx) {
    // Harness crash, not an agent outcome: no reward file was written, so
    // the thrown error exits non-zero and harbor records a trial exception.
    throw new Error(
      "environment module failed to load — grading cannot run. No reward " +
        "file was written; harbor will raise RewardFileNotFoundError (trial " +
        "exception = infrastructure failure, not an agent score)."
    );
  }

  const problems = await ctx.env.listProblems();
  const problemId =
    opts.problemId || problems.find((p) => p.default)?.id || problems[0]?.id;
  if (!problemId) {
    throw new Error("No problem specified and no default problem found");
  }

  // Same code path as `env-orchestrator test` (skipExit: we own the exit
  // code). It writes the harbor rewards dict (flat {testId: score}) under
  // <logsDir>/<problemId>/harbor/ via env-base's writeHarborRewards. Test
  // failures do NOT throw — they come back as real 0 scores; only harness
  // crashes propagate as exceptions.
  const testResults = await handleTestCommand(ctx.env, /* skipExit */ true);

  const logsDir = getLogsDir();
  const harborDir = path.join(logsDir, problemId, "harbor");
  const rewardSrc = path.join(harborDir, "reward.json");
  if (!fs.existsSync(rewardSrc) || fs.statSync(rewardSrc).size === 0) {
    // Harness completed but never emitted rewards (e.g. emission failure
    // logged as non-fatal by the test command) — that is an infra failure.
    throw new Error(
      `test run finished but no harbor reward file was emitted at ${rewardSrc} — ` +
        "writing nothing to the verifier dir and exiting non-zero so harbor " +
        "records a trial exception instead of a fake agent score."
    );
  }

  // Aggregate "reward" key (Wave 1 step 4): reward.json keeps its flat
  // per-test keys AND gains one explicit composite — the SAME signed
  // weighted mean the platform records as the rollout score
  // (aggregateTestResults: skipped/errored excluded, weight-defaulted,
  // unclamped). Harbor's own tooling (viewer, analyzer, min_reward gating)
  // keys on rewards["reward"], and customers get the composite without
  // re-deriving our aggregation. Per-test keys and the telemetry-driven UI
  // breakdown are unchanged.
  const perTest: Record<string, number> = JSON.parse(
    fs.readFileSync(rewardSrc, "utf-8")
  );
  if (Object.prototype.hasOwnProperty.call(perTest, "reward")) {
    // A test literally named "reward" would collide with the composite key
    // and silently misreport — refuse loudly (no reward file written; the
    // trial records an infra exception, not a fake score).
    throw new Error(
      'a test id is literally "reward", which collides with the aggregate ' +
        "reward key in harbor's reward.json contract — rename the test."
    );
  }
  const { rolloutScore } = aggregateTestResults(testResults);
  fs.writeFileSync(
    path.join(verifierDir, "reward.json"),
    JSON.stringify({ reward: rolloutScore, ...perTest }, null, 2) + "\n"
  );
  copyIfExists(path.join(harborDir, "rewards-meta.json"), verifierDir);
  log(
    `harbor rewards -> ${path.join(verifierDir, "reward.json")} ` +
      `(aggregate reward ${rolloutScore})`
  );

  // Preserve the NATIVE test telemetry next to the harbor reward files.
  // reward.json is deliberately lossy (flat {testId: score}); the tests
  // JSONL carries per-test names, durations, output, and rubric rationale.
  // Harbor persists everything under /logs to the host trial dir, so
  // downstream consumers (Hyperfocal's platform converts trial dirs into
  // its rollout telemetry) get full test fidelity instead of bare scores.
  const testsDstDir = path.join(verifierDir, "tests");
  fs.mkdirSync(testsDstDir, { recursive: true });
  const testsSrcDir = path.join(logsDir, problemId, "tests");
  if (fs.existsSync(testsSrcDir)) {
    for (const entry of fs.readdirSync(testsSrcDir)) {
      if (entry.endsWith(".jsonl")) {
        copyIfExists(path.join(testsSrcDir, entry), testsDstDir);
      }
    }
  }
  copyIfExists(path.join(logsDir, problemId, "metadata.json"), testsDstDir);
  log(`native test telemetry -> ${testsDstDir}`);

  // Grading succeeded — the CLI wrapper exits 0 regardless of test scores
  // (agent failures are real rewards, not verifier errors).
}
