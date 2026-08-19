import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { BatchTest, TestResult } from "@hyperfocal/env-base";
import { runSubprocessTests } from "@hyperfocal/env-base";
import { BEHAVIORAL_TESTS_DIR, graderPython, workspacePath } from "../paths.js";

// Generous ceiling: design() runs greedy search against a CPU oracle; a full 3-track grade is a few
// minutes. Raised well above observed wall-clock so a slow host never spuriously errors the suite.
const GRADE_TIMEOUT_MS = 20 * 60 * 1000;

const ID_PREFIX = "pytest-";

/**
 * Held-out-activity tests emit a continuous held-out z; env-base's runSubprocessTests derives score
 * from JUnit pass/fail only (1.0 / 0.0), so the test emits its z via seqdesign_fixtures.emit_metric()
 * and we map it here to a SKILL SCORE:
 *
 *   score = clamp((z - floor) / (ref - floor), 0, 1)
 *
 * with `floor` = the naive anchor (random-start raw greedy) and `ref` = floor + SATURATION*(reference
 * - floor), where `reference` is the shipped reference design() at the same budget. SATURATION>1 puts
 * the reference at score 1/SATURATION (headroom above it), never guessed — fit to the archetype ladder.
 *
 * CALIBRATION: floor/reference are the probe-measured anchors at the grade budget; re-confirm through
 * the harness archetype ladder before freezing (F-12: imputebench scored a non-imputer 0.38 by
 * fitting a metric before measuring the ladder).
 */
interface ContinuousMetric {
  test: string;
  floor: number;
  ref: number;
}

const SATURATION = 1 / 0.70;

/** floor = naive anchor; the reference lands at score 1/SATURATION. */
function skillAnchors(naive: number, reference: number): { floor: number; ref: number } {
  return { floor: naive, ref: naive + SATURATION * (reference - naive) };
}

// DeepSTARR oracle, Dev track, budget 40k, 3-WAY ensemble-min (evalA,B maxpool-CNN + evalC dilated-
// residual). Re-measured archetype ladder (held-out z; F-12: measure before fitting):
//   overfitter(hack) -0.61 · random 0.09 · natural 1.37 · careless greedy 2.79 ·
//   competent greedy 5.13 · gold design() reference 18.57.
// floor = 0.5 (the random/hack "no better than chance" level) so the reward-hack AND random both map
// to 0 — the anti-hack property is carried by the ensemble-min (overfitter -0.6), NOT a high floor.
// The prior floor of 3.0 sat ABOVE careless greedy, crushing all honest non-population effort to ~0
// (rollout QA 2026-07-17); at floor 0.5 competent greedy earns ~0.19 and the greedy->population climb
// has real gradient. reference = the actual shipped gold design() (18.57, verified through this grade
// path).
//
// RE-ANCHORED 2026-08-19 (grelu-v2). The previous SATURATION=1.27 capped the scale at z=23.4 and 3 of
// 7 valid Opus-5 rollouts clipped at exactly 1.0 (observed z up to 27.2), so the top of the scale had
// no gradient. Those rollouts were run while the model weights sat in the agent's workspace, i.e. with
// unbounded offline search; the session meter now bounds total queries, which lowers what is reachable.
// SATURATION = 1/0.70 places the gold reference at exactly 0.70 on this test (overall grade ~0.72 once
// the invalid-input block, weight 0.05, is folded in) and saturates at z = 0.5 + (18.57-0.5)/0.70 =
// 26.3, i.e. ~42% of held-out z above gold still buys score. Hk dropped (its 2-way signal was a
// shared-CNN artifact — see fixtures).
const CONTINUOUS_METRICS: Record<string, ContinuousMetric> = {
  heldout_Dev: { test: "pytest-test_heldout_Dev", ...skillAnchors(0.5, 18.57) },
};

/** Parse the JSONL metrics side-channel. Absent/garbled lines are ignored, not fatal. */
function readMetrics(file: string, logger?: { info: (m: string) => void }): Map<string, number> {
  const out = new Map<string, number>();
  let raw: string;
  try {
    raw = fs.readFileSync(file, "utf8");
  } catch {
    return out;
  }
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const { name, value } = JSON.parse(line) as { name: string; value: number };
      if (typeof name === "string" && Number.isFinite(value)) out.set(name, value);
    } catch {
      logger?.info(`[pytest-suite] ignoring malformed metric line: ${line.slice(0, 120)}`);
    }
  }
  return out;
}

/** Replace the binary score of each emitting test with its continuous skill score. */
function applyContinuousScores(
  results: TestResult[],
  metrics: Map<string, number>,
  logger?: { info: (m: string) => void },
): TestResult[] {
  const byTest = new Map<string, { metric: string; cfg: ContinuousMetric }>();
  for (const [metric, cfg] of Object.entries(CONTINUOUS_METRICS)) byTest.set(cfg.test, { metric, cfg });

  return results.map((r) => {
    const entry = byTest.get(r.id);
    if (!entry || r.status !== "passed") return r;
    const value = metrics.get(entry.metric);
    if (value === undefined) {
      logger?.info(`[pytest-suite] ${r.id} passed but emitted no ${entry.metric}; leaving binary`);
      return r;
    }
    const { floor, ref } = entry.cfg;
    const score = Math.max(0, Math.min(1, (value - floor) / (ref - floor)));
    logger?.info(`[pytest-suite] ${entry.metric}=${value.toFixed(4)} -> score ${score.toFixed(4)}`);
    return { ...r, score };
  });
}

/** Prefix each JUnit-derived test id with "pytest-" so it maps to the registry / CONTINUOUS_METRICS. */
function withPrefix(results: TestResult[]): TestResult[] {
  return results.map((r) => {
    if (r.status === "errored" || r.id.startsWith(ID_PREFIX)) return r;
    const id = `${ID_PREFIX}${r.id}`;
    return { ...r, id, name: r.name.startsWith(ID_PREFIX) ? r.name : id };
  });
}

/**
 * Runs the hidden grader-owned held-out evaluation harness (environment/behavioral-tests/) host-side
 * against the agent's workspace varseq, imported directly. Each pytest test becomes one grader item.
 * The suite is absent during the agent session and staged only at grade time.
 */
export const pytestSuite: BatchTest = {
  id: "seq-design-harness",
  name: "seq-design held-out evaluation harness",
  description:
    "Host-side pytest over environment/behavioral-tests/ against the agent's workspace varseq; one grader item per test.",
  runBatch: async (logger): Promise<TestResult[]> => {
    const ws = workspacePath();
    const py = graderPython();
    const resultsFile = path.join(os.tmpdir(), `seqdesign-grade-${process.pid}.xml`);
    const metricsFile = path.join(os.tmpdir(), `seqdesign-metrics-${process.pid}.jsonl`);
    try {
      fs.rmSync(metricsFile, { force: true });
    } catch {
      /* best effort */
    }

    const command = [
      py, "-m", "pytest", BEHAVIORAL_TESTS_DIR,
      "-p", "no:cacheprovider", "-q", "--no-header",
      `--junitxml=${resultsFile}`,
    ].join(" ");

    const results = await runSubprocessTests(command, {
      timeout: GRADE_TIMEOUT_MS,
      cwd: BEHAVIORAL_TESTS_DIR,
      env: {
        ...process.env,
        WORKSPACE_PATH: ws,
        PYTHONPATH: path.join(ws, "src"),
        SEQDESIGN_METRICS_FILE: metricsFile,
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
      },
      logger,
      resultsFile,
    });

    return applyContinuousScores(withPrefix(results), readMetrics(metricsFile, logger), logger);
  },
};
