
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { EnvironmentDefinition, Logger, Problem, TestResult } from "@hyperfocal/env-base";
import { loadProblemsFromDirectory } from "@hyperfocal/env-base";

interface ContractStatus {
  verdict: "contract_met" | "contract_invalid" | "contract_partial";
  valid_suites: number;
  total_suites: number;
  faults: string[];
  fault_counts: Record<string, number>;
}

interface RewardDetail {
  schema_version: string;
  scoring_version: string;
  status: string;
  contract_status: ContractStatus;
  problem_id: string;
  score: number;
  reward: number;
  passed: boolean;
  category_pass: Record<string, boolean>;
  category_weights: Record<string, number>;
  subscores: Array<{ name: string; score: number }>;
  additional_data: {
    included_suites: number;
    category_scores: Record<string, number>;
    raw_category_scores: Record<string, number>;
    category_weights: Record<string, number>;
    lower_tail_quality: number;
    breadth_factor: number;
    suite_details: unknown[];
  };
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ENV_DIR = path.join(__dirname, "..");
const REPO_ROOT = path.join(ENV_DIR, "..");
const WORKSPACE = path.join(REPO_ROOT, "workspace");
const SUBMISSION = path.join(WORKSPACE, "submission");
const HARNESS_CONFIG = path.join(REPO_ROOT, "grader", "harness.json");
const PROVISION = path.join(ENV_DIR, "scripts", "provision.sh");
const MAKE_STUB = path.join(ENV_DIR, "scripts", "make_format_stub.py");
const PYTHON = "/opt/hyperfocal/manifold-bench/bin/python";
const OUTPUT_LIMIT = 1024 * 1024;
const INSTALL_TIMEOUT_MS = 30 * 60 * 1000;
const GRADER_TIMEOUT_MS = 25 * 60 * 1000;
const REWARD_CONTRACT_TOLERANCE = 1e-9;
const CATEGORY_NAMES = [
  "reconstruction",
  "factor_recovery",
  "factor_discovery",
  "structural_coherence",
  "compression",
  "efficiency",
].sort();
const THRESHOLDED_CATEGORY_NAMES = [
  "reconstruction",
  "factor_recovery",
  "factor_discovery",
  "structural_coherence",
  "compression",
].sort();
const CATEGORY_WEIGHTS: Record<string, number> = {
  reconstruction: 0.00,
  factor_recovery: 0.32,
  factor_discovery: 0.21,
  structural_coherence: 0.26,
  compression: 0.18,
  efficiency: 0.03,
};

function sameNumericRecord(left: Record<string, number>, right: Record<string, number>): boolean {
  const keys = Object.keys(right).sort();
  return JSON.stringify(Object.keys(left).sort()) === JSON.stringify(keys)
    && keys.every((key) => left[key] === right[key]);
}

function isUnitInterval(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function hasCategoryKeys(value: unknown): value is Record<string, number> {
  return typeof value === "object"
    && value !== null
    && !Array.isArray(value)
    && JSON.stringify(Object.keys(value).sort()) === JSON.stringify(CATEGORY_NAMES);
}

function weightedCategoryMean(categories: Record<string, number>): number {
  return Object.entries(categories).reduce(
    (sum, [name, score]) => sum + score * CATEGORY_WEIGHTS[name],
    0,
  );
}

const PROCESS_ENV: NodeJS.ProcessEnv = {
  PATH: "/opt/hyperfocal/manifold-bench/bin:/usr/local/bin:/usr/bin:/bin",
  HOME: "/root",
  TMPDIR: "/tmp",
  LANG: "C.UTF-8",
  LC_ALL: "C.UTF-8",
  PYTHONHASHSEED: "0",
  OMP_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
  OPENBLAS_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
  MKL_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
  ...(process.env.MANIFOLD_BENCH_SANDBOX
    ? { MANIFOLD_BENCH_SANDBOX: process.env.MANIFOLD_BENCH_SANDBOX }
    : {}),
};

function loadProblems(): Problem[] {
  const loaded = loadProblemsFromDirectory(ENV_DIR);
  if (loaded.length === 0) throw new Error("problems.yaml must contain at least one problem");
  const ids = new Set<string>();
  for (const problem of loaded) {
    if (!problem.id || !problem.prompt) throw new Error("every problem requires an id and prompt");
    if (ids.has(problem.id)) throw new Error(`duplicate problem id: ${problem.id}`);
    ids.add(problem.id);
  }
  if (loaded.filter((problem) => problem.default === true).length !== 1) {
    throw new Error("problems.yaml must declare exactly one default problem");
  }
  return loaded;
}

const problems = loadProblems();

function requireProblem(problemId: string): void {
  if (!problems.some((problem) => problem.id === problemId)) {
    throw new Error(`unknown manifold-bench problem: ${problemId}`);
  }
}

function appendCapped(chunks: Buffer[], chunk: Buffer): void {
  chunks.push(chunk);
  let bytes = chunks.reduce((total, item) => total + item.length, 0);
  while (bytes > OUTPUT_LIMIT && chunks.length > 1) bytes -= chunks.shift()!.length;
  if (bytes > OUTPUT_LIMIT) chunks[0] = chunks[0].subarray(bytes - OUTPUT_LIMIT);
}

async function runProcess(
  executable: string,
  args: string[],
  timeoutMs: number,
  logger?: Logger,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd: REPO_ROOT,
      env: PROCESS_ENV,
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => appendCapped(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => appendCapped(stderr, chunk));
    const timer = setTimeout(() => {
      if (child.pid && process.platform !== "win32") process.kill(-child.pid, "SIGKILL");
      else child.kill("SIGKILL");
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code, signal) => {
      clearTimeout(timer);
      const out = Buffer.concat(stdout).toString("utf8");
      const err = Buffer.concat(stderr).toString("utf8");
      if (out && logger) logger.info(out.slice(-8_000));
      if (code === 0) resolve();
      else reject(new Error(`${executable} exited code=${code} signal=${signal ?? "none"}: ${err.slice(-8_000)}`));
    });
  });
}

async function provision(): Promise<void> {
  await runProcess("/bin/bash", [PROVISION], INSTALL_TIMEOUT_MS);
  await runProcess(PYTHON, ["-c", "import numpy, scipy, torch"], 30_000);
}

function sealPrivateState(): void {
  assertSealedDirsPresent();
  for (const entry of fs.readdirSync(REPO_ROOT, { withFileTypes: true })) {
    const target = path.join(REPO_ROOT, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`top-level symlink is not sealable: ${entry.name}`);
    if (entry.name === "workspace" || entry.name === "packages") {
      if (!entry.isDirectory()) throw new Error(`${entry.name} must be a directory`);
      fs.chmodSync(target, 0o755);
    } else {
      fs.chmodSync(target, entry.isDirectory() ? 0o700 : 0o600);
    }
  }
}

const SOLVE_SEALED_DIRS = ["grader", "calibration"];

function assertSealedDirsPresent(): void {
  for (const name of SOLVE_SEALED_DIRS) {
    const target = path.join(REPO_ROOT, name);
    if (!fs.existsSync(target)) {
      throw new Error(
        `required sealed directory is missing (tree trim and SOLVE_SEALED_DIRS must change together): ${target}`,
      );
    }
  }
}

function sealTreeRootOnly(target: string): void {
  const stat = fs.lstatSync(target);
  if (stat.isSymbolicLink()) {
    throw new Error(`private path is a symlink and cannot be sealed: ${target}`);
  }
  fs.chownSync(target, 0, 0);
  if (stat.isDirectory()) {
    fs.chmodSync(target, 0o700);
    for (const entry of fs.readdirSync(target)) sealTreeRootOnly(path.join(target, entry));
  } else {
    fs.chmodSync(target, 0o600);
  }
}

function sealPrivateStateForSolve(): void {
  assertSealedDirsPresent();
  for (const name of SOLVE_SEALED_DIRS) {
    sealTreeRootOnly(path.join(REPO_ROOT, name));
  }
}

async function assertPrivateStateUnreadable(): Promise<void> {
  for (const privatePath of SOLVE_SEALED_DIRS.map((name) => path.join(REPO_ROOT, name))) {
    await runProcess(
      "/usr/bin/setpriv",
      [
        "--reuid", "manifoldsub",
        "--regid", "manifoldsub",
        "--clear-groups",
        "--inh-caps=-all",
        "--no-new-privs",
        "/usr/bin/test", "!", "-r", privatePath,
      ],
      30_000,
    );
  }
}

function validateDetail(detail: RewardDetail, problemId: string): Record<string, number> {
  if (
    detail.schema_version !== "1.0"
    || detail.scoring_version !== "manifold-bench-v14"
    || detail.status !== "ok"
    || detail.problem_id !== problemId
  ) {
    throw new Error("grader returned an unexpected reward contract");
  }
  if (!isUnitInterval(detail.reward) || detail.reward !== detail.score) {
    throw new Error("reward and score must be identical numbers in [0, 1]");
  }
  if (
    !detail.additional_data
    || !Number.isInteger(detail.additional_data.included_suites)
    || detail.additional_data.included_suites < 1
  ) {
    throw new Error("reward detail contains no evaluated suites");
  }
  const categories = detail.additional_data.category_scores;
  const rawCategories = detail.additional_data.raw_category_scores;
  if (!hasCategoryKeys(categories) || !hasCategoryKeys(rawCategories)) {
    throw new Error("raw and breadth-adjusted reward categories must match the manifold-bench contract");
  }
  if (!sameNumericRecord(detail.category_weights, CATEGORY_WEIGHTS)
      || !sameNumericRecord(detail.additional_data.category_weights, CATEGORY_WEIGHTS)) {
    throw new Error("reward category weights do not match the manifold-bench contract");
  }
  if (
    !Object.values(categories).every(isUnitInterval)
    || !Object.values(rawCategories).every(isUnitInterval)
  ) {
    throw new Error("reward contains a raw or breadth-adjusted category score outside [0, 1]");
  }
  const breadthFactor = detail.additional_data.breadth_factor;
  const lowerTailQuality = detail.additional_data.lower_tail_quality;
  if (!isUnitInterval(breadthFactor) || !isUnitInterval(lowerTailQuality)) {
    throw new Error("breadth factor and lower-tail quality must be finite numbers in [0, 1]");
  }
  const BREADTH_FLOOR = 0.15;
  const TAIL_FRACTION = 0.25;
  const suiteDetails = detail.additional_data.suite_details as SuiteDetail[];
  if (!Array.isArray(suiteDetails) || suiteDetails.length === 0) {
    throw new Error("reward detail carries no suite_details to reconcile the breadth factor");
  }
  const qualities = suiteDetails.map((suite) => {
    const c = suite.categories;
    if (!c) throw new Error(`suite ${suite.suite} is missing categories for breadth reconciliation`);
    const core =
      Math.max(0, c.reconstruction) *
      Math.max(0, c.factor_recovery) *
      Math.max(0, c.factor_discovery) *
      Math.max(0, c.structural_coherence);
    return core ** 0.25;
  });
  const ordered = [...qualities].sort((a, b) => a - b);
  const tailCount = Math.max(1, Math.ceil(TAIL_FRACTION * ordered.length));
  const recomputedTail = ordered.slice(0, tailCount).reduce((s, v) => s + v, 0) / tailCount;
  if (Math.abs(recomputedTail - lowerTailQuality) > REWARD_CONTRACT_TOLERANCE) {
    throw new Error("lower-tail quality does not match the per-suite qualities");
  }
  const expectedFactor =
    Math.max(...qualities) <= 0 ? 0 : Math.max(BREADTH_FLOOR, Math.sqrt(recomputedTail));
  if (Math.abs(breadthFactor - expectedFactor) > REWARD_CONTRACT_TOLERANCE) {
    throw new Error("breadth factor is inconsistent with the per-suite lower-tail quality");
  }
  for (const name of CATEGORY_NAMES) {
    const expected = rawCategories[name] * breadthFactor;
    if (Math.abs(categories[name] - expected) > REWARD_CONTRACT_TOLERANCE) {
      throw new Error(`breadth-adjusted category mismatch for ${name}`);
    }
  }
  const subscore = new Map(detail.subscores.map((entry) => [entry.name, entry.score]));
  for (const [name, score] of Object.entries(categories)) {
    if (subscore.get(`category:${name}`) !== score) throw new Error(`subscore mismatch for ${name}`);
  }
  const weighted = weightedCategoryMean(categories);
  if (Math.abs(weighted - detail.reward) > REWARD_CONTRACT_TOLERANCE) {
    throw new Error("headline reward is not the weighted breadth-adjusted category score");
  }
  return categories;
}

function errored(error: unknown, duration: number): TestResult[] {
  return [{
    id: "outcome",
    name: "representation system outcome",
    status: "errored",
    duration,
    score: 0,
    weight: 1,
    error: error instanceof Error ? error.message : String(error),
  }];
}

interface SuiteDetail {
  suite: string;
  suite_category: string;
  weight: number;
  valid: boolean;
  error_type?: string;
  contract_fault?: string;
  error?: string;
  categories?: Record<string, number>;
  integrity?: {
    additive_error: number;
    support_agreement: number;
    permutation_error: number;
    factor: number;
  };
  candidate?: Record<string, number>;
  floor?: Record<string, number>;
  target?: Record<string, number>;
  duration_seconds?: number;
}

function auditLines(detail: RewardDetail): string[] {
  const suites = detail.additional_data.suite_details as SuiteDetail[];
  if (!Array.isArray(suites)) throw new Error("reward detail carries no suite_details to audit");
  const f = (value: number | undefined, digits = 3): string =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "na";
  const lines: string[] = [];
  for (const suite of suites) {
    if (!suite.valid) {
      const message = (suite.error ?? "").replace(/\s+/g, " ").slice(0, 600);
      lines.push(`suite ${suite.suite}[${suite.suite_category}] INVALID fault=${suite.contract_fault ?? "unclassified"} ${suite.error_type ?? "error"}: ${message}`);
      continue;
    }
    const integrity = suite.integrity;
    const candidate = suite.candidate;
    const floor = suite.floor;
    const target = suite.target;
    const category = suite.categories;
    if (!integrity || !candidate || !floor || !target || !category) {
      throw new Error(`valid suite ${suite.suite} is missing required audit evidence`);
    }
    const required = (record: Record<string, number>, key: string, label: string): number => {
      const value = record[key];
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`valid suite ${suite.suite} has invalid ${label}.${key}`);
      }
      return value;
    };
    for (const [key, value] of Object.entries(integrity)) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`valid suite ${suite.suite} has invalid integrity.${key}`);
      }
    }
    if (!hasCategoryKeys(category) || !Object.values(category).every(isUnitInterval)) {
      throw new Error(`valid suite ${suite.suite} has invalid category evidence`);
    }
    lines.push(
      `suite ${suite.suite}[${suite.suite_category}] valid w=${f(suite.weight, 2)}`
      + ` gate=${f(integrity.factor)}(add=${f(integrity.additive_error)} sup=${f(integrity.support_agreement)} perm=${f(integrity.permutation_error)})`
      + ` recon[f=${f(required(floor, "reconstruction_r2", "floor"))} c=${f(required(candidate, "reconstruction_r2", "candidate"))} t=${f(required(target, "reconstruction_r2", "target"))}]`
      + ` recov[f=${f(required(floor, "contribution_r2", "floor"))} c=${f(required(candidate, "contribution_r2", "candidate"))} t=${f(required(target, "contribution_r2", "target"))}]`
      + ` disc[f=${f(required(floor, "adjusted_average_precision", "floor"))} c=${f(required(candidate, "adjusted_average_precision", "candidate"))} t=${f(required(target, "adjusted_average_precision", "target"))}]`
      + ` geom[f=${f(required(floor, "geometry_score", "floor"))} c=${f(required(candidate, "geometry_score", "candidate"))} t=${f(required(target, "geometry_score", "target"))}]`
      + ` cf[f=${f(required(floor, "counterfactual_r2", "floor"))} c=${f(required(candidate, "counterfactual_r2", "candidate"))} t=${f(required(target, "counterfactual_r2", "target"))}]`
      + ` youden[f=${f(required(floor, "support_youden", "floor"))} c=${f(required(candidate, "support_youden", "candidate"))} t=${f(required(target, "support_youden", "target"))}]`
      + ` assign[f=${f(required(floor, "assignment_coherence", "floor"))} c=${f(required(candidate, "assignment_coherence", "candidate"))} t=${f(required(target, "assignment_coherence", "target"))}]`
      + ` cat[rc=${f(category.reconstruction)} rv=${f(category.factor_recovery)} ds=${f(category.factor_discovery)} st=${f(category.structural_coherence)} cp=${f(category.compression)} ef=${f(category.efficiency)}]`
      + ` t=${f(suite.duration_seconds, 1)}s`,
    );
  }
  return lines;
}

class Environment implements EnvironmentDefinition {
  async listProblems(): Promise<Problem[]> {
    return problems;
  }

  async setupProblem(problemId: string): Promise<void> {
    requireProblem(problemId);
    await provision();
    sealPrivateStateForSolve();
    await assertPrivateStateUnreadable();
    fs.rmSync(WORKSPACE, { recursive: true, force: true });
    fs.mkdirSync(SUBMISSION, { recursive: true });
    await runProcess(PYTHON, [MAKE_STUB, WORKSPACE], 60_000);
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    requireProblem(problemId);
    const started = Date.now();
    const rewardDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "manifold-bench-reward-"));
    try {
      await provision();
      sealPrivateState();
      await assertPrivateStateUnreadable();
      fs.writeFileSync(HARNESS_CONFIG, JSON.stringify({
        submission: SUBMISSION,
        problem_id: problemId,
        reward_json: path.join(rewardDirectory, "reward.json"),
        sandboxed: true,
      }) + "\n", { mode: 0o600 });
      await runProcess(PYTHON, [path.join(REPO_ROOT, "grader", "verifier.py")], GRADER_TIMEOUT_MS, logger);
      const detail = JSON.parse(
        fs.readFileSync(path.join(rewardDirectory, "reward_detail.json"), "utf8"),
      ) as RewardDetail;
      const categories = validateDetail(detail, problemId);
      const duration = Math.round((Date.now() - started) / CATEGORY_NAMES.length);
      const contract = detail.contract_status;
      const rawReward = weightedCategoryMean(detail.additional_data.raw_category_scores);
      logger.info([
        ...auditLines(detail),
        `breadth: lower_tail=${detail.additional_data.lower_tail_quality.toFixed(5)}`
          + ` factor=${detail.additional_data.breadth_factor.toFixed(5)}`
          + ` raw_reward=${rawReward.toFixed(5)}`
          + ` adjusted_reward=${detail.reward.toFixed(5)}`,
        `contract: ${contract.verdict} valid=${contract.valid_suites}/${contract.total_suites}`
          + (contract.faults.length ? ` faults=${JSON.stringify(contract.fault_counts)}` : ""),
        `suite audit: reward=${detail.reward.toFixed(5)} passed=${detail.passed}`
          + ` suites=${detail.additional_data.included_suites}`,
        "representation system checks completed",
      ].join("\n"));
      const categoryPass = detail.category_pass;
      if (
        typeof categoryPass !== "object"
        || categoryPass === null
        || Array.isArray(categoryPass)
        || JSON.stringify(Object.keys(categoryPass).sort())
          !== JSON.stringify(THRESHOLDED_CATEGORY_NAMES)
        || !Object.values(categoryPass).every((value) => typeof value === "boolean")
      ) {
        throw new Error("reward detail carries an invalid category_pass contract");
      }
      const categoryRows = Object.entries(categories)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([category, score]): TestResult => {
          const hasThreshold = category in categoryPass;
          return {
            id: `category:${category}`,
            name: `${category} outcome`,
            description: hasThreshold
              ? `${category} continuous score; row status reflects this category's own mastery threshold`
              : `${category} continuous score; informational, no binary threshold`,
            status: hasThreshold ? (categoryPass[category] ? "passed" : "failed") : "passed",
            duration,
            score,
            weight: CATEGORY_WEIGHTS[category],
          };
        });
      const contractRow: TestResult = {
        id: "contract",
        name: "program contract",
        description:
          contract.verdict === "contract_met"
            ? "every suite ran and produced a well-formed decomposition"
            : `${contract.total_suites - contract.valid_suites}/${contract.total_suites} suite(s) `
              + `never produced a scorable result: ${contract.faults.join(", ")}. `
              + "This is a contract failure, not a measurement of scientific quality.",
        status: contract.verdict === "contract_met" ? "passed" : "failed",
        duration,
        score: contract.valid_suites / Math.max(contract.total_suites, 1),
        weight: 0,
      };
      const masteryRow: TestResult = {
        id: "mastery",
        name: "complete operating criteria",
        description: "the joint mastery event: aggregate quality, robust lower tail, and no abandoned suite",
        status: detail.passed ? "passed" : "failed",
        duration,
        score: detail.passed ? 1 : 0,
        weight: 0,
      };
      return [contractRow, masteryRow, ...categoryRows];
    } catch (error) {
      logger.error(error instanceof Error ? error.message : String(error));
      return errored(error, Date.now() - started);
    } finally {
      fs.rmSync(HARNESS_CONFIG, { force: true });
      fs.rmSync(rewardDirectory, { recursive: true, force: true });
    }
  }
}

export default new Environment();
