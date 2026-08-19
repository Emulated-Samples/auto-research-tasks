/**
 * svpgsbench Hyperfocal environment.
 *
 * The solver agent builds a from-scratch polygenic-scoring library (./fit,
 * ./predict, optional ./build.sh) in its workspace. runTests grades it with the
 * hidden Python grader (repo grader/) over the dataset corpus, and reports one
 * continuous-score TestResult per capability category. The grader supplies the
 * exact category coefficients for its mean/lower-tail headline, and Hyperfocal's
 * weighted aggregation therefore reproduces the hardened reward exactly.
 *
 * Implements EnvironmentDefinition from @hyperfocal/env-base.
 */

import {
  EnvironmentDefinition,
  Logger,
  TestResult,
  execute,
  loadProblemsFromDirectory,
} from "@hyperfocal/env-base";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as crypto from "crypto";
import { fileURLToPath } from "url";
import {
  AGGREGATION_SCHEMA_VERSION,
  AggregationContract,
  aggregationMatches,
  datasetRewardsMatchCategories,
  gradedStatus,
  MAX_DATASET_REWARD,
  MIN_DATASET_REWARD,
  masteryContractError,
  toPlatformScore,
} from "./aggregation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// environment/ holds problems.yaml; the repo root is one level up.
const ENV_DIR = path.join(__dirname, "..");
const REPO_ROOT = path.join(ENV_DIR, "..");
const WORKSPACE_PATH = path.join(REPO_ROOT, "workspace");
const PROBLEM_ID = "from_scratch_svpgs";
const PYTHON = "/opt/svpgs-venv/bin/python";
const CORPUS_DIR = path.join(REPO_ROOT, "corpus");
// The checked-in source credential lives under grader/, which is sealed from the
// solver account before the workspace is exposed. Setup materializes its decoded
// 32 bytes at this fixed, owner-only runtime path. This keeps corpus authentication
// deployable on Hyperfocal without accepting an environment-controlled fallback.
const CORPUS_KEY_SOURCE = path.join(REPO_ROOT, "grader", "corpus-key.hex");
const CORPUS_KEY_FILE = "/run/secrets/svpgsbench-corpus.key";
const PRIVATE_STATE_DIRS = [
  "corpus",
  "datagen",
  "grader",
  "reference",
  "tasks",
  "validation",
];
const INSTALL_SCRIPT = path.join(ENV_DIR, "scripts", "install-toolchains.sh");
const DEPLOYMENT_PREFLIGHT = path.join(REPO_ROOT, "grader", "preflight.py");
const RELEASE_LOCK_PATH = path.join(REPO_ROOT, "validation", "release_lock.json");
const PROVISION_MARKER = "/opt/svpgsbench/.provisioned-v3";
const INSTALL_TIMEOUT_MS = 60 * 60 * 1000;
const WRAPPER_GRADER_TIMEOUT_MS = 14_400 * 1000;

const problems = loadProblemsFromDirectory(ENV_DIR);

interface RewardDetail {
  schema_version: number;
  reward: number;
  // Headline score at/above which the benchmark counts as MASTERED. The grader
  // sends the active threshold (currently the frontier-ceiling anchor); if it ever
  // sends null the wrapper emits no mastery event rather than inventing one.
  mastery_threshold: number | null;
  // The pass event requires the bottom-tail aggregate to clear this too, so a
  // lopsided rollout (high headline, one broken capability) cannot master.
  mastery_tail_threshold: number | null;
  mastery_tail_value: number;
  score_bounds: {
    min_dataset_reward: number;
    max_dataset_reward: number;
  };
  additional_data: {
    per_category: Record<string, number>;
    aggregation: AggregationContract;
  };
  // Integrity fields the grader stamps onto a fully-complete reward file; the
  // wrapper refuses to derive a score unless all of them check out.
  complete?: boolean;
  grader_nonce?: string;
  manifest_sha256?: string;
  datasets?: unknown[];
  dataset_ids?: string[];
}

interface DatasetDetail {
  dataset: string;
  category: string;
  weight: number;
  status: DatasetStatus;
  reward: number;
  accuracy: number;
  raw_skill: number;
  perf: number;
  t_fit: number;
  t_predict: number;
}

interface ReleaseProvenance {
  release_status: "candidate" | "production";
  scientific_tuple_sha256: string;
  executable_tuple_sha256: string;
  manifest_sha256: string;
  scoring_contract_sha256: string;
  wrapper_sha256: string;
  candidate_lock_sha256: string;
}

function releaseProvenance(expectedManifestSha: string): ReleaseProvenance {
  const bytes = fs.readFileSync(RELEASE_LOCK_PATH);
  const lock = JSON.parse(bytes.toString("utf8")) as Record<string, unknown>;
  const digest = crypto.createHash("sha256").update(bytes).digest("hex");
  const releaseStatus = lock.release_status;
  const candidateLockSha = releaseStatus === "candidate"
    ? digest
    : lock.candidate_lock_sha256;
  const sha = /^[0-9a-f]{64}$/;
  if ((releaseStatus !== "candidate" && releaseStatus !== "production")
      || lock.manifest_sha256 !== expectedManifestSha
      || typeof lock.scientific_tuple_sha256 !== "string"
      || !sha.test(lock.scientific_tuple_sha256)
      || typeof lock.executable_tuple_sha256 !== "string"
      || !sha.test(lock.executable_tuple_sha256)
      || typeof lock.scoring_contract_sha256 !== "string"
      || !sha.test(lock.scoring_contract_sha256)
      || typeof lock.wrapper_sha256 !== "string"
      || !sha.test(lock.wrapper_sha256)
      || typeof candidateLockSha !== "string"
      || !sha.test(candidateLockSha)) {
    throw new Error("authenticated release lock provenance is malformed");
  }
  return {
    release_status: releaseStatus,
    scientific_tuple_sha256: lock.scientific_tuple_sha256,
    executable_tuple_sha256: lock.executable_tuple_sha256,
    manifest_sha256: expectedManifestSha,
    scoring_contract_sha256: lock.scoring_contract_sha256,
    wrapper_sha256: lock.wrapper_sha256,
    candidate_lock_sha256: candidateLockSha,
  };
}

// A dataset the submission actually executed to a valid, in-contract pred.csv.
// Any other status is a contract failure (crash, timeout, malformed or
// out-of-range output), which is reported separately from the science score.
const CONTRACT_OK_STATUS = "ok";
const DATASET_STATUSES = new Set([
  CONTRACT_OK_STATUS,
  "submission_reject",
  "no_executable",
  "fit_failed",
  "no_model",
  "predict_failed",
  "unsafe_pred",
  "bad_pred",
  "missing_col",
  "row_count",
  "nonfinite",
  "prob_range",
  "build_failed",
] as const);
type DatasetStatus = typeof DATASET_STATUSES extends Set<infer T> ? T : never;

interface CorpusDataset {
  path: string;
  category: string;
  weight: number;
}

function validDatasetDetail(
  detail: unknown,
  expected: CorpusDataset | undefined,
): detail is DatasetDetail {
  if (!expected || typeof detail !== "object" || detail === null) return false;
  const candidate = detail as Partial<DatasetDetail>;
  const expectedId = path.basename(expected.path);
  return candidate.dataset === expectedId
    && candidate.category === expected.category
    && Number.isFinite(candidate.weight)
    && Math.abs(candidate.weight as number - expected.weight) <= 1e-9
    && typeof candidate.status === "string"
    && DATASET_STATUSES.has(candidate.status as DatasetStatus)
    && Number.isFinite(candidate.reward)
    && (candidate.reward as number) >= MIN_DATASET_REWARD
    && (candidate.reward as number) <= MAX_DATASET_REWARD
    && Number.isFinite(candidate.raw_skill)
    && (candidate.raw_skill as number) >= MIN_DATASET_REWARD
    && (candidate.raw_skill as number) <= MAX_DATASET_REWARD
    && Number.isFinite(candidate.accuracy)
    && (candidate.accuracy as number) >= 0
    && Number.isFinite(candidate.perf)
    && (candidate.perf as number) >= 0
    && (candidate.perf as number) <= 1.0
    && Number.isFinite(candidate.t_fit)
    && (candidate.t_fit as number) >= 0
    && Number.isFinite(candidate.t_predict)
    && (candidate.t_predict as number) >= 0
    && (candidate.status === CONTRACT_OK_STATUS
      || ((candidate.reward as number) === MIN_DATASET_REWARD
        && (candidate.raw_skill as number) === MIN_DATASET_REWARD
        && (candidate.perf as number) === 0));
}

function categoryOutput(members: DatasetDetail[], score: number): string {
  return JSON.stringify({
    // DESCRIPTIVE, not a mastery verdict: a continuous score on the naive=0 /
    // reference=1 scale (negative = worse than the naive baseline), plus whether
    // every dataset in the category actually honoured the program contract.
    native_score: score,
    platform_score: toPlatformScore(score),
    contract_ok: members.every((detail) => detail.status === CONTRACT_OK_STATUS),
    datasets: members.map((detail) => ({
      dataset: detail.dataset,
      status: detail.status,
      reward: detail.reward,
      skill: detail.raw_skill,
      accuracy: detail.accuracy,
      performance: detail.perf,
      fit_seconds: detail.t_fit,
      predict_seconds: detail.t_predict,
    })),
  });
}


// PLATFORM-FACING reward transform. The trusted NATIVE reward is signed on the
// naive=0 / reference=1 scale, per-dataset winsorized to [MIN_DATASET_REWARD,
// MAX_DATASET_REWARD] = [-0.5, 1.5]; the headline shares that range. MEASURED
// (run_019f71e7): the platform ingests the emitted `score` and CLAMPS it to >= 0, so
// a below-naive native headline of -0.0509 was stored as 0 -- erasing the negative RL
// gradient (every below-naive model looked identical to naive). This maps the signed
// reward through the ONE monotone affine transform with reference=1 so the full
// ordering survives ingestion: invalid floor -> 0, naive -> 1/3, reference -> 1,
// and scientific headroom 1.5 -> 4/3 (not clamped). It is applied ONLY to emitted
// `score` fields; the native
// signed values stay in each record's `output`/`reward_detail`, and mastery / pass@k
// read the native headline, so the human-facing scale is unchanged.
// Put a tiny SCHEMA SAMPLE in the workspace, generated through the SAME
// datagen/materialize path as the graded corpus.
//
// Measured 2026-07-15 (run_019f6689): the workspace used to be empty, so the
// solver fabricated its own fixture -- a 61-byte formula.txt against the
// corpus's ~145 -- whose pgs(...) held no categorical annotation. Its code
// passed against its own fiction and then hit `float('snv')` on the real
// formula, scoring the invalid floor on all 45 datasets. A 45-dataset corpus,
// a 170 s fit budget and 15 capability categories measured ONE float() call.
//
// This is a SCHEMA sample, never a dev corpus: ~60 samples over ~25 variants,
// public files only (no truth), signal too weak to fit. It discloses no method,
// no prior, no LD structure and no answer -- the schema is already disclosed in
// the prompt AND in dgp.json's annotation_types, so this only makes it
// EXECUTABLE rather than described. See datagen/build_sample.py for the full
// rationale and the "do not grow this" contract.
async function provisionSchemaSample(): Promise<void> {
  await execute(
    `${PYTHON} ${JSON.stringify(path.join(REPO_ROOT, "datagen", "build_sample.py"))} ` +
      `${JSON.stringify(WORKSPACE_PATH)}`,
    {
      cwd: REPO_ROOT,
      timeout: 120_000,
      env: {
        PATH: "/opt/svpgs-venv/bin:/usr/local/bin:/usr/bin:/bin",
        PYTHONPATH: REPO_ROOT,
      },
    },
  );
  // Fail LOUDLY if the sample did not land. A silently absent sample returns the
  // env to the exact state that produced run_019f6689 -- the solver inventing a
  // fixture -- and that failure would look like a capability result.
  const formula = path.join(WORKSPACE_PATH, "sample_dataset", "public", "formula.txt");
  if (!fs.existsSync(formula)) {
    throw new Error(`schema sample missing after generation: ${formula}`);
  }
}

async function provisionToolchains(): Promise<void> {
  if (fs.existsSync(PROVISION_MARKER)) return;
  await execute(`bash ${JSON.stringify(INSTALL_SCRIPT)}`, { timeout: INSTALL_TIMEOUT_MS });
  if (!fs.existsSync(PROVISION_MARKER)) {
    throw new Error(`toolchain provisioner completed without ${PROVISION_MARKER}`);
  }
}

function provisionCorpusKey(): void {
  const source = fs.lstatSync(CORPUS_KEY_SOURCE);
  if (!source.isFile() || source.isSymbolicLink()) {
    throw new Error(`corpus key source must be a regular file: ${CORPUS_KEY_SOURCE}`);
  }
  const sourceBytes = fs.readFileSync(CORPUS_KEY_SOURCE, "utf8");
  if (!/^[0-9a-f]{64}\n$/.test(sourceBytes)) {
    throw new Error("corpus key source must contain exactly 64 lowercase hexadecimal characters and one LF");
  }
  const key = Buffer.from(sourceBytes.slice(0, -1), "hex");
  if (key.length !== 32) {
    throw new Error("decoded corpus key must contain exactly 32 bytes");
  }

  const directory = path.dirname(CORPUS_KEY_FILE);
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  fs.chmodSync(directory, 0o700);
  const temporary = path.join(
    directory,
    `.svpgsbench-corpus.${process.pid}.${crypto.randomBytes(8).toString("hex")}`,
  );
  try {
    fs.writeFileSync(temporary, key, { flag: "wx", mode: 0o600 });
    fs.renameSync(temporary, CORPUS_KEY_FILE);
    fs.chmodSync(CORPUS_KEY_FILE, 0o600);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
}

// Forward the platform sandbox-owner switch (packaging.image.env
// SVPGSBENCH_SANDBOX=external) into the preflight/grader subprocesses. They are
// launched with an EXPLICIT env dict, which would otherwise drop the variable so
// the runner takes the bwrap path -- exactly what cannot work inside harbor (no
// CAP_SYS_ADMIN) at BAKE (the deployment preflight) and at grade time. The image
// bakes the ENV before the setup RUN, so it is in scope here in both phases;
// native EC2 rollouts never set it and keep the full bwrap jail.
function sandboxSwitchEnv(): Record<string, string> {
  return process.env.SVPGSBENCH_SANDBOX
    ? { SVPGSBENCH_SANDBOX: process.env.SVPGSBENCH_SANDBOX }
    : {};
}

async function validateDeployment(): Promise<void> {
  await execute(
    `${PYTHON} ${JSON.stringify(DEPLOYMENT_PREFLIGHT)} `
      + `${JSON.stringify(CORPUS_DIR)} --key-file ${JSON.stringify(CORPUS_KEY_FILE)}`,
    {
      cwd: REPO_ROOT,
      timeout: 30 * 60 * 1000,
      env: {
        PATH: "/opt/svpgs-venv/bin:/usr/local/bin:/usr/bin:/bin",
        HOME: "/root",
        TMPDIR: "/tmp",
        LANG: "C.UTF-8",
        PYTHONPATH: REPO_ROOT,
        OMP_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        VECLIB_MAXIMUM_THREADS: "1",
        ...sandboxSwitchEnv(),
      },
    },
  );
}

function sealPrivateState(): void {
  for (const relative of PRIVATE_STATE_DIRS) {
    const target = path.join(REPO_ROOT, relative);
    const stat = fs.lstatSync(target);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new Error(`private state must be a real directory: ${target}`);
    }
    // One root-owned, non-traversable top-level directory is sufficient to make
    // every descendant inaccessible to the solver uid while leaving the trusted
    // root grader able to read it. Reapply before grading in case host setup
    // changed a mode after setupProblem.
    fs.chmodSync(target, 0o700);
  }
}

async function assertSubmissionCannotReadPrivateState(): Promise<void> {
  for (const relative of PRIVATE_STATE_DIRS) {
    const target = path.join(REPO_ROOT, relative);
    const command = [
      "/usr/bin/setpriv",
      "--reuid", "svpgsub",
      "--regid", "svpgsub",
      "--clear-groups",
      "--inh-caps=-all",
      "--no-new-privs",
      "/usr/bin/test", "!", "-r", target,
    ].map((part) => JSON.stringify(part)).join(" ");
    await execute(command, { timeout: 30_000 });
  }
}

class Environment implements EnvironmentDefinition {
  async listProblems() {
    return problems;
  }

  async setupProblem(problemId: string): Promise<void> {
    if (problemId !== PROBLEM_ID) {
      throw new Error(`unknown problem ${problemId}; expected ${PROBLEM_ID}`);
    }
    console.log(`🔧 svpgsbench setup: ${problemId}`);
    await provisionToolchains();
    provisionCorpusKey();
    // Seal BEFORE the deployment preflight. validateDeployment() runs
    // preflight_sandbox(), whose probe reads the corpus truth AS the svpgsub uid
    // and must not succeed. Under the bwrap jail the truth was hidden by the mount
    // namespace regardless of file modes, so sealing could follow. Under
    // SVPGSBENCH_SANDBOX=external the setpriv-only jail has no mount namespace, so
    // read-protection is the 0700 root seal alone — it must already be in place
    // when the probe runs, or the probe reads the still-world-readable truth and
    // fails closed (LEAK_FS). Sealing first is harmless to the bwrap path and to
    // the corpus validation (which runs as root and reads 0700 root-owned state
    // fine). The runTests/grade path already seals before its preflight.
    sealPrivateState();
    await validateDeployment();
    await assertSubmissionCannotReadPrivateState();
    // Fresh workspace each problem: the agent builds the library from scratch.
    fs.rmSync(WORKSPACE_PATH, { recursive: true, force: true });
    fs.mkdirSync(WORKSPACE_PATH, { recursive: true });
    await provisionSchemaSample();
    console.log("✅ setup complete");
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    if (problemId !== PROBLEM_ID) {
      throw new Error(`unknown problem ${problemId}; expected ${PROBLEM_ID}`);
    }
    await provisionToolchains();
    provisionCorpusKey();
    sealPrivateState();
    await assertSubmissionCannotReadPrivateState();
    const rewardDir = fs.mkdtempSync(path.join(os.tmpdir(), "svpgsbench-reward-"));

    logger.info(`Grading ${problemId} submission at ${WORKSPACE_PATH} over corpus ${CORPUS_DIR} ...`);
    try {
      return await this.gradeInto(rewardDir, logger);
    } finally {
      // Never leave per-run reward/feedback dirs behind in the host /tmp where a
      // later same-uid process could read another trial's per-dataset detail.
      fs.rmSync(rewardDir, { recursive: true, force: true });
    }
  }

  private async gradeInto(rewardDir: string, logger: Logger): Promise<TestResult[]> {
    const errored = (error: string): TestResult[] => [
      {
        id: "grade",
        name: "svpgsbench grade",
        status: "errored",
        duration: 0,
        score: 0,
        weight: 1,
        error,
      },
    ];

    // Expected corpus shape, read from the same manifest.json the grader hashes,
    // so an incomplete or forged reward file can be rejected.
    let expectedDatasets: CorpusDataset[];
    let expectedDatasetIds: string[];
    let expectedCategories: string[];
    let manifestSha: string;
    let release: ReleaseProvenance;
    try {
      const manifestBytes = fs.readFileSync(path.join(CORPUS_DIR, "manifest.json"));
      const manifest = JSON.parse(manifestBytes.toString("utf8"));
      if (!Array.isArray(manifest.datasets) || manifest.datasets.length === 0) {
        return errored("corpus manifest has no datasets");
      }
      expectedDatasets = manifest.datasets as CorpusDataset[];
      expectedDatasetIds = expectedDatasets.map((d) => path.basename(d.path));
      expectedCategories = [...new Set(
        expectedDatasets.map((d) => d.category),
      )].sort() as string[];
      manifestSha = crypto.createHash("sha256").update(manifestBytes).digest("hex");
      release = releaseProvenance(manifestSha);
    } catch (e) {
      return errored(`cannot read corpus manifest at ${CORPUS_DIR}: ${(e as Error).message}`);
    }

    // Per-run nonce the grader must echo into reward_detail.json: proves the file
    // was produced by THIS grader invocation, not pre-planted / forged.
    const nonce = crypto.randomBytes(16).toString("hex");

    let graderFailed = false;
    try {
      await execute(
        `${PYTHON} ${JSON.stringify(path.join(REPO_ROOT, "grader", "grade.py"))} ` +
          `${JSON.stringify(CORPUS_DIR)} ${JSON.stringify(WORKSPACE_PATH)} ` +
          `--key-file ${JSON.stringify(CORPUS_KEY_FILE)} ` +
          `--out ${JSON.stringify(rewardDir)}`,
        {
          cwd: REPO_ROOT,
          timeout: WRAPPER_GRADER_TIMEOUT_MS,
          env: {
            PATH: "/opt/svpgs-venv/bin:/usr/local/bin:/usr/bin:/bin",
            HOME: "/root",
            TMPDIR: "/tmp",
            LANG: "C.UTF-8",
            PYTHONPATH: REPO_ROOT,
            SVPGSBENCH_GRADER_NONCE: nonce,
            OMP_NUM_THREADS: "1",
            OPENBLAS_NUM_THREADS: "1",
            MKL_NUM_THREADS: "1",
            VECLIB_MAXIMUM_THREADS: "1",
            ...sandboxSwitchEnv(),
          },
        },
      );
    } catch (e) {
      // Nonzero exit / timeout / kill is a HARD failure: do NOT parse or accept
      // any reward file -- a killed or aborted grader must never yield a
      // (partial or forged) score.
      graderFailed = true;
      logger.error(`grader process failed (nonzero exit / timeout / kill): ${(e as Error).message}`);
    }
    if (graderFailed) {
      return errored("grader process failed (nonzero exit / timeout / kill); reward file not trusted");
    }

    const detailPath = path.join(rewardDir, "reward_detail.json");
    let detail: RewardDetail;
    try {
      detail = JSON.parse(fs.readFileSync(detailPath, "utf8"));
    } catch {
      return errored(`no reward_detail.json produced at ${detailPath}`);
    }

    // Integrity gate: accept a score ONLY from a fully-complete, nonce-matched,
    // corpus-matched reward file. This kills reward forgery (an attacker cannot
    // know the per-run nonce) and partial-output acceptance (an incomplete or
    // killed grade never sets complete=true / all datasets).
    const nDatasets = Array.isArray(detail.datasets) ? detail.datasets.length : -1;
    const nonceOk = detail.grader_nonce === nonce;
    const manifestOk = detail.manifest_sha256 === manifestSha;
    const idsOk = Array.isArray(detail.dataset_ids)
      && JSON.stringify(detail.dataset_ids) === JSON.stringify(expectedDatasetIds);
    if (detail.complete !== true || !nonceOk || !manifestOk
        || nDatasets !== expectedDatasetIds.length || !idsOk) {
      return errored(
        `reward_detail.json failed integrity check ` +
          `(complete=${detail.complete}, datasets=${nDatasets}/${expectedDatasetIds.length}, ` +
          `ids_ok=${idsOk}, nonce_ok=${nonceOk}, manifest_ok=${manifestOk})`,
      );
    }

    const perCat = detail.additional_data?.per_category;
    const aggregation = detail.additional_data?.aggregation;
    if (detail.schema_version !== AGGREGATION_SCHEMA_VERSION
        || !perCat || typeof perCat !== "object" || !aggregation) {
      return errored(
        `reward_detail.json does not satisfy aggregation schema ${AGGREGATION_SCHEMA_VERSION}`,
      );
    }
    const cats = Object.keys(perCat).sort();
    const categoriesOk = JSON.stringify(cats) === JSON.stringify(expectedCategories);
    const boundedScores = cats.every((cat) => Number.isFinite(perCat[cat])
      && perCat[cat] >= MIN_DATASET_REWARD
      && perCat[cat] <= MAX_DATASET_REWARD);
    const aggregationOk = boundedScores && aggregationMatches(perCat, aggregation);
    const derivedReward = aggregationOk ? aggregation.headline : Number.NaN;
    const rewardOk = Number.isFinite(detail.reward)
      && detail.reward >= MIN_DATASET_REWARD
      && detail.reward <= MAX_DATASET_REWARD
      && Math.abs(detail.reward - derivedReward) <= 1e-9;
    if (!categoriesOk || !boundedScores || !aggregationOk || !rewardOk) {
      return errored(
        `reward_detail.json category/reward integrity failed ` +
          `(categories_ok=${categoriesOk}, bounded=${boundedScores}, ` +
          `aggregation_ok=${aggregationOk}, reward_ok=${rewardOk})`,
      );
    }
    const datasetDetails = detail.datasets as unknown[];
    const datasetDetailsOk = datasetDetails.every(
      (dataset, index) => validDatasetDetail(dataset, expectedDatasets[index]),
    );
    if (!datasetDetailsOk) {
      return errored("reward_detail.json contains invalid per-dataset reporting fields");
    }
    const trustedDatasetDetails = datasetDetails as DatasetDetail[];
    const categoriesReproduce = datasetRewardsMatchCategories(
      trustedDatasetDetails,
      perCat,
    );
    if (!categoriesReproduce) {
      return errored("per-category rewards do not reproduce the authenticated dataset rewards");
    }
    if (detail.score_bounds?.min_dataset_reward !== MIN_DATASET_REWARD
        || detail.score_bounds?.max_dataset_reward !== MAX_DATASET_REWARD) {
      return errored("reward_detail.json score bounds disagree with the wrapper contract");
    }
    // A mastery threshold must be an explicit, audited number or an explicit
    // absence. Anything else (a missing field, a string, NaN) would leave the
    // wrapper guessing at what "mastered" means, so it is a hard failure.
    if (!Object.prototype.hasOwnProperty.call(detail, "mastery_threshold")
        || !Object.prototype.hasOwnProperty.call(detail, "mastery_tail_threshold")
        || !Object.prototype.hasOwnProperty.call(detail, "mastery_tail_value")) {
      return errored("reward_detail.json is missing explicit mastery fields");
    }
    const mastery = detail.mastery_threshold;
    if (mastery !== null && !Number.isFinite(mastery)) {
      return errored(`reward_detail.json carries an invalid mastery_threshold: ${mastery}`);
    }
    const masteryTail = detail.mastery_tail_threshold;
    const masteryTailValue = detail.mastery_tail_value;
    // The tail VALUE remains mandatory while the release is uncalibrated; only the
    // two decision thresholds become null. The pure helper is unit-tested directly.
    const masteryError = masteryContractError(
      mastery,
      masteryTail,
      masteryTailValue,
      aggregation,
    );
    if (masteryError !== null) {
      return errored(`reward_detail.json ${masteryError}`);
    }
    // The headline masters ONLY if the bottom-tail aggregate also clears its floor.
    // Encode this by handing gradedStatus an effective mastery of +Infinity (never
    // passes) when the tail is short, so a lopsided rollout cannot pass on headline.
    const headlineMastery = mastery === null ? null
      : ((masteryTailValue as number) >= (masteryTail as number)
          ? mastery : Number.POSITIVE_INFINITY);
    const contractOk = trustedDatasetDetails.every(
      (dataset) => dataset.status === CONTRACT_OK_STATUS,
    );
    logger.info(`reward=${detail.reward.toFixed(4)} categories=${JSON.stringify(perCat)}`);

    // These coefficients exactly decompose the grader's 60% category mean +
    // 40% bottom-tail mean, so Hyperfocal reproduces the trusted headline.
    const categoryResults = cats.map((cat) => {
      const score = perCat[cat];
      const members = trustedDatasetDetails.filter((dataset) => dataset.category === cat);
      const duration = Math.round(members.reduce(
        (milliseconds, dataset) => milliseconds
          + 1000 * (dataset.t_fit + dataset.t_predict),
        0,
      ));
      return {
        id: `category:${cat}`,
        name: `${cat} category`,
        description: `${cat}: native reward is naive=0/reference=1 in output; `
          + `TestResult score is its monotone non-negative platform transform`,
        // There is no calibrated per-category mastery bar. A valid continuous result
        // is therefore `partially_passed`; `failed` is reserved for a contract or
        // execution failure. Calling every non-crashing category "passed" made even a
        // native -0.5 scientific result look successful in status dashboards.
        status: gradedStatus(
          score,
          members.every((dataset) => dataset.status === CONTRACT_OK_STATUS),
          null,
        ),
        duration,
        // Platform-facing score is the monotone reference=1 transform (survives the
        // >=0 clamp and intentionally preserves >1 headroom); native stays in output.
        score: toPlatformScore(score),
        weight: aggregation.category_coefficients[cat],
        output: categoryOutput(members, score),
      } as TestResult;
    });

    // The ONE trusted scalar. The platform's top-level score is a weighted mean
    // over the results below; carrying the headline as its own result gives that
    // scalar an explicit home (a rollout previously reported an EMPTY top-level
    // score while only telemetry held the real reward). The category coefficients
    // sum to 1 and decompose the headline exactly, so adding this weight-1 record
    // leaves the weighted mean equal to the headline instead of skewing it.
    const headline = detail.reward;
    const results: TestResult[] = [
      {
        id: "benchmark",
        name: "svpgsbench headline reward",
        description:
          "Trusted native headline (in output): 0.6 x category mean + 0.4 x the mean "
          + "of the weakest fifth, with naive=0/reference=1. The TestResult score is "
          + "its monotone non-negative platform transform. "
          + (mastery === null
            ? "no calibrated mastery threshold exists, so this is not a mastery verdict."
            : `mastery = headline >= ${mastery} AND bottom-tail >= ${masteryTail}.`),
        status: gradedStatus(headline, contractOk, headlineMastery),
        duration: 0,
        // Platform reward = monotone reference=1 transform of the native headline
        // (the platform clamps <0 to 0). Native signed headline stays in `output`.
        score: toPlatformScore(headline),
        weight: 1,
        output: JSON.stringify({
          native_score: headline,
          platform_score: toPlatformScore(headline),
          integrity_ok: true,
          contract_ok: contractOk,
          mastery: headlineMastery === null
            ? null
            : contractOk && headline >= headlineMastery,
          mastery_threshold: mastery,
          mastery_tail_threshold: masteryTail,
          mastery_tail_value: masteryTailValue,
          per_category: perCat,
          ...release,
        }),
      } as TestResult,
      ...categoryResults,
    ];

    // Fail loudly rather than report a score the platform would aggregate into a
    // DIFFERENT number than the transformed trusted headline. Because toPlatformScore
    // is affine and the category coefficients sum to 1, the weighted mean of the
    // emitted (transformed) scores equals toPlatformScore(headline) exactly.
    const platformHeadline = toPlatformScore(headline);
    const weightTotal = results.reduce((sum, result) => sum + (result.weight ?? 1), 0);
    const weighted = results.reduce(
      (sum, result) => sum + (result.weight ?? 1) * (result.score ?? 0),
      0,
    ) / weightTotal;
    if (!Number.isFinite(weighted) || Math.abs(weighted - platformHeadline) > 1e-9) {
      return errored(
        `emitted scores do not reproduce the transformed trusted headline `
          + `(aggregate=${weighted}, platform_headline=${platformHeadline}, `
          + `native_reward_detail=${headline})`,
      );
    }
    return results;
  }
}

export default new Environment();
