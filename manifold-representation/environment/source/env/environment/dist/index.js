/** Hyperfocal adapter for manifold-bench's full nine-suite problem. */
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { loadProblemsFromDirectory } from "@hyperfocal/env-base";
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
// Reconstruction is deliberately weightless: an unmasked full-width linear
// projection reconstructs better than the learned reference does, so no level of
// it distinguishes a solution from a non-solution. It gates joint quality and the
// pass event instead of paying.
const CATEGORY_WEIGHTS = {
    reconstruction: 0.00,
    factor_recovery: 0.32,
    factor_discovery: 0.21,
    structural_coherence: 0.26,
    compression: 0.18,
    efficiency: 0.03,
};
function sameNumericRecord(left, right) {
    const keys = Object.keys(right).sort();
    return JSON.stringify(Object.keys(left).sort()) === JSON.stringify(keys)
        && keys.every((key) => left[key] === right[key]);
}
function isUnitInterval(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}
function hasCategoryKeys(value) {
    return typeof value === "object"
        && value !== null
        && !Array.isArray(value)
        && JSON.stringify(Object.keys(value).sort()) === JSON.stringify(CATEGORY_NAMES);
}
function weightedCategoryMean(categories) {
    return Object.entries(categories).reduce((sum, [name, score]) => sum + score * CATEGORY_WEIGHTS[name], 0);
}
const PROCESS_ENV = {
    PATH: "/opt/hyperfocal/manifold-bench/bin:/usr/local/bin:/usr/bin:/bin",
    HOME: "/root",
    TMPDIR: "/tmp",
    LANG: "C.UTF-8",
    LC_ALL: "C.UTF-8",
    PYTHONHASHSEED: "0",
    OMP_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
    OPENBLAS_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
    MKL_NUM_THREADS: String(Math.min(8, os.availableParallelism())),
    // HYPERFOCAL PATCH(sandbox-env-passthrough): this env is deliberately a
    // fixed, fully-enumerated dictionary so grading is hermetic — but that
    // scrubbing also dropped the sandbox-owner switch, and the switch must
    // reach the verifier to mean anything. Packaged harbor images bake
    // MANIFOLD_BENCH_SANDBOX=external (hyperfocal.yaml packaging.image.env)
    // into the CONTAINER env, and grader/runner.py reads it from its OWN
    // process env to select the setpriv-only uid jail. Without this
    // passthrough the packaged verify phase silently took the bwrap path,
    // which needs CAP_SYS_ADMIN that harbor never grants the task container,
    // so every packaged grade died in preflight (SandboxUnavailable,
    // grader_error, empty stderr, ~2s) regardless of submission content —
    // observed identically on check runs before and after the preflight
    // network fix, and reproduced locally by scrubbing the variable. Native
    // EC2 rollouts never set the variable, so this spread is a no-op there
    // and the bwrap jail keeps its loud failure.
    ...(process.env.MANIFOLD_BENCH_SANDBOX
        ? { MANIFOLD_BENCH_SANDBOX: process.env.MANIFOLD_BENCH_SANDBOX }
        : {}),
};
function loadProblems() {
    const loaded = loadProblemsFromDirectory(ENV_DIR);
    if (loaded.length === 0)
        throw new Error("problems.yaml must contain at least one problem");
    const ids = new Set();
    for (const problem of loaded) {
        if (!problem.id || !problem.prompt)
            throw new Error("every problem requires an id and prompt");
        if (ids.has(problem.id))
            throw new Error(`duplicate problem id: ${problem.id}`);
        ids.add(problem.id);
    }
    if (loaded.filter((problem) => problem.default === true).length !== 1) {
        throw new Error("problems.yaml must declare exactly one default problem");
    }
    return loaded;
}
const problems = loadProblems();
function requireProblem(problemId) {
    if (!problems.some((problem) => problem.id === problemId)) {
        throw new Error(`unknown manifold-bench problem: ${problemId}`);
    }
}
function appendCapped(chunks, chunk) {
    chunks.push(chunk);
    let bytes = chunks.reduce((total, item) => total + item.length, 0);
    while (bytes > OUTPUT_LIMIT && chunks.length > 1)
        bytes -= chunks.shift().length;
    if (bytes > OUTPUT_LIMIT)
        chunks[0] = chunks[0].subarray(bytes - OUTPUT_LIMIT);
}
async function runProcess(executable, args, timeoutMs, logger) {
    await new Promise((resolve, reject) => {
        const child = spawn(executable, args, {
            cwd: REPO_ROOT,
            env: PROCESS_ENV,
            detached: process.platform !== "win32",
            stdio: ["ignore", "pipe", "pipe"],
        });
        const stdout = [];
        const stderr = [];
        child.stdout.on("data", (chunk) => appendCapped(stdout, chunk));
        child.stderr.on("data", (chunk) => appendCapped(stderr, chunk));
        const timer = setTimeout(() => {
            if (child.pid && process.platform !== "win32")
                process.kill(-child.pid, "SIGKILL");
            else
                child.kill("SIGKILL");
        }, timeoutMs);
        child.once("error", (error) => {
            clearTimeout(timer);
            reject(error);
        });
        child.once("close", (code, signal) => {
            clearTimeout(timer);
            const out = Buffer.concat(stdout).toString("utf8");
            const err = Buffer.concat(stderr).toString("utf8");
            if (out && logger)
                logger.info(out.slice(-8_000));
            if (code === 0)
                resolve();
            else
                reject(new Error(`${executable} exited code=${code} signal=${signal ?? "none"}: ${err.slice(-8_000)}`));
        });
    });
}
async function provision() {
    await runProcess("/bin/bash", [PROVISION], INSTALL_TIMEOUT_MS);
    await runProcess(PYTHON, ["-c", "import numpy, scipy, torch"], 30_000);
}
function sealPrivateState() {
    assertSealedDirsPresent();
    for (const entry of fs.readdirSync(REPO_ROOT, { withFileTypes: true })) {
        const target = path.join(REPO_ROOT, entry.name);
        if (entry.isSymbolicLink())
            throw new Error(`top-level symlink is not sealable: ${entry.name}`);
        if (entry.name === "workspace" || entry.name === "packages") {
            if (!entry.isDirectory())
                throw new Error(`${entry.name} must be a directory`);
            fs.chmodSync(target, 0o755);
        }
        else {
            fs.chmodSync(target, entry.isDirectory() ? 0o700 : 0o600);
        }
    }
}
// The private scientific state the untrusted SOLVE phase must never read: the
// grader (specs.py carries n_factors and every generative parameter) and the
// committed targets/floors/private_bsf in calibration/ -- the grade-time truth
// grader/targets.py reads. These are sealed root-only before the agent phase
// begins. (The prompt-only reference/ and the retained rollout_analysis/ were
// TRIMMED out of the shipped tree in the 2026-07-20 customer-snapshot wave and
// removed from this list in the same change -- trimming a dir and editing the
// seal list must always travel together; see assertSealedDirsPresent.)
// Crucially this is an ALLOWLIST, not
// "everything but workspace": the total seal (sealPrivateState, used before
// grading in runTests) also chmods the compiled adapter runtime and
// hyperfocal.yaml to root-only, and the agent runs UNPRIVILEGED -- so applying
// it before the agent phase makes the agent's own runtime unreadable, the agent
// dies on EACCES, and the control plane terminalizes the rollout at setup_end
// as a spurious pass with no grade (the v16-v23 regression; v15
// sealed only in runTests, which is why it graded). packages/ is deliberately
// NOT in this list: it carries the env-orchestrator and env-base runtime the
// unprivileged agent phase executes, and the abandoned first cut of this fix
// sealed it -- recreating exactly the spurious-pass class it was written to
// close. environment/ holds only the public scoring weights
// (CATEGORY_WEIGHTS), never a generative secret, so it too stays readable
// during solve. runTests re-seals everything, as root, right before the
// verifier runs.
const SOLVE_SEALED_DIRS = ["grader", "calibration"];
// Every allowlisted private directory must EXIST before any seal runs. This is
// a deliberate tripwire, not a convenience check: trimming a sealed directory
// out of the shipped tree without editing SOLVE_SEALED_DIRS in the same change
// must fail loudly here, never silently narrow the seal. (The 2026-07-20 trim
// that removed reference/ and rollout_analysis/ edited the tree and this list
// together; a future trim of grader/ or calibration/ must do the same.) The
// guard also protects the seal PROOF: `test ! -r` on a missing path passes
// vacuously, so assertPrivateStateUnreadable alone could not tell "sealed"
// from "trimmed away". sealTreeRootOnly's lstatSync would throw ENOENT on the
// solve path, but the total seal (sealPrivateState, run before grading)
// iterates whatever is present and needs this explicit check to trip too.
function assertSealedDirsPresent() {
    for (const name of SOLVE_SEALED_DIRS) {
        const target = path.join(REPO_ROOT, name);
        if (!fs.existsSync(target)) {
            throw new Error(`required sealed directory is missing (tree trim and SOLVE_SEALED_DIRS must change together): ${target}`);
        }
    }
}
function sealTreeRootOnly(target) {
    const stat = fs.lstatSync(target);
    if (stat.isSymbolicLink()) {
        throw new Error(`private path is a symlink and cannot be sealed: ${target}`);
    }
    fs.chownSync(target, 0, 0);
    if (stat.isDirectory()) {
        fs.chmodSync(target, 0o700);
        for (const entry of fs.readdirSync(target))
            sealTreeRootOnly(path.join(target, entry));
    }
    else {
        fs.chmodSync(target, 0o600);
    }
}
function sealPrivateStateForSolve() {
    // assertSealedDirsPresent is the trim/seal-list coupling tripwire (see its
    // comment); the per-entry lstatSync in sealTreeRootOnly would also throw
    // ENOENT, but the explicit guard keeps the failure message diagnostic.
    assertSealedDirsPresent();
    for (const name of SOLVE_SEALED_DIRS) {
        sealTreeRootOnly(path.join(REPO_ROOT, name));
    }
}
async function assertPrivateStateUnreadable() {
    // Re-prove the seal AS an unprivileged uid, over exactly the allowlist that
    // was sealed -- deriving the probe list from SOLVE_SEALED_DIRS means it can
    // never drift from the seal itself. `test ! -r` on a MISSING path would pass
    // vacuously, which is why assertSealedDirsPresent runs before any seal.
    for (const privatePath of SOLVE_SEALED_DIRS.map((name) => path.join(REPO_ROOT, name))) {
        await runProcess("/usr/bin/setpriv", [
            "--reuid", "manifoldsub",
            "--regid", "manifoldsub",
            "--clear-groups",
            "--inh-caps=-all",
            "--no-new-privs",
            "/usr/bin/test", "!", "-r", privatePath,
        ], 30_000);
    }
}
function validateDetail(detail, problemId) {
    if (detail.schema_version !== "1.0"
        || detail.scoring_version !== "manifold-bench-v14"
        || detail.status !== "ok"
        || detail.problem_id !== problemId) {
        throw new Error("grader returned an unexpected reward contract");
    }
    if (!isUnitInterval(detail.reward) || detail.reward !== detail.score) {
        throw new Error("reward and score must be identical numbers in [0, 1]");
    }
    if (!detail.additional_data
        || !Number.isInteger(detail.additional_data.included_suites)
        || detail.additional_data.included_suites < 1) {
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
    if (!Object.values(categories).every(isUnitInterval)
        || !Object.values(rawCategories).every(isUnitInterval)) {
        throw new Error("reward contains a raw or breadth-adjusted category score outside [0, 1]");
    }
    const breadthFactor = detail.additional_data.breadth_factor;
    const lowerTailQuality = detail.additional_data.lower_tail_quality;
    if (!isUnitInterval(breadthFactor) || !isUnitInterval(lowerTailQuality)) {
        throw new Error("breadth factor and lower-tail quality must be finite numbers in [0, 1]");
    }
    // Recompute the breadth factor from the per-suite categories, mirroring
    // grader/scoring.py exactly, rather than checking the old `factor**2 == tail`
    // identity. That identity held only while `breadth = sqrt(tail)`; the floor
    // (`max(BREADTH_FLOOR, sqrt(tail))`, gated to 0 when no suite has any quality)
    // broke it, and the scalar `lower_tail_quality` alone cannot tell the gate
    // (factor 0, every suite dead) from the floor (factor 0.15, some suite alive)
    // because both can carry tail 0. Recomputing from suite_details reconciles the
    // whole breadth pipeline, which is strictly stronger than the identity was.
    const BREADTH_FLOOR = 0.15;
    const TAIL_FRACTION = 0.25;
    const suiteDetails = detail.additional_data.suite_details;
    if (!Array.isArray(suiteDetails) || suiteDetails.length === 0) {
        throw new Error("reward detail carries no suite_details to reconcile the breadth factor");
    }
    const qualities = suiteDetails.map((suite) => {
        const c = suite.categories;
        if (!c)
            throw new Error(`suite ${suite.suite} is missing categories for breadth reconciliation`);
        const core = Math.max(0, c.reconstruction) *
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
    const expectedFactor = Math.max(...qualities) <= 0 ? 0 : Math.max(BREADTH_FLOOR, Math.sqrt(recomputedTail));
    if (Math.abs(breadthFactor - expectedFactor) > REWARD_CONTRACT_TOLERANCE) {
        throw new Error("breadth factor is inconsistent with the per-suite lower-tail quality");
    }
    // The platform pays the reported category rows, so those rows must contain
    // the one and only application of the breadth factor. Keeping both maps in
    // the contract makes an omitted or duplicated penalty observable here rather
    // than silently changing the benchmark's reward surface.
    for (const name of CATEGORY_NAMES) {
        const expected = rawCategories[name] * breadthFactor;
        if (Math.abs(categories[name] - expected) > REWARD_CONTRACT_TOLERANCE) {
            throw new Error(`breadth-adjusted category mismatch for ${name}`);
        }
    }
    const subscore = new Map(detail.subscores.map((entry) => [entry.name, entry.score]));
    for (const [name, score] of Object.entries(categories)) {
        if (subscore.get(`category:${name}`) !== score)
            throw new Error(`subscore mismatch for ${name}`);
    }
    const weighted = weightedCategoryMean(categories);
    if (Math.abs(weighted - detail.reward) > REWARD_CONTRACT_TOLERANCE) {
        throw new Error("headline reward is not the weighted breadth-adjusted category score");
    }
    return categories;
}
function errored(error, duration) {
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
/**
 * Compact, one-line-per-suite audit of the private reward detail.
 *
 * Telemetry retains the final info record but may discard earlier adjacent info
 * records. Emit one bounded multiline record rather than one call per suite, or
 * the final generic success line can erase every useful audit line. Each suite
 * carries validity, the multiplicative integrity gate and its factors, and
 * candidate-vs-full-credit-target behavioral metrics. This is post-agent telemetry; the
 * agent never observes it, and it exposes aggregate scalars rather than private
 * factor truth. Each behavioral metric labels floor, candidate and target: a
 * category score is the fraction of that span crossed, and without both ends a
 * reader cannot tell a strong run from a run whose target was simply low. The
 * labels matter because candidates may lie below the floor or above the target;
 * inequality glyphs would assert an ordering that is not necessarily true.
 */
function auditLines(detail) {
    // No `?? []` here. This audit is the only forensic record of why a submission
    // scored what it scored, and a default would turn a missing field into an
    // empty audit -- indistinguishable from a run with nothing to report. Absence
    // of information must not read as information. An absent field is a fault, and
    // the caller's try/catch turns it into a loud errored result.
    const suites = detail.additional_data.suite_details;
    if (!Array.isArray(suites))
        throw new Error("reward detail carries no suite_details to audit");
    const f = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "na";
    const lines = [];
    for (const suite of suites) {
        if (!suite.valid) {
            // An invalid suite is the ONLY record of why a submission scored zero, so
            // truncating it to a tidy 100 characters throws away the entire diagnosis.
            // A Haiku rollout scored 0.000 on all nine suites and the retained log said
            // only `./fit exited 1: Traceback (most recent call last): File "/work/./fit",
            // line 19, in <module> output_f` -- cut off exactly before the KeyError that
            // named the cause. Keep enough to read the exception.
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
        const required = (record, key, label) => {
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
        lines.push(`suite ${suite.suite}[${suite.suite_category}] valid w=${f(suite.weight, 2)}`
            + ` gate=${f(integrity.factor)}(add=${f(integrity.additive_error)} sup=${f(integrity.support_agreement)} perm=${f(integrity.permutation_error)})`
            + ` recon[f=${f(required(floor, "reconstruction_r2", "floor"))} c=${f(required(candidate, "reconstruction_r2", "candidate"))} t=${f(required(target, "reconstruction_r2", "target"))}]`
            + ` recov[f=${f(required(floor, "contribution_r2", "floor"))} c=${f(required(candidate, "contribution_r2", "candidate"))} t=${f(required(target, "contribution_r2", "target"))}]`
            + ` disc[f=${f(required(floor, "adjusted_average_precision", "floor"))} c=${f(required(candidate, "adjusted_average_precision", "candidate"))} t=${f(required(target, "adjusted_average_precision", "target"))}]`
            + ` geom[f=${f(required(floor, "geometry_score", "floor"))} c=${f(required(candidate, "geometry_score", "candidate"))} t=${f(required(target, "geometry_score", "target"))}]`
            + ` cf[f=${f(required(floor, "counterfactual_r2", "floor"))} c=${f(required(candidate, "counterfactual_r2", "candidate"))} t=${f(required(target, "counterfactual_r2", "target"))}]`
            + ` youden[f=${f(required(floor, "support_youden", "floor"))} c=${f(required(candidate, "support_youden", "candidate"))} t=${f(required(target, "support_youden", "target"))}]`
            + ` assign[f=${f(required(floor, "assignment_coherence", "floor"))} c=${f(required(candidate, "assignment_coherence", "candidate"))} t=${f(required(target, "assignment_coherence", "target"))}]`
            + ` cat[rc=${f(category.reconstruction)} rv=${f(category.factor_recovery)} ds=${f(category.factor_discovery)} st=${f(category.structural_coherence)} cp=${f(category.compression)} ef=${f(category.efficiency)}]`
            + ` t=${f(suite.duration_seconds, 1)}s`);
    }
    return lines;
}
class Environment {
    async listProblems() {
        return problems;
    }
    async setupProblem(problemId) {
        requireProblem(problemId);
        await provision();
        // Seal the private scientific state before the solve workspace is handed to
        // the agent. Waiting until runTests() is too late: the solve phase itself
        // is when an untrusted submission can inspect world-readable grader and
        // calibration state. This is the ALLOWLIST
        // seal, not the total one -- the total seal (runTests) also locks the
        // agent's own unprivileged runtime and would orphan the agent at setup,
        // spurious-passing the rollout; see sealPrivateStateForSolve. Ordering
        // matters: provision() must run first because it creates the manifoldsub
        // account and installs setpriv, both of which assertPrivateStateUnreadable
        // needs to re-prove the seal AS an unprivileged uid.
        sealPrivateStateForSolve();
        await assertPrivateStateUnreadable();
        fs.rmSync(WORKSPACE, { recursive: true, force: true });
        fs.mkdirSync(SUBMISSION, { recursive: true });
        // A FORMAT STUB, not sample data. Structureless Gaussian noise at a shape the
        // prompt already discloses (`ambient_dim`, `train_rows` are in config.json),
        // so a solver can exercise its own entry points end to end -- the file names,
        // the config keys, the output schema, and the wall-clock cost of its fit at
        // the real shape -- and learn nothing whatsoever about the generative model.
        //
        // Both Haiku rollouts on this env scored exactly 0.000 for want of this. The
        // first read config["output_file"] (the key is `predictions_file`) and
        // crashed on every suite; the second blew the disclosed 75s fit budget on
        // every suite. Neither is a measurement of scientific ability, and neither
        // was discoverable: with no data of any shape, a submission cannot be run
        // even once before it is graded. Guessing an exact multi-array contract from
        // prose is variance uncorrelated with capability -- noise, not difficulty.
        //
        // There is no structure to leak. The noise has no latent objects, no sparse
        // support, no manifold geometry and no correlations; fitting it teaches a
        // solver nothing a careful reader of the prompt does not already have.
        await runProcess(PYTHON, [MAKE_STUB, WORKSPACE], 60_000);
    }
    async runTests(problemId, logger) {
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
            const detail = JSON.parse(fs.readFileSync(path.join(rewardDirectory, "reward_detail.json"), "utf8"));
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
            // Each category row reports its own continuous score and a status that
            // reflects that category's own aggregate threshold -- not the single
            // joint mastery event.  Categories without a threshold (efficiency) are
            // informational.  The joint benchmark pass/fail is exposed once, as its
            // own dedicated row, so a strong-but-not-passing run no longer renders as
            // though every scientific dimension independently failed.
            //
            // No `?? {}` default. That default fabricated a verdict: with the field
            // absent, every category silently read as not-passed, so the ONE real
            // fault -- a missing field -- rendered as every scientific dimension
            // independently failing. A fabricated verdict is worse than a crash,
            // because a crash is investigated and a verdict is believed.
            const categoryPass = detail.category_pass;
            if (typeof categoryPass !== "object"
                || categoryPass === null
                || Array.isArray(categoryPass)
                || JSON.stringify(Object.keys(categoryPass).sort())
                    !== JSON.stringify(THRESHOLDED_CATEGORY_NAMES)
                || !Object.values(categoryPass).every((value) => typeof value === "boolean")) {
                throw new Error("reward detail carries an invalid category_pass contract");
            }
            const categoryRows = Object.entries(categories)
                .sort(([left], [right]) => left.localeCompare(right))
                .map(([category, score]) => {
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
            const contractRow = {
                id: "contract",
                name: "program contract",
                description: contract.verdict === "contract_met"
                    ? "every suite ran and produced a well-formed decomposition"
                    : `${contract.total_suites - contract.valid_suites}/${contract.total_suites} suite(s) `
                        + `never produced a scorable result: ${contract.faults.join(", ")}. `
                        + "This is a contract failure, not a measurement of scientific quality.",
                status: contract.verdict === "contract_met" ? "passed" : "failed",
                duration,
                score: contract.valid_suites / Math.max(contract.total_suites, 1),
                weight: 0,
            };
            const masteryRow = {
                id: "mastery",
                name: "complete operating criteria",
                description: "the joint mastery event: aggregate quality, robust lower tail, and no abandoned suite",
                status: detail.passed ? "passed" : "failed",
                duration,
                score: detail.passed ? 1 : 0,
                weight: 0,
            };
            return [contractRow, masteryRow, ...categoryRows];
        }
        catch (error) {
            logger.error(error instanceof Error ? error.message : String(error));
            return errored(error, Date.now() - started);
        }
        finally {
            fs.rmSync(HARNESS_CONFIG, { force: true });
            fs.rmSync(rewardDirectory, { recursive: true, force: true });
        }
    }
}
export default new Environment();
//# sourceMappingURL=index.js.map