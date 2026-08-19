import * as fs from "fs";
import * as path from "path";
import type { Logger } from "@hyperfocal/env-base";
import { executeWithExitCode } from "@hyperfocal/env-base";
import {
  GRADE_VENV_DIR,
  REQUIREMENTS_TEST,
  REQUIREMENTS_SOLVE,
  agentVenvDir,
  graderPython,
  workspacePath,
} from "../paths.js";

/**
 * Grade-time import probe. If these resolve under the grade interpreter, the
 * pinned closure is already present and we skip the (slow) install. varseq itself
 * is imported from PYTHONPATH=workspace/src, not pip — deliberately absent here.
 */
const DEP_PROBE = [
  "torch",
  "numpy",
  "pandas",
  "anndata",
  "pytorch_lightning",
  "statsmodels",
  "scipy",
  "pyfaidx",
  "pytest",
];

/**
 * Prepare the environment for grading.
 *
 * The run image ships NO prebuilt venv — the packaged repo is only the env/
 * contents, so the previously hardcoded /hyperfocal/.venv does not exist at
 * rollout (it crashed the launched run). We self-provision at setup time.
 *
 * We install into a DEDICATED venv rather than the ambient /usr/bin/python3:
 * the distro python carries rpm-managed packages (requests/urllib3/idna) with no
 * RECORD file, so pip cannot upgrade them and the install aborts. An isolated
 * venv avoids that collision. Install uses full dependency resolution against the
 * pinned closure in requirements-test.txt — a --no-deps trim to the "imported" set
 * was tried and broke (sklearn is a transitive-only Stage-D import). GRADER_PYTHON
 * overrides the whole mechanism for a pre-baked image.
 */
export async function setupProblem(
  problemId: string | undefined,
  logger: Logger
): Promise<void> {
  const ws = workspacePath();
  const varseqInit = path.join(ws, "src", "varseq", "__init__.py");
  if (!fs.existsSync(varseqInit)) {
    throw new Error(
      `workspace varseq not found at ${varseqInit} — cannot grade problem '${problemId ?? "default"}'`
    );
  }
  logger.info(`Setup: workspace varseq at ${ws}/src/varseq`);

  await ensurePythonDeps(logger);
  await provisionAgentVenv(ws, logger);

  logger.info("Setup OK");
}

/**
 * Provision the AGENT's solve-time interpreter.
 *
 * The prompt routes the agent to `varseq.oracle.load_oracle()` as its development
 * anchor — design a sequence, score it locally, see what transfers, refine. But the
 * run image ships no agent runtime: the system python has no numpy/torch, the solve
 * phase has no internet, and the grade venv is locked under environment/ (o-rwx at
 * image build). Without this, the agent can never execute its own design() and ships
 * blind (rollout QA 2026-07-17: score capped at the floor for exactly this reason).
 *
 * We build a SLIM venv (numpy + CPU torch, requirements-solve.txt) at setup time —
 * i.e. at image build, where internet is available — inside workspace/, which the
 * Dockerfile later `chown`s to the agent. varseq is put on its path via a .pth so
 * `import varseq` picks up the agent's own edits. Best-effort: a failure here must
 * never break grading (which uses its own .grade-venv), so we log and continue.
 */
async function provisionAgentVenv(ws: string, logger: Logger): Promise<void> {
  const venvDir = agentVenvDir();
  const py = path.join(venvDir, "bin", "python");
  try {
    if (fs.existsSync(py)) {
      logger.info(`Agent venv already present at ${venvDir}`);
      return;
    }
    if (!fs.existsSync(REQUIREMENTS_SOLVE)) {
      logger.info(`No ${REQUIREMENTS_SOLVE}; skipping agent venv provisioning`);
      return;
    }
    logger.info(`Provisioning agent solve-time venv at ${venvDir}...`);
    const created = await executeWithExitCode(`python3 -m venv ${shellQuote(venvDir)}`, {
      timeout: 120_000,
    });
    if (!created.success || !fs.existsSync(py)) {
      logger.info(`Agent venv creation failed (non-fatal): ${created.output}`);
      return;
    }
    await ensurePipAvailable(py, logger);
    const install = await executeWithExitCode(
      `${py} -m pip install --disable-pip-version-check -r ${shellQuote(REQUIREMENTS_SOLVE)}`,
      { timeout: 1_800_000 }
    );
    if (!install.success) {
      logger.info(`Agent dep install failed (non-fatal): ${install.output}`);
      return;
    }
    // Put the workspace varseq on the agent interpreter's path (editable-style),
    // so `import varseq` resolves to the code under test and tracks the agent's edits.
    const sp = await executeWithExitCode(
      `${py} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`,
      { silent: true }
    );
    const purelib = (sp.output || "").trim().split("\n").pop() ?? "";
    if (purelib && fs.existsSync(purelib)) {
      fs.writeFileSync(path.join(purelib, "varseq_workspace.pth"), `${path.join(ws, "src")}\n`);
    }
    const probe = await executeWithExitCode(
      `${py} -c ${shellQuote("import numpy, torch, varseq.oracle")}`,
      { silent: true }
    );
    if (probe.success) {
      logger.info("Agent solve-time venv ready (numpy + torch + varseq importable)");
    } else {
      logger.info(`Agent venv provisioned but probe failed (non-fatal): ${probe.output}`);
    }
  } catch (err) {
    logger.info(`Agent venv provisioning error (non-fatal): ${String(err)}`);
  }
}

/** Ensure the grade interpreter exists and has the pinned closure importable. */
async function ensurePythonDeps(logger: Logger): Promise<void> {
  // 1. If the current grade interpreter already imports the closure, we're done.
  if (await probeOk(graderPython())) {
    logger.info("Grade-time Python dependencies available");
    return;
  }

  if (!fs.existsSync(REQUIREMENTS_TEST)) {
    throw new Error(
      `Python deps missing and ${REQUIREMENTS_TEST} not found — cannot provision the grade environment`
    );
  }

  // 2. Pick the interpreter to install into. An explicit GRADER_PYTHON is honored
  //    as-is; otherwise we build an isolated venv to dodge the distro rpm packages.
  let installPy: string;
  if (process.env.GRADER_PYTHON) {
    installPy = process.env.GRADER_PYTHON;
  } else {
    installPy = await ensureVenv(logger);
  }

  // 3. Install the pinned closure with full dependency resolution (into the clean
  //    venv, so no rpm-managed distro package blocks the install).
  logger.info("Installing grade-time Python dependencies (this may take a few minutes)...");
  await ensurePipAvailable(installPy, logger);
  const install = await executeWithExitCode(
    `${installPy} -m pip install --disable-pip-version-check -r ${shellQuote(REQUIREMENTS_TEST)}`,
    { timeout: 1_800_000 }
  );
  if (!install.success) {
    throw new Error(`Failed to install grade-time Python dependencies: ${install.output}`);
  }

  if (!(await probeOk(graderPython()))) {
    throw new Error(
      "Grade-time dependencies still unimportable after install (closure incomplete?)"
    );
  }
  logger.info("Grade-time Python dependencies installed");
}

/** True if every DEP_PROBE module imports under `py`. */
async function probeOk(py: string): Promise<boolean> {
  const probe = DEP_PROBE.map((m) => `import ${m}`).join("; ");
  const res = await executeWithExitCode(`${py} -c ${shellQuote(probe)}`, {
    silent: true,
  });
  return res.success;
}

/** Create (idempotently) the isolated grade venv and return its interpreter. */
async function ensureVenv(logger: Logger): Promise<string> {
  const vpy = path.join(GRADE_VENV_DIR, "bin", "python");
  if (fs.existsSync(vpy)) return vpy;

  logger.info(`Creating isolated grade venv at ${GRADE_VENV_DIR}...`);
  const created = await executeWithExitCode(
    `python3 -m venv ${shellQuote(GRADE_VENV_DIR)}`,
    { timeout: 120_000 }
  );
  if (!created.success || !fs.existsSync(vpy)) {
    throw new Error(`Failed to create grade venv: ${created.output}`);
  }
  return vpy;
}

/** Bootstrap pip via ensurepip if the interpreter has none. */
async function ensurePipAvailable(py: string, logger: Logger): Promise<void> {
  const hasPip = await executeWithExitCode(`${py} -m pip --version`, {
    silent: true,
  });
  if (hasPip.success) return;

  logger.info("Bootstrapping pip with ensurepip...");
  const boot = await executeWithExitCode(`${py} -m ensurepip --upgrade`, {
    timeout: 120_000,
  });
  if (!boot.success) {
    throw new Error(`Failed to bootstrap pip: ${boot.output}`);
  }
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}
