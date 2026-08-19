import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** environment/ root (parent of dist/, where this file compiles to dist/paths.js). */
export const ENV_DIR = path.resolve(__dirname, "..");

/**
 * Hidden grader-owned pytest suite. NEVER present in the agent's workspace —
 * it is a sibling of dist/ under environment/, absent during the agent session
 * and run just-in-time at grade time (verifier_design.md §2, §7).
 */
export const BEHAVIORAL_TESTS_DIR = path.join(ENV_DIR, "behavioral-tests");

/**
 * The agent's workspace. The orchestrator sets WORKSPACE_PATH in-process before
 * calling runTests (env-orchestrator main.ts); fall back to the sibling
 * workspace/ for standalone local runs.
 */
export function workspacePath(): string {
  return process.env.WORKSPACE_PATH || path.join(ENV_DIR, "workspace");
}

/** environment/requirements-test.txt — the pinned grade-time dependency closure. */
export const REQUIREMENTS_TEST = path.join(ENV_DIR, "requirements-test.txt");

/**
 * environment/requirements-solve.txt — the SLIM agent-side closure (numpy + CPU
 * torch). Installed at setup (image-build) time into the agent venv so the prompt's
 * `load_oracle()` dev loop actually runs; the solve phase has no internet.
 */
export const REQUIREMENTS_SOLVE = path.join(ENV_DIR, "requirements-solve.txt");

/**
 * Agent-facing virtualenv, provisioned at setup time so the agent can execute its
 * own `design()` against a local `load_oracle()` during the solve phase. It lives
 * INSIDE workspace/ (not under environment/, which the image build locks o-rwx) so
 * the Dockerfile's `chown -R agent workspace` hands the agent full ownership. It is
 * gitignored — never committed, so the problem..gold diff stays design_task.py-only.
 */
export function agentVenvDir(): string {
  return path.join(workspacePath(), ".venv");
}

/**
 * Dedicated grade virtualenv. We do NOT install into the ambient /usr/bin/python3:
 * the run image's distro python has rpm-managed packages (requests/urllib3/idna)
 * with no RECORD file, so pip cannot upgrade them and the install aborts. An
 * isolated venv sidesteps that entirely. Created at setupProblem time.
 */
export const GRADE_VENV_DIR = path.join(ENV_DIR, ".grade-venv");

function venvPython(): string {
  return path.join(GRADE_VENV_DIR, "bin", "python");
}

/**
 * Python interpreter for the grade suite. The run image ships NO prebuilt venv
 * (the packaged repo is only the env/ contents — a hardcoded /hyperfocal/.venv
 * does not exist at rollout and crashed the launched run). Resolution order:
 *   1. GRADER_PYTHON — explicit override for a custom / pre-baked image.
 *   2. the dedicated grade venv, once setupProblem has built it.
 *   3. ambient python3 — only before the venv exists (e.g. the first dep probe).
 */
export function graderPython(): string {
  if (process.env.GRADER_PYTHON) return process.env.GRADER_PYTHON;
  const vpy = venvPython();
  return fs.existsSync(vpy) ? vpy : "python3";
}
