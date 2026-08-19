/**
 * Anti-cheat catastrophe gates for seq-design (verifier §4.1, §6).
 *
 * Each gate is a weight-0 SimpleTest whose id starts with `cat-`. A CLEAN gate passes; a FIRED gate
 * fails, and applyCatastropheMultiplier (grader/catastrophe.ts) then scales every other item's score
 * by CATASTROPHE_FACTOR. A gate that cannot run returns `errored`, so an infra hiccup never triggers a
 * spurious catastrophe.
 *
 * Gates:
 *  - cat-design-under-workspace  : the graded design() resolves to a workspace file.
 *  - cat-no-pypi-reshadow        : no site-packages varseq shadows the workspace copy.
 *  - cat-evaluator-not-in-workspace : no answer-key artifact (evaluator/z-norm/natural set) leaked
 *                                     into the workspace.
 *  - cat-determinism             : design() is reproducible given its args (verifier §7). A
 *                                   non-deterministic submission has no well-defined grade.
 */
import type { SimpleTest, SimpleTestResult } from "@hyperfocal/env-base";
import { executeWithExitCode } from "@hyperfocal/env-base";
import * as path from "path";
import { BEHAVIORAL_TESTS_DIR, graderPython, workspacePath } from "../paths.js";

const CAT_WEIGHT = 0;
const FIXTURES_DIR = path.join(BEHAVIORAL_TESTS_DIR, "fixtures");

/** Run a python snippet (base64-wrapped to dodge shell quoting) with the workspace on path. */
async function runPy(code: string, extraPath: string[] = []) {
  const ws = workspacePath();
  const b64 = Buffer.from(code, "utf-8").toString("base64");
  const cmd = `${graderPython()} -c "import base64;exec(base64.b64decode('${b64}').decode())"`;
  const pythonPath = [path.join(ws, "src"), ...extraPath].join(path.delimiter);
  return executeWithExitCode(cmd, {
    env: { ...process.env, WORKSPACE_PATH: ws, PYTHONPATH: pythonPath,
           OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1" },
    silent: true,
    timeout: 120_000,
  });
}

/** Map a python check that prints `CLEAN` / `FIRE <reason>` into a gate result. */
function fromPyOutput(output: string, exitCode: number): SimpleTestResult {
  const text = (output || "").trim();
  const line = text.split("\n").filter(Boolean).pop() ?? "";
  if (line.startsWith("CLEAN")) return { success: true };
  if (line.startsWith("FIRE")) return { success: false, error: line.slice("FIRE".length).trim() };
  return {
    success: false,
    errored: true,
    error: `gate check inconclusive (exit ${exitCode}): ${text.slice(-300)}`,
  };
}

const designUnderWorkspace: SimpleTest = {
  id: "cat-design-under-workspace",
  name: "design() resolves to the agent workspace",
  description:
    "The graded design must be defined in a file under <workspace>/src, not a site-packages / PyPI copy.",
  weight: CAT_WEIGHT,
  run: async (): Promise<SimpleTestResult> => {
    const code = `
import inspect, os
ws_src = os.path.realpath(os.path.join(os.environ["WORKSPACE_PATH"], "src"))
import varseq
from varseq.design_task import design
f1 = os.path.realpath(varseq.__file__)
f2 = os.path.realpath(inspect.getfile(design))
print("CLEAN" if f1.startswith(ws_src) and f2.startswith(ws_src)
      else "FIRE varseq=%s design=%s not under %s" % (f1, f2, ws_src))
`;
    const r = await runPy(code);
    return fromPyOutput(r.output, r.exitCode);
  },
};

const noPypiReshadow: SimpleTest = {
  id: "cat-no-pypi-reshadow",
  name: "No site-packages varseq shadow",
  description:
    "No real varseq/__init__.py exists in site-packages outside the workspace. An editable .pth pointing at the workspace is fine.",
  weight: CAT_WEIGHT,
  run: async (): Promise<SimpleTestResult> => {
    const code = `
import os, site
ws_src = os.path.realpath(os.path.join(os.environ["WORKSPACE_PATH"], "src"))
dirs = list(getattr(site, "getsitepackages", lambda: [])())
try:
    dirs.append(site.getusersitepackages())
except Exception:
    pass
hits = [os.path.realpath(os.path.join(d, "varseq", "__init__.py")) for d in dirs
        if os.path.exists(os.path.join(d, "varseq", "__init__.py"))
        and not os.path.realpath(os.path.join(d, "varseq", "__init__.py")).startswith(ws_src)]
print("FIRE site-packages varseq shadow: %s" % hits if hits else "CLEAN")
`;
    const r = await runPy(code);
    return fromPyOutput(r.output, r.exitCode);
  },
};

const evaluatorNotInWorkspace: SimpleTest = {
  id: "cat-evaluator-not-in-workspace",
  name: "No answer-key artifact in the workspace",
  description:
    "The held-out evaluators / z-norm stats / natural-peak set (the grade's answer key) must never appear under the agent workspace.",
  weight: CAT_WEIGHT,
  run: async (): Promise<SimpleTestResult> => {
    const ws = workspacePath();
    // Distinctive answer-key filenames — the grader-owned fixtures, not the agent-given oracle.
    const patterns = ["evaluatorA.pt", "evaluatorB.pt", "evaluatorC.pt", "znorm_stats.json",
                      "natural_peaks.npy", "seqdesign_fixtures"];
    const grepExpr = patterns.map((p) => `-e ${JSON.stringify(p)}`).join(" ");
    // Skip the agent venv and node_modules: multi-thousand-file dependency trees the agent
    // never authors and which cannot contain grader-owned answer-key artifacts (those never
    // ship in the image) — scanning them only slows the gate.
    const excludes = "--exclude-dir=.git --exclude-dir=.venv --exclude-dir=node_modules";
    const cmd = `grep -rIl ${excludes} ${grepExpr} ${JSON.stringify(ws)} 2>/dev/null || true`;
    const r = await executeWithExitCode(cmd, { silent: true, timeout: 30_000 });
    const hits = (r.output || "").split("\n").map((s) => s.trim()).filter(Boolean);
    if (hits.length > 0) {
      return { success: false, error: `answer-key markers found in workspace: ${hits.slice(0, 5).join(", ")}` };
    }
    return { success: true };
  },
};

const determinism: SimpleTest = {
  id: "cat-determinism",
  name: "design() is deterministic given its arguments",
  description:
    "Two identical design() calls (fixed cheap config) must return byte-identical sequences. A non-deterministic submission has no reproducible grade (verifier §7).",
  weight: CAT_WEIGHT,
  run: async (): Promise<SimpleTestResult> => {
    const code = `
from varseq.design_task import design
import seqdesign_fixtures as F
args = (F.REGIMES["Dev"]["track"], 6000, 249, 2, 0)
d1 = design(F.build_oracle(6000), *args)
d2 = design(F.build_oracle(6000), *args)
print("CLEAN" if list(d1) == list(d2) else "FIRE design() is non-deterministic across identical calls")
`;
    const r = await runPy(code, [FIXTURES_DIR]);
    return fromPyOutput(r.output, r.exitCode);
  },
};

export const antiCheatTests: SimpleTest[] = [
  designUnderWorkspace,
  noPypiReshadow,
  evaluatorNotInWorkspace,
  determinism,
];
