/**
 * ar-cherimoya Environment
 *
 * Implements EnvironmentDefinition interface from @hyperfocal/env-base
 *
 * CURRENT STATE: scaffolding for the `gpu-diagnostics` problem only — a
 * measurement run, not a training task. The real problem (reimplement the Cheri
 * block's Triton kernels from a pure-PyTorch reference, graded on accuracy +
 * speed) is blocked on gold's self-contradicting precision contract.
 * See env-docs/02-gold-state-defects.md section 1 and env-docs/03-task-design.md.
 */

import {
  EnvironmentDefinition,
  Logger,
  TestResult,
  SimpleTest,
  runSimpleTests,
  // executeWithExitCode, not execute: `execute` THROWS on non-zero exit and
  // returns only stdout (env-base execute.ts:60-75). A diagnostic that fails
  // is exactly when we need its output, so we want the non-throwing variant.
  executeWithExitCode,
  loadProblemsFromDirectory,
} from "@hyperfocal/env-base";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Problems loaded from YAML file (located at environment/problems.yaml)
// We use ".." because this file runs from either src/ or dist/
const problems = loadProblemsFromDirectory(path.join(__dirname, ".."));

// Workspace path - configured via env-orchestrator
const WORKSPACE_PATH = process.env.WORKSPACE_PATH || "/root/hyperfocal/workspace";

// Same ".." trick as the problems loader: resolves from src/ or dist/.
const ASSETS_PATH = path.join(__dirname, "..", "assets");

const MINUTES = 60_000;

/**
 * cheri.py needs ONLY torch + triton (cheri.py:34-46). We deliberately do NOT
 * `pip install -e .`: the package's __init__ drags in modisco, tangermeme,
 * macs3, bam2bw and bpnet-lite (pyproject.toml:20-34), which are heavy and
 * compile-happy and can sink a sandbox build. bench_kernels.py loads cheri.py
 * directly via importlib to sidestep the package __init__ entirely.
 *
 * pytest is here so we can run the existing suite and see which of gold's own
 * tests actually fail on a GPU — the headline question.
 */
const PIP_DEPS = "torch triton pytest numpy";

/**
 * Grader deps live in a dedicated venv, NOT system python3.
 *
 * The previous bootstrap (ensurepip / dnf python3-pip, then
 * `python3 -m pip install --break-system-packages`) is NOT portable to the
 * packaged GPU image and errored both 2026-07-20 redraws at the deps step:
 *   - kernel  → `python3 -m pip` → "No module named pip" (rc=1): no pip at all.
 *   - training → bootstrap found a pip, but that pip PREDATES
 *     `--break-system-packages` → "no such option: --break-system-packages"
 *     (rc=2).
 * Both left torch uninstalled and crashed the grader at `import torch`.
 *
 * A venv sidesteps both: `python3 -m venv` seeds a fresh, self-owned pip from
 * the stdlib's bundled wheels, and installing INTO the venv needs no
 * `--break-system-packages` (a venv is not PEP-668 externally-managed). This is
 * exactly the path the agent used to run torch+triton on Modal L4. Idempotent:
 * the docker-build bake (setupProblem) and the verifier both call it, reusing an
 * existing venv. The venv lives outside workspace/ and stays hidden from agents.
 * The unused param keeps the historical `pipInstall(py)` call sites compiling.
 */
const GRADER_VENV = "/opt/cheri-grader-venv";
const VENV_PY = `${GRADER_VENV}/bin/python`;
const pipInstall = (_py?: string) =>
  `([ -x ${VENV_PY} ] || python3 -m venv ${GRADER_VENV}) && ` +
  `${VENV_PY} -m pip install --quiet ${PIP_DEPS}`;

class Environment implements EnvironmentDefinition {
  async listProblems() {
    return problems;
  }

  /**
   * Prepare the starting state for a problem.
   *
   * Runs during the docker build at package time and owns each problem's
   * starting state: whatever this leaves in workspace/ is what the agent
   * wakes up in.
   */
  async setupProblem(problemId: string, logger?: Logger): Promise<void> {
    console.log(`Setting up problem: ${problemId}`);

    if (!fs.existsSync(WORKSPACE_PATH)) {
      fs.mkdirSync(WORKSPACE_PATH, { recursive: true });
    }

    // The bench script is a diagnostic aid; the real problem does not stage it
    // (the grader lives in environment/assets, hidden from the agent, and is run
    // straight from there by the verifier).
    if (problemId === "gpu-diagnostics" || problemId === "gpu-vanilla") {
      const dest = path.join(WORKSPACE_PATH, "..", "bench_kernels.py");
      fs.copyFileSync(path.join(ASSETS_PATH, "bench_kernels.py"), dest);
      console.log(`staged bench_kernels.py -> ${dest}`);
    }

    // Warm the deps into the image for every GPU problem so the verifier phase
    // is measurement, not install. `python3 -m pip` + --break-system-packages:
    // the base image has no bare `python`/`pip` and is PEP-668 externally
    // managed. Never throw — a docker-build failure here is opaque; the `deps`
    // test in runTests reinstalls and reports the full output.
    if (
      problemId === "gpu-diagnostics" ||
      problemId === "gpu-vanilla" ||
      problemId === "cheri-triton-kernel" ||
      problemId === "cheri-triton-training"
    ) {
      const install = await executeWithExitCode(pipInstall("python3"), {
        timeout: 30 * MINUTES,
      });
      console.log(
        install.success
          ? "deps installed at setup"
          : `setup install did not succeed (rc=${install.exitCode}); the deps ` +
            `test will retry and report:\n${install.output}`
      );
    }

    console.log("Problem setup completed");
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    if (problemId === "gpu-diagnostics") {
      return runSimpleTests(this.diagnosticTests(), logger);
    }
    if (problemId === "gpu-vanilla") {
      return runSimpleTests(this.vanillaTests(), logger);
    }
    if (problemId === "cheri-triton-kernel") {
      return runSimpleTests(this.cheriInferenceTests(), logger);
    }
    if (problemId === "cheri-triton-training") {
      return runSimpleTests(this.cheriTrainingTests(), logger);
    }

    // No other problem is wired yet.
    return runSimpleTests(
      [
        {
          id: "not-implemented",
          name: "Problem not wired",
          description: `runTests has no implementation for '${problemId}'`,
          run: async () => ({
            success: false,
            error: `No tests defined for problem '${problemId}'. Only 'gpu-diagnostics' is wired.`,
          }),
        },
      ],
      logger
    );
  }

  /**
   * The real grader for `cheri-triton-kernel`, iteration 1 (inference).
   *
   * Runs environment/assets/grade_inference.py — which lives OUTSIDE workspace/
   * and is unreadable to the agent (environment/ is chmod o-rwx; the agent runs
   * unprivileged). It loads the agent's cheri.py, checks the no_grad CUDA
   * forward against an independent pure-PyTorch reference across a randomized
   * grid, times the speedup, and prints one `GRADE_JSON: {...}` line. Backward
   * is not graded in iteration 1.
   */
  private cheriInferenceTests(): SimpleTest[] {
    const PY = VENV_PY;
    const grader = path.join(ASSETS_PATH, "grade_inference.py");
    return [
      {
        id: "deps",
        name: "Install torch + triton (grader deps)",
        description:
          "Verifier-side deps. NOT `pip install -e .` — the grader loads " +
          "cheri.py via importlib and needs only torch + triton.",
        run: async (logger: Logger) => {
          const r = await executeWithExitCode(
            pipInstall(PY),
            { timeout: 30 * MINUTES }
          );
          logger.info(r.output);
          // weight 0: infrastructure, not skill. It must succeed for the grader
          // to run, but it must not inflate the reward (a status-only pass would
          // otherwise average a 1.0 into the rollout score).
          return r.success
            ? { success: true, weight: 0 }
            : { success: false, weight: 0, error: `deps install rc=${r.exitCode}` };
        },
      },
      {
        id: "inference-kernel",
        name: "Cheri inference kernel: correctness + speedup",
        description:
          "Grades the agent's no_grad CUDA forward vs the pure-PyTorch " +
          "reference (accuracy across a randomized shape/dtype grid) and its " +
          "speedup. Correctness gates perf. Continuous score in [0,1]. See " +
          "env-docs/07-iteration-1-inference.md.",
        run: async (logger: Logger) => {
          const cmd = `${PY} ${grader} --workspace ${WORKSPACE_PATH} --seed 0`;
          logger.info(`[grader] ${cmd}`);
          const r = await executeWithExitCode(cmd, {
            cwd: WORKSPACE_PATH,
            timeout: 30 * MINUTES,
          });
          logger.info(r.output);

          // The grader is authoritative via its GRADE_JSON line. A crash (no
          // line) is an infra/grader error, distinct from a low score.
          const m = r.output.match(/GRADE_JSON:\s*(\{.*\})\s*$/m);
          if (!m) {
            return {
              success: false,
              errored: true,
              error: `grader produced no GRADE_JSON (rc=${r.exitCode})`,
            };
          }
          let g: any;
          try {
            g = JSON.parse(m[1]);
          } catch (e) {
            return { success: false, errored: true, error: `bad GRADE_JSON: ${e}` };
          }

          const score: number = typeof g.score === "number" ? g.score : 0;
          // Set success:true once the score clears the pass threshold so the CLI
          // doesn't force `partially_passed` for a legitimate partial credit.
          // Threshold mirrors minReplayScore intent; tune during calibration.
          const PASS = 0.5;
          return {
            success: score >= PASS,
            score,
            rationale: g.reason || "",
            output: r.output.slice(-4000),
          };
        },
      },
    ];
  }

  /**
   * The real grader for `cheri-triton-training`, iteration 2 (training).
   *
   * Runs environment/assets/grade_training.py (hidden, outside workspace/). It
   * loads the agent's cheri.py, checks the GRAD-ENABLED CUDA forward against an
   * independent pure-PyTorch reference, checks BACKWARD parity of the gradients
   * (input, conv weight, both linears) against autograd of that reference across
   * a randomized grid, times the fwd+bwd speedup vs vanilla (anchored on live
   * gold), and prints one `GRADE_JSON: {...}` line. Correctness gates perf.
   */
  private cheriTrainingTests(): SimpleTest[] {
    const PY = VENV_PY;
    const grader = path.join(ASSETS_PATH, "grade_training.py");
    return [
      {
        id: "deps",
        name: "Install torch + triton (grader deps)",
        description:
          "Verifier-side deps. NOT `pip install -e .` — the grader loads " +
          "cheri.py via importlib and needs only torch + triton.",
        run: async (logger: Logger) => {
          const r = await executeWithExitCode(
            pipInstall(PY),
            { timeout: 30 * MINUTES }
          );
          logger.info(r.output);
          // weight 0: infrastructure, not skill (see cheriInferenceTests).
          return r.success
            ? { success: true, weight: 0 }
            : { success: false, weight: 0, error: `deps install rc=${r.exitCode}` };
        },
      },
      {
        id: "training-kernel",
        name: "Cheri training kernel: fwd+bwd correctness + speedup",
        description:
          "Grades the agent's grad-enabled CUDA forward+backward: forward " +
          "parity and gradient parity (input + all parameters) vs the " +
          "pure-PyTorch autograd reference across a randomized shape/dtype grid, " +
          "and the fwd+bwd speedup. Correctness gates perf. Continuous score in " +
          "[0,1]. See env-docs/08-iteration-2-training.md.",
        run: async (logger: Logger) => {
          const cmd = `${PY} ${grader} --workspace ${WORKSPACE_PATH} --seed 0`;
          logger.info(`[grader] ${cmd}`);
          const r = await executeWithExitCode(cmd, {
            cwd: WORKSPACE_PATH,
            timeout: 30 * MINUTES,
          });
          logger.info(r.output);

          const m = r.output.match(/GRADE_JSON:\s*(\{.*\})\s*$/m);
          if (!m) {
            return {
              success: false,
              errored: true,
              error: `grader produced no GRADE_JSON (rc=${r.exitCode})`,
            };
          }
          let g: any;
          try {
            g = JSON.parse(m[1]);
          } catch (e) {
            return { success: false, errored: true, error: `bad GRADE_JSON: ${e}` };
          }

          const score: number = typeof g.score === "number" ? g.score : 0;
          const PASS = 0.5;
          return {
            success: score >= PASS,
            score,
            rationale: g.reason || "",
            output: r.output.slice(-4000),
          };
        },
      },
    ];
  }

  /**
   * Lean run: interpreter probe + deps + the vanilla-vs-Triton speedup/accuracy
   * comparison only. Exists to answer one question fast without re-running the
   * full ~15-minute diagnostic grid.
   */
  private vanillaTests(): SimpleTest[] {
    const bench = path.join(WORKSPACE_PATH, "..", "bench_kernels.py");
    let PY = "python3";
    const run = async (logger: Logger, cmd: string, label: string, ms: number) => {
      logger.info(`[${label}] ${cmd}`);
      const r = await executeWithExitCode(cmd, { cwd: WORKSPACE_PATH, timeout: ms });
      logger.info(r.output);
      return {
        success: r.success,
        output: r.output,
        ...(r.success ? {} : { error: `[${label}] rc=${r.exitCode}` }),
      };
    };
    return [
      {
        id: "deps",
        name: "Install torch + triton",
        description: "torch/triton only; bench loads cheri.py via importlib.",
        run: async (logger: Logger) =>
          run(
            logger,
            pipInstall(PY),
            "deps",
            30 * MINUTES
          ),
      },
      {
        id: "vanilla",
        name: "Gold Triton kernels vs vanilla PyTorch (speedup + accuracy)",
        description:
          "Times the SAME block with HAS_TRITON toggled: gold's hand-written " +
          "kernels vs the pure-PyTorch reference, on the same GPU/shape/weights, " +
          "for train-fwd, fwd+bwd, and no_grad inference. Reports speedup ratios " +
          "and the max-abs fwd/grad disagreement. This is the number that " +
          "justifies the whole task.",
        run: async (logger: Logger) =>
          run(
            logger,
            `${PY} ${bench} vanilla --workspace ${WORKSPACE_PATH} --iters 50`,
            "vanilla",
            30 * MINUTES
          ),
      },
    ];
  }

  /**
   * Diagnostics. Every one of these REPORTS rather than gates — the payload is
   * the captured stdout, not the pass/fail. Read them back with:
   *   hyperfocal run tests <run-id>
   *   hyperfocal run pull <run-id> --only tests
   *
   * A test "fails" here only if the command could not run at all.
   *
   * Note on error strings: isEnvironmentError (env-base testing.ts:11-52)
   * regex-matches /timeout/i, /timed out/i, /limit.*exceeded/i and reclassifies
   * the test as `errored`, dropping it from scoring entirely. The wording below
   * deliberately avoids those words.
   */
  private diagnosticTests(): SimpleTest[] {
    const bench = path.join(WORKSPACE_PATH, "..", "bench_kernels.py");

    // Resolved by the shell-probe test, reused by every later test. Run 1 of
    // this problem hardcoded `python` and every command returned rc=127
    // (command not found) — we learned nothing and burned a GPU. Never assume
    // the base image's interpreter name.
    let PY = "python3";

    const runCmd = async (
      logger: Logger,
      cmd: string,
      label: string,
      timeoutMs: number
    ) => {
      logger.info(`[${label}] ${cmd}`);
      const r = await executeWithExitCode(cmd, {
        cwd: WORKSPACE_PATH,
        timeout: timeoutMs,
      });
      logger.info(r.output);
      return {
        success: r.success,
        output: r.output,
        ...(r.success
          ? {}
          : { error: `[${label}] command returned rc=${r.exitCode}` }),
      };
    };

    return [
      {
        id: "shell-probe",
        name: "What is actually on this box?",
        description:
          "Run 1 failed with rc=127 on every command: `python` does not exist " +
          "here. This probe reports the box before anything depends on it, and " +
          "is written to never fail so its output always survives.",
        run: async (logger: Logger) => {
          const cmd = [
            "echo '--- uname:'; uname -a",
            "echo '--- whoami:'; id",
            "echo '--- interpreters:'; which python python3 pip pip3 uv 2>&1 || true",
            "echo '--- python3 -V:'; python3 -V 2>&1 || true",
            "echo '--- nvidia-smi:'; nvidia-smi 2>&1 || echo 'nvidia-smi ABSENT'",
            "echo '--- cwd:'; pwd; ls -la",
          ].join("; ");
          const r = await executeWithExitCode(`sh -c ${JSON.stringify(cmd)}`, {
            cwd: WORKSPACE_PATH,
            timeout: 5 * MINUTES,
          });
          logger.info(r.output);
          // Pick the interpreter for every later test.
          if (r.output.includes("/python3")) PY = "python3";
          else if (r.output.includes("/python")) PY = "python";
          logger.info(`resolved interpreter: ${PY}`);
          return { success: true, output: r.output };
        },
      },
      {
        id: "git-leak-probe",
        name: "Can the workspace read gold out of git?",
        description:
          "Packaging COPYs the whole env repo clone, .git included, with every " +
          "branch materialized — so `git show main:workspace/cherimoya/cheri.py` " +
          "returns gold verbatim. The harness mitigates by chmod + " +
          "GIT_CEILING_DIRECTORIES + a bash denylist, all self-described as " +
          "best-effort and all void if the agent runs as root. This probe runs " +
          "in the VERIFIER, which is root — so it establishes the upper bound, " +
          "not the agent's actual reach. Compare against the agent trace.",
        run: async (logger: Logger) => {
          const cmd = [
            "echo '--- id:'; id",
            "echo '--- .git present?'; ls -la /hyperfocal/env/.git 2>&1 | head -5 || echo ABSENT",
            "echo '--- branches:'; git -C /hyperfocal/env branch -a 2>&1 | head -20 || echo BLOCKED",
            "echo '--- can we read gold from another branch?';" +
              " git -C /hyperfocal/env show main:workspace/cherimoya/cheri.py 2>&1 | head -5 || echo BLOCKED",
            "echo '--- history for cheri.py:';" +
              " git -C /hyperfocal/env log --all --oneline -- workspace/cherimoya/cheri.py 2>&1 | head -10 || echo BLOCKED",
          ].join("; ");
          const r = await executeWithExitCode(`sh -c ${JSON.stringify(cmd)}`, {
            timeout: 5 * MINUTES,
          });
          logger.info(r.output);
          return { success: true, output: r.output };
        },
      },
      {
        id: "deps",
        name: "Install torch + triton",
        description:
          "Deliberately NOT `pip install -e .`: the package __init__ drags in " +
          "modisco/tangermeme/macs3/bam2bw/bpnet-lite. cheri.py needs only " +
          "torch + triton, and bench_kernels.py importlib-loads it directly.",
        run: async (logger: Logger) =>
          // --break-system-packages: the base image is PEP-668
          // externally-managed (Debian 3.12), so a plain pip install fails
          // with "externally-managed-environment". We own this throwaway
          // sandbox, so writing into the system site-packages is fine.
          runCmd(
            logger,
            pipInstall(PY),
            "deps",
            30 * MINUTES
          ),
      },
      {
        id: "env-probe",
        name: "GPU / torch / triton probe",
        description:
          "bf16 support is the load-bearing bit: T4/Turing has none, and the " +
          "whole precision question is a bf16 question (env-docs/04 0b).",
        run: async (logger: Logger) => {
          // Semicolons, not newlines: a `\n` inside the shell-quoted -c string
          // reaches Python literally and raises SyntaxError (run 2's bug). Keep
          // `ok=` first so the later conditionals can reference it on one line.
          const py = [
            "import torch",
            "print('torch', torch.__version__)",
            "import triton",
            "print('triton', triton.__version__)",
            "ok = torch.cuda.is_available()",
            "print('cuda', ok)",
            "print('device', torch.cuda.get_device_name() if ok else 'NONE')",
            "print('cc', torch.cuda.get_device_capability() if ok else 'NONE')",
            "print('bf16', torch.cuda.is_bf16_supported() if ok else 'NONE')",
          ].join("; ");
          return runCmd(
            logger,
            `${PY} -c ${JSON.stringify(py)}`,
            "env-probe",
            5 * MINUTES
          );
        },
      },
      {
        id: "autotune-configs",
        name: "Are the training autotune configs collapsed?",
        description:
          "env-docs/02 section 3: _autotune_configs (cheri.py:106) puts " +
          "num_warps/num_stages in the Config kwargs dict instead of the " +
          "constructor, unlike the inference path (cheri.py:501). If every " +
          "config prints the same values, the sweep is dead.",
        run: async (logger: Logger) =>
          runCmd(
            logger,
            `${PY} ${bench} autotune --workspace ${WORKSPACE_PATH}`,
            "autotune",
            5 * MINUTES
          ),
      },
      {
        id: "gold-test-suite",
        name: "Does gold pass its own test suite?",
        description:
          "THE headline question. env-docs/02 predicts 4 failures, all of them " +
          "1e-4 assertions colliding with the bf16 downcast. Invisible on CPU " +
          "because every cuda/triton test auto-skips (conftest.py:39-48). " +
          "NOTE: run 1 piped this to `tail`, so the exit code was tail's and " +
          "the test reported PASSED while measuring nothing. No pipe here.",
        run: async (logger: Logger) =>
          runCmd(
            logger,
            `${PY} -m pytest tests/test_cheri.py -v --no-header -rA`,
            "pytest",
            30 * MINUTES
          ),
      },
      {
        id: "drift",
        name: "Real max-abs drift between the three forward paths",
        description:
          "Resolves env-docs/02 section 1: two tests pin the SAME comparison " +
          "at 1e-4 (test_cheri.py:402) and 1e-2 (test_cheri.py:583). This " +
          "prints the actual number and says which survives.",
        run: async (logger: Logger) =>
          runCmd(
            logger,
            `${PY} ${bench} drift --workspace ${WORKSPACE_PATH}`,
            "drift",
            30 * MINUTES
          ),
      },
      {
        id: "bench-published",
        name: "Can we reproduce docs/benchmarks.rst?",
        description:
          "benchmarks.rst credits numbers to a bench_kernels.py that does not " +
          "exist in the tree (env-docs/02 section 5), measured on torch 2.12 / " +
          "triton 3.7 which pyproject does not pin. Expect divergence: the " +
          "published table is H200 and this is an L4.",
        run: async (logger: Logger) =>
          runCmd(
            logger,
            `${PY} ${bench} bench --grid published --workspace ${WORKSPACE_PATH} --iters 50`,
            "bench-published",
            45 * MINUTES
          ),
      },
      {
        id: "bench-prod",
        name: "Gold baseline at production shapes",
        description:
          "fwd/bwd/fwd+bwd at (64, 2114, 128) across the real dilation " +
          "schedule. This GPU is not the rollout GPU, so treat as shape " +
          "coverage, not the anchor. N=512 may exhaust memory on a 24GB card; " +
          "the script skips those cases and continues.",
        run: async (logger: Logger) =>
          runCmd(
            logger,
            `${PY} ${bench} bench --grid prod --workspace ${WORKSPACE_PATH} --iters 50`,
            "bench-prod",
            60 * MINUTES
          ),
      },
    ];
  }
}

export default new Environment();
