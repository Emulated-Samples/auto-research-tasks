import fs from "node:fs";
import path from "node:path";

const [scenario, root] = process.argv.slice(2);

if (!scenario || !root) {
  throw new Error("usage: rollout-telemetry-scenario.js <scenario> <root>");
}

const workspace = path.join(root, "workspace");
fs.mkdirSync(workspace, { recursive: true });
fs.writeFileSync(
  path.join(root, "hyperfocal.yaml"),
  [
    'version: "1.0"',
    "environment:",
    "  name: rollout-telemetry-test",
    "paths:",
    `  root: ${JSON.stringify(root)}`,
    "  environmentDist: environment/dist",
    "  workspace: workspace",
    "agent:",
    "  awsAccess: false",
    "  defaultModel: test-model",
    "",
  ].join("\n")
);

process.chdir(root);

const { handleRolloutCommand } = await import(
  "../../dist/cli/commands/rollout.js"
);

const environment = {
  async listProblems() {
    return [{
      id: "phase-telemetry",
      prompt: "Exercise rollout phase telemetry.",
      default: true,
    }];
  },

  async setupProblem(_problemId, logger) {
    logger?.info("setup entered");
    if (scenario === "setup-failure") {
      throw new Error("No module named flashinfer.jit.aot_config");
    }
  },

  async runTests() {
    if (scenario === "tests-exception") {
      throw new Error("grader assertion exploded");
    }
    if (scenario !== "test-failure" && scenario !== "agent-timeout") {
      throw new Error(`runTests unexpectedly reached for ${scenario}`);
    }
    return [{
      id: "intentional-failure",
      name: "intentional test failure",
      status: "failed",
      duration: 1,
    }];
  },
};

await handleRolloutCommand(environment, {
  agentType: "anthropic-coding",
  permissionsMode: "claude-permissions",
});
