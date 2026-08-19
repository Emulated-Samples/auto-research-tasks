import assert from "node:assert/strict";
import test from "node:test";

import { ensureSetupSolverIdentity } from "../dist/cli/commands/setup.js";

test("linux-user provisions the solver identity before setup", () => {
  let calls = 0;
  ensureSetupSolverIdentity("linux-user", () => {
    calls += 1;
  });
  assert.equal(calls, 1);
});

test("claude-permissions does not create the legacy solver identity", () => {
  let calls = 0;
  ensureSetupSolverIdentity("claude-permissions", () => {
    calls += 1;
  });
  assert.equal(calls, 0);
});
