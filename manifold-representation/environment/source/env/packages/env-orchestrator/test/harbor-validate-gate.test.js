import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  readMinReplayScore,
  readNetworkModes,
} from "../dist/packager/runtime/taskInfo.js";

const TASK_TOML = `version = "1.0"

[agent]
timeout_sec = 7200
user = "agent"

[verifier]
timeout_sec = 900
user = "root"
network_mode = "no-network"

[environment]
docker_image = "img"
network_mode = "public"
`;

async function makeTaskDir(files) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "harbor-validate-gate-"));
  for (const [name, content] of Object.entries(files)) {
    await writeFile(path.join(dir, name), content);
  }
  cleanup.push(dir);
  return dir;
}

const cleanup = [];
test.after(async () => {
  for (const dir of cleanup) {
    await rm(dir, { recursive: true, force: true });
  }
});

test("readNetworkModes returns every declared network_mode", async () => {
  const dir = await makeTaskDir({ "task.toml": TASK_TOML });
  assert.deepEqual(readNetworkModes(dir), ["no-network", "public"]);
});

test("readNetworkModes is empty for a toml without network fields", async () => {
  const dir = await makeTaskDir({ "task.toml": 'version = "1.0"\n' });
  assert.deepEqual(readNetworkModes(dir), []);
});

test("readMinReplayScore defaults to 1.0 without hyperfocal-validate.json", async () => {
  const dir = await makeTaskDir({ "task.toml": TASK_TOML });
  assert.equal(readMinReplayScore(dir), 1.0);
});

test("readMinReplayScore reads [metadata] min_replay_score from task.toml", async () => {
  // Current emission folds the gate into task.toml [metadata] (harbor 0.18
  // tolerates unknown metadata keys; the hyperfocal-validate.json sidecar is
  // retired). The reader must prefer this over the legacy sidecar.
  const dir = await makeTaskDir({
    "task.toml": `version = "1.0"

[metadata]
source = "hyperfocal"
min_replay_score = 0.85

[environment]
docker_image = "img"
`,
  });
  assert.equal(readMinReplayScore(dir), 0.85);
});

test("readMinReplayScore reads the packaged threshold", async () => {
  const dir = await makeTaskDir({
    "task.toml": TASK_TOML,
    "hyperfocal-validate.json": JSON.stringify({
      problemId: "E1_smooth",
      minReplayScore: 0.8,
    }),
  });
  assert.equal(readMinReplayScore(dir), 0.8);
});

test("readMinReplayScore falls back to the pre-v2 oracleMinScore key", async () => {
  // Task dirs packaged before the contract v2 rename still exist in S3
  // bundles — the reader must keep understanding them.
  const dir = await makeTaskDir({
    "task.toml": TASK_TOML,
    "hyperfocal-validate.json": JSON.stringify({
      problemId: "E1_smooth",
      oracleMinScore: 0.7,
    }),
  });
  assert.equal(readMinReplayScore(dir), 0.7);
});

test("readMinReplayScore defaults to 1.0 when the field is absent", async () => {
  const dir = await makeTaskDir({
    "hyperfocal-validate.json": JSON.stringify({ problemId: "p" }),
  });
  assert.equal(readMinReplayScore(dir), 1.0);
});

test("readMinReplayScore rejects out-of-range thresholds", async () => {
  for (const bad of [0, -1, 1.5, "0.8"]) {
    const dir = await makeTaskDir({
      "hyperfocal-validate.json": JSON.stringify({ minReplayScore: bad }),
    });
    assert.throws(
      () => readMinReplayScore(dir),
      /invalid minReplayScore/,
      `expected rejection for minReplayScore ${JSON.stringify(bad)}`
    );
  }
});
