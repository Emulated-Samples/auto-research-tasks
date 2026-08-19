/**
 * `--cloud modal` dev loop: run an orchestrator cycle (setup/test/rollout)
 * inside an ephemeral Modal sandbox instead of on this machine, against the
 * author's CURRENT WORKING TREE (dirty state included — that is the point:
 * iterate on a GPU env from a CPU dev box without committing or launching a
 * platform rollout).
 *
 * Flow: tar the env repo (node_modules excluded, .git included) → create a
 * sandbox from hyperfocal-modal-base with the compute: block's GPU/sizing →
 * upload + extract + build packages → run the requested command, streaming
 * output → terminate (or keep with --keep for `modal shell` debugging).
 *
 * The `modal` npm SDK is a DYNAMIC import on purpose: it requires Node 22,
 * and env-orchestrator must stay installable on rollout EC2 boxes (Node 18)
 * without pulling it. Dev boxes opt in:
 *   npm install --no-save modal    # in packages/env-orchestrator
 * Auth: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or ~/.modal.toml (modal token set).
 */

import { execFileSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as YAML from "yaml";
import { GPU_TYPES_ANY, normalizeGpuTypes } from "../../config/yaml-config.js";

const DEV_APP_NAME = "hyperfocal-dev";
const ECR_PULL_SECRET = "hyperfocal-ecr-pull";
const SANDBOX_TIMEOUT_MS = 2 * 60 * 60 * 1000;

function log(msg: string): void {
  console.log(`[cloud:modal] ${msg}`);
}

interface ComputeBlock {
  gpus?: number;
  /**
   * Normalized allowlist: `gpuTypes: [T4, A10]`, the scalar `gpuTypes: any`,
   * and the singular sugar `gpuType: T4` all land here as a list ("any"
   * stays a sentinel — resolved in devLoopGpuType).
   */
  gpuTypes?: string[];
  cpus?: number;
  memoryMb?: number;
}

function readCompute(repoRoot: string): ComputeBlock {
  const yamlPath = path.join(repoRoot, "hyperfocal.yaml");
  if (!fs.existsSync(yamlPath)) {
    return {};
  }
  const doc = YAML.parse(fs.readFileSync(yamlPath, "utf-8"));
  const raw = (doc?.compute ?? {}) as Record<string, unknown>;
  return {
    gpus: raw.gpus as number | undefined,
    gpuTypes: normalizeGpuTypes(raw.gpuType, raw.gpuTypes),
    cpus: raw.cpus as number | undefined,
    memoryMb: raw.memoryMb as number | undefined,
  };
}

/**
 * The control plane resolves the allowlist to the cheapest allowed type
 * per venue; this Modal-only dev loop keeps it simple and takes the FIRST
 * concrete type. "any" resolves to T4 — the cheapest Modal type.
 */
function devLoopGpuType(gpuTypes: string[]): string {
  return gpuTypes.find((t) => t.toLowerCase() !== GPU_TYPES_ANY) ?? "T4";
}

function resolveBaseImage(): string {
  if (process.env.HYPERFOCAL_MODAL_BASE_IMAGE) {
    return process.env.HYPERFOCAL_MODAL_BASE_IMAGE;
  }
  try {
    const value = execFileSync(
      "aws",
      [
        "ssm",
        "get-parameter",
        "--name",
        "/hyperfocal/dev/modal/base-image",
        "--query",
        "Parameter.Value",
        "--output",
        "text",
      ],
      { encoding: "utf-8" }
    ).trim();
    if (value && value !== "UNSET") {
      return value;
    }
  } catch {
    /* fall through to the error below */
  }
  throw new Error(
    "Cannot resolve the Modal base image: set HYPERFOCAL_MODAL_BASE_IMAGE or make SSM /hyperfocal/dev/modal/base-image readable (aws cli) — see control-plane infra/modal/README.md"
  );
}

// Deliberately untyped: `modal` is not a dependency (see header), so the
// compiler must not try to resolve its types.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function importModalSdk(): Promise<any> {
  try {
    return await import("modal" as string);
  } catch {
    throw new Error(
      "The `modal` npm SDK is not installed (it needs Node >= 22 and is deliberately not a dependency — rollout boxes run Node 18). " +
        "Run `npm install --no-save modal` in packages/env-orchestrator, then retry."
    );
  }
}

/** Pipe a sandbox output stream to a local one without buffering the run. */
async function pump(
  stream: AsyncIterable<string>,
  out: NodeJS.WriteStream
): Promise<void> {
  for await (const chunk of stream) {
    out.write(chunk);
  }
}

export async function handleCloudCommand(): Promise<void> {
  // Strip our flags; everything left after the node/script prefix is the
  // inner orchestrator invocation to replay inside the sandbox.
  const argv = process.argv.slice(2);
  const keep = argv.includes("--keep");
  const innerArgs = argv.filter(
    (a, i) => a !== "--keep" && a !== "--cloud" && argv[i - 1] !== "--cloud"
  );

  const repoRoot = process.cwd();
  if (!fs.existsSync(path.join(repoRoot, "hyperfocal.yaml"))) {
    throw new Error("--cloud modal must run from an env repo root (no hyperfocal.yaml here)");
  }

  const compute = readCompute(repoRoot);
  // Silent-CPU footgun guard: gpus > 0 with a missing/typo'd gpuTypes would
  // otherwise launch a CPU sandbox — and the developer would believe they
  // just validated their env on a GPU, inverting this tool's purpose.
  // Mirrors the control plane's run-creation validation.
  if ((compute.gpus ?? 0) > 0 && !compute.gpuTypes?.length) {
    throw new Error(
      "hyperfocal.yaml declares compute.gpus > 0 but no compute.gpuTypes — refusing to launch a CPU sandbox for a GPU env. Set compute.gpuTypes (e.g. [\"T4\"] or \"any\")."
    );
  }
  const imageRef = resolveBaseImage();
  const modal = await importModalSdk();

  log(`base image: ${imageRef}`);
  log(
    `compute: gpus=${compute.gpus ?? 0}${compute.gpuTypes?.length ? ` (${compute.gpuTypes.join(",")})` : ""} cpus=${compute.cpus ?? "default"} memoryMb=${compute.memoryMb ?? "default"}`
  );

  const client = new modal.ModalClient();
  const app = await client.apps.fromName(DEV_APP_NAME, { createIfMissing: true });
  const image = imageRef.includes(".dkr.ecr.")
    ? client.images.fromAwsEcr(imageRef, await client.secrets.fromName(ECR_PULL_SECRET))
    : client.images.fromRegistry(imageRef);

  // Tar the dirty working tree. .git comes along (setup/test need branch
  // ops); node_modules are rebuilt inside and excluded for size.
  const tarPath = path.join(os.tmpdir(), `hyperfocal-devtree-${Date.now()}.tgz`);
  log("packing working tree (node_modules excluded)...");
  execFileSync("tar", ["czf", tarPath, "--exclude=node_modules", "-C", repoRoot, "."]);
  const tarMb = (fs.statSync(tarPath).size / 1024 / 1024).toFixed(1);
  log(`working tree: ${tarMb} MB`);

  const sandbox = await client.sandboxes.create(app, image, {
    gpu:
      compute.gpus && compute.gpuTypes?.length
        ? `${devLoopGpuType(compute.gpuTypes)}:${compute.gpus}`
        : undefined,
    cpu: compute.cpus,
    memoryMiB: compute.memoryMb,
    timeoutMs: SANDBOX_TIMEOUT_MS,
    tags: { "hyperfocal-dev": "true" },
  });
  log(`sandbox ${sandbox.sandboxId} created`);

  try {
    await sandbox.filesystem.copyFromLocal(tarPath, "/tmp/devtree.tgz");
    log("working tree uploaded; building packages in sandbox...");

    const setup = await sandbox.exec(
      [
        "bash",
        "-c",
        `set -e
mkdir -p /hyperfocal/env /hyperfocal/logs && chmod 777 /hyperfocal/logs
tar xzf /tmp/devtree.tgz -C /hyperfocal/env
cd /hyperfocal/env
(cd packages/env-base && npm install --silent && npm run build --silent)
(cd packages/env-orchestrator && npm install --silent && npm run build --silent)
if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --silent && npm run build --silent); fi
(cd environment && npm install --silent && npm run build --silent)
echo "[sandbox] build done"`,
      ],
      { stdout: "pipe", stderr: "pipe" }
    );
    await Promise.all([pump(setup.stdout, process.stdout), pump(setup.stderr, process.stderr)]);
    const setupRc = await setup.wait();
    if (setupRc !== 0) {
      throw new Error(`sandbox package build failed (rc=${setupRc})`);
    }

    log(`running: env-orchestrator ${innerArgs.join(" ")}`);
    const run = await sandbox.exec(
      [
        "bash",
        "-c",
        `cd /hyperfocal/env && node packages/env-orchestrator/bin/env-orchestrator.js ${innerArgs
          .map((a) => `'${a.replace(/'/g, "'\\''")}'`)
          .join(" ")}`,
      ],
      { stdout: "pipe", stderr: "pipe" }
    );
    await Promise.all([pump(run.stdout, process.stdout), pump(run.stderr, process.stderr)]);
    const rc = await run.wait();
    log(`command exited rc=${rc}`);
    process.exitCode = rc;
  } finally {
    fs.rmSync(tarPath, { force: true });
    if (keep) {
      log(`--keep: sandbox left running (times out in ${SANDBOX_TIMEOUT_MS / 60000} min)`);
      log(`debug with: modal shell ${sandbox.sandboxId}`);
      log(`terminate with: modal sandbox terminate ${sandbox.sandboxId} (or let it time out)`);
    } else {
      await sandbox.terminate();
      log("sandbox terminated");
    }
  }
}
