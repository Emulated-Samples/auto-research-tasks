/**
 * Parallel image builds for `harbor package` via `docker buildx bake`
 * (design docs/7-build-optimization, decision 7.1): render every problem's
 * Dockerfile up front, drop targets whose immutable tag already exists in
 * ECR, then build the rest as bake targets over the ONE shared staged
 * context. Each target still runs the env's full setupProblem() on a
 * pristine base — parallelism changes wall clock and nothing else.
 *
 * Throttle: bake would otherwise start every target at once; 36 unthrottled
 * silentbench builds would OOM/thrash a builder. Targets are chunked into
 * groups of min(floor(vcpu/4), 8) — override with HYPERFOCAL_BAKE_CONCURRENCY
 * — and baked group by group. Chunking (not buildkitd max-parallelism) so
 * there is no daemon-config dependency and behavior is deterministic.
 *
 * Failure attribution: bake fails as one invocation, but release
 * status_detail must NAME the failing problem. On a failed group, targets
 * whose image now exists (registry tag when pushing, local image otherwise)
 * are done; the missing ones are re-run individually with `docker build` —
 * completed vertices are cached, so the isolation pass costs minutes — and
 * each real per-problem error is captured and reported by problem id.
 */

import { spawnSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  ecrTagExists,
  ensureRegistryLogin,
  localImageExists,
  publishImage,
  pullPublishedImage,
} from "./ecr.js";
import { ensureBuildx } from "./buildx.js";
import {
  bakeTargetName,
  renderBakeFile,
  type BakeTargetSpec,
} from "../render/bakefile.js";

type Log = (msg: string) => void;

export interface ImageBuildTarget extends BakeTargetSpec {
  /** Absolute path of the pre-rendered Dockerfile (inside the context dir). */
  dockerfilePath: string;
}

export interface ImageBuildResult {
  /** problemIds whose immutable tag already existed (pulled, not rebuilt). */
  reused: string[];
  /** problemIds actually built this run. */
  built: string[];
  /** problemId -> image digest, from the bake metadata (built targets only). */
  digests: Map<string, string>;
}

/**
 * Bake group size: min(floor(vcpu / 4), 8) — each full build wants ~4 vCPU
 * during provision. HYPERFOCAL_BAKE_CONCURRENCY overrides on the builder.
 */
export function resolveBakeConcurrency(log?: Log): number {
  const raw = process.env.HYPERFOCAL_BAKE_CONCURRENCY;
  if (raw !== undefined && raw !== "") {
    const n = Number(raw);
    if (Number.isInteger(n) && n > 0) return n;
    if (log) {
      log(
        `ignoring invalid HYPERFOCAL_BAKE_CONCURRENCY="${raw}" (want a ` +
          `positive integer)`
      );
    }
  }
  return Math.max(1, Math.min(Math.floor(os.cpus().length / 4), 8));
}

function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

/**
 * Build (and, when pushing, publish) every target image. Returns which
 * targets were reused from the registry vs built, plus built digests.
 * Throws with per-problem attribution when any target cannot be built.
 */
export function buildImages(opts: {
  targets: ImageBuildTarget[];
  contextDir: string;
  /** Where bake files + metadata land (kept for auditability). */
  bakeDir: string;
  baseImage?: string;
  registryPrefix?: string;
  /**
   * packaging.bakeNeedsGpu: the env's setup executes CUDA work during
   * docker build, and dockerd-BuildKit ignores the daemon's nvidia
   * default-runtime for RUN steps (docs/7-build-optimization 5-findings.md
   * S3-F2 — proven empirically both directions). The LEGACY docker builder
   * honors default-runtime, so these envs skip bake entirely and build
   * serially with DOCKER_BUILDKIT=0 (per-problem attribution comes free;
   * the tag-exists skip above still applies).
   */
  bakeNeedsGpu?: boolean;
  log?: Log;
}): ImageBuildResult {
  const log: Log =
    opts.log ?? ((msg) => console.log(`[harbor:package] ${msg}`));
  const push = Boolean(opts.registryPrefix);

  // Tag-exists skip BEFORE building (immutable spec-hash tags make this
  // sound): already-published targets are pulled for downstream solution
  // production and never rebuilt. Previously the tag was consulted only
  // after a full build (push-vs-pull), so a same-commit-same-spec
  // re-release paid the whole build again.
  const reused: ImageBuildTarget[] = [];
  let toBuild: ImageBuildTarget[] = opts.targets;
  if (push) {
    ensureRegistryLogin(opts.targets[0].imageTag);
    toBuild = [];
    for (const target of opts.targets) {
      if (ecrTagExists(target.imageTag)) {
        log(
          `  ${target.imageTag} already published (immutable) — pulling it ` +
            `instead of rebuilding`
        );
        pullPublishedImage(target.imageTag);
        reused.push(target);
      } else {
        toBuild.push(target);
      }
    }
  }

  const result: ImageBuildResult = {
    reused: reused.map((t) => t.problemId),
    built: toBuild.map((t) => t.problemId),
    digests: new Map(),
  };

  if (toBuild.length === 0) {
    log(
      `all ${opts.targets.length} image tag(s) already published — ` +
        `skipping the build phase entirely`
    );
    return result;
  }

  if (opts.baseImage) ensureRegistryLogin(opts.baseImage);

  if (opts.bakeNeedsGpu) {
    buildSerialLegacy({
      toBuild,
      contextDir: opts.contextDir,
      push,
      digests: result.digests,
      log,
    });
    return result;
  }

  ensureBuildx(log);

  const concurrency = resolveBakeConcurrency(log);
  const groups = chunk(toBuild, concurrency);
  log(
    `building ${toBuild.length} image(s) via docker buildx bake ` +
      `(${groups.length} group(s) of up to ${concurrency} parallel target(s))`
  );

  fs.mkdirSync(opts.bakeDir, { recursive: true });
  for (const [i, group] of groups.entries()) {
    const bakeFile = path.join(opts.bakeDir, `bake-${i + 1}.json`);
    const metadataFile = path.join(opts.bakeDir, `bake-${i + 1}.metadata.json`);
    fs.writeFileSync(
      bakeFile,
      JSON.stringify(
        renderBakeFile({
          targets: group,
          contextDir: opts.contextDir,
          push,
        }),
        null,
        2
      ) + "\n"
    );
    log(
      `  bake group ${i + 1}/${groups.length}: ` +
        group.map((t) => t.problemId).join(", ")
    );
    const rc = spawnSync(
      "docker",
      [
        "buildx",
        "bake",
        "--file",
        bakeFile,
        "--metadata-file",
        metadataFile,
        // Attestation manifests would change what registries/consumers see
        // vs the historical plain `docker build && docker push`.
        "--provenance=false",
      ],
      {
        stdio: "inherit",
        env: {
          ...process.env,
          // buildx >= 0.17 gates filesystem access outside the invocation
          // cwd behind an interactive entitlement prompt; our per-problem
          // Dockerfiles and the shared context are sibling paths the
          // packager itself staged, and builders are headless — the prompt
          // would hang forever. Disable the gate for this invocation only.
          BUILDX_BAKE_ENTITLEMENTS_FS: "0",
        },
      }
    );
    if (rc.status !== 0) {
      isolateBakeFailure({ group, contextDir: opts.contextDir, push, log });
      // isolateBakeFailure throws when any target is truly broken; reaching
      // here means every missing target rebuilt (and published) fine — the
      // bake failure was transient (e.g. a flaky push).
      continue;
    }
    recordDigests(metadataFile, group, result.digests, log);
  }

  return result;
}

/**
 * `docker build` invocation for the legacy-builder path — exported so tests
 * can pin the exact command shape without spawning docker. DOCKER_BUILDKIT=0
 * is the whole point: the classic builder runs RUN steps through the
 * daemon's default runtime (nvidia on GPU builders — #109/#110), which
 * BuildKit provably does not (S3-F2). The generated Dockerfiles are plain
 * FROM/ENV/COPY/WORKDIR/RUN/CMD (render/dockerfile.ts + taskBase.ts), so
 * nothing BuildKit-specific is lost.
 */
export function legacyBuildInvocation(
  target: Pick<ImageBuildTarget, "imageTag" | "dockerfilePath">,
  contextDir: string
): { command: string; args: string[]; env: Record<string, string> } {
  return {
    command: "docker",
    args: [
      "build",
      "-t",
      target.imageTag,
      "-f",
      target.dockerfilePath,
      contextDir,
    ],
    env: { DOCKER_BUILDKIT: "0" },
  };
}

/**
 * packaging.bakeNeedsGpu build phase: serial per-problem legacy `docker
 * build` + push. Serial on purpose — each build runs a real CUDA workload
 * (e.g. flashinfer's warm benchmark + JIT compile) that wants the whole
 * GPU; parallel legacy builds would contend for device memory and skew any
 * bake-time perf baselines. Failure attribution is natural: the failing
 * problem is the one whose build just ran (named in the thrown error).
 *
 * Known cost, accepted: the legacy builder has no stage pruning, so the
 * orphaned task-base stage of a dockerfileLines-FROM override (the S3-F4
 * idiom) is pulled/built instead of skipped. For the fleet base that is one
 * image pull; correctness is unchanged (the last stage is the image).
 *
 * stdio is inherited so the CUDA work streams into the builder's build.log
 * live (builderd re-uploads it periodically; a captured-then-dumped buffer
 * would look like a dead job for the whole JIT warm — the S3 stale-log
 * false-incident class).
 */
function buildSerialLegacy(opts: {
  toBuild: ImageBuildTarget[];
  contextDir: string;
  push: boolean;
  digests: Map<string, string>;
  log: Log;
}): void {
  const { toBuild, contextDir, push, digests, log } = opts;
  log(
    `packaging.bakeNeedsGpu: on — building ${toBuild.length} image(s) ` +
      `SERIALLY with the legacy docker builder (DOCKER_BUILDKIT=0) so the ` +
      `daemon's nvidia default-runtime applies to RUN steps (BuildKit ` +
      `ignores it — S3-F2)`
  );
  for (const [i, target] of toBuild.entries()) {
    log(
      `  legacy build ${i + 1}/${toBuild.length}: ${target.problemId} -> ` +
        target.imageTag
    );
    const invocation = legacyBuildInvocation(target, contextDir);
    const rc = spawnSync(invocation.command, invocation.args, {
      stdio: "inherit",
      env: { ...process.env, ...invocation.env },
    });
    if (rc.status !== 0) {
      throw new Error(
        `image build failed for problem '${target.problemId}' ` +
          `(legacy serial GPU build ${i + 1}/${toBuild.length}, exit ` +
          `${rc.status}) — the docker build output above/in build.log is ` +
          `this problem's error`
      );
    }
    if (push) publishImage(target.imageTag);
    const digest = localImageDigest(target.imageTag);
    if (digest) {
      digests.set(target.problemId, digest);
      log(`  built ${target.problemId}: ${digest}`);
    }
  }
}

/** RepoDigest of a local (pushed) image tag, or null (diagnostics only). */
function localImageDigest(imageTag: string): string | null {
  const rc = spawnSync(
    "docker",
    ["image", "inspect", "--format", "{{index .RepoDigests 0}}", imageTag],
    { encoding: "utf-8" }
  );
  if (rc.status !== 0) return null;
  const out = rc.stdout.trim();
  return /@sha256:[0-9a-f]{64}$/.test(out)
    ? out.slice(out.indexOf("@") + 1)
    : null;
}

/** Read per-target digests out of a bake metadata file (best-effort). */
function recordDigests(
  metadataFile: string,
  group: ImageBuildTarget[],
  digests: Map<string, string>,
  log: Log
): void {
  let metadata: Record<string, Record<string, unknown>>;
  try {
    metadata = JSON.parse(fs.readFileSync(metadataFile, "utf-8"));
  } catch {
    return; // metadata is diagnostics, never load-bearing
  }
  for (const target of group) {
    const digest =
      metadata[bakeTargetName(target.problemId)]?.["containerimage.digest"];
    if (typeof digest === "string") {
      digests.set(target.problemId, digest);
      log(`  built ${target.problemId}: ${digest}`);
    }
  }
}

/**
 * A bake group failed as one invocation. Figure out which targets actually
 * completed, re-run the missing ones individually (`docker build` — the
 * completed vertices are cached, so this costs minutes), and throw an error
 * that NAMES each failing problem with its real error output.
 */
function isolateBakeFailure(opts: {
  group: ImageBuildTarget[];
  contextDir: string;
  push: boolean;
  log: Log;
}): void {
  const { group, contextDir, push, log } = opts;
  const missing = group.filter((t) =>
    push ? !ecrTagExists(t.imageTag) : !localImageExists(t.imageTag)
  );
  if (missing.length === 0) {
    log(
      "bake exited non-zero but every target in the group exists — " +
        "continuing (transient bake failure)"
    );
    return;
  }
  log(
    `bake group failed — isolating: re-running ${missing.length} missing ` +
      `target(s) individually to attribute the error ` +
      `(${missing.map((t) => t.problemId).join(", ")})`
  );

  const failures: Array<{ problemId: string; detail: string }> = [];
  for (const target of missing) {
    log(`  docker build ${target.imageTag} (isolation re-run)...`);
    const rc = spawnSync(
      "docker",
      [
        "build",
        "-t",
        target.imageTag,
        "-f",
        target.dockerfilePath,
        contextDir,
      ],
      { encoding: "utf-8", maxBuffer: 256 * 1024 * 1024 }
    );
    if (rc.status === 0) {
      log(`  ${target.problemId}: isolation re-run succeeded`);
      if (push) publishImage(target.imageTag);
      continue;
    }
    const output = `${rc.stdout ?? ""}\n${rc.stderr ?? ""}`.trim();
    const tail = output.split("\n").slice(-40).join("\n");
    failures.push({ problemId: target.problemId, detail: tail });
  }

  if (failures.length > 0) {
    throw new Error(
      `image build failed for problem(s) ` +
        failures.map((f) => `'${f.problemId}'`).join(", ") +
        ` — per-problem errors:\n` +
        failures
          .map(
            (f) =>
              `--- problem ${f.problemId} (docker build output, last 40 lines) ---\n${f.detail}`
          )
          .join("\n")
    );
  }
}
