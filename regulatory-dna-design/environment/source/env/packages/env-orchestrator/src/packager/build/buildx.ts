/**
 * docker buildx presence check + stopgap self-install.
 *
 * The bake build path (build/bake.ts) requires the buildx CLI plugin. Docker
 * Desktop bundles it; AL2023's docker package does NOT, and builder AMIs
 * baked before the buildx rollout lack it. Until the AMI ships buildx, the
 * packager installs a pinned static binary into ~/.docker/cli-plugins on
 * first use.
 *
 * TODO(ami-bake): remove the download path once buildx is baked into the
 * builder AMI (W0 rebake, decision 7.16) — the presence check stays.
 */

import { execFileSync, spawnSync } from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/** Pinned buildx release (latest stable at pin time, 2026-07-18). */
export const BUILDX_VERSION = "v0.35.0";

/**
 * sha256 of the static release binaries, from the release's checksums.txt
 * (https://github.com/docker/buildx/releases/tag/v0.35.0). Only Linux is
 * pinned: macOS/Windows dev boxes run Docker Desktop, which bundles buildx.
 */
const BUILDX_SHA256: Record<string, string> = {
  "linux-amd64":
    "d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda",
  "linux-arm64":
    "c4248d6cbc4a619a7e0b4609c11e509ad4ac0b475e1c64817c0ac20c5d90c766",
};

/** buildx version string (e.g. "github.com/docker/buildx v0.35.0 ..."), or null. */
export function buildxVersion(): string | null {
  const rc = spawnSync("docker", ["buildx", "version"], {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "ignore"],
  });
  if (rc.status !== 0) return null;
  return (rc.stdout || "").trim() || null;
}

/**
 * Minimum buildx the bake path is validated against. Builder AMIs have
 * shipped docker with a BUNDLED buildx (0.12.1 observed on the live pool,
 * Stage-1 finding S1-F1), so a presence-only check is not enough — an old
 * bundled plugin parses our bake files differently (and 0.12-era bake
 * predates the behaviors we rely on). Anything below this floor is treated
 * as missing and the pinned static build is installed to
 * ~/.docker/cli-plugins, which the docker CLI resolves AHEAD of the distro
 * plugin dir, shadowing the bundled copy.
 */
export const MIN_BUILDX_SEMVER = "0.17.0";

/** Parse "github.com/docker/buildx v0.35.0 <sha>" → [0, 35, 0], or null. */
function parseBuildxSemver(version: string | null): number[] | null {
  const m = version?.match(/v(\d+)\.(\d+)\.(\d+)/);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

function semverAtLeast(actual: number[], min: string): boolean {
  const want = min.split(".").map(Number);
  for (let i = 0; i < 3; i++) {
    if (actual[i] !== want[i]) return actual[i] > want[i];
  }
  return true;
}

/**
 * Ensure `docker buildx` works, downloading the pinned static plugin into
 * ~/.docker/cli-plugins when missing (Linux builders only — anywhere else a
 * missing buildx is an actionable setup error, not something to paper over).
 */
export function ensureBuildx(
  log: (msg: string) => void = (msg) => console.log(`[harbor:package] ${msg}`)
): void {
  const existing = parseBuildxSemver(buildxVersion());
  if (existing && semverAtLeast(existing, MIN_BUILDX_SEMVER)) return;
  if (existing) {
    log(
      `docker buildx ${existing.join(".")} is below the supported minimum ` +
        `${MIN_BUILDX_SEMVER} — installing the pinned static build over it`
    );
  }

  const archMap: Record<string, string> = { x64: "amd64", arm64: "arm64" };
  const arch = archMap[process.arch];
  const key = `linux-${arch}`;
  if (process.platform !== "linux" || !arch || !BUILDX_SHA256[key]) {
    throw new Error(
      "docker buildx is not available and cannot be auto-installed on " +
        `${process.platform}/${process.arch} — install the buildx CLI plugin ` +
        "(Docker Desktop bundles it; on Linux drop the static binary into " +
        "~/.docker/cli-plugins/docker-buildx)"
    );
  }

  // TODO(ami-bake): remove once buildx is baked into the builder AMI.
  const url =
    `https://github.com/docker/buildx/releases/download/` +
    `${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.${key}`;
  const pluginDir = path.join(os.homedir(), ".docker", "cli-plugins");
  const pluginPath = path.join(pluginDir, "docker-buildx");
  const tmpPath = `${pluginPath}.download-${process.pid}`;
  log(
    `docker buildx missing — installing pinned static buildx ` +
      `${BUILDX_VERSION} (${key}) to ${pluginPath}`
  );
  fs.mkdirSync(pluginDir, { recursive: true });
  try {
    execFileSync("curl", ["-fsSL", "--retry", "3", "-o", tmpPath, url], {
      stdio: ["ignore", "ignore", "inherit"],
    });
    const digest = crypto
      .createHash("sha256")
      .update(fs.readFileSync(tmpPath))
      .digest("hex");
    if (digest !== BUILDX_SHA256[key]) {
      throw new Error(
        `downloaded buildx sha256 mismatch (got ${digest}, want ` +
          `${BUILDX_SHA256[key]}) — refusing to install`
      );
    }
    fs.chmodSync(tmpPath, 0o755);
    fs.renameSync(tmpPath, pluginPath);
  } catch (err) {
    fs.rmSync(tmpPath, { force: true });
    throw new Error(
      `buildx self-install failed: ${err instanceof Error ? err.message : err}`
    );
  }

  const version = buildxVersion();
  const installed = parseBuildxSemver(version);
  if (!installed || !semverAtLeast(installed, MIN_BUILDX_SEMVER)) {
    throw new Error(
      `installed buildx to ${pluginPath} but \`docker buildx version\` ` +
        `reports "${version ?? "nothing"}" — the docker CLI is not resolving ` +
        `the user plugin dir ahead of the bundled plugin on this box`
    );
  }
  log(`buildx installed: ${version}`);
}
