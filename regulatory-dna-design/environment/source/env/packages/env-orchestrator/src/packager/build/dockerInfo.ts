/**
 * Docker daemon introspection for the packaging.bakeNeedsGpu path.
 *
 * The legacy-builder GPU bake (build/bake.ts buildSerialLegacy) relies on
 * the daemon's nvidia default-runtime applying to RUN steps. Whether the
 * runtime is even REGISTERED is a daemon property (`docker info`
 * .Runtimes), baked on GPU builders by the nvidia-container-toolkit AMI
 * (control-plane #110) and activated as default at boot by the guarded
 * userdata (#109). `harbor doctor` uses this to warn — never fail — when a
 * bakeNeedsGpu env is doctored on a box without it (local dev, CPU boxes).
 */

import { spawnSync } from "child_process";

/**
 * Whether docker has an "nvidia" runtime registered. Returns null when
 * `docker info` itself fails (no daemon) — callers treat that as unknown,
 * not as absent.
 */
export function dockerHasNvidiaRuntime(): boolean | null {
  const rc = spawnSync(
    "docker",
    ["info", "--format", "{{json .Runtimes}}"],
    { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] }
  );
  if (rc.status !== 0) return null;
  try {
    const runtimes = JSON.parse(rc.stdout.trim());
    return runtimes !== null &&
      typeof runtimes === "object" &&
      Object.prototype.hasOwnProperty.call(runtimes, "nvidia");
  } catch {
    return null;
  }
}
