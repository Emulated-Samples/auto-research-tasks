/**
 * Provision (once) and reuse a Python venv holding the PINNED harbor CLI.
 *
 * Location: ~/.cache/hyperfocal/harbor-venv-<version>. Provisioning order:
 *   1. $HARBOR_SOURCE (local path or pip URL) if set — for environments
 *      where the pinned version is not yet on PyPI.
 *   2. pip install "<pipSpec>" from PyPI.
 */

import { execFileSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { HARBOR_LOCK } from "./lock.js";

function findPython(): string {
  // harbor's pyproject requires Python >= 3.12.
  for (const candidate of ["python3.12", "python3.13", "python3"]) {
    try {
      const out = execFileSync(candidate, ["--version"], {
        encoding: "utf-8",
      });
      const match = out.match(/Python 3\.(\d+)/);
      if (match && Number(match[1]) >= 12) return candidate;
    } catch {
      /* try next */
    }
  }
  throw new Error(
    "No python >= 3.12 found (tried python3.12, python3.13, python3) — required by the harbor CLI"
  );
}

export function ensureHarborCli(log: (msg: string) => void): string {
  const venvDir = path.join(
    process.env.HOME || "/root",
    ".cache",
    "hyperfocal",
    `harbor-venv-${HARBOR_LOCK.version}`
  );
  const harborBin = path.join(venvDir, "bin", "harbor");

  if (fs.existsSync(harborBin)) {
    return harborBin;
  }

  const python = findPython();
  log(`Provisioning harbor ${HARBOR_LOCK.version} venv at ${venvDir}...`);
  fs.mkdirSync(path.dirname(venvDir), { recursive: true });
  execFileSync(python, ["-m", "venv", venvDir], { stdio: "inherit" });

  const pip = path.join(venvDir, "bin", "pip");
  execFileSync(pip, ["install", "--quiet", "--upgrade", "pip"], {
    stdio: "inherit",
  });

  const source = process.env[HARBOR_LOCK.sourceEnvVar];
  const spec = source ? [source] : [HARBOR_LOCK.pipSpec];
  try {
    execFileSync(pip, ["install", "--quiet", ...spec], { stdio: "inherit" });
  } catch (error) {
    fs.rmSync(venvDir, { recursive: true, force: true });
    throw new Error(
      `Failed to install harbor (${spec.join(" ")}). If the pinned version ` +
        `is not on PyPI, set ${HARBOR_LOCK.sourceEnvVar} to a local checkout ` +
        `(e.g. /hyperfocal/ref/harbor). Underlying error: ${error}`
    );
  }

  // Verify the installed version matches the lock (a HARBOR_SOURCE checkout
  // may drift from the pin — fail loudly rather than validate against the
  // wrong contract).
  const shown = execFileSync(pip, ["show", "harbor"], { encoding: "utf-8" });
  const version = shown.match(/^Version:\s*(\S+)/m)?.[1];
  if (version !== HARBOR_LOCK.version) {
    throw new Error(
      `Installed harbor ${version} != pinned ${HARBOR_LOCK.version} ` +
        `(source: ${source ?? HARBOR_LOCK.pipSpec}). Align the source with ` +
        `the lock or update lock.ts deliberately.`
    );
  }

  log(`harbor ${version} ready: ${harborBin}`);
  return harborBin;
}
