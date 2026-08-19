/**
 * Per-problem Dockerfile rendering for `harbor package`.
 *
 * Contracts this encodes (see harbor-packaging/04-harbor-spec.md):
 *  - The Dockerfile bakes the problem state via the env's own setup
 *    (decision D1: setup is the provisioner) and does all content-heavy
 *    work in a single RUN so lockdown chmod/chown never duplicate layers.
 *  - The fallback base recipe comes from taskBase.ts — the single source —
 *    never an inline copy (finding F13's lockstep hazard died there).
 *
 * TODO(live-infra-envs): D1 only covers envs whose setup materializes
 * files/deps. Envs whose setupProblem() starts live services (e.g.
 * pg-engine-feature-cutover's 6-container docker compose sandbox) cannot
 * bake setup inside `docker build` (no daemon at build time). Both harbor
 * paths were validated empirically on harbor 0.18.0 (2026-07-08, EC2):
 *   (B) privileged main container running dockerd at runtime — harbor's
 *       local docker provider passes the task compose file through
 *       verbatim, so `privileged: true` on main works; entrypoint starts
 *       dockerd then runs setupProblem() unmodified, with
 *       [environment].healthcheck gating the agent on a setup-complete
 *       sentinel. Lower effort, faithful port — preferred first step.
 *   (A) map sandbox containers onto harbor task compose sidecars
 *       (environment/docker-compose.yaml; main service is hardcoded
 *       "main"; sidecars reachable by service DNS name; verifier collect
 *       hooks + separate verifier mode give cleaner isolation) — better
 *       end state, but requires re-expressing host-side imperative setup
 *       as entrypoints/volumes.
 * Requires a runtime-setup phase concept in packaging (split setup into
 * bakeable vs runtime), plus pre-seeded inner image caches to keep task
 * start latency sane.
 */

import * as path from "path";
import { TASK_BASE_FALLBACK_RECIPE } from "./taskBase.js";

/**
 * In-image repo root. Fixed by the generated Dockerfile's `COPY env/
 * /hyperfocal/env/` regardless of what paths.root says on the source repo,
 * so templates join workspace paths against this literal, not config.
 */
export const IMAGE_ENV_ROOT = "/hyperfocal/env";

export interface DockerfileParams {
  problemId: string;
  /**
   * Shared-setup-layer toggle (packaging.sharedSetupLayer, design docs/
   * 7-build-optimization §2): the env's canonical problem id — the FIRST
   * problem in problems.yaml order across ALL problems (never the packaged
   * selection), so the warm-setup RUN below is the same command string for
   * every problem of the env and docker layer cache shares it. Only set
   * when the env owner asserted setupProblem is reset-complete: warming
   * with the canonical problem's setup and then running the real problem's
   * setup must yield the same final state as a fresh single setup. Unset =>
   * the historical single-RUN layout, byte-identical to before the toggle
   * existed.
   */
  canonicalProblemId?: string;
  /**
   * Environment variables baked into the image as Docker ENV — visible in
   * every phase (setup, agent, verifier, oracle). From the final TaskSpec
   * (hyperfocal.yaml `packaging.image.env`, possibly amended by the env's
   * packageProblem() hook). Rendered before dockerfileLines so the lines
   * can reference them.
   */
  imageEnv?: Record<string, string>;
  /**
   * Verbatim Dockerfile lines from the final TaskSpec (hyperfocal.yaml
   * `packaging.image.dockerfileLines`, possibly amended by the env's
   * packageProblem() hook) — the escape hatch for system deps the env's
   * setup cannot express (e.g. a rust toolchain). Inserted after the fleet
   * toolchain, before the repo COPY, so they cache well.
   */
  dockerfileLines?: string[];
  /**
   * Shared fleet base image (hyperfocal-task-base in ECR — the single
   * opinionated AL2023 base, decision D3; built from the control-plane
   * repo's infra/packer/task-base/Dockerfile) that already contains the
   * toolchain layer below. When set, we FROM it and skip the dnf install —
   * fleet-wide builds then share one warm cached layer. When unset (dev
   * boxes, offline use), the public amazonlinux:2023 image + inline
   * toolchain (taskBase.ts, the single source) keeps packaging
   * self-contained.
   */
  baseImage?: string;
  /**
   * Repo-relative workspace dir (hyperfocal.yaml paths.workspace, default
   * "workspace") — the only agent-owned subtree; chown'd at image build.
   */
  workspacePath?: string;
}

/**
 * Zero-comment rule for every GENERATED Dockerfile (owner decision,
 * docs/11-build-finalization-fourth-draft/4-implementation-plan.md §4 step
 * 3): shipped and internal recipes carry no comment lines at all — no
 * headers, no section markers, no provenance stamps. Comments in generated
 * files read as clutter to the customer and drift from the truth over
 * time; the Dockerfile explains itself or the README does. The strip runs
 * over the ASSEMBLED text so single-source blocks (taskBase.ts) keep their
 * in-code documentation without leaking it into artifacts. Parser
 * directives (`# syntax=`, `# escape=`), which are only meaningful before
 * the first instruction, are preserved. Blank-line runs left behind by
 * removed comments are collapsed.
 */
export function stripDockerfileComments(text: string): string {
  const lines = text.split("\n");
  const kept: string[] = [];
  let sawInstruction = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#")) {
      const isDirective =
        !sawInstruction && /^#\s*(syntax|escape)\s*=/i.test(trimmed);
      if (!isDirective) continue;
    } else if (trimmed !== "") {
      sawInstruction = true;
    }
    kept.push(line);
  }
  return (
    kept
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/^\n+/, "")
      .replace(/\n*$/, "") + "\n"
  );
}

/** One `ENV NAME="value"` line per var, with Dockerfile-safe quoting. */
function renderEnvLines(imageEnv: Record<string, string>): string[] {
  return Object.entries(imageEnv).map(([name, value]) => {
    const escaped = value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    return `ENV ${name}="${escaped}"`;
  });
}

/** Env-specific image-edit block (imageEnv + dockerfileLines), or "". */
export function renderEnvSpecificBlock(p: {
  imageEnv?: Record<string, string>;
  dockerfileLines?: string[];
}): string {
  const envLines = p.imageEnv ? renderEnvLines(p.imageEnv) : [];
  const extraLines = [
    ...envLines,
    ...(p.dockerfileLines ?? []).map((l) => l.trim()).filter(Boolean),
  ];
  return extraLines.length
    ? `\n# --- env-specific image edits (packaging.image / packageProblem hook) ---\n${extraLines.join("\n")}\n# --- end env-specific ---\n`
    : "";
}

/**
 * The single content-heavy provision sequence (npm builds + the env's own
 * setup + users + trust lockdown), as individual shell steps. SINGLE SOURCE:
 * both the real per-problem Dockerfile (renderDockerfile below) and the
 * shipped environment/Dockerfile (environmentDockerfile.ts) assemble their
 * RUN from exactly this list, so the two recipes cannot drift apart (the
 * F13/taskBase.ts lockstep lesson). Every step lands in ONE RUN with the
 * content it covers — chmod/chown in a later layer than the content would
 * duplicate whole files in overlayfs (decision D1 lineage).
 */
export function provisionRunSteps(
  problemId: string,
  workspacePath?: string
): string[] {
  return [
    "mkdir -p /hyperfocal/logs && chmod 777 /hyperfocal/logs",
    "touch environment/.env",
    "(cd packages/env-base && npm install --no-audit --no-fund && npm run build)",
    "(cd packages/env-orchestrator && npm install --no-audit --no-fund && npm run build)",
    "if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --no-audit --no-fund && npm run build); fi",
    "(cd environment && npm install --no-audit --no-fund && npm run build)",
    `node packages/env-orchestrator/bin/env-orchestrator.js setup --problem ${problemId}`,
    "useradd --create-home --shell /bin/bash agent",
    "mkdir -p /tests /solution /logs/verifier",
    "chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier",
    "chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages",
    "chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true",
    `chown -R agent:agent ${path.posix.join(IMAGE_ENV_ROOT, workspacePath ?? "workspace")}`,
    "npm cache clean --force >/dev/null 2>&1 || true",
  ];
}

/** Join shell steps into the packager's canonical single-RUN form. */
export function renderRun(steps: string[]): string {
  return `RUN set -e; \\\n    ${steps.join("; \\\n    ")}`;
}

export function renderDockerfile(p: DockerfileParams): string {
  const extra = renderEnvSpecificBlock(p);
  const base = p.baseImage?.trim()
    ? `# Shared fleet base (toolchain preinstalled; see hyperfocal-task-base).
FROM ${p.baseImage.trim()}
`
    : TASK_BASE_FALLBACK_RECIPE;
  if (p.canonicalProblemId) {
    return stripDockerfileComments(
      renderSharedSetupLayerDockerfile(p, base, extra)
    );
  }
  // The template's comments (and the base recipe's) are documentation for
  // THIS code's readers; stripDockerfileComments removes every one of them
  // from the emitted artifact (zero-comment rule, step 3).
  return stripDockerfileComments(`# Generated by \`env-orchestrator harbor package\` — do not edit by hand.
# Build context: a tokenless clone of the env repo under ./env/
${base}${extra}
COPY env/ /hyperfocal/env/
WORKDIR /hyperfocal/env

# Single content-heavy layer on purpose: npm builds + the env's own setup
# (bakes the problem state + deps — decision D1) + users + trust lockdown.
# Doing chmod/chown in the same layer that creates the content avoids the
# overlayfs full-file duplication that added +1.3GB in experiment 01.
${renderRun(provisionRunSteps(p.problemId, p.workspacePath))}

# Keepalive so exec-driven harnesses can operate the container.
CMD ["sleep", "infinity"]
`);
}

/**
 * Two-RUN layout for packaging.sharedSetupLayer (design docs/
 * 7-build-optimization §2, validated in the layer-split experiment):
 *
 *  - Warm RUN — the content-heavy work that is identical for every problem
 *    of the env: npm builds, the env's provisioning (via a warm setup of
 *    the CANONICAL problem, whose marker-gated provision does all toolchain
 *    work), users, and ALL trust-lockdown chmods. Its command string is
 *    identical across problems, so BuildKit computes the vertex once per
 *    bake and the docker layer cache shares it across builds.
 *  - Thin RUN — the real problem's setup (provision already done — marker
 *    hit) plus the workspace chown, which must follow the LAST setup
 *    because setup recreates the workspace.
 *
 * The chmod-with-content rule holds: every lockdown chmod stays in the warm
 * layer with the content it covers (overlayfs duplicates a whole file when
 * a later layer touches its metadata — experiment 01's +1.3GB); the thin
 * layer only chowns the workspace its own setup created.
 *
 * Correctness rests on the env owner's reset-complete assertion (see
 * PackagingConfig.sharedSetupLayer): the canonical problem's starting state
 * remains in a lower layer, overwritten in the final filesystem (FAQ #9 —
 * accepted as cosmetic; starting states ship to labs anyway).
 */
function renderSharedSetupLayerDockerfile(
  p: DockerfileParams,
  base: string,
  extra: string
): string {
  const canonical = p.canonicalProblemId!;
  const workspaceAbs = path.posix.join(
    IMAGE_ENV_ROOT,
    p.workspacePath ?? "workspace"
  );
  // When the problem IS the canonical one, the warm layer's setup already
  // baked exactly this problem's state — re-running it would rebuild the
  // same state a second time for nothing, so the thin layer is chown-only.
  const thinLayer =
    p.problemId === canonical
      ? `# Per-PROBLEM layer (thin): this problem IS the canonical problem — its
# setup already ran in the warm layer above, so a second setup would only
# recreate the identical state; just the agent handoff chown remains.
RUN set -e; \\
    chown -R agent:agent ${workspaceAbs}`
      : `# Per-PROBLEM layer (thin): the real problem's setup — provision is
# marker-gated and instant now — then the agent handoff chown, after the
# final setup since setup recreates the workspace.
RUN set -e; \\
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem ${p.problemId}; \\
    chown -R agent:agent ${workspaceAbs}`;
  return `# Generated by \`env-orchestrator harbor package\` — do not edit by hand.
# Build context: a tokenless clone of the env repo under ./env/
${base}${extra}
COPY env/ /hyperfocal/env/
WORKDIR /hyperfocal/env

# Problem-INDEPENDENT layer (packaging.sharedSetupLayer — identical command
# for every problem of this env): npm builds + warm setup of the canonical
# problem (${canonical}) — whose marker-gated provision bakes all
# toolchains — + users + trust lockdown. chmods stay in the layer that
# creates the content (overlayfs duplication, experiment 01); the o-rwx on
# the env-owned dirs also denies the agent anything created under them
# later.
RUN set -e; \\
    mkdir -p /hyperfocal/logs && chmod 777 /hyperfocal/logs; \\
    touch environment/.env; \\
    (cd packages/env-base && npm install --no-audit --no-fund && npm run build); \\
    (cd packages/env-orchestrator && npm install --no-audit --no-fund && npm run build); \\
    if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --no-audit --no-fund && npm run build); fi; \\
    (cd environment && npm install --no-audit --no-fund && npm run build); \\
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem ${canonical}; \\
    useradd --create-home --shell /bin/bash agent; \\
    mkdir -p /tests /solution /logs/verifier; \\
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \\
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages; \\
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true; \\
    npm cache clean --force >/dev/null 2>&1 || true

${thinLayer}

# Keepalive so exec-driven harnesses can operate the container.
CMD ["sleep", "infinity"]
`;
}
