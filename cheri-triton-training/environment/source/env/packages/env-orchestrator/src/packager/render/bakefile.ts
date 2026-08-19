/**
 * buildx bake file (JSON) generation: one target per problem, all sharing
 * the ONE staged build context, each with its own pre-rendered
 * Dockerfile.<problemId>. The bake file is written next to the package
 * output (not inside the build context) and kept after the run for
 * auditability.
 *
 * Output type: `registry` when pushing (push overlaps with still-running
 * builds for free) and `docker` for local/dev builds (loads into the local
 * daemon exactly like `docker build -t` did) — same code path either way.
 */

export interface BakeTargetSpec {
  /** Problem id (target names are derived from this, sanitized). */
  problemId: string;
  /** Full image tag the target builds (and pushes, when pushing). */
  imageTag: string;
  /** Dockerfile filename, relative to the shared context dir. */
  dockerfileName: string;
}

/** Bake target names must be valid identifiers; problem ids may not be. */
export function bakeTargetName(problemId: string): string {
  return problemId.replace(/[^a-zA-Z0-9_-]/g, "-");
}

/**
 * Render the bake file object for one group of targets. Serialized with
 * JSON.stringify by the caller (build/bake.ts) — buildx accepts JSON bake
 * files directly.
 */
export function renderBakeFile(opts: {
  targets: BakeTargetSpec[];
  contextDir: string;
  push: boolean;
}): Record<string, unknown> {
  const target: Record<string, unknown> = {};
  for (const t of opts.targets) {
    target[bakeTargetName(t.problemId)] = {
      context: opts.contextDir,
      dockerfile: t.dockerfileName,
      tags: [t.imageTag],
      // String form, not {type: ...} object form: object-form output entries
      // require buildx >= 0.13, and builder AMIs have shipped docker with an
      // older bundled buildx (0.12.1 observed) — the object form makes bake
      // fail parse there and every build silently rides the serial isolation
      // fallback (Stage-1 finding S1-F1). String form parses on all versions.
      output: [opts.push ? "type=registry" : "type=docker"],
    };
  }
  return {
    group: {
      default: {
        targets: opts.targets.map((t) => bakeTargetName(t.problemId)),
      },
    },
    target,
  };
}
