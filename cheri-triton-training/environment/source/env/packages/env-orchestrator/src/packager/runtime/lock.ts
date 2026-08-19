/**
 * Pinned Harbor harness version.
 *
 * The reward contract and oracle semantics DRIFT between harbor versions
 * (observed: reward.json envelope-vs-flat and oracle exec-bit handling both
 * changed 0.17.0 -> 0.17.1; 0.18.0 kept the flat reward.json contract but
 * deprecated allow_internet in favor of network_mode). All packaging
 * validation must run through THIS version; bump deliberately, per
 * customer, never implicitly.
 *
 * TODO(separate-verifier): harbor 0.18 already ships the machinery — a
 * [verifier] environment_mode = "shared" | "separate" plus an optional
 * [verifier.environment] table (VerifierConfig,
 * src/harbor/models/task/config.py in the pinned source). We deliberately
 * emit neither this round (owner decision): when separation lands it must
 * be a PER-ENV OPTION, never a global default, because some envs need the
 * agent and the verifier in the SAME image — the dind-cluster envs'
 * verifiers probe the docker-in-docker cluster the agent stood up, and a
 * separate verifier container would probe an empty world. The emission
 * seam is in emit/taskDir.ts renderTaskToml (see the marker there); any
 * bump of this pin should re-check that VerifierConfig's shape still
 * matches.
 */
export const HARBOR_LOCK = {
  version: "0.18.0",
  /** pip requirement spec used to provision the validation venv. */
  pipSpec: "harbor==0.18.0",
  /**
   * Override source for environments where the pinned version is not on
   * PyPI (or a local checkout should be used): set HARBOR_SOURCE to a local
   * path or pip-installable URL and it takes precedence over pipSpec.
   */
  sourceEnvVar: "HARBOR_SOURCE",
} as const;
