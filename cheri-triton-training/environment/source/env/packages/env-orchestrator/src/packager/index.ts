/**
 * The packager's ONE public API. `cli/` (and any future export target) may
 * import ONLY this module — never packager internals — and the packager
 * never reads argv or prints on its own authority: entry points take typed
 * options (with an optional logger) and return typed results / throw typed
 * errors.
 *
 * The packager lives inside env-orchestrator on purpose: reward semantics,
 * problem setup, and packaging move in lockstep with the versions each env
 * repo pins, and the release pipeline runs each env's OWN pinned
 * orchestrator rather than a platform-wide one (which could silently skew
 * from what the env was developed and tested against).
 * TODO(extract-packager): lift into a standalone env-packager package when
 * a second export target appears.
 */

export {
  packageRelease,
  findEnvRepo,
  slugifyBranch,
  deriveTaskBranch,
  type HarborPackageOptions,
} from "./package.js";
// Publish-side README re-rendering (the control plane stamps live platform
// state into the [Internal] task-status table at publish time).
export {
  renderEnvReadme,
  type RenderEnvReadmeParams,
  type ReadmeTaskRow,
} from "./emit/readme.js";
export { doctor, type HarborDoctorOptions } from "./checks/doctor.js";
export { validate, type HarborValidateOptions } from "./checks/validate.js";
export { grade, type HarborGradeOptions } from "./runtime/grade.js";
export { run, type HarborRunOptions } from "./runtime/run.js";
