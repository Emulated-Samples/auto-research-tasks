/**
 * The ONE source of the inline fallback task-base recipe — the Dockerfile
 * block used when no shared fleet base image (`--base-image`) is supplied.
 *
 * Every renderer that needs the fallback toolchain (the per-problem
 * Dockerfile today; the self-contained audit Dockerfile when W4 lands) MUST
 * consume this constant instead of carrying its own copy, so the "two package
 * lists drifting apart" failure class (dress-rehearsal finding F13: the
 * curl/curl-minimal lockstep bug) is structurally impossible on the
 * orchestrator side. The remaining lockstep partner is the real
 * hyperfocal-task-base image (control-plane repo:
 * infra/packer/task-base/Dockerfile) — if you change one, change the other.
 */

/**
 * Fallback base recipe: public amazonlinux:2023 + the fleet toolchain,
 * byte-identical to what `renderDockerfile` historically inlined.
 */
export const TASK_BASE_FALLBACK_RECIPE = `# Inline fallback mirroring the AL2023 hyperfocal-task-base (control-plane
# repo: infra/packer/task-base/Dockerfile) — keep the two package lists in
# lockstep. Passing --base-image (the fleet default) skips this entirely.
FROM amazonlinux:2023

# Fleet toolchain: node for env-orchestrator/env-base builds, git for
# workspace state ops, python for env verifiers and venv bootstraps (venv is
# stdlib on AL2023 — no separate -venv package like Debian's), both 3.11 and
# 3.12 (mle pins 3.12), gcc/gcc-c++/make for source builds during pip
# installs (build-essential equivalent). FULL \`curl\` is swapped over the
# preinstalled curl-minimal (--allowerasing): harbor's agent-setup phase
# runs \`yum install -y curl procps-ng\` inside every task container and dies
# on the curl/curl-minimal conflict otherwise (dress-rehearsal finding F13,
# 2026-07-17). weak deps disabled + cache purged to keep the image lean.
RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \\
      curl \\
      nodejs20 nodejs20-npm \\
      git tar gzip unzip rsync procps-ng openssh-clients ca-certificates \\
      which \\
      python3.11 python3.11-pip python3.11-devel \\
      python3.12 python3.12-pip python3.12-devel \\
      gcc gcc-c++ make \\
    && dnf clean all \\
    && rm -rf /var/cache/dnf

# AL2023's versioned nodejs20 packages install /usr/bin/node-20 and
# /usr/bin/npm-20 without claiming the unversioned names. Register them via
# alternatives (guarded, in case a future package revision starts providing
# them), then fail the build loudly if \`node\`/\`npm\` still don't resolve.
RUN if ! command -v node >/dev/null 2>&1; then \\
      alternatives --install /usr/bin/node node /usr/bin/node-20 100; \\
    fi \\
    && if ! command -v npm >/dev/null 2>&1; then \\
      alternatives --install /usr/bin/npm npm /usr/bin/npm-20 100; \\
    fi \\
    && node --version && npm --version
`;
