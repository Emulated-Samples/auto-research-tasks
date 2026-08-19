FROM amazonlinux:2023

# GPU DEVICE INJECTION -- added 2026-08-15 (shard 2, wave 2). MEASURED, not assumed.
# harbor runs GPU tasks with `--override-gpus 0` (harbor 0.18.0's docker environment declares
# gpus=False as a STATIC capability, so it cannot allocate a GPU by any route). The container
# therefore gets its GPU ONLY from the daemon's DEFAULT nvidia runtime -- and that runtime
# injects /dev/nvidia* ONLY when NVIDIA_VISIBLE_DEVICES is present in the container config.
# amazonlinux:2023 sets neither var, so a plain `docker run` here saw ZERO devices: the grader
# exited in ~1.4 s with "CUDA not available in grader" and scored 0.0 -- real agent work (13-49
# turns, up to 6 file edits) recorded as indistinguishable from an empty submission. That zeroed
# all 8 wave-1 cheri-triton-training rollouts.
# MEASURED on a live g5.2xlarge (default runtime = nvidia in both arms):
#   docker run amazonlinux:2023                   -> 0 nvidia devices
#   docker run -e NVIDIA_VISIBLE_DEVICES=all ...  -> 5 nvidia devices
# This is NOT a new mechanism: the CUDA-derived tasks in this repo were never affected precisely
# because their base images already bake these two vars in -- verified by docker inspect on
# nvidia/cuda:12.8.1-devel-ubuntu22.04 and pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime, both
# NVIDIA_VISIBLE_DEVICES=all + NVIDIA_DRIVER_CAPABILITIES=compute,utility, both yielding 5
# devices and torch.cuda.is_available()==True with no explicit flag. This makes the
# amazonlinux-based tasks match them.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \
      curl \
      nodejs20 nodejs20-npm \
      git tar gzip unzip rsync patch procps-ng openssh-clients ca-certificates \
      which \
      python3.11 python3.11-pip python3.11-devel \
      python3.12 python3.12-pip python3.12-devel \
      gcc gcc-c++ make \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN if ! command -v node >/dev/null 2>&1; then \
      alternatives --install /usr/bin/node node /usr/bin/node-20 100; \
    fi \
    && if ! command -v npm >/dev/null 2>&1; then \
      alternatives --install /usr/bin/npm npm /usr/bin/npm-20 100; \
    fi \
    && node --version && npm --version

RUN dnf install -y --setopt=install_weak_deps=False python3-devel && dnf clean all && rm -rf /var/cache/dnf

COPY source/env/ /hyperfocal/env/
COPY source/repo.bundle /hyperfocal/git-bundles/repo.bundle
COPY source/git-submodules/ /hyperfocal/git-bundles/git-submodules/
WORKDIR /hyperfocal/env

RUN set -e; \
    git init -q .; \
    git fetch -q /hyperfocal/git-bundles/repo.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git symbolic-ref HEAD refs/heads/harbor-release/problem/cheri-triton-training/cheri-triton-training; \
    git read-tree HEAD; \
    git update-ref -d refs/hyperfocal/bundle-head; \
    git remote add origin "https://github.com/Hyperfocal-AI-Environments/ar-cherimoya"; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-base /hyperfocal/env/packages/env-base; \
    git -C /hyperfocal/env/packages/env-base fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-base.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-base update-ref --no-deref HEAD 17eb75cdf518c108dc77340da75e5cd1ee24f0d6; \
    git -C /hyperfocal/env/packages/env-base read-tree HEAD; \
    git -C /hyperfocal/env/packages/env-base update-ref -d refs/hyperfocal/bundle-head; \
    git -C /hyperfocal/env config submodule.packages/env-base.url "https://github.com/Hyperfocal-AI-Packages/env-base.git"; \
    git -C /hyperfocal/env config submodule.packages/env-base.active true; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-orchestrator /hyperfocal/env/packages/env-orchestrator; \
    git -C /hyperfocal/env/packages/env-orchestrator fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-orchestrator.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-orchestrator update-ref --no-deref HEAD 28feff8febd94c65636c145854a3d5fb7b6fbc03; \
    git -C /hyperfocal/env/packages/env-orchestrator read-tree HEAD; \
    git -C /hyperfocal/env/packages/env-orchestrator update-ref -d refs/hyperfocal/bundle-head; \
    git -C /hyperfocal/env config submodule.packages/env-orchestrator.url "https://github.com/Hyperfocal-AI-Packages/env-orchestrator.git"; \
    git -C /hyperfocal/env config submodule.packages/env-orchestrator.active true; \
    rm -rf /hyperfocal/git-bundles; \
    mkdir -p /hyperfocal/logs && chmod 777 /hyperfocal/logs; \
    touch environment/.env; \
    (cd packages/env-base && npm install --no-audit --no-fund && npm run build); \
    (cd packages/env-orchestrator && npm install --no-audit --no-fund && npm run build); \
    if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --no-audit --no-fund && npm run build); fi; \
    (cd environment && npm install --no-audit --no-fund && npm run build); \
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem cheri-triton-training; \
    useradd --create-home --shell /bin/bash agent; \
    mkdir -p /tests /solution /logs/verifier; \
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages; \
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true; \
    chown -R agent:agent /hyperfocal/env/workspace; \
    npm cache clean --force >/dev/null 2>&1 || true

CMD ["sleep", "infinity"]
