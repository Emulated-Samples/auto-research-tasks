FROM amazonlinux:2023

RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \
      curl \
      nodejs20 nodejs20-npm \
      git tar gzip unzip rsync patch procps-ng openssh-clients ca-certificates \
      which \
      python3.11 python3.11-pip python3.11-devel \
      python3.12 python3.12-pip python3.12-devel \
      gcc gcc-c++ make \
      zlib-devel bzip2-devel xz-devel \
      python3-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN if ! command -v node >/dev/null 2>&1; then \
      alternatives --install /usr/bin/node node /usr/bin/node-20 100; \
    fi \
    && if ! command -v npm >/dev/null 2>&1; then \
      alternatives --install /usr/bin/npm npm /usr/bin/npm-20 100; \
    fi \
    && node --version && npm --version

COPY source/env/ /hyperfocal/env/
COPY source/repo.bundle /hyperfocal/git-bundles/repo.bundle
COPY source/git-submodules/ /hyperfocal/git-bundles/git-submodules/
# Scoring-service payload: the activity-model weights, the daemon that serves them and the
# container entrypoint that starts it. Deliberately COPYed OUTSIDE /hyperfocal/env/workspace --
# the weights used to sit in the agent's own workspace, which made the query budget decorative
# (load the weights, search offline without limit, return constants). They are locked down below.
COPY source/seqdesign/ /opt/seqdesign/
WORKDIR /hyperfocal/env

RUN set -e; \
    git init -q .; \
    git fetch -q /hyperfocal/git-bundles/repo.bundle "+refs/heads/harbor-release/problem/seq-design-ds/seq-design-ds:refs/heads/harbor-release/problem/seq-design-ds/seq-design-ds" --update-head-ok; \
    git symbolic-ref HEAD refs/heads/harbor-release/problem/seq-design-ds/seq-design-ds; \
    git read-tree HEAD; \
    git update-ref -d refs/hyperfocal/bundle-head; \
    git remote add origin "https://github.com/Hyperfocal-AI-Environments/grelu-test"; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-base /hyperfocal/env/packages/env-base; \
    git -C /hyperfocal/env/packages/env-base fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-base.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-base update-ref --no-deref HEAD 14e43e3b44fe660701c709f28933940ae3cf62bc; \
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
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem seq-design; \
    useradd --create-home --shell /bin/bash agent; \
    mkdir -p /tests /solution /logs/verifier; \
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages; \
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true; \
    chown -R agent:agent /hyperfocal/env/workspace; \
    # Bake-and-close (2026-08-09, EGRESS-SWEEP.md). The pinned grade-time closure
    # (environment/requirements-test.txt, torch==2.8.0+cpu et al) is ALREADY installed
    # at build time into environment/.grade-venv by the setup step above -- but that
    # tree is root-only, so the agent phase had no usable Python at all (no torch, no
    # numpy, no pip under any interpreter) and every rollout rebuilt the whole closure
    # from PyPI at agent time. That is why these tasks appeared to "need" egress.
    # Three of the four instruction.md files already PROMISE "no internet access" and
    # "a venv with the dependencies you need is already set up in the workspace"; this
    # makes that promise true instead of installing anything new.
    #   o+x on environment/ grants TRAVERSAL only -- it is not o+r, so the agent still
    #   cannot list it, and behavioral-tests/ src/ dist/ requirements-test.txt keep
    #   their own o-rwx and stay unreadable. Only .grade-venv is opened, read-only.
    # The symlink is created AFTER the chown above on purpose: `chown -R` without -h
    # would follow it and hand the grade venv to the agent uid.
    # workspace/.gitignore already ignores `.venv*/`, so it cannot pollute the agent diff.
    chmod o+x /hyperfocal/env/environment; \
    chmod -R o+rX /hyperfocal/env/environment/.grade-venv; \
    ln -sfn /hyperfocal/env/environment/.grade-venv /hyperfocal/env/workspace/.venv; \
    chown -h agent:agent /hyperfocal/env/workspace/.venv; \
    npm cache clean --force >/dev/null 2>&1 || true

# Scoring-service lockdown. /opt/seqdesign is root-owned 0700: the agent uid can neither read the
# model weights nor read or edit the query counter the verifier trusts. The only agent-reachable
# surface is the unix socket the daemon binds at runtime (0666, in a 0755 dir) -- it answers with
# activity predictions and nothing else. The entrypoint lives in /usr/local/bin (0755) so the
# container still starts if a runtime launches it as a non-root user.
RUN set -e; \
    mv /opt/seqdesign/entrypoint.sh /usr/local/bin/seqdesign-entrypoint.sh; \
    chmod 755 /usr/local/bin/seqdesign-entrypoint.sh; \
    chown -R root:root /opt/seqdesign; \
    chmod 700 /opt/seqdesign; \
    chmod 600 /opt/seqdesign/oracle.pth /opt/seqdesign/oracled.py

# Total sequences the scoring service may be asked for across one session (development included).
# The verifier compares the root-owned counter against this. Set here so the image is the single
# place the allowance is defined; the daemon reports it to clients but never enforces it.
ENV SEQDESIGN_SESSION_BUDGET=200000

# The daemon must be up for both the agent phase and the grade phase. harbor's compose template
# overrides `command:` but never `entrypoint:`, so starting it here is the one hook that survives
# the harbor flow, a bare `docker run`, and a `docker compose up` (single container throughout --
# multi-service setups are unsupported, harbor-format.md).
ENTRYPOINT ["/usr/local/bin/seqdesign-entrypoint.sh"]
CMD ["sh", "-c", "sleep infinity"]
