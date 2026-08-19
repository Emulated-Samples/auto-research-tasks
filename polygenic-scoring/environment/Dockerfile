FROM amazonlinux:2023

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

COPY source/env/ /hyperfocal/env/
WORKDIR /hyperfocal/env

RUN set -e; \
    git init -q .; \
    git config user.name "Hyperfocal Packaging"; \
    git config user.email "packaging@hyperfocal.so"; \
    git config commit.gpgsign false; \
    git add -A -f; \
    GIT_AUTHOR_DATE="2020-01-01T00:00:00 +0000" GIT_COMMITTER_DATE="2020-01-01T00:00:00 +0000" git commit -q -m "Snapshot at packaging pin (no upstream history)" --no-verify; \
    mkdir -p /hyperfocal/logs && chmod 777 /hyperfocal/logs; \
    touch environment/.env; \
    (cd packages/env-base && npm install --no-audit --no-fund && npm run build); \
    (cd packages/env-orchestrator && npm install --no-audit --no-fund && npm run build); \
    if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --no-audit --no-fund && npm run build); fi; \
    (cd environment && npm install --no-audit --no-fund && npm run build); \
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem from_scratch_svpgs; \
    useradd --create-home --shell /bin/bash agent; \
    mkdir -p /tests /solution /logs/verifier; \
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages; \
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true; \
    for d in README.md PACKAGING.md RELEASE.md SECURITY.md; do \
      chmod o-rwx "/hyperfocal/env/$d" 2>/dev/null || true; \
    done; \
    chown -R agent:agent /hyperfocal/env/workspace; \
    npm cache clean --force >/dev/null 2>&1 || true

CMD ["sleep", "infinity"]
