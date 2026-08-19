#!/bin/bash
set -euo pipefail

# The rollout harness invokes provisioning with a restricted PATH that omits the
# system sbin directories, so bare `useradd`/`groupadd` and friends resolve to
# "command not found" and drop the whole rollout during setup with no score
# (this exact failure — provision line "useradd: command not found" — killed
# runs 019f5e98 and 019f5e99 before any agent ran). Absolute paths are still used
# below for the specific tools we depend on; this guards the general class.
export PATH="$PATH:/usr/sbin:/sbin"

PREFIX=/opt/hyperfocal/manifold-bench
MARKER="$PREFIX/.provisioned-v4"

if [[ -f "$MARKER" ]]; then
  exit 0
fi

mkdir -p "$PREFIX"
if command -v dnf >/dev/null 2>&1; then
  dnf install -y python3.12 python3.12-pip util-linux bubblewrap
elif ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 is required" >&2
  exit 1
fi
if [[ ! -x "$PREFIX/bin/python" ]]; then
  python3.12 -m venv "$PREFIX"
fi
"$PREFIX/bin/pip" install --disable-pip-version-check --no-cache-dir \
  numpy==2.2.6 scipy==1.15.3
"$PREFIX/bin/pip" install --disable-pip-version-check --no-cache-dir \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
"$PREFIX/bin/python" -c 'import numpy, scipy, torch; assert not torch.cuda.is_available()'

# Invoking a virtualenv interpreter through an external symlink can lose its
# pyvenv.cfg prefix and therefore its installed packages. Use wrappers that
# execute the interpreter at its real path. Leave the system python3.12 binary
# untouched because the virtualenv may itself link to it.
for command in python python3; do
  rm -f "/usr/local/bin/$command"
  printf '#!/bin/sh\nexec /opt/hyperfocal/manifold-bench/bin/python "$@"\n' \
    > "/usr/local/bin/$command"
  chmod 0755 "/usr/local/bin/$command"
done

if ! id manifoldsub >/dev/null 2>&1; then
  /usr/sbin/useradd --system --no-create-home --shell /usr/sbin/nologin manifoldsub
fi

touch "$MARKER"
