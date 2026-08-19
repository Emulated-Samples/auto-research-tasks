#!/usr/bin/env bash
# Provision the fixed Linux substrate used by both the solver and hidden grader.
set -euo pipefail

MARKER=/opt/svpgsbench/.provisioned-v3
if [[ -f "$MARKER" ]]; then
  exit 0
fi

mkdir -p /opt/svpgsbench

# bubblewrap is required by the fail-closed grader. The remaining packages are
# the documented generic numeric/build substrate; no PGS/GWAS/GAM/penalized-
# regression implementation is installed.
dnf -y install \
  bubblewrap util-linux shadow-utils \
  gcc gcc-c++ gcc-gfortran make cmake \
  openblas-devel lapack-devel \
  python3.12 python3.12-devel

python3.12 -m venv /opt/svpgs-venv
/opt/svpgs-venv/bin/python -m pip install "pip==24.3.1"
/opt/svpgs-venv/bin/python -m pip install \
  "numpy==2.1.3" \
  "scipy==1.14.1"

if ! id -u svpgsub >/dev/null 2>&1; then
  useradd --system --no-create-home --shell "$(command -v nologin)" svpgsub
fi

for tool in bwrap setpriv gcc g++ gfortran make cmake python3.12; do
  command -v "$tool" >/dev/null
done
id -u svpgsub >/dev/null
/opt/svpgs-venv/bin/python - <<'PY'
import numpy
import scipy
assert numpy.__version__ == "2.1.3", numpy.__version__
assert scipy.__version__ == "1.14.1", scipy.__version__
PY

touch "$MARKER"
