#!/usr/bin/env bash
# Create/refresh .venv with PyTorch XPU on the test PC. Run from the repo root
# on the PC (e.g. `scripts/remote.sh scripts/setup_env.sh` from the Mac).
set -euo pipefail

PY="${PYI_PYTHON:-$HOME/miniforge3/bin/python3.13}"

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate

python -m pip install -q -U pip
python -m pip install -q torch --index-url https://download.pytorch.org/whl/xpu
python -m pip install -q -r requirements.txt

# Triton XPU compiles its launcher against Level Zero (level_zero/ze_api.h,
# -lze_loader) and oneCCL (xccl) dlopens libze_loader.so at runtime. Without
# the system package libze-dev, unpack that package's headers into the venv and
# add an unversioned libze_loader.so symlink there, instead of installing
# anything system-wide.
sed -i '/Level Zero dev files\|ze-link\|ze-dev/d' .venv/bin/activate
if [ ! -f /usr/include/level_zero/ze_api.h ]; then
    if [ ! -f .venv/ze-dev/usr/include/level_zero/ze_api.h ]; then
        tmp=$(mktemp -d)
        (cd "$tmp" && apt-get download -q libze-dev >/dev/null)
        dpkg -x "$tmp"/libze-dev_*.deb .venv/ze-dev
        rm -rf "$tmp"
    fi
    mkdir -p .venv/lib/ze-link
    ln -sf /usr/lib/x86_64-linux-gnu/libze_loader.so.1 .venv/lib/ze-link/libze_loader.so
    cat >> .venv/bin/activate <<'ACT'
# Level Zero dev files for Triton XPU and oneCCL (see scripts/setup_env.sh)
export LIBRARY_PATH="$VIRTUAL_ENV/lib/ze-link${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/ze-link${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$VIRTUAL_ENV/ze-dev/usr/include${CPATH:+:$CPATH}"
ACT
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "| xpu available:", torch.xpu.is_available(),
      "| devices:", torch.xpu.device_count())
PY
