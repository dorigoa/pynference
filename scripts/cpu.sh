#!/usr/bin/env bash
# Run a command on the test PC with the GPUs hidden: sync the repo (like
# scripts/remote.sh), then run COMMAND in ~/pynference with the venv active.
#
#   scripts/cpu.sh 'python -m pytest -q tests/cpu'
#
# ZE_AFFINITY_MASK=99 hides the GPUs from PyTorch/Level Zero (torch.xpu.device_count()
# is 0) and ONEAPI_DEVICE_SELECTOR=opencl:cpu keeps the SYCL runtime off the GPUs, so
# nothing here can take GPU memory from the LLM server (Strata) that holds both cards.
# GPU work goes through GPU jobs (docs/plans/pynference-plan.md, Parte 2.4).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: scripts/cpu.sh 'COMMAND'" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/remote.sh" "export ZE_AFFINITY_MASK=99 ONEAPI_DEVICE_SELECTOR=opencl:cpu; $*"
