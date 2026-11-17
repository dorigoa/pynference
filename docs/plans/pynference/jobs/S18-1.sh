#!/usr/bin/env bash
# job: S18-1
# timeout: 45
# scopo: (1) profilo di partenza del decode gpt-oss con profilo per kernel
#   (bench/decode_profile.py --mode fused --ctx 8192, contratto C9) e
#   statistiche ExpertStore; (2) sonda GPUActive: perche' il nostro processo
#   blocca ~1 GiB di RAM host per ogni GiB riservato su GPU (S17-6: 40 GiB
#   con 2x20 riservate) mentre Strata con ~30 GiB/GPU ne blocca 7? Confronto
#   un processo senza oneCCL (come Strata) vs due con TP/xccl, allocazioni
#   grandi vs piccole; GPUActive / torch reserved / torch allocated a ogni
#   stadio. Nessuna modifica di codice: e' la baseline del passo 1.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf

echo "== baseline meminfo (Strata fermo, GPU fredde):"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# ---- part 2 first: the probes are short and must run before the model load
cat > "$OUT/probe_hostmem.py" <<'PYEOF'
"""GPUActive per reserved/allocated byte: 1 proc without CCL vs 2 with (S18).

Usage: probe_hostmem.py single   (one process, xpu:0, no torch.distributed)
       probe_hostmem.py ccl      (torchrun nproc_per_node=2, TP -> oneCCL)
Stages print rank, stage, /proc/meminfo GPUActive (system-wide), torch
bytes_reserved and bytes_allocated. Interpretation: GPUActive delta follows
allocated -> host RAM mirrors touched VRAM; follows reserved -> allocator
caching; only the ccl probe inflates -> oneCCL context.
"""
import os
import subprocess
import sys
import time

import torch


def gpu_active_bytes():
    for line in open("/proc/meminfo"):
        if line.startswith("GPUActive"):
            return int(line.split()[1]) * 1024
    return -1


def torch_bytes(dev):
    return (torch.xpu.memory_reserved(dev), torch.xpu.memory_allocated(dev))


def line(rank, stage):
    resv, alloc = torch_bytes(dev) if torch.xpu.is_initialized() else (0, 0)
    print(f"P{rank} {stage:28s} gpu_active {gpu_active_bytes() / 2**30:7.2f} GiB "
          f"reserved {resv / 2**30:6.2f} allocated {alloc / 2**30:6.2f}", flush=True)


mode = sys.argv[1]
rank = int(os.environ.get("RANK", 0))
if mode == "ccl":
    from pynference.parallel import TP
    dev = torch.device(f"xpu:{rank % torch.xpu.device_count()}")
    line(rank, "pre-TP")
    tp = TP()                       # xccl context + persistent buffers
    line(rank, "post-TP")

    def sync():                     # proven path: TP's persistent all_reduce
        tp.all_reduce(torch.ones(8, device=dev))

    sync()
    line(rank, "post-sync")

    def stage_ccl(tag, blocks, block_mib):
        bufs = [torch.empty(int(block_mib * 2**20), dtype=torch.uint8, device=dev)
                for _ in range(blocks)]
        for b in bufs:              # touch: pages must be real
            b.zero_()
        time.sleep(4)               # both ranks settle; GPUActive is global
        line(rank, tag)
        del bufs
        torch.xpu.empty_cache()
        time.sleep(4)
        line(rank, tag + " freed")
        sync()

    stage_ccl("ccl 8x1GiB", 8, 1024)
    stage_ccl("ccl 512x16MiB", 512, 16)
    tp.close()
    line(rank, "post-TP.close")
else:
    dev = torch.device("xpu:0")
    line(rank, "pre-alloc")

    def stage_single(tag, blocks, block_mib):
        bufs = [torch.empty(int(block_mib * 2**20), dtype=torch.uint8, device=dev)
                for _ in range(blocks)]
        for b in bufs:
            b.zero_()
        line(rank, tag)
        del bufs
        torch.xpu.empty_cache()
        line(rank, tag + " freed")

    stage_single("single 8x1GiB", 8, 1024)
    stage_single("single 512x16MiB", 512, 16)
    stage_single("single 64x128MiB", 64, 128)
PYEOF

echo "== probe A: one process, no oneCCL ($(date +%T))"
python "$OUT/probe_hostmem.py" single 2>&1 | grep -a -E "^P[0-9] " | tee "$OUT/probe_single.log"

echo "== probe B: two processes with oneCCL ($(date +%T))"
timeout 300 torchrun --nproc_per_node=2 "$OUT/probe_hostmem.py" ccl 2>&1 \
  | grep -a -E "^P[0-9] |Error|error" | tee "$OUT/probe_ccl.log" | head -n 40

echo "== meminfo after probes:"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo

# ---- part 1: baseline decode profile
cat > "$OUT/run_bench.py" <<'PYEOF'
"""Run a bench/ script with load_tokenizer as its tokenizer factory (S17)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from pynference.tokenizer import load_tokenizer  # noqa: E402

module, argv = sys.argv[1], sys.argv[2:]
mod = __import__(module, fromlist=["main"])
mod.Tokenizer = load_tokenizer
sys.argv = [module] + argv
mod.main()
PYEOF

echo "== decode_profile fused --ctx 8192 --steps 32 ($(date +%T))"
PYI_MEMDIAG="$OUT/memdiag.log" timeout 1200 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.decode_profile -m "$M" --mode fused \
  --ctx 8192 --steps 32 > "$OUT/profile.log" 2>&1
rc=$?
echo "== decode_profile exit $rc ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|load done|wall time|store: hits|GPU kernel time' \
  "$OUT/profile.log" | head -n 8
echo "-- top kernels (ms/token):"
sed -n '/kernel.*ms.token/,/^$/p' "$OUT/profile.log" | head -n 28
echo "-- profile tail:"
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/profile.log" | tail -n 6

echo "-- kernel oom lines (job safety net):"
journalctl -k --no-pager --since "30 min ago" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || true

echo "== summary: profile=$rc"
echo "== end $(date +%T)"
