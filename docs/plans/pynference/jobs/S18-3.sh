#!/usr/bin/env bash
# job: S18-3
# timeout: 75
# scopo: job finale di S18 (passi 3 e 4 insieme, un solo job): (1) microbench
#   della GEMV indicizzata MXFP4 su forme reali degli esperti (N=K=2880,
#   row_bytes=1530): banda GB/s per ogni (BLOCK_N, BLOCK_K, num_warps) della
#   lista estesa in quant/kernels.py, E = esperti per rank (1/2/4), e la
#   configurazione scelta dall'autotune intatto; (2) pytest tests/gpu completo
#   con le config estese (2.9.2); (3) velocità C9: decode graph --ctx 8192 su
#   prompt corto (32 step) e dopo prompt di ~2000 token (16 step, prefill
#   misurato), sempre --no-profile (il profiler si impicca sul grafo, S18-2);
#   (4) regressioni: scripts/regress.sh gpt-oss-120b e gemma4 (C10).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

echo "== baseline meminfo:"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# ---- step 3: indexed MXFP4 GEMV bandwidth, real expert shapes, forced configs
cat > "$OUT/bench_gemv.py" <<'PYEOF'
"""Indexed MXFP4 GEMV bandwidth on gpt-oss expert shapes (S18 step 3).

Forces one (_GEMV_CONFIGS) config at a time on _gemv_indexed_kernel and
times gemv_indexed_native at N=K=2880, row_bytes=1530, for E experts per
call (a rank sees ~2 of the 4 routed experts per token). One process, xpu:0.
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch
import triton

from pynference.quant.kernels import _GEMV_CONFIGS, _gemv_indexed_kernel, gemv_indexed_native

dev = torch.device(f"xpu:{int(os.environ.get('RANK', 0)) % torch.xpu.device_count()}")
N = K = 2880                       # gpt-oss expert: gate/up/down are all square
RB = (K // 32) * 17                # MXFP4: 16 fp4 bytes + 1 e8m0 scale per 32
S = 12                             # w_stack slots, distinct experts in L2 range
torch.manual_seed(0)
w = torch.randint(0, 256, (S, N, RB), dtype=torch.uint8, device=dev)


def bench(E, cfg):
    _gemv_indexed_kernel.configs = [cfg]   # len==1 -> Autotuner uses configs[0]
    _gemv_indexed_kernel.cache.clear()
    x = torch.randn(E, K, device=dev)
    slots = torch.arange(E, dtype=torch.int32, device=dev) % S
    out = torch.empty(E, N, device=dev)
    for _ in range(3):
        gemv_indexed_native("MXFP4", w, x, slots, out)
    torch.xpu.synchronize(dev)
    n_it = 20
    t = time.perf_counter()
    for _ in range(n_it):
        gemv_indexed_native("MXFP4", w, x, slots, out)
    torch.xpu.synchronize(dev)
    ms = (time.perf_counter() - t) / n_it * 1e3
    gb = E * N * RB / 2**30        # weight bytes read dominate everything else
    return ms * 1e3, gb / (ms / 1e3)


for E in (1, 2, 4):
    rows = []
    for cfg in _GEMV_CONFIGS:
        bn, bk = cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"]
        try:
            us, gbps = bench(E, cfg)
        except Exception as e:     # OutOfResources: autotune would skip it too
            print(f"E={E} BN={bn:2d} BK={bk:4d} W={cfg.num_warps:2d}  SKIP {type(e).__name__}", flush=True)
            continue
        rows.append((gbps, bn, bk, cfg.num_warps, us))
        print(f"E={E} BN={bn:2d} BK={bk:4d} W={cfg.num_warps:2d}  {us:8.1f} us  {gbps:6.1f} GB/s", flush=True)
    gbps, bn, bk, nw, us = max(rows)
    print(f"E={E} BEST BN={bn} BK={bk} W={nw} {gbps:.1f} GB/s ({us:.1f} us)", flush=True)

# the untouched autotune on the full list: one benchmark pick per (shape, type) key
_gemv_indexed_kernel.configs = list(_GEMV_CONFIGS)
_gemv_indexed_kernel.cache.clear()
x = torch.randn(2, K, device=dev)
slots = torch.arange(2, dtype=torch.int32, device=dev)
out = torch.empty(2, N, device=dev)
gemv_indexed_native("MXFP4", w, x, slots, out)
for cfg in _gemv_indexed_kernel.cache.values():
    print(f"AUTOTUNE-PICK BN={cfg.kwargs['BLOCK_N']} BK={cfg.kwargs['BLOCK_K']} W={cfg.num_warps}", flush=True)
PYEOF

echo "== gemv microbench ($(date +%T))"
timeout 1800 python "$OUT/bench_gemv.py" > "$OUT/gemv_bench.log" 2>&1
rc_gb=$?
echo "== gemv microbench exit $rc_gb ($(date +%T))"
grep -aE 'BEST|AUTOTUNE|SKIP|Error|error' "$OUT/gemv_bench.log" | head -n 30
echo "-- best lines per E:"
grep -a 'BEST' "$OUT/gemv_bench.log" || true
echo "-- full table tail:"
tail -n 8 "$OUT/gemv_bench.log"

# ---- full GPU test suite with the extended config list (2.9.2)
echo "== pytest tests/gpu ($(date +%T))"
timeout 2400 python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest_gpu.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest_gpu.log" | head -n 12

# ---- bench runner (tokenizer factory rebinder, as in regress.sh / S18-2)
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

M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf

# ---- C9 speed on the graph path; no profiler (hangs on graph replay, S18-2)
echo "== decode graph short ctx 8192 ($(date +%T))"
PYI_MEMDIAG="$OUT/memdiag_short.log" timeout 1800 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.decode_profile -m "$M" --mode graph \
  --ctx 8192 --steps 32 --no-profile > "$OUT/graph_short.log" 2>&1
rc_gs=$?
echo "== graph short exit $rc_gs ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|load done|prefill|wall time|store: hits' \
  "$OUT/graph_short.log" | head -n 8

echo "== decode graph prompt 2000 tok ($(date +%T))"
PYI_MEMDIAG="$OUT/memdiag_long.log" timeout 1800 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.decode_profile -m "$M" --mode graph \
  --ctx 8192 --prompt-tokens 2000 --steps 16 --no-profile > "$OUT/graph_long.log" 2>&1
rc_gl=$?
echo "== graph long exit $rc_gl ($(date +%T))"
grep -a -E 'load done|prefill|wall time|store: hits' "$OUT/graph_long.log" | head -n 8

# ---- regressions (step 4): gpt-oss golden + Gemma (C10, kernels.py touched)
echo "== regress gpt-oss-120b ($(date +%T))"
bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_rg=$?
echo "== regress gpt-oss exit $rc_rg ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning" "$OUT/regress_gptoss.log" | tail -n 10

echo "== regress gemma4 ($(date +%T))"
bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rgm=$?
echo "== regress gemma4 exit $rc_rgm ($(date +%T))"
tail -n 6 "$OUT/regress_gemma4.log"

echo "-- oom lines (job safety net):"
journalctl -k --no-pager --since "75 min ago" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || true

echo "== summary: gemv=$rc_gb pytest=$rc_pytest graph_short=$rc_gs graph_long=$rc_gl regress_gptoss=$rc_rg regress_gemma4=$rc_rgm"
echo "== end $(date +%T)"
