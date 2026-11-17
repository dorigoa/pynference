#!/usr/bin/env bash
# job: S18-2
# timeout: 60
# scopo: verifica del decode segmentato a grafi del passo 2. (1) probe B di
#   S18-1 foldata con sys.path.insert (GPUActive a 2 processi con oneCCL:
#   chiude il capitolo budget); (2) selftest kernel su xpu + tests/gpu/
#   test_kernels.py con i nuovi bias_rope/swiglu_oai; (3) timing fused
#   (controllo di non-regressione dopo le modifiche) e timing graph
#   --ctx 8192 con profilo per kernel (misura C9 + mappa per il passo 3);
#   (4) numerica: compare_probs --decode fused e graph contro
#   tests/golden/gpt-oss-120b/ref_probs.json (contratto C8) e graph vs
#   fused. Nessuna rigenerazione di ours_baseline.json in questo job.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
G=tests/golden/gpt-oss-120b

echo "== baseline meminfo (Strata fermo, GPU fredde):"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# ---- probe B folded from S18-1: the fix is sys.path.insert(0, getcwd()):
# torchrun runs this file from $OUT, where pynference is not importable.
cat > "$OUT/probe_hostmem.py" <<'PYEOF'
"""GPUActive with 2 processes + oneCCL (S18-1 probe B, fixed import path)."""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402


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


rank = int(os.environ.get("RANK", 0))
from pynference.parallel import TP  # noqa: E402
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
PYEOF

echo "== probe B: two processes with oneCCL ($(date +%T))"
timeout 300 torchrun --nproc_per_node=2 "$OUT/probe_hostmem.py" 2>&1 \
  | grep -a -E "^P[0-9] |Error|error" | tee "$OUT/probe_ccl.log" | head -n 40

echo "== meminfo after probe B:"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo

# ---- kernel selftest on device (runs _selftest on xpu:0, small + real shapes)
echo "== kernels selftest ($(date +%T))"
timeout 900 python -m pynference.kernels > "$OUT/kernels_selftest.log" 2>&1
rc_st=$?
echo "== kernels selftest exit $rc_st"
tail -n 22 "$OUT/kernels_selftest.log"

echo "== pytest tests/gpu/test_kernels.py ($(date +%T))"
timeout 1800 python -m pytest -q tests/gpu/test_kernels.py > "$OUT/tests_gpu_kernels.log" 2>&1
rc_tk=$?
echo "== pytest exit $rc_tk"
tail -n 6 "$OUT/tests_gpu_kernels.log"

# ---- bench runner (tokenizer factory rebinder, as in S18-1 / regress.sh)
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

echo "== decode_profile fused --ctx 8192 --steps 16 --no-profile ($(date +%T))"
timeout 1800 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode fused --ctx 8192 --steps 16 --no-profile > "$OUT/fused.log" 2>&1
rc_f=$?
echo "== fused exit $rc_f ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|load done|wall time|store: hits' "$OUT/fused.log" | head -n 6

echo "== decode_profile graph --ctx 8192 --steps 32 ($(date +%T))"
PYI_MEMDIAG="$OUT/memdiag_graph.log" timeout 2400 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.decode_profile -m "$M" --mode graph \
  --ctx 8192 --steps 32 > "$OUT/graph.log" 2>&1
rc_g=$?
echo "== graph exit $rc_g ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|load done|wall time|store: hits|GPU kernel time' \
  "$OUT/graph.log" | head -n 8
echo "-- graph top kernels (ms/token):"
sed -n '/kernel.*ms.token/,/^$/p' "$OUT/graph.log" | head -n 28
echo "-- graph log tail:"
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/graph.log" | tail -n 6

# ---- numerics: graph path against the llama.cpp reference (C8) and vs fused
echo "== compare_probs ours --decode fused ($(date +%T))"
timeout 1200 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode fused -o "$OUT/ours_fused.json" \
  > "$OUT/cp_fused.log" 2>&1
rc_cf=$?
echo "== ours fused exit $rc_cf ($(date +%T))"

echo "== compare_probs ours --decode graph ($(date +%T))"
timeout 1200 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode graph -o "$OUT/ours_graph.json" \
  > "$OUT/cp_graph.log" 2>&1
rc_cg=$?
echo "== ours graph exit $rc_cg ($(date +%T))"

echo "== diffs (C8 margins; strict only vs ref):"
python bench/compare_probs.py diff "$G/ref_probs.json" "$OUT/ours_fused.json" --strict \
  > "$OUT/diff_ref_fused.log" 2>&1; rc_d1=$?
echo "ref vs fused  exit $rc_d1: $(tail -n 1 "$OUT/diff_ref_fused.log")"
python bench/compare_probs.py diff "$G/ref_probs.json" "$OUT/ours_graph.json" --strict \
  > "$OUT/diff_ref_graph.log" 2>&1; rc_d2=$?
echo "ref vs graph  exit $rc_d2: $(tail -n 1 "$OUT/diff_ref_graph.log")"
python bench/compare_probs.py diff "$OUT/ours_fused.json" "$OUT/ours_graph.json" \
  > "$OUT/diff_fused_graph.log" 2>&1; rc_d3=$?
echo "fused vs graph exit $rc_d3: $(tail -n 1 "$OUT/diff_fused_graph.log")"

echo "-- kernel oom lines (job safety net):"
journalctl -k --no-pager --since "60 min ago" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || true

echo "== summary: selftest=$rc_st kernels_pytest=$rc_tk fused=$rc_f graph=$rc_g ours_f=$rc_cf ours_g=$rc_cg diff=$rc_d1/$rc_d2/$rc_d3"
echo "== end $(date +%T)"
