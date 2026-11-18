#!/usr/bin/env bash
# job: S26-3
# timeout: 75
# scopo: (ripresa S26 passo 1) diagnosticare il SIGABRT di regress gpt-oss
#   sotto PYTORCH_ALLOC_CONF=expandable_segments:True (S26-2) e verificare la
#   via chirurgica. 4 sonde a 2 processi torchrun+oneCCL (8 GiB/rank, ctrl
#   picco host ~32 GiB sicuro): ctrl (no conf), global (env ON dall'avvio),
#   setter (torch._C._accelerator_setAllocatorSettings ON -> 8 GiB -> OFF ->
#   allreduce nuovo), setterkeep (ON resta anche sull'allreduce nuovo).
#   Branch: global verde -> ours --ref + decode_profile graph --no-profile
#   sotto conf (log interi in $OUT); global rosso + setter verde -> via
#   chirurgica per moe.py; entrambi rossi -> nessuna leva con CCL.
set -uo pipefail
OUT=$PYI_JOB_OUT
MODEL=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
G=tests/golden/gpt-oss-120b
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable" /proc/meminfo

cat > "$OUT/ccl_probe.py" <<'PY'
# CCL + allocator-settings probe, fresh pair of processes per mode.
# Usage: ccl_probe.py MODE   (MODE = ctrl|global|setter|setterkeep)
import os
import sys

import torch
import torch.distributed as dist

mode = sys.argv[1]
rank = int(os.environ["RANK"])
dev = torch.device("xpu", int(os.environ["LOCAL_RANK"]))
torch.xpu.set_device(dev)


def settings():
    try:
        return torch._C._accelerator_getAllocatorSettings()
    except Exception as exc:
        return f"unavailable({type(exc).__name__})"


def mi():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        d[k] = int(v.split()[0])  # kB
    return d


def stage(tag, base):
    m = mi()
    print(f"MARK {mode} r{rank} {tag} "
          f"dGPUActive={(m['GPUActive'] - base['GPUActive']) / 2**20:+.2f} "
          f"dMemAvail={(m['MemAvailable'] - base['MemAvailable']) / 2**20:+.2f} "
          f"reserved={torch.xpu.max_memory_reserved() / 2**30:.2f}", flush=True)


def setg(v):
    torch._C._accelerator_setAllocatorSettings(v)
    print(f"MARK {mode} r{rank} setter:{v} settings={settings()}", flush=True)


print(f"MARK {mode} r{rank} import-ok settings={settings()}", flush=True)

dist.init_process_group("xccl", device_id=dev)
print(f"MARK {mode} r{rank} pginit-ok", flush=True)
t = torch.zeros(1024, device=dev)
dist.all_reduce(t)
torch.xpu.synchronize()
print(f"MARK {mode} r{rank} ar1-ok", flush=True)

if mode in ("setter", "setterkeep"):
    setg("expandable_segments:True")
    torch.xpu.empty_cache()  # drop cached segments so the big alloc gets a new one

base = mi()
n = int(8 * 2**30)
t = torch.empty(n, dtype=torch.uint8, device=dev)
stage("big-alloc", base)
src = torch.zeros(n, dtype=torch.uint8)
t.copy_(src)  # pageable H2D, the ExpertStore path
torch.xpu.synchronize()
stage("big-h2d", base)
del src
t.zero_()
torch.xpu.synchronize()
stage("big-kernel", base)

if mode == "setter":
    setg("expandable_segments:False")

u = torch.ones(1 << 22, device=dev)  # fresh buffer: expandable if setterkeep/global
dist.all_reduce(u)
torch.xpu.synchronize()
print(f"MARK {mode} r{rank} ar2-ok sum={u[0].item():.1f}", flush=True)
dist.barrier()
print(f"VERDICT {mode} r{rank} ok=1", flush=True)
dist.destroy_process_group()
PY
python -m py_compile "$OUT/ccl_probe.py" && echo "== ccl_probe.py compiles"

# bench/ rebinder for non-Gemma models, same as scripts/regress.sh (S17).
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

run_probe() { # run_probe NAME [ENV=VAL ...]
  local name=$1
  shift
  echo "-- probe $name ${*:-env: none} ($(date +%T))"
  timeout 300 env "$@" torchrun --nproc_per_node=2 "$OUT/ccl_probe.py" "$name" \
      > "$OUT/probe_$name.log" 2>&1
  echo "   exit $? ($(date +%T))"
  grep -E "^MARK|^VERDICT|rror|Abort|bort" "$OUT/probe_$name.log" | tail -25
}

run_probe ctrl
run_probe global PYTORCH_ALLOC_CONF=expandable_segments:True
run_probe setter
run_probe setterkeep

n_ok() { grep -c "^VERDICT $1 r[01] ok=1" "$OUT/probe_$1.log" || true; }

if [ "$(n_ok global)" = 2 ]; then
  echo "== global green: ours --ref under conf ($(date +%T))"
  grep -E "GPUActive|MemAvailable" /proc/meminfo
  PYTORCH_ALLOC_CONF=expandable_segments:True timeout 1200 \
      torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
      --ref "$G/ref_probs.json" -m "$MODEL" --decode fused -o "$OUT/ours_now.json" \
      > "$OUT/ours_global.log" 2>&1
  echo "== ours exit $? ($(date +%T))"
  grep -v -E "CCL_WARN|^W1" "$OUT/ours_global.log" | tail -n 15
  if [ -f "$OUT/ours_now.json" ]; then
    echo "== ours green: decode_profile graph --no-profile under conf ($(date +%T))"
    grep -E "GPUActive|MemAvailable" /proc/meminfo
    PYTORCH_ALLOC_CONF=expandable_segments:True timeout 1200 \
        torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
        -m "$MODEL" --mode graph --no-profile --ctx 8192 > "$OUT/decode_global.log" 2>&1
    echo "== decode exit $? ($(date +%T))"
    grep -v -E "CCL_WARN|^W1" "$OUT/decode_global.log" | tail -n 20
  fi
elif [ "$(n_ok setter)" = 2 ]; then
  echo "== global red, setter green: surgical path for moe.py is the way ($(date +%T))"
else
  echo "== NO LEVER: neither global nor setter green under CCL ($(date +%T))"
fi

grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
