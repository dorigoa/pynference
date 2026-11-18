#!/usr/bin/env bash
# job: S26-2
# timeout: 50
# scopo: (ripresa S26 passo 1) capire QUANDO compare lo specchio RAM 1:1
#   nelle allocazioni torch-XPU: all'alloc (senza touch), alla copia H2D,
#   o alla scrittura kernel GPU; e se PYTORCH_ALLOC_CONF=
#   expandable_segments:True lo evita. Nota: la preparazione di S26-1 aveva
#   grepato solo libtorch_xpu.so; la conf generica PYTORCH_ALLOC_CONF e la
#   stringa expandable_segments esistono in libc10.so, libtorch_python.so e
#   libc10_xpu.so. Se il VERDICT dice nomirror=1 anche a 24 GiB: regress.sh
#   gpt-oss-120b sotto quella conf (C8, stabilità, niente alza budget qui).
set -uo pipefail
OUT=$PYI_JOB_OUT
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

cat > "$OUT/probe.py" <<'PY'
# One staged allocation probe: fresh process, env already set by the caller.
# Usage: probe.py NAME GIB
import sys
name = sys.argv[1]
gib = float(sys.argv[2])
import torch

def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        d[k] = int(v.split()[0])  # kB
    return d

if "GPUActive" not in meminfo():
    print("PROBE-ERR no GPUActive in /proc/meminfo:", sorted(meminfo())[:10])
    sys.exit(2)

# Bring up the device context BEFORE the baseline so context pages do not
# count as mirror.
_ = torch.empty(16 * 2**20, dtype=torch.uint8, device="xpu")
torch.xpu.synchronize()

try:
    print(f"PROBE {name} expandable={torch.xpu.get_allocator_settings('expandable_segments')!r}")
except Exception as exc:
    print(f"PROBE {name} conf-check unavailable: {type(exc).__name__}: {exc}")

def stage(tag, base):
    m = meminfo()
    da = (m["GPUActive"] - base["GPUActive"]) / 2**20
    dm = (m["MemAvailable"] - base["MemAvailable"]) / 2**20
    print(f"PROBE {name} stage={tag} dGPUActive={da:+.2f} dMemAvail={dm:+.2f} "
          f"reserved={torch.xpu.max_memory_reserved() / 2**30:.2f}", flush=True)
    return dm

base = meminfo()
stage("init", base)
n = int(gib * 2**30)
t = torch.empty(n, dtype=torch.uint8, device="xpu")
dm_a = stage("alloc", base)          # no touch yet
src = torch.zeros(n, dtype=torch.uint8)
t.copy_(src)                          # H2D copy, exactly the ExpertStore path
torch.xpu.synchronize()
dm_h = stage("h2d", base)
del src
t.zero_()                             # whole-buffer GPU kernel write
torch.xpu.synchronize()
dm_k = stage("kernel", base)
worst = max(dm_a, dm_h, dm_k)
print(f"VERDICT {name} nomirror={1 if worst > -2.0 else 0} worst_dMemAvail={worst:+.2f}")
PY
python -m py_compile "$OUT/probe.py" && echo "== probe.py compiles"

run_probe() { # run_probe NAME GIB [ENV=VAL ...]
  local name=$1 gib=$2
  shift 2
  echo "-- probe $name $gib GiB ${*:-env: none} ($(date +%T))"
  env "$@" python "$OUT/probe.py" "$name" "$gib" > "$OUT/probe_$name.log" 2>&1
  echo "   exit $?"
  grep -E "^PROBE|^VERDICT|Error|Warning|rror" "$OUT/probe_$name.log" | head -10
}

run_probe staged8 8
run_probe staged24 24
run_probe expandable8 8 PYTORCH_ALLOC_CONF=expandable_segments:True
run_probe expandable24 24 PYTORCH_ALLOC_CONF=expandable_segments:True

# Branch: lever only if BOTH expandable probes say nomirror.
if grep -q "^VERDICT expandable8 nomirror=1" "$OUT/probe_expandable8.log" \
   && grep -q "^VERDICT expandable24 nomirror=1" "$OUT/probe_expandable24.log"; then
  echo "== LEVER: expandable_segments nomirror — gpt-oss regress under the conf ($(date +%T))"
  grep -E "GPUActive|MemAvailable" /proc/meminfo
  PYTORCH_ALLOC_CONF=expandable_segments:True timeout 2400 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
  echo "== regress exit $? ($(date +%T))"
  grep -v -E "CCL_WARN|^W1" "$OUT/regress_gptoss.log" | tail -20
  grep -E "GPUActive|MemAvailable" /proc/meminfo
else
  echo "== NO LEVER under expandable_segments — no regress branch ($(date +%T))"
fi

grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
