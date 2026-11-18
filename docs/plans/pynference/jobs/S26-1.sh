#!/usr/bin/env bash
# job: S26-1
# timeout: 30
# scopo: trovare un percorso di allocazione torch-XPU che non blocchi RAM
#   dell'host (GPUActive in /proc/meminfo). Ogni variante in un processo
#   nuovo (le variabili d'ambiente del runtime si leggono all'inizializzazione).
#   Misure: GPUActive e MemAvailable prima/dopo, torch reserved, 8 e 24 GiB.
set -uo pipefail
OUT=$PYI_JOB_OUT
V=$HOME/pynference/.venv/lib
echo "== start $(date +%T)"

echo "== baseline meminfo =="
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

echo "== xe module params =="
ls /sys/module/xe/parameters/ 2>/dev/null | tr '\n' ' '; echo

echo "== devices =="
python - <<'PY'
import torch
print("devices:", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    print(i, p.name, f"{p.total_memory / 2**30:.1f} GiB")
print("xpu.memory attrs:", [a for a in dir(torch.xpu.memory) if not a.startswith("__")])
PY
echo "== devices exit $? =="

# Which runtime env knobs actually exist in the bundled stack?
echo "== UR log env strings =="
grep -haoE "UR_LOG[A-Z_]+" $V/libur_adapter_level_zero.so.0.12.0 $V/libur_adapter_level_zero_v2.so.0.12.0 2>/dev/null | sort -u | head
echo "== UR loader env strings =="
grep -haoE "UR_ADAPTERS[A-Z_]*|UR_[A-Z]*POOL[A-Z_]*" $V/libur_loader.so.0.12.0 2>/dev/null | sort -u | head

cat > "$OUT/probe.py" <<'PY'
# One allocation probe: fresh process, env already set by the caller.
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

if name.startswith("pool"):
    pool = torch.xpu.MemPool()
    ctx = torch.xpu.use_mem_pool(pool)
    ctx.__enter__()

before = meminfo()
n = int(gib * 2**30)
t = torch.empty(n, dtype=torch.uint8, device="xpu")
t.fill_(1)  # touch every page so the driver counts real pages, not reserved VA
torch.xpu.synchronize()
after = meminfo()
res = torch.xpu.max_memory_reserved()
da = (after["GPUActive"] - before["GPUActive"]) / 2**20
dm = (after["MemAvailable"] - before["MemAvailable"]) / 2**20
print(f"PROBE {name} gib={gib:g} GPUActive {before['GPUActive'] / 2**20:.2f}"
      f" -> {after['GPUActive'] / 2**20:.2f} GiB (delta {da:+.2f})"
      f" MemAvail delta {dm:+.2f} GiB reserved {res / 2**30:.2f} GiB")
PY

probe() { # probe NAME GIB [ENV=VAL ...]
  local name=$1 gib=$2
  shift 2
  echo "-- probe $name $gib GiB ${*:-env: none}"
  env "$@" python "$OUT/probe.py" "$name" "$gib" 2>&1 | grep -E "^PROBE|^PROBE-ERR|Error" | head -3
}

# Controls first: the known 1:1 mirror, then the runtime-knob variants.
probe baseline 8
probe sysman 8 ZES_ENABLE_SYSMAN=1
probe flat 8 ZE_FLAT_DEVICE_HIERARCHY=flat
probe allocconf 8 PYTORCH_XPU_ALLOC_CONF=backend:cudaMallocAsync
probe pool 8

# Full UR level-zero trace on one probe: which USM calls does torch make?
echo "-- probe urlog 8 GiB UR_LOG_LEVEL_ZERO=debug (log -> \$OUT/urlog.log)"
UR_LOG_LEVEL_ZERO=debug python "$OUT/probe.py" urlog 8 > "$OUT/urlog.log" 2>&1
echo "urlog exit $?"
grep -E "^PROBE" "$OUT/urlog.log" | head -2
grep -iE "usmDeviceAlloc|usmHostAlloc|usmPool|pool" "$OUT/urlog.log" | head -15
echo "urlog lines: $(wc -l < "$OUT/urlog.log")"

# Big-size confirmation at 24 GiB: baseline (is the mirror still ~1:1 up
# there?) plus the env variants that looked alive at 8 GiB.
probe baseline24 24
probe sysman24 24 ZES_ENABLE_SYSMAN=1
probe flat24 24 ZE_FLAT_DEVICE_HIERARCHY=flat

grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
