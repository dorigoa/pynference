#!/usr/bin/env bash
# job: S20-2
# timeout: 45
# scopo: ripetere S20 passo 4 dopo i fix di S20-1 (collisione variabili r/xv
#   nel ramo if di _conv_prefill_k; _ref_pipeline chiamato senza S in
#   _chunked_case). (1) pytest tests/gpu completo: kernel decode e prefill
#   DeltaNet contro il riferimento f64 (forme piccole, FULL 16/48/128,
#   per-rank 8/24/128, pesi reali layer 0 Qwen3.8-27B) piu' le suite delle
#   sezioni precedenti (sub-timeout tarato su S18-3: 685 s di suite);
#   (2) tempi decode/prefill su forme reali; (3) regress.sh gemma4 (C10: la
#   sezione aggiunge codice GPU, 2.9.2).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

echo "== pytest tests/gpu ($(date +%T))"
timeout 2400 python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest_gpu.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED\|^ERROR' "$OUT/pytest_gpu.log" | head -n 12

echo "== timing deltanet ($(date +%T))"
timeout 600 python - <<'PY' > "$OUT/timing.log" 2>&1
import time

import torch

from pynference import deltanet

DEV = torch.device("xpu", 0)

def bench(tag, fn, iters, warm=10):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    print(f"t {tag} {(time.perf_counter() - t0) / iters * 1000:.3f} ms", flush=True)

for H_k, H_v, S in ((16, 48, 128), (8, 24, 128)):
    C = (2 * H_k + H_v) * S
    torch.manual_seed(0)
    w = torch.randn(C, 4, device=DEV)
    g = -(torch.rand(H_v, device=DEV) + 1.0)      # decay in (e^-2, e^-1]
    beta = torch.rand(H_v, device=DEV)
    x = torch.randn(C, device=DEV)
    hist = torch.randn(3, C, device=DEV)
    st = torch.randn(H_v, S, S, device=DEV)
    bench(f"decode {H_k}/{H_v}/S{S}", lambda: deltanet.decode(x, hist, w, g, beta, st, H_k), 300)
    for T in (128, 512):
        xx = torch.randn(T, C, device=DEV)
        gg = -(torch.rand(T, H_v, device=DEV) + 1.0)
        bb = torch.rand(T, H_v, device=DEV)
        def run():
            hh = torch.randn(3, C, device=DEV)
            ss = torch.randn(H_v, S, S, device=DEV)
            deltanet.prefill(xx, hh, w, gg, bb, ss, H_k)
        bench(f"prefill {H_k}/{H_v}/S{S} T={T}", run, 20, warm=3)
print("t timing done", flush=True)
PY
rc_time=$?
echo "== timing exit $rc_time ($(date +%T))"
grep -a '^t ' "$OUT/timing.log" || tail -n 15 "$OUT/timing.log"

echo "== regress gemma4 ($(date +%T))"
timeout 600 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rgm=$?
echo "== regress gemma4 exit $rc_rgm ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning" "$OUT/regress_gemma4.log" | tail -n 8

echo "-- oom lines (job safety net):"
journalctl -k --no-pager --since "45 min ago" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || true

echo "== summary: pytest=$rc_pytest timing=$rc_time regress_gemma4=$rc_rgm"
echo "== end $(date +%T)"
