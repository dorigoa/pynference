#!/usr/bin/env bash
# job: S6-1
# timeout: 45
# scopo: prima finestra GPU: selftest quant/kernels; golden di Gemma da llama-server
#        (tokens, template, ref_probs, ref_ppl); nostri risultati (ours_baseline,
#        ours_ppl, decode profile) in $PYI_JOB_OUT
set -uo pipefail
echo "== start $(date +%T)"
M=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf
T=docs/plans/pynference/tools/llama_ref.sh
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pynference.quant 2>&1 | tail -n 3
echo "== quant selftest exit ${PIPESTATUS[0]}"
python -m pynference.kernels 2>&1 | tail -n 3
echo "== kernels selftest exit ${PIPESTATUS[0]}"

$T server-start $M 8091
echo "== server-start exit $?"
python bench/llama_golden.py --url http://127.0.0.1:8091 --out "$OUT" 2>&1 | tail -n 15
echo "== llama_golden exit ${PIPESTATUS[0]}"
$T server-stop 8091
echo "== server-stop exit $?"

$T ppl $M "$OUT/ref_ppl.log" > /dev/null 2>&1
echo "== ref ppl exit $?"
grep -E "Final estimate|^\|" "$OUT/ref_ppl.log" > "$OUT/ref_ppl.txt" || true
tail -n 2 "$OUT/ref_ppl.txt"

torchrun --nproc_per_node=2 bench/compare_probs.py ours --ref "$OUT/ref_probs.json" -m "$M" \
    --decode graph -o "$OUT/ours_baseline.json" 2>&1 | grep -v -E "CCL_WARN|^W1" | tail -n 12
echo "== ours exit ${PIPESTATUS[0]}"
python bench/compare_probs.py diff --strict "$OUT/ref_probs.json" "$OUT/ours_baseline.json" 2>&1 | tail -n 14
echo "== diff strict exit ${PIPESTATUS[0]}"

torchrun --nproc_per_node=2 bench/ppl.py -m "$M" -c 512 --chunks 8 > "$OUT/ours_ppl.log" 2>&1
echo "== ppl exit $?"
grep -E "Final estimate|^\[" "$OUT/ours_ppl.log" | tail -n 3 | tee "$OUT/ours_ppl.txt"

torchrun --nproc_per_node=2 bench/decode_profile.py -m "$M" --mode graph --no-profile \
    2>&1 | grep -v -E "CCL_WARN|^W1" | tail -n 8 | tee "$OUT/decode_profile.txt"
echo "== decode exit ${PIPESTATUS[0]}"

ls -la "$OUT"
echo "== end $(date +%T)"
