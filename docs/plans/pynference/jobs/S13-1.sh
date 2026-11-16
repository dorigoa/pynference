#!/usr/bin/env bash
# job: S13-1
# timeout: 90
# scopo: golden gpt-oss-120b: llama-server con --n-cpu-moe 3, llama_golden.py, ppl stesso --n-cpu-moe
set -uo pipefail
echo "== start $(date +%T)"
T=docs/plans/pynference/tools/llama_ref.sh
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
$T server-start $M 8091 --n-cpu-moe 3
echo "== server-start exit $?"
echo "-- swa/pattern lines from server log:"
grep -i -m 10 "swa\|pattern" "${PYI_JOB_OUT}/llama-server-8091.log" || echo "(no swa lines)"
echo "-- memory lines:"
grep -i -m 8 "VRAM\|buffer size\|model size\|offloaded" "${PYI_JOB_OUT}/llama-server-8091.log" | tail -n 8 || true
python bench/llama_golden.py --url http://127.0.0.1:8091 --out "$PYI_JOB_OUT"
echo "== golden exit $?"
$T server-stop 8091
echo "== server-stop exit $?"
$T ppl $M "$PYI_JOB_OUT/ref_ppl.log" --n-cpu-moe 3
echo "== ppl exit $?"
ls -l "$PYI_JOB_OUT"
echo "== end $(date +%T)"
