#!/usr/bin/env bash
# job: S7-3
# timeout: 40
# scopo: ripetizione di S7-1/S7-2 dopo la fix di cli.py (riuso del prefisso
#        anche con risposta tagliata da -n; [stats] = token nuovi / token del
#        prompt). Chat CLI con --script di 3 turni greedy su Gemma (2 GPU):
#        il turno 1 prefilla tutto il prompt (nuovi == totale), i turni 2 e 3
#        solo la coda nuova; poi pytest tests/gpu e regressione Gemma C10.
set -uo pipefail
echo "== start $(date +%T)"
M=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

git -C ~/pynference log --oneline -1 || true
if ! grep -q last_truncated ~/pynference/pynference/cli.py ~/pynference/pynference/engine.py; then
    echo "== code check FAIL: ~/pynference lacks the prefix-reuse fix, abort"
    exit 1
fi
echo "== code check ok"

cat > "$OUT/script.txt" <<'EOF'
Cosa hai fatto oggi? Rispondi in una frase.
E ieri? Una frase.
Grazie, ciao!
/exit
EOF

torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
    --script "$OUT/script.txt" --temp 0 -n 128 --ctx 4096 \
    > "$OUT/chat.out" 2> "$OUT/chat.err"
echo "== chat exit $? ($(date +%T))"
grep "\[stats\]" "$OUT/chat.out" | tee "$OUT/chat_stats.txt"
tail -n 5 "$OUT/chat.err"

python - "$OUT/chat_stats.txt" <<'EOF'
import re, sys
stats = re.findall(r"new (\d+)/(\d+)", open(sys.argv[1]).read())
print("stats:", stats)
assert len(stats) == 3, f"expected 3 turns, got {len(stats)}"
assert stats[0][0] == stats[0][1] and int(stats[0][0]) > 100, \
    f"turn 1 should prefill the whole prompt: {stats[0]}"
for turn, (new, tot) in enumerate(stats[1:], start=2):
    assert int(new) < int(tot), f"turn {turn}: no prefix reuse ({new}/{tot})"
    assert int(new) < 60, f"turn {turn}: tail too big ({new})"
assert [int(t) for _, t in stats] == sorted(int(t) for _, t in stats), "total must grow"
print("== prefix reuse OK")
EOF
echo "== reuse check exit $?"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
