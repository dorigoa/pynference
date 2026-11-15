#!/usr/bin/env bash
# job: S7-4
# timeout: 40
# scopo: ripetizione di S7-1/S7-3 dopo la continuazione a livello di token
#        (continuation_ids in cli.py): con il template Gemma, che non sa
#        re-renderizzare una risposta con thought channel, il turno 1
#        prefilla tutto il prompt (nuovi == totale), i turni 2 e 3 solo la
#        coda (chiusura turno + blocco utente ~20 tok); il prompt totale
#        cresce di ~128/turno perche' la risposta resta nel contesto.
#        Poi pytest tests/gpu e regressione Gemma C10.
set -uo pipefail
echo "== start $(date +%T)"
M=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

git -C ~/pynference log --oneline -1 || true
if ! grep -q continuation_ids ~/pynference/pynference/cli.py \
   || ! grep -q last_truncated ~/pynference/pynference/engine.py; then
    echo "== code check FAIL: ~/pynference lacks the token-level continuation, abort"
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
first_new, first_tot = (int(v) for v in stats[0])
assert first_new == first_tot, f"turn 1 must prefill the whole prompt: {stats[0]}"
assert first_tot >= 20, f"turn 1 prompt suspiciously small: {stats[0]}"
prev_tot = first_tot
for turn, (new, tot) in enumerate(((int(a), int(b)) for a, b in stats[1:]), start=2):
    assert new < tot, f"turn {turn}: no prefix reuse ({new}/{tot})"
    assert new < 60, f"turn {turn}: tail too big ({new})"
    assert tot - prev_tot >= 100, f"turn {turn}: answer lost from the prompt ({prev_tot}->{tot})"
    prev_tot = tot
print("== prefix reuse OK")
EOF
echo "== reuse check exit $?"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
