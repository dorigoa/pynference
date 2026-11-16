#!/usr/bin/env bash
# job: S11-3
# timeout: 90
# scopo: catturare finalmente i valori RES di test_quant_s11 (banda GEMV per
#        tipo, dequant, GEMV, GEMV indicizzata, tensori reali): in S11-1 il
#        tail -n 120 era stato occupato dal traceback, in S11-2 il filtro
#        grep '^RES ' ha perso quasi tutte le righe perche' i print di
#        pytest -s finiscono interlacciati ai puntini sulla stessa riga
#        (nessun newline prima di "RES"). Qui il log grezzo di pytest va in
#        $OUT e le righe RES si estraggono con grep -o senza àncora.
#        Servono a fissare i pavimenti MIN_GBPS per tipo come in S10.
#        Stessa sequenza di S11-2 (suite tests/gpu + regress gemma4), cosi'
#        l'ultimo job della sezione copre anche il requisito 2.9.2.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_s11.py > "$OUT/s11-pytest.log" 2>&1
echo "== pytest s11 exit $? ($(date +%T))"
grep -aoE 'RES [A-Za-z0-9_., =x()/-]+' "$OUT/s11-pytest.log" | head -n 40
grep -aE 'passed|failed|FAILED|error' "$OUT/s11-pytest.log" | tail -n 5

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 15 "$OUT/regress.log"

echo "== end $(date +%T)"
