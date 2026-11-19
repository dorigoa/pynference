#!/usr/bin/env bash
# job: S26-10
# timeout: 60
# scopo: rerun identico di S26-9 (kill esterno: SIGTERM su torchrun ~50 s
#   dal via a caricamento riuscito, nessun errore nostro — vedi report).
#   (S26 passo 5) localizzare il primo nodo in divergenza vs dump
#   llama.cpp sul prompt del dump (5 token "The capital of France is",
#   dump S23-1) in regime produzione S26-8: TP world=2, modello completo
#   48 layer, ExpertStore reale SENZA cap di layer (84/512 residenti +
#   swap), nessun PYI_* impostato.
#   tests/cpu/s26_nodes.py xpu (torchrun):
#   1) PROBE: forward completo a 48 layer (exit mixer + lm_head +
#      all_gather, mai eseguito su GPU per questo modello) e decode di
#      1 token (gemv/swap dell'ExpertStore e QSA.decode, mai eseguiti)
#      → top5 logits e conteggio non-finiti;
#   2) replay del forward sui blocchi 0..3 con confronto di OGNI nodo
#      dumpato da cb() (tests/golden/qwen4exp/{dump_nodes,qsa_dump_nodes}
#      .json) → righe NODE e FIRST_DIVERGING (regola 2.5: modello ≠
#      golden → primo layer divergente).
# Run CPU speculare (TP1, cap 4 layer, MoE stub CPU) verde: 28/31 nodi
# OK fino ad attn_output-3; i 3 FAIL (Qcur_reshaped/normed/Qcur-3) sono
# artefatti di layout del dump (l'uscita attn_output-3 è cos 0,9989), e
# su world=2 quei nodi sono rank-local e saltati. Attesa: se ffn_out-0
# FAIL → bug store/swap; se hc_*-0/hc_init FAIL → bug embed/parity TP;
# se tutto verde → bug oltre blk.3 (blocchi 4..47, exit, lm_head, decode).
set -uo pipefail
OUT=$PYI_JOB_OUT
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

poll() { # poll PID LOG: campiona meminfo ogni 3 s in LOG finché PID vivo
  # (morto o zombie: il figlio resta zombie finché non lo si aspetta)
  local pid=$1 log=$2 st
  while st=$(awk -F') ' 'NR==1 {print substr($2, 1, 1)}' "/proc/$pid/stat" 2>/dev/null) \
         && [ -n "$st" ] && [ "$st" != "Z" ]; do
    printf '%s ' "$(date +%T)" >> "$log"
    grep -E "MemAvailable|GPUActive" /proc/meminfo | tr '\n' ' ' >> "$log"
    echo >> "$log"
    sleep 3
  done
}

summarize() { # summarize LOG: floor MemAvailable e max GPUActive (GiB)
  if [ ! -s "$1" ]; then echo "(no samples: $1)"; return; fi
  awk '/MemAvailable/ {if (m == "" || $3 < m) m = $3}
       /GPUActive/    {if ($6 > g) g = $6}
       END {printf "floor MemAvailable %.2f GiB, max GPUActive %.2f GiB\n",
            m / 2**20, g / 2**20}' "$1"
}

echo "== node replay xpu ($(date +%T))"
timeout 3000 torchrun --nproc_per_node=2 tests/cpu/s26_nodes.py xpu \
    > "$OUT/nodes.log" 2>&1 &
NPID=$!
poll "$NPID" "$OUT/meminfo_nodes.log"
wait $NPID; RN=$?
echo "== nodes exit $RN ($(date +%T)) (atteso 0)"
grep -E "PROBE|FAIL|compared|Traceback|Error" "$OUT/nodes.log" | head -n 40
echo "OK=$(grep -cE 'OK$' "$OUT/nodes.log") \
SKIP=$(grep -cE 'SKIP' "$OUT/nodes.log")"
summarize "$OUT/meminfo_nodes.log"

echo "== summary nodes=$RN $(date +%T)"
grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
