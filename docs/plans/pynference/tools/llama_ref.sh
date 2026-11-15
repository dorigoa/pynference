#!/usr/bin/env bash
# llama.cpp come riferimento, SOLO dentro un job GPU (gira sul PC, GPU libere).
# Imposta il percorso di libumf (senza, il runtime oneAPI vede le GPU solo via
# OpenCL e llama.cpp si blocca sulle copie tra GPU) e marca il server con
# --alias pynference-ref, così la finestra GPU può chiuderlo se resta acceso.
#
#   llama_ref.sh server-start MODEL [PORT] [argomenti di llama-server...]
#       avvia llama-server in background (porta 8091) e aspetta /health (fino a 20 min)
#   llama_ref.sh server-stop [PORT]
#   llama_ref.sh ppl MODEL OUT_LOG [argomenti...]
#       perplexity su /usr/share/common-licenses/GPL-3, 8 blocchi da 512, batch 512
#   llama_ref.sh tokenize MODEL TEXT_FILE
#       stampa gli id (una riga Python-like) del testo, con BOS se il modello lo aggiunge
#   llama_ref.sh dump MODEL PROMPT OUT_FILE [argomenti...]
#       llama-eval-callback: tensori intermedi di una valutazione del prompt
#
# Argomenti di default per i dispositivi: --device SYCL0,SYCL1 -sm layer -ngl 999.
# Per modelli che non entrano nelle GPU aggiungi per esempio --n-cpu-moe 12.
set -euo pipefail

LL=${PYI_LLAMA:-/opt/llama.cpp}
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ONEAPI_DEVICE_SELECTOR=level_zero:*
DEV=(--device SYCL0,SYCL1 -sm layer -ngl 999)
TEXT=/usr/share/common-licenses/GPL-3

cmd=${1:-}
shift || true
case $cmd in
  server-start)
    model=$1; port=${2:-8091}; shift 2 2>/dev/null || shift $#
    log=${PYI_JOB_OUT:-/tmp}/llama-server-$port.log
    nohup "$LL/llama-server" -m "$model" "${DEV[@]}" -c 4096 --port "$port" --host 127.0.0.1 \
        --alias pynference-ref "$@" > "$log" 2>&1 < /dev/null &
    for i in $(seq 240); do
      if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then echo "llama-server up on $port"; exit 0; fi
      if ! pgrep -f -- "alias pynference-re[f]" > /dev/null; then
        echo "llama-server exited during load; last lines of $log:"; tail -n 30 "$log"; exit 1
      fi
      sleep 5
    done
    echo "llama-server not ready after 20 min"; tail -n 30 "$log"; exit 1 ;;
  server-stop)
    pkill -TERM -f -- "alias pynference-re[f]" || true
    for i in $(seq 30); do pgrep -f -- "alias pynference-re[f]" > /dev/null || exit 0; sleep 2; done
    pkill -KILL -f -- "alias pynference-re[f]" || true ;;
  ppl)
    model=$1; out=$2; shift 2
    "$LL/llama-perplexity" -m "$model" -f "$TEXT" -c 512 -b 512 --chunks 8 "${DEV[@]}" "$@" > "$out" 2>&1
    grep -E "Final estimate" "$out" ;;
  tokenize)
    model=$1; file=$2
    "$LL/llama-tokenize" -m "$model" -f "$file" --ids --log-disable 2>/dev/null | tail -n 1 ;;
  dump)
    model=$1; prompt=$2; out=$3; shift 3
    "$LL/llama-eval-callback" -m "$model" -p "$prompt" -n 1 --no-warmup "${DEV[@]}" "$@" > "$out" 2>&1
    echo "dump: $(wc -l < "$out") lines in $out" ;;
  *)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
