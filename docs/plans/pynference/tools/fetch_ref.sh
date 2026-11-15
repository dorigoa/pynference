#!/bin/bash
# Scarica i sorgenti di riferimento di llama.cpp al commit del binario sul PC
# (2d8a32502) in scratch/ref/llama.cpp/, con gli stessi percorsi del repository.
# File già presenti non vengono riscaricati.
#
#   fetch_ref.sh                 l'elenco di default (Parte 3.5 del piano)
#   fetch_ref.sh PATH [PATH...]  altri file del repository llama.cpp
#
# File grandi (sopra 400 righe): non leggerli interi, usa grep -n e sed -n.
set -u
COMMIT=2d8a32502909ca065c41c4cf5f2fb1c427c047fc
BASE=https://raw.githubusercontent.com/ggml-org/llama.cpp/$COMMIT
cd "$(git rev-parse --show-toplevel)" || exit 2
DEST=scratch/ref/llama.cpp

DEFAULT="
src/models/gemma4.cpp
src/models/openai-moe.cpp
src/models/qwen35.cpp
src/models/qwen4exp.cpp
src/models/delta-net-base.cpp
src/models/models.h
src/llama-graph.cpp
src/llama-model.cpp
src/llama-arch.cpp
src/llama-hparams.cpp
src/llama-hparams.h
src/llama-vocab.cpp
src/llama-chat.cpp
src/unicode.cpp
ggml/include/ggml.h
ggml/src/ggml.c
ggml/src/ggml-common.h
ggml/src/ggml-quants.c
ggml/src/ggml-cpu/ops.cpp
gguf-py/gguf/quants.py
gguf-py/gguf/constants.py
tools/perplexity/perplexity.cpp
examples/eval-callback/eval-callback.cpp
models/templates/openai-gpt-oss-120b.jinja
models/templates/google-gemma-4-31B-it.jinja
"

[ $# -gt 0 ] && LIST="$*" || LIST=$DEFAULT
fail=0
for p in $LIST; do
  case $p in /*|*..*) echo "percorso non valido: $p"; fail=1; continue ;; esac
  f=$DEST/$p
  if [ -s "$f" ]; then echo "ok (già presente) $p"; continue; fi
  mkdir -p "$(dirname "$f")"
  if curl -sfL -o "$f.tmp" "$BASE/$p"; then
    mv "$f.tmp" "$f"; echo "ok $p ($(wc -l < "$f" | tr -d ' ') righe)"
  else
    rm -f "$f.tmp"; echo "MANCA $p"; fail=1
  fi
done
exit $fail
