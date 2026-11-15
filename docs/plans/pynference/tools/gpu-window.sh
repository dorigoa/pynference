#!/bin/bash
# Finestra GPU: ferma Strata sul PC, esegue un job GPU di pynference, riporta
# i risultati sul Mac e riavvia Strata. Lo usa run-sections.sh; puoi lanciarlo
# anche a mano, dalla radice del repository.
#
#   gpu-window.sh run JOB [--no-restart] [--force]
#       esegue docs/plans/pynference/jobs/JOB.sh sul PC. Risultati:
#         docs/plans/pynference/jobs/JOB.log    output del job (tagliato a ~3000 righe)
#         docs/plans/pynference/jobs/JOB.exit   exit code
#         docs/plans/pynference/jobs/JOB.out/   copia di results/jobs/JOB/ del PC (se < 50 MB)
#       --no-restart: non riavvia Strata alla fine; --force: riesegue un job già eseguito
#   gpu-window.sh strata-start     avvia Strata se non risponde e aspetta /health
#   gpu-window.sh strata-stop      ferma Strata e aspetta che le GPU siano libere
#   gpu-window.sh strata-status    "up", "loading" oppure "down"
#
# Exit code: 0 finestra riuscita (anche se il job è fallito: vedi JOB.exit);
# 2 uso sbagliato; 3 Strata non si ferma o non riparte; 4 GPU occupate da altri.
#
# Variabili:
#   PYI_HOST (ubuntu), PYI_DIR (pynference, relativo alla home del PC)
#   PYI_SETTINGS (~/.claude/strata.json): da qui l'URL del server (ANTHROPIC_BASE_URL)
#   PYI_STRATA_START (~/Strata/run-unsloth-ud-iq4_xs-v31.sh, percorso sul PC)
#   PYI_STRATA_PATTERN (regex di pgrep -f per i processi di Strata)
#   PYI_STRATA_TIMEOUT (1800 s per tornare su), PYI_JOB_MAX_MIN (120)

set -u

HOST=${PYI_HOST:-ubuntu}
RDIR=${PYI_DIR:-pynference}
SETTINGS=${PYI_SETTINGS:-$HOME/.claude/strata.json}
STRATA_START=${PYI_STRATA_START:-'$HOME/Strata/run-unsloth-ud-iq4_xs-v31.sh'}
# (le parentesi quadre evitano che pgrep/pkill trovino la shell remota che li esegue)
STRATA_PATTERN=${PYI_STRATA_PATTERN:-'sycl/serve/server_inte[l][.]py|/Strata/engine/strat[a] --serve'}
STRATA_TIMEOUT=${PYI_STRATA_TIMEOUT:-1800}
JOB_MAX_MIN=${PYI_JOB_MAX_MIN:-120}
JOBS=docs/plans/pynference/jobs
# GPU processes that are not ours and may stay (desktop)
DESKTOP='gnome-shell|gjs|gnome-remote-desktop|Xwayland|Xorg|mutter'

say() { printf '%s [gpu-window] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "ERRORE: $2" >&2; exit "$1"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$@"; }

strata_url() {
  local u
  u=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("env",{}).get("ANTHROPIC_BASE_URL",""))' \
      "$SETTINGS" 2>/dev/null)
  echo "${u:-http://192.168.1.200:8000}"
}

strata_status() {
  local code
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$(strata_url)/health")
  if [ "$code" = 200 ]; then echo up; return; fi
  if rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null"; then echo loading; else echo down; fi
}

# Processi sulle GPU che non sono il desktop (xpu-smi ps), una riga per processo.
gpu_users() {
  rsh "xpu-smi ps 2>/dev/null | awk 'NR>1 && \$1 ~ /^[0-9]+\$/ {print \$1, \$2}' | grep -v -E '$DESKTOP' | sort -u" 2>/dev/null
}

strata_stop() {
  local i
  if ! rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null"; then
    say "Strata non è attivo"
  else
    say "fermo Strata"
    rsh "pkill -TERM -f -- 'sycl/serve/server_inte[l][.]py'" || true
    for i in $(seq 45); do rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null" || break; sleep 2; done
    if rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null"; then
      rsh "pkill -TERM -f -- '$STRATA_PATTERN'" || true
      for i in $(seq 30); do rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null" || break; sleep 2; done
    fi
    if rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null"; then
      rsh "pkill -KILL -f -- '$STRATA_PATTERN'" || true
      sleep 5
    fi
    rsh "pgrep -f -- '$STRATA_PATTERN' >/dev/null" && die 3 "Strata non si ferma"
  fi
  for i in $(seq 30); do
    [ -z "$(gpu_users)" ] && break
    sleep 2
  done
  [ -z "$(gpu_users)" ] || die 3 "dopo lo stop di Strata le GPU sono ancora occupate: $(echo $(gpu_users))"
  say "GPU libere"
}

# Altri processi sulle GPU oltre a Strata e al desktop: la finestra non parte.
check_other_users() {
  local others
  others=$(gpu_users | grep -v -E '^[0-9]+ strata' || true)
  [ -z "$others" ] || die 4 "le GPU sono usate da altri processi (non li tocco): $(echo $others)"
}

strata_start() {
  local st i
  st=$(strata_status)
  if [ "$st" = up ]; then say "Strata è già attivo"; return 0; fi
  if [ "$st" = down ]; then
    say "avvio Strata ($STRATA_START)"
    rsh "mkdir -p ~/.cache && cd ~ && setsid nohup $STRATA_START > ~/.cache/pynference-strata.log 2>&1 < /dev/null &" \
      || die 3 "avvio di Strata non riuscito"
  else
    say "Strata si sta caricando"
  fi
  for i in $(seq 0 10 "$STRATA_TIMEOUT"); do
    [ "$(strata_status)" = up ] && { say "Strata risponde su $(strata_url)"; return 0; }
    [ $((i % 120)) -eq 0 ] && [ "$i" -gt 0 ] && say "aspetto Strata ($i s)"
    if [ "$(strata_status)" = down ] && [ "$i" -ge 30 ]; then
      die 3 "Strata si è chiuso durante l'avvio: guarda ~/.cache/pynference-strata.log sul PC"
    fi
    sleep 10
  done
  die 3 "Strata non risponde dopo $STRATA_TIMEOUT s: guarda ~/.cache/pynference-strata.log sul PC"
}

run_job() {
  local job=$1 restart=$2 force=$3 script log exitf out tmin rc lines t0
  case $job in ''|*/*|.*) die 2 "nome del job non valido: $job" ;; esac
  script=$JOBS/$job.sh; log=$JOBS/$job.log; exitf=$JOBS/$job.exit; out=$JOBS/$job.out
  [ -f "$script" ] || die 2 "manca $script"
  if [ -f "$exitf" ] && [ "$force" != 1 ]; then
    say "il job $job è già stato eseguito (exit $(cat "$exitf")): niente da fare"
    return 0
  fi
  tmin=$(sed -n 's/^# *timeout: *\([0-9][0-9]*\).*/\1/p' "$script" | head -n 1)
  tmin=${tmin:-30}
  [ "$tmin" -le "$JOB_MAX_MIN" ] || tmin=$JOB_MAX_MIN

  scripts/remote.sh || die 2 "sincronizzazione con il PC non riuscita"
  check_other_users
  strata_stop
  say "eseguo il job $job (timeout $tmin min)"
  t0=$(date +%s)
  rm -f "$exitf" "$log.full"
  rsh "cd '$RDIR' && . .venv/bin/activate && rm -rf 'results/jobs/$job' && mkdir -p 'results/jobs/$job' && \
       PYI_JOB='$job' PYI_JOB_OUT='results/jobs/$job' timeout -k 60 ${tmin}m bash '$script'" > "$log.full" 2>&1
  rc=$?
  [ "$rc" = 124 ] && echo "[gpu-window] timeout dopo $tmin minuti" >> "$log.full"
  say "job $job finito: exit $rc in $(( ($(date +%s) - t0) / 60 )) min"
  # processi rimasti del job: solo i nostri (venv di pynference, llama.cpp avviato da llama_ref.sh)
  rsh "pkill -TERM -f -- '/$RDIR/[.]venv/bin/python' ; pkill -TERM -f -- 'alias pynference-re[f]'" 2>/dev/null || true
  sleep 3
  lines=$(wc -l < "$log.full" | tr -d ' ')
  if [ "$lines" -gt 3000 ]; then
    { head -n 300 "$log.full"; echo "[gpu-window] ... $((lines - 2800)) righe tolte ..."; tail -n 2500 "$log.full"; } > "$log"
    rm -f "$log.full"
  else
    mv "$log.full" "$log"
  fi
  rm -rf "$out"
  local kb
  kb=$(rsh "du -sk '$RDIR/results/jobs/$job' 2>/dev/null | cut -f1")
  if [ -n "$kb" ] && [ "$kb" -gt 4 ]; then
    if [ "$kb" -lt 51200 ]; then
      mkdir -p "$out" && rsync -az "$HOST:$RDIR/results/jobs/$job/" "$out/" \
        || say "copia di results/jobs/$job non riuscita"
    else
      say "results/jobs/$job è troppo grande ($((kb / 1024)) MB): resta solo sul PC"
    fi
  fi
  echo "$rc" > "$exitf"
  if [ "$restart" = 1 ]; then strata_start; else say "Strata resta fermo (--no-restart)"; fi
  return 0
}

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || die 2 "non sono in un repository git"
case ${1:-} in
  run)
    job=${2:-}; restart=1; force=0
    shift 2 2>/dev/null || true
    for a in "$@"; do
      case $a in --no-restart) restart=0 ;; --force) force=1 ;; *) die 2 "opzione sconosciuta: $a" ;; esac
    done
    run_job "$job" "$restart" "$force" ;;
  strata-start) strata_start ;;
  strata-stop) check_other_users; strata_stop ;;
  strata-status) strata_status ;;
  *) sed -n '2,26p' "$0"; exit 2 ;;
esac
