#!/bin/bash
# Esegue di fila le sezioni del piano docs/plans/pynference-plan.md con Claude
# CLI e il modello locale servito da Strata (profilo ~/.claude/strata.json).
# Ogni sezione, e ogni ripresa, gira in un processo `claude -p` nuovo: il
# contesto riparte sempre da zero.
#
# Uso, dalla radice del repository (~/pynference), sul branch engine:
#   docs/plans/pynference/tools/run-sections.sh S6        # da S6 fino al prossimo ⛔
#   docs/plans/pynference/tools/run-sections.sh S9 S12    # da S9 a S12
#   docs/plans/pynference/tools/run-sections.sh --status  # stato di tutte le sezioni
#
# Finestre GPU: quando il report di una sezione dice "Stato: In attesa GPU (job
# NOME)", lo script esegue tools/gpu-window.sh run NOME (ferma Strata, esegue il
# job sul PC, riporta log e risultati, riavvia Strata) e poi riprende la sezione.
# Al massimo MAX_GPU_WINDOWS finestre per sezione.
#
# Se una sezione si interrompe a metà, la riprende finché fa progressi (nuovi
# commit o file modificati). Se Claude CLI finisce con un errore (per esempio
# del server del modello), aspetta ERROR_PAUSE secondi e controlla che Strata
# risponda (se è giù lo riavvia) prima di riprovare.
# Se il turno si chiude con un messaggio di solo ragionamento (risultato vuoto),
# lo script continua la stessa sessione con --resume, al massimo MAX_NUDGES
# volte di fila senza progressi, poi una ripresa nuova. Nelle schede con una
# sessione per passo, dopo un nuovo commit si riparte sempre da una sessione nuova.
# Si ferma quando:
#   - una sezione diventa "Bloccata" (un blocco già scritto prima del tentativo,
#     che l'agente non ha nemmeno toccato, non conta);
#   - MAX_STALLS riprese di fila non producono progressi, oppure la sezione
#     supera MAX_ATTEMPTS tentativi (le continuazioni e le finestre GPU non contano);
#   - una finestra GPU non riesce (Strata non si ferma o non riparte, GPU usate
#     da altri processi);
#   - finisce una sezione con il tuo controllo ⛔ (S6, S17, S21, S26, S28; vedi 1.7);
#   - un controllo iniziale fallisce (branch, working tree, server, altra sessione).
# Le sezioni già "Completata" vengono saltate.
#
# Variabili: PYI_SETTINGS (default ~/.claude/strata.json), MAX_STALLS (3),
# MAX_ATTEMPTS (40), MAX_NUDGES (5), MAX_GPU_WINDOWS (15), ERROR_PAUSE (60),
# PYI_RUNNER_NOTIFY=0 per niente notifiche di macOS; per Strata e il PC vedi
# tools/gpu-window.sh (PYI_STRATA_START, PYI_HOST, ...).
# Log completi: scratch/logs/runner/ (un file .jsonl per tentativo).
#
# I comandi che l'agente può usare sono in ALLOW. Se un tentativo stampa
# "permessi negati" per un comando che serve, aggiungilo lì e rilancia.

set -u

BRANCH=engine
PLAN=docs/plans/pynference-plan.md
REPORTS=docs/plans/pynference/reports
TOOLS=docs/plans/pynference/tools
JOBS=docs/plans/pynference/jobs
LOGDIR=scratch/logs/runner
SETTINGS=${PYI_SETTINGS:-$HOME/.claude/strata.json}
export PYI_SETTINGS=$SETTINGS
MAX_STALLS=${MAX_STALLS:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}
MAX_NUDGES=${MAX_NUDGES:-5}
MAX_GPU_WINDOWS=${MAX_GPU_WINDOWS:-15}
ERROR_PAUSE=${ERROR_PAUSE:-60}
NOTIFY=${PYI_RUNNER_NOTIFY:-1}
CHECKPOINTS=" 6 17 21 26 28 "
LAST_SECTION=28

ALLOW=(
  "Read" "Edit" "Write"
  "Bash(cd *)" "Bash(pwd)" "Bash(ls*)" "Bash(cat *)" "Bash(head *)" "Bash(tail *)"
  "Bash(grep *)" "Bash(egrep *)" "Bash(sed *)" "Bash(awk *)" "Bash(wc *)"
  "Bash(sort*)" "Bash(uniq*)" "Bash(cut *)" "Bash(tr *)" "Bash(find *)" "Bash(diff *)"
  "Bash(cmp *)" "Bash(du *)" "Bash(df *)" "Bash(stat *)" "Bash(file *)" "Bash(xxd *)"
  "Bash(od *)" "Bash(shasum *)" "Bash(seq *)" "Bash(basename *)" "Bash(dirname *)"
  "Bash(realpath *)" "Bash(readlink *)" "Bash(echo *)" "Bash(printf *)" "Bash(test *)"
  "Bash(true)" "Bash(date*)" "Bash(sleep *)" "Bash(which *)" "Bash(tee *)" "Bash(bc*)"
  "Bash(mkdir *)" "Bash(touch *)" "Bash(cp *)" "Bash(mv *)" "Bash(ln -s *)"
  "Bash(rm scratch/*)" "Bash(rm -f scratch/*)" "Bash(rm -rf scratch/*)"
  "Bash(uname*)" "Bash(sysctl *)" "Bash(pgrep *)" "Bash(ps *)" "Bash(curl *)"
  "Bash(git status*)" "Bash(git log*)" "Bash(git diff*)" "Bash(git show*)" "Bash(git add *)"
  "Bash(git commit *)" "Bash(git rev-parse *)" "Bash(git branch)" "Bash(git branch --show-current)"
  "Bash(git ls-files*)" "Bash(git ls-tree *)" "Bash(git cat-file *)" "Bash(git grep *)"
  "Bash(git mv *)" "Bash(git rm *)"
  "Bash(python3 *)" "Bash(scripts/cpu.sh *)"
  "Bash($TOOLS/bg.sh *)" "Bash($TOOLS/fetch_ref.sh*)"
)

DENY=(
  "Bash(ssh*)" "Bash(scp*)" "Bash(rsync*)" "Bash(scripts/remote.sh*)"
  "Bash($TOOLS/gpu-window.sh*)" "Bash($TOOLS/run-sections.sh*)" "Bash($TOOLS/llama_ref.sh*)"
  "Bash(git push*)" "Bash(git pull*)" "Bash(git fetch*)" "Bash(git reset*)" "Bash(git clean*)"
  "Bash(git stash*)" "Bash(git restore*)" "Bash(git checkout*)" "Bash(git switch*)"
  "Bash(git rebase*)" "Bash(git merge*)" "Bash(git branch -*)" "Bash(git tag*)"
  "Bash(git commit --amend*)" "Bash(git remote*)"
  "Bash(kill*)" "Bash(pkill*)" "Bash(killall*)" "Bash(sudo*)"
)

# Riassume lo stream JSON di Claude CLI: testo, comandi, errori, fine turno.
VIEWER=$(cat <<'PY'
import json, sys

def short(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"

def arg(i):
    if not isinstance(i, dict):
        return i
    return i.get("command") or i.get("file_path") or i.get("pattern") or i

def show(d, uses):
    t = d.get("type")
    msg = d.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if t == "assistant" and isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and str(p.get("text", "")).strip():
                print("   · " + short(p["text"]), flush=True)
            elif p.get("type") == "tool_use":
                uses[p.get("id")] = p.get("input")
                print("   ▸ %s: %s" % (p.get("name"), short(arg(p.get("input")), 140)), flush=True)
    elif t == "user" and isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "tool_result" and p.get("is_error"):
                c = p.get("content")
                if isinstance(c, list):
                    c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
                print("   ✗ " + short(c), flush=True)
    elif t == "system" and d.get("subtype") == "permission_denied":
        cmd = arg(uses.get(d.get("tool_use_id"), "?"))
        print("   ⚠ permesso negato, %s: %s" % (d.get("tool_name"), short(cmd, 140)), flush=True)
    elif t == "result":
        mins = (d.get("duration_ms") or 0) / 60000
        how = "errore (%s)" % (d.get("terminal_reason") or d.get("subtype")) if d.get("is_error") else d.get("subtype")
        print("   ⏹ fine turno: %s, %s turni, %.0f min" % (how, d.get("num_turns"), mins), flush=True)
        denials = d.get("permission_denials") or []
        if denials:
            print("   ⚠ permessi negati (se servono, aggiungili in ALLOW):", flush=True)
            for x in denials:
                if isinstance(x, dict):
                    print("     - %s: %s" % (x.get("tool_name"), short(arg(x.get("tool_input")), 140)), flush=True)

uses = {}
for line in sys.stdin:
    try:
        d = json.loads(line)
        if isinstance(d, dict):
            show(d, uses)
    except Exception as e:
        print("   (riga del log non riassunta: %s)" % short(e, 100), flush=True)
PY
)

# Dal log di un tentativo: l'errore con cui Claude CLI è finito, o niente.
LAST_ERROR=$(cat <<'PY'
import json, sys
last = None
try:
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            last = d
except OSError:
    pass
if last is None:
    print("Claude CLI è uscito senza risultato (exit %s)" % sys.argv[2])
elif last.get("is_error"):
    why = last.get("terminal_reason") or last.get("subtype") or "errore"
    text = " ".join(str(last.get("result") or "").split())
    print(("%s: %s" % (why, text))[:300])
PY
)

# Dal log di un tentativo: l'ID della sessione se il turno si è chiuso senza
# errore e senza testo finale (il modello ha scritto solo ragionamento), o niente.
EMPTY_END=$(cat <<'PY'
import json, sys
last = None
try:
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            last = d
except OSError:
    pass
if last and not last.get("is_error") and last.get("subtype") == "success" \
        and not str(last.get("result") or "").strip() and last.get("session_id"):
    print(last["session_id"])
PY
)

say()    { printf '%s [runner] %s\n' "$(date +%H:%M:%S)" "$*"; }
notify() {
  [ "$NOTIFY" = 1 ] || return 0
  osascript -e 'on run argv' -e 'display notification (item 1 of argv) with title "pynference runner"' \
    -e 'end run' "$1" >/dev/null 2>&1 || true
}
stop()   { say "STOP: $*" >&2; notify "STOP: $*"; exit 1; }

parse_section() {
  local n=${1#[Ss]}
  case $n in ''|*[!0-9]*) stop "sezione non valida: $1" ;; esac
  n=$((10#$n))
  { [ "$n" -ge 1 ] && [ "$n" -le "$LAST_SECTION" ]; } || stop "sezione fuori da S1–S$LAST_SECTION: $1"
  echo "$n"
}

status_of() { sed -n 's/^- Stato:[[:space:]]*//p' "$REPORTS/S$1.md" 2>/dev/null | head -n 1; }
job_of()    { status_of "$1" | sed -n 's/.*job \([A-Za-z0-9_.-]*\).*/\1/p'; }
clean()     { [ -z "$(git status --porcelain)" ]; }

auto_rules() {
  cat <<EOF
Esecuzione automatica: nessuno legge i tuoi messaggi né risponde, e i comandi
non permessi vengono rifiutati.
- Non fare domande. Non chiudere il turno prima della fine della sezione,
  tranne in due casi: dopo aver chiesto un job GPU (Parte 2.4: report "In
  attesa GPU (job NOME)", commit, una riga di saluto) e, nelle schede con una
  sessione per passo, dopo il commit di ogni passo.
- Le GPU del PC le usa il modello che ti guida: non usarle mai fuori da un job
  GPU. Sul PC lavori solo con scripts/cpu.sh; ssh, scp, rsync,
  scripts/remote.sh, gpu-window.sh e llama_ref.sh fuori dai job sono vietati.
- Dove la Parte 2 dice di chiedere o avvisare l'utente, o se un comando viene
  rifiutato e il piano non offre un'alternativa, segna "Stato: Bloccata
  (motivo)" nel report, scrivi i dettagli in "Problemi aperti", fai commit e
  fermati.
- Dopo ogni passo spunta il passo nel report, poi commit: una ripresa riparte
  dal primo passo non spuntato.
- Non scrivere mai i marker della chat con le loro parentesi, né nelle
  risposte né nel ragionamento, nemmeno quando citi codice, template o piano:
  il server li prende per segnali e il turno si interrompe. Nel ragionamento
  chiamali per nome in maiuscolo (IM_START, IM_END, ENDOFTEXT, THINK, /THINK,
  TOOL_CALL, TOOL_RESPONSE). Nei file scrivi il primo carattere di ogni token
  speciale come escape: \\x3c in Python, \\u003c in JSON (Parte 2.1).
- Se un comando viene rifiutato, prima riprova in forma più semplice: un
  comando per volta; codice Python di più righe in un file sotto scratch/
  (con Write) da lanciare con python3, invece di python3 -c.
- Comandi che possono superare 9 minuti: non usare run_in_background. Avviali con
  $TOOLS/bg.sh start NOME COMANDO ARGOMENTI...
  e poi ripeti
  $TOOLS/bg.sh wait NOME
  finché non stampa "finished". L'output completo è in scratch/logs/NOME.log.
- Vietati: kill, pkill, git reset, git clean, git stash, git checkout, git push,
  altri branch, sudo.
EOF
}

run_prompt() {
  cat <<EOF
Esegui la sezione S$1 del piano $PLAN.
Leggi prima la Parte 2 del piano e seguila alla lettera; poi leggi la scheda
S$1 nella Parte 4 e, della Parte 3, solo ciò che la scheda indica. Il piano è
lungo: trova le righe di ogni parte con
grep -n '^# Parte\|^## \|^### S' $PLAN
e leggi solo quegli intervalli.
Lavora solo sul branch $BRANCH e modifica solo i file che la scheda permette.
Dopo ogni passo aggiorna $REPORTS/S$1.md e fai commit.
Alla fine segui la Parte 2.9 (report finale, commit) e fermati: non iniziare
la sezione successiva.

EOF
  auto_rules
}

resume_prompt() {
  cat <<EOF
Riprendi la sezione S$1 del piano $PLAN.
Leggi la Parte 2 del piano (in particolare 2.4, job GPU, e 2.8, ripresa), la
scheda S$1 e il report $REPORTS/S$1.md; poi continua dal primo passo non
completato. Il piano è lungo: trova le righe di ogni parte con
grep -n '^# Parte\|^## \|^### S' $PLAN
e leggi solo gli intervalli che servono.
Se il report dice "In attesa GPU (job NOME)" e $JOBS/NOME.exit esiste, il job
è stato eseguito: leggi exit code e log (con grep e tail, non per intero),
rimetti "Stato: In corso" e prosegui.
Eccezione a 2.8: le modifiche non committate a file che la scheda S$1 permette
(e i risultati dei job in $JOBS/) vengono dalla sessione interrotta o dalla
finestra GPU: tienile, controllale e prosegui senza chiedere. Modifiche ad
altri file: Bloccata.
Se il report dice Bloccata, il problema è stato risolto: rimetti "Stato: In
corso" e riprendi dal passo che si era bloccato.

EOF
  auto_rules
}

nudge_prompt() {
  cat <<EOF
Il tuo ultimo messaggio conteneva solo ragionamento, senza testo né chiamate a
strumenti, e il turno si è chiuso. Nessuno ti ha risposto: prosegui la sezione
S$1 da dove eri, con lo strumento che avevi pianificato. Non chiudere il turno
finché la sezione non è finita, salvo dopo aver chiesto un job GPU o, se la
scheda chiede una sessione per passo, dopo il commit del passo. Valgono le
stesse regole di prima.
EOF
}

# La scheda S$1 chiede una sessione per passo?
per_step() {
  awk -v h="### S$1 — " 'index($0, h) == 1 { on = 1; next } on && /^### S/ { exit }
    on && /\*\*Una sessione per passo\.\*\*/ { found = 1 } END { exit !found }' "$PLAN"
}

# Impronta del working tree: cambia se l'agente ha modificato file.
tree_sig() { { git status --porcelain; git diff HEAD; } 2>/dev/null | shasum | cut -c1-40; }

ensure_llm() {
  "$TOOLS/gpu-window.sh" strata-start || stop "Strata non risponde e non riparte (vedi ~/.cache/pynference-strata.log sul PC)"
}

gpu_window() { # $1 = sezione
  local job
  job=$(job_of "$1")
  [ -n "$job" ] || stop "S$1: il report dice In attesa GPU ma non nomina il job"
  [ -f "$JOBS/$job.sh" ] || stop "S$1: il report chiede il job $job ma manca $JOBS/$job.sh"
  say "S$1: finestra GPU per il job $job"
  notify "Finestra GPU: $job"
  "$TOOLS/gpu-window.sh" run "$job"
  local rc=$?
  [ "$rc" = 0 ] || stop "S$1: finestra GPU per $job non riuscita (exit $rc); Strata potrebbe essere fermo"
  say "S$1: job $job eseguito (exit $(cat "$JOBS/$job.exit" 2>/dev/null)); riprendo la sezione"
}

run_claude() { # $1 = numero della sezione, $2 = prompt, $3 = sessione da continuare (opzionale)
  local log rc
  local resume=()
  [ -z "${3:-}" ] || resume=(--resume "$3")
  log="$LOGDIR/S$1-$(date +%Y%m%d-%H%M%S).jsonl"
  say "S$1: avvio Claude CLI (log completo: $log)"
  # Se il riassunto si interrompe, cat consuma il resto: claude non deve
  # ricevere SIGPIPE e morire a metà sezione.
  printf '%s\n' "$2" | claude -p --settings "$SETTINGS" ${resume[@]+"${resume[@]}"} \
      --permission-mode acceptEdits --permission-prompts none \
      --allowedTools "${ALLOW[@]}" --disallowedTools "${DENY[@]}" \
      --output-format stream-json --verbose \
    | tee "$log" | { python3 -I -X utf8 -u -c "$VIEWER"; cat >/dev/null; }
  rc=${PIPESTATUS[1]}
  say "S$1: Claude CLI terminato (exit $rc)"
  CLI_ERROR=$(python3 -I -X utf8 -c "$LAST_ERROR" "$log" "$rc" 2>/dev/null)
  [ -z "$CLI_ERROR" ] || say "S$1: il tentativo è finito con un errore: $CLI_ERROR"
  EMPTY_SID=""
  [ -n "$CLI_ERROR" ] || EMPTY_SID=$(python3 -I -X utf8 -c "$EMPTY_END" "$log" 2>/dev/null)
}

# --- Controlli iniziali ---------------------------------------------------

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || stop "non sono in un repository git"

if [ "${1:-}" = --status ]; then
  for n in $(seq 1 "$LAST_SECTION"); do
    title=$(sed -n "s/^### S$n — //p" "$PLAN" 2>/dev/null | head -n 1)
    printf 'S%-3s %-34s %s\n' "$n" "$(status_of "$n" | cut -c1-34)" "$title"
  done
  exit 0
fi
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  sed -n '2,42p' "$0"; exit 2
fi
start=$(parse_section "$1") || exit 1
end=$(parse_section "${2:-S$LAST_SECTION}") || exit 1
[ "$start" -le "$end" ] || stop "S$start viene dopo S$end"

# Variabili ereditate da un'altra sessione di Claude (per esempio il terminale
# dell'app desktop) prevalgono sul profilo: le togliamo, così server e modello
# vengono solo da $SETTINGS.
for v in $(env | sed -n -E 's/^((ANTHROPIC_|CLAUDE_CODE_|CLAUDE_AGENT_SDK_)[A-Za-z0-9_]*|CLAUDECODE)=.*/\1/p'); do
  unset "$v"
done

[ -f "$PLAN" ] || stop "manca $PLAN"
[ -f "$SETTINGS" ] || stop "manca il profilo $SETTINGS (imposta PYI_SETTINGS)"
command -v claude >/dev/null || stop "claude non è nel PATH"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || stop "il branch attivo non è $BRANCH"
if pgrep -f -- "--settings $SETTINGS" >/dev/null 2>&1; then
  stop "c'è già una sessione di Claude CLI con $SETTINGS: chiudila prima"
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 "${PYI_HOST:-ubuntu}" true 2>/dev/null \
  || stop "il PC (${PYI_HOST:-ubuntu}) non risponde via ssh"

mkdir -p "$LOGDIR"
mkdir "$LOGDIR/lock" 2>/dev/null \
  || stop "c'è già un run-sections.sh in corso (se non è vero: rmdir $LOGDIR/lock)"
trap 'rmdir "$LOGDIR/lock" 2>/dev/null' EXIT
caffeinate -i -w $$ >/dev/null 2>&1 &   # il Mac non va in stop finché lo script gira

# --- Sezioni --------------------------------------------------------------

say "Sezioni S${start}–S${end}, branch $BRANCH, profilo $SETTINGS"
for n in $(seq "$start" "$end"); do
  st=$(status_of "$n")
  case $st in Completata*) say "S$n è già completata: la salto"; continue ;; esac
  if [ "$n" -gt 1 ]; then
    prev=$(status_of $((n - 1)))
    case $prev in
      Completata*) ;;
      *) stop "S$((n - 1)) non è completata (stato: ${prev:-nessun report})" ;;
    esac
  fi
  if [ -z "$st" ]; then
    mode=run
    clean || stop "working tree non pulito prima di S$n"
  else
    mode=resume
    say "S$n ha già un report (stato: $st): la riprendo"
  fi

  finished=0
  attempt=0
  stalls=0
  nudges=0
  windows=0
  sid=""
  CLI_ERROR=""
  while :; do
    # Job GPU in attesa: prima la finestra, poi una ripresa nuova.
    case $(status_of "$n") in
      "In attesa GPU"*)
        if [ ! -f "$JOBS/$(job_of "$n").exit" ]; then
          windows=$((windows + 1))
          [ "$windows" -le "$MAX_GPU_WINDOWS" ] || stop "S$n: più di $MAX_GPU_WINDOWS finestre GPU"
          gpu_window "$n"
          stalls=0; nudges=0
        fi
        sid=""; mode=resume ;;
    esac
    ensure_llm
    if [ -n "$sid" ]; then
      was_nudge=1
      nudges=$((nudges + 1))
      say "S$n: continuo la stessa sessione (continuazione $nudges di $MAX_NUDGES senza progressi)"
      prompt=$(nudge_prompt "$n")
    else
      was_nudge=0
      attempt=$((attempt + 1))
      say "S$n: tentativo $attempt ($mode)"
      if [ "$mode" = run ]; then prompt=$(run_prompt "$n"); else prompt=$(resume_prompt "$n"); fi
    fi
    st_before=$(status_of "$n")
    head_before=$(git rev-parse HEAD)
    sig_before=$(tree_sig)
    run_claude "$n" "$prompt" "$sid"
    committed=0
    [ "$(git rev-parse HEAD)" = "$head_before" ] || committed=1
    progressed=$committed
    [ "$(tree_sig)" = "$sig_before" ] || progressed=1
    if [ "$progressed" = 1 ]; then
      stalls=0
      nudges=0
    elif [ "$was_nudge" = 0 ]; then
      stalls=$((stalls + 1))
    fi
    st=$(status_of "$n")
    sid=""
    case $st in
      Completata*)
        if clean; then finished=1; break; fi
        say "S$n: il report dice Completata, ma ci sono modifiche non committate" ;;
      Bloccata*)
        if [ "$progressed" = 0 ] && [ "$st" = "$st_before" ]; then
          say "S$n: il report dice ancora Bloccata come prima del tentativo: l'agente non l'ha ripresa"
        else
          stop "S$n: $st. Leggi $REPORTS/S$n.md e la 1.6, poi rilancia da S$n"
        fi ;;
      "In attesa GPU"*)
        say "S$n: l'agente chiede il job $(job_of "$n")"
        mode=resume
        [ "$stalls" -lt "$MAX_STALLS" ] || break
        [ "$attempt" -lt "$MAX_ATTEMPTS" ] || break
        continue ;;
      *)
        say "S$n non è finita (stato: ${st:-nessun report}; tentativi di fila senza progressi: $stalls)" ;;
    esac
    # Turno chiuso con solo ragionamento: si continua la stessa sessione, salvo
    # un passo appena committato in una scheda con una sessione per passo.
    if [ -n "$EMPTY_SID" ]; then
      if [ "$committed" = 1 ] && per_step "$n"; then
        say "S$n: passo committato, la scheda chiede una sessione per passo: sessione nuova"
      elif [ "$nudges" -lt "$MAX_NUDGES" ]; then
        say "S$n: il turno si è chiuso con solo ragionamento, senza testo né strumenti"
        sid=$EMPTY_SID
      else
        say "S$n: $MAX_NUDGES continuazioni di fila senza progressi: riparto con una sessione nuova"
        nudges=0
      fi
    fi
    if [ -z "$sid" ]; then
      [ "$stalls" -lt "$MAX_STALLS" ] || break
      [ "$attempt" -lt "$MAX_ATTEMPTS" ] || break
    fi
    if [ -n "$CLI_ERROR" ]; then
      say "S$n: riprovo fra $ERROR_PAUSE secondi"
      sleep "$ERROR_PAUSE"
    fi
    mode=resume
  done
  if [ "$finished" != 1 ]; then
    if [ "$stalls" -ge "$MAX_STALLS" ]; then why="$stalls tentativi di fila senza progressi"
    else why="raggiunto il limite di $MAX_ATTEMPTS tentativi"; fi
    [ -z "$CLI_ERROR" ] || why="$why; l'ultimo è finito con un errore: $CLI_ERROR"
    stop "S$n non completata ($why): leggi $REPORTS/S$n.md e i log in $LOGDIR"
  fi

  say "S$n completata: $(git log -1 --format='%h %s')"
  notify "S$n completata"
  case $CHECKPOINTS in
    *" $n "*)
      if [ "$n" -lt "$LAST_SECTION" ]; then
        say "⛔ Dopo S$n tocca a te: fai i controlli della 1.7, poi rilancia da S$((n + 1))"
      else
        say "⛔ Piano finito: fai i controlli finali della 1.7"
      fi
      notify "⛔ Controllo dopo S$n"
      exit 0 ;;
  esac
done
say "Fatto: S${start}–S${end} completate"
notify "S${start}–S${end} completate"
