# Piano: pynference, da Gemma-4-31B ai quattro modelli

- Versione: 1 (2026-11-15)
- Commit di base: `5027733` su `main` (fasi 0a e 0b)
- Branch di lavoro: **`engine`**, uno solo per tutto il piano. Il repository
  non ha remote: niente push, niente pull
- Esecutore: Claude Code CLI sul Mac Studio, con `--settings
  ~/.claude/strata.json` (agente `strata-mini`, modello Qwen3.8-Flash-Next
  servito da Strata sul PC, 262.144 token di contesto, compattazione
  automatica a 120.000)
- Report delle sezioni: `docs/plans/pynference/reports/SN.md`
- Strumenti: `docs/plans/pynference/tools/`
- Job GPU: `docs/plans/pynference/jobs/`

## In breve

pynference è un motore di inferenza in Python (PyTorch XPU, Triton, oneCCL)
per le due Intel Arc Pro B70 del PC Ubuntu, con un processo per GPU e i pesi
divisi tra le due (tensor parallel). Le sezioni S1–S5, fatte prima di questo
piano, lo portano fino a Gemma-4-31B Q8_0: ~30 token/s in generazione,
~790 token/s sul prompt, risultati identici a llama.cpp.

Questo piano aggiunge, in ordine:

1. un banco di prova: test su CPU e su GPU, golden di llama.cpp, regressione
   di Gemma;
2. un motore di generazione riusabile: sessioni con riuso del prefisso,
   campionamento, chat da terminale, server compatibile OpenAI;
3. tutti i tipi di quantizzazione GGUF presenti nei modelli scelti;
4. **GPT-OSS-120B** (MoE, 128 esperti MXFP4, esperti tra VRAM e RAM);
5. **Qwen3.8-27B** (ibrido Gated DeltaNet + attention);
6. **Qwen3.8-Flash-Next** (MoE da 512 esperti, DeltaNet, attention
   compressa con indexer, hyper-connections, embedding n-gram per layer).

**Il vincolo delle GPU.** Il modello che guida l'agente gira su Strata, che
occupa entrambe le B70. L'agente quindi non usa mai le GPU: sul PC lavora solo
con `scripts/cpu.sh` (GPU nascoste). Quando un passo richiede le GPU scrive un
**job GPU**, segna il report "In attesa GPU" e chiude il turno; lo script
`run-sections.sh` ferma Strata, esegue il job, riporta log e risultati,
riavvia Strata e riprende la sezione. Ogni finestra GPU costa il riavvio di
Strata (alcuni minuti): i job raggruppano tutto il lavoro GPU di un passo.

Il lavoro è diviso in **28 sezioni**: S1–S5 completate, S6–S28 da eseguire
in ordine, una per sessione di Claude CLI (un passo per sessione dove la
scheda lo dice). Ogni sezione finisce con il report e un commit.

## Indice

- Parte 1 — Per te: come usare il piano
- Parte 2 — Regole per l'agente
- Parte 3 — Riferimento tecnico
- Parte 4 — Le sezioni S1–S28
- Appendici A–C

## Glossario minimo

| Termine | Significato |
| --- | --- |
| PC | Il PC Ubuntu con le due B70 (`ssh ubuntu`, 192.168.1.200), dove gira il codice |
| Mac | Il Mac Studio, dove sta il repository e gira l'agente |
| Job GPU | Script bash in `docs/plans/pynference/jobs/` eseguito sul PC con le GPU libere |
| Finestra GPU | Strata fermo, job eseguito, Strata riavviato (`tools/gpu-window.sh`) |
| Golden | Risultati di riferimento prodotti da llama.cpp (o da `gguf-py`), salvati in `tests/golden/` |
| TP | Tensor parallel: ogni matrice divisa tra le due GPU, somme con all-reduce |
| EP | Expert parallel: ogni esperto MoE sta su una sola GPU (C5) |
| GEMV | Prodotto matrice-vettore: il calcolo dominante della generazione (decode) |
| Prefill | Elaborazione del prompt, molti token insieme |
| Riferimento | llama.cpp al commit `2d8a32502` (lo stesso del binario sul PC) |

---

# Parte 1 — Per te: come usare il piano

## 1.1 Le sezioni

| Sezione | Cosa fa | Job GPU | Download | Tuo controllo |
| --- | --- | --- | --- | --- |
| S1 | Ambiente del PC e misure dell'hardware | – | – | ✅ fatta |
| S2 | Lettore GGUF e tokenizer di Gemma 4 | – | – | ✅ fatta |
| S3 | Kernel Q8_0 e tensor parallel con xccl | – | – | ✅ fatta |
| S4 | Gemma-4-31B e verifica con llama.cpp | – | Gemma Q8_0 | ✅ fatta |
| S5 | Decode veloce: kernel fusi e XPUGraph | – | – | ✅ fatta |
| S6 | Banco di prova e prima finestra GPU | sì | pacchetti pip | ⛔ |
| S7 | Motore di generazione e chat da terminale | sì | – | |
| S8 | Server compatibile OpenAI | sì | – | |
| S9 | Quantizzazioni 1: registro, riferimento, blocchi da 32 | sì | gguf-py | |
| S10 | Quantizzazioni 2: K-quant | sì | – | |
| S11 | Quantizzazioni 3: MXFP4, IQ4_XS, Q2_0 | sì | – | |
| S12 | Quantizzazioni 4: IQ3_XXS, IQ2_S | sì | – | |
| S13 | GPT-OSS-120B: download, fatti, golden | sì | 59 GiB | |
| S14 | Tokenizer BPE byte-level e template chat | sì | – | |
| S15 | Esperti tra VRAM e RAM, MoE generico | sì | – | |
| S16 | GPT-OSS 1: attention con sinks, finestra, YaRN | sì | – | |
| S17 | GPT-OSS 2: MoE, modello completo, verifica | sì | – | ⛔ |
| S18 | GPT-OSS 3: velocità | sì | – | |
| S19 | Qwen3.8-27B: fatti, golden, mappa del riferimento | sì | – | |
| S20 | Gated DeltaNet: kernel | sì | – | |
| S21 | Qwen3.8-27B: modello completo e verifica | sì | – | ⛔ |
| S22 | Qwen3.8-27B: velocità | sì | – | |
| S23 | Flash-Next: fatti, golden, mappa del riferimento | sì | – | |
| S24 | Flash-Next 1: hyper-connections ed embedding n-gram | sì | – | |
| S25 | Flash-Next 2: attention compressa e indexer | sì | – | |
| S26 | Flash-Next 3: MoE, modello completo, verifica | sì | – | ⛔ |
| S27 | Flash-Next 4: velocità | sì | – | |
| S28 | Validazione finale, benchmark, documentazione | sì | – | ⛔ |

Durata indicativa: le sezioni di sola scrittura di codice qualche ora; le
schede dei modelli (S16–S17, S21, S23–S26) anche un giorno, per via dei
golden e delle finestre GPU.

## 1.2 Preparazione (una volta sola)

1. Il piano e i suoi strumenti stanno su `main`. Crea il branch di lavoro
   (o chiedi a Claude di farlo):

   ```bash
   cd ~/pynference && git checkout -b engine
   ```

   `git status` deve essere pulito: ogni sezione lo controlla.
2. Sul PC deve esistere il venv (`scripts/remote.sh scripts/setup_env.sh`,
   già fatto in S1).
3. Strata: lo script lo avvia, quando serve, con
   `~/Strata/run-unsloth-ud-iq4_xs-v31.sh` sul PC (il modello di
   `strata.json`). Se usi un altro script, esporta
   `PYI_STRATA_START='$HOME/Strata/altro-script.sh'` prima di lanciare
   `run-sections.sh`. Il log di Strata avviato dallo script è in
   `~/.cache/pynference-strata.log` sul PC.
4. Sul PC non devono esserci altri processi sulle GPU oltre a Strata e al
   desktop (per esempio un `llama-server` tuo): una finestra GPU li trova e
   si ferma senza toccarli.
5. Claude CLI sul Mac: `claude` nel PATH. Non serve altro.

## 1.3 Eseguire le sezioni

**Con lo script (consigliato).** Dalla radice del repository:

```bash
docs/plans/pynference/tools/run-sections.sh S6        # da S6 fino al prossimo ⛔
docs/plans/pynference/tools/run-sections.sh S9 S12    # da S9 a S12
docs/plans/pynference/tools/run-sections.sh --status  # stato di tutte le sezioni
docs/plans/pynference/tools/run-sections.sh --check   # PC, Strata, tunnel, risposta di prova
```

Ogni sezione gira in una sessione nuova di Claude CLI (contesto vuoto). Lo
script avvia Strata se non risponde, esegue le finestre GPU, riprende le
sezioni interrotte finché fanno progressi e si ferma dopo ogni sezione ⛔.
Le istruzioni complete sono in testa allo script; i log in
`scratch/logs/runner/`.

Claude CLI raggiunge Strata attraverso un tunnel ssh su `127.0.0.1:18000`,
aperto e chiuso dallo script: su macOS il permesso "Rete locale" a volte
smette di valere per Claude CLI ("Can't reach the API server",
FailedToOpenSocket), mentre ssh e curl continuano a funzionare.
`PYI_TUNNEL=0` usa il profilo così com'è.

**A mano.** Avvia `claude --settings ~/.claude/strata.json` dalla radice del
repository e incolla, con il numero della sezione al posto di `N`:

```text
Esegui la sezione SN del piano docs/plans/pynference-plan.md.
Leggi prima la Parte 2 del piano e seguila alla lettera; poi leggi la scheda
SN nella Parte 4 e, della Parte 3, solo ciò che la scheda indica. Il piano è
lungo: trova le righe di ogni parte con
grep -n '^# Parte\|^## \|^### S' docs/plans/pynference-plan.md
e leggi solo quegli intervalli.
Lavora solo sul branch engine e modifica solo i file che la scheda permette.
Dopo ogni passo aggiorna docs/plans/pynference/reports/SN.md e fai commit.
Alla fine segui la Parte 2.9 (report finale, commit) e fermati: non iniziare
la sezione successiva.
```

Quando l'agente si ferma con "In attesa GPU (job NOME)", chiudi Claude CLI ed
esegui la finestra:

```bash
docs/plans/pynference/tools/gpu-window.sh run NOME
```

poi riprendi la sezione con il prompt di 1.4.

## 1.4 Riprendere una sezione interrotta

Crash, contesto esaurito, Claude CLI chiuso, finestra GPU appena eseguita:
apri una sessione nuova e incolla

```text
Riprendi la sezione SN del piano docs/plans/pynference-plan.md.
Leggi la Parte 2 del piano (in particolare 2.4, job GPU, e 2.8, ripresa), la
scheda SN e il report docs/plans/pynference/reports/SN.md; poi continua dal
primo passo non completato.
```

## 1.5 Le finestre GPU

`gpu-window.sh run NOME`:

1. sincronizza il repository sul PC;
2. controlla che sulle GPU ci siano solo Strata e il desktop (altrimenti si
   ferma con exit 4, senza toccare nulla);
3. ferma Strata (SIGTERM, poi SIGKILL dopo ~2 minuti) e aspetta le GPU
   libere;
4. esegue `jobs/NOME.sh` sul PC nel venv, con il timeout scritto nel job
   (default 30 minuti, massimo 120);
5. riporta sul Mac `jobs/NOME.log` (tagliato a ~3000 righe), `jobs/NOME.exit`
   e la cartella `results/jobs/NOME/` del PC in `jobs/NOME.out/` (se sotto
   50 MB; `.out/` non va in git);
6. riavvia Strata e aspetta `/health` (fino a 30 minuti).

Altri comandi utili: `gpu-window.sh strata-status`, `strata-start`,
`strata-stop`. Se una finestra fallisce (exit 3), Strata potrebbe essere
fermo: guarda il log sul PC e riavvialo a mano.

## 1.6 Se una sezione si blocca

- **Manca qualcosa** (un file, spazio, Strata non riparte): sistemalo e
  rilancia lo script dalla stessa sezione.
- **Il piano è sbagliato** (un fatto, una formula, un contratto): fai
  correggere il piano (per esempio a Claude), commit su `engine`, poi
  rilancia. Il report spiega cosa non torna.
- **Un test non passa dopo 3 tentativi**: decidi tu se riprendere, rivedere
  la scheda o spezzarla.
- **Il tunnel ssh non parte** perché la chiave del PC non è quella in
  `~/.ssh/known_hosts` (succede dopo una reinstallazione: con una chiave che
  non torna ssh esegue i comandi, ma blocca il port forwarding). Leggi
  l'impronta sul PC con `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`,
  poi sul Mac `ssh-keygen -R 192.168.1.200` e
  `ssh -o StrictHostKeyChecking=accept-new -o ControlPath=none ubuntu true`:
  l'impronta che ssh salva deve essere la stessa. Verifica con
  `run-sections.sh --check`.

## 1.7 I tuoi controlli (⛔)

| Dopo | Controlla |
| --- | --- |
| S6 | Le due finestre GPU del report sono riuscite, Strata è ripartito da solo, i tempi delle finestre sono accettabili; la regressione di Gemma è verde |
| S17 | GPT-OSS-120B rispetta le tolleranze C8 contro llama.cpp; la chat da terminale con GPT-OSS risponde in modo sensato (prova tu con il comando del report) |
| S21 | Qwen3.8-27B rispetta C8; prova la chat con e senza ragionamento |
| S26 | Flash-Next rispetta C8; memoria e velocità del report sono plausibili |
| S28 | Tabella finale dei benchmark, prova del server con un client OpenAI; poi decidi tu il merge di `engine` in `main` |

## 1.8 Decisioni già prese

| # | Decisione |
| --- | --- |
| D1 | Formato GGUF. File dei modelli: Gemma `/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf`; GPT-OSS `/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf` (S13); Qwen3.8-27B `/nvme/gguf/Qwen3.8-27B-Q8_0.gguf`; Flash-Next `/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` (sola lettura: lo usa Strata) |
| D2 | Download approvati, e solo questi: GPT-OSS-120B da `ggml-org/gpt-oss-120b-GGUF`, file `gpt-oss-120b-MXFP4.gguf` (59 GiB, S13); sorgenti di llama.cpp al commit `2d8a32502` in `scratch/ref/` (`tools/fetch_ref.sh`); nel venv del PC i pacchetti pip `pytest`, `jinja2`, `regex` e `gguf` (gguf-py di llama.cpp allo stesso commit, S9) |
| D3 | Interfacce: comando di generazione, chat da terminale e server compatibile OpenAI (`/v1/chat/completions` con e senza streaming, `/v1/models`, `/health`), una richiesta alla volta, solo testo |
| D4 | Esclusi: decodifica speculativa e MTP (i layer `nextn` si ignorano), immagini e audio, più richieste in parallelo, modelli diversi dai quattro |
| D5 | Precisione: attivazioni fp32 nella generazione, bf16 nei prodotti del prefill, KV cache fp16. La correttezza si misura contro llama.cpp (C8) |
| D6 | Un solo branch, `engine`; un commit per passo; nessun remote, push o merge. Il merge in `main` lo decidi tu dopo S28 |
| D7 | Strata lo fermano e lo riavviano solo `gpu-window.sh` e `run-sections.sh`. L'agente non usa le GPU fuori dai job |
| D8 | Memoria: almeno 1 GB libero su ogni GPU (decisione iniziale). Budget: 30,9 GB su GPU 1, 30,0 GB su GPU 0 (che ha anche il desktop) |
| D9 | Contesto: default `--ctx 8192` in CLI e server; i modelli devono reggere almeno 32768 dove la memoria lo consente |
| D10 | Ordine dei modelli: GPT-OSS-120B, Qwen3.8-27B, Flash-Next. Gemma resta identico (regressione C10) |

---

# Parte 2 — Regole per l'agente

Questa parte vale per **tutte** le sezioni. Leggila per intero all'inizio di
ogni sessione.

## 2.1 Contesto

- Sei Claude CLI sul Mac, nella radice del repository pynference
  (`/Volumes/Home/Alvise/pynference`). Il codice gira sul PC Ubuntu, in
  `~/pynference` con il venv. Il modello che ti guida ha 262.144 token di
  contesto e la compattazione scatta a 120.000: ogni sezione ci sta se segui
  le regole di 2.3 sul contesto.
- Sul PC lavori in due modi soltanto:
  1. `scripts/cpu.sh 'COMANDO'`: sincronizza il repository ed esegue COMANDO
     sul PC, nel venv, con le GPU nascoste (`torch.xpu.device_count()` vale
     0). Per test su CPU, riferimenti numpy, lettura dei GGUF, `pip install`
     dei pacchetti di D2, download di S13. `scratch/` non viene copiato sul
     PC: uno script di servizio lo passi su stdin,
     `scripts/cpu.sh 'python - ARGOMENTI' < scratch/script.py` (niente
     kernel Triton così: quelli stanno in file del repository).
  2. Job GPU (2.4): l'unico modo di usare le GPU.
- **Le GPU le usa il modello che ti guida.** Usarle fuori da un job può far
  cadere Strata, e con lui la tua sessione. Vietati: `ssh`, `scp`, `rsync`,
  `scripts/remote.sh`, `gpu-window.sh`, `run-sections.sh`, `llama_ref.sh` fuori
  dai job, avviare o fermare Strata o llama.cpp.
- Codice, commenti, nomi, messaggi di commit: in inglese, nello stile del
  repository. Report: in italiano.
- Se un report precedente corregge esplicitamente il piano, vale il report.
- **Marker della chat.** Il modello che ti guida usa i token speciali di
  ChatML (IM_START, IM_END, ENDOFTEXT, THINK, TOOL_CALL, …): se li scrivi
  letteralmente, il server interrompe il turno. Nei file scrivi il primo
  carattere di **ogni** token speciale, di qualunque modello (Gemma, GPT-OSS,
  Qwen), come escape: `"\x3c|im_end|>"` in Python, `<` in JSON. Nel
  ragionamento chiamali per nome. Leggere file che li contengono va bene.

## 2.2 Inizio sezione (sempre, in quest'ordine)

1. Controlla l'ambiente; se un controllo fallisce, fermati (2.6):

   ```bash
   uname -s                                   # Darwin
   ls pynference/gguf.py scripts/cpu.sh       # devono esistere
   scripts/cpu.sh 'python -c "import torch; print(torch.__version__, torch.xpu.device_count())"'
   ```

   L'ultimo deve stampare la versione di torch e `0` (GPU nascoste).
2. `git status --porcelain` deve essere vuoto. Se non lo è: fermati ed elenca
   i file (non scartare modifiche di altri).
3. `git rev-parse --abbrev-ref HEAD` deve stampare `engine`.
4. Il report della sezione precedente,
   `docs/plans/pynference/reports/S<N-1>.md`, deve esistere e dire
   `Stato: Completata`. Se no, fermati.
5. Se `docs/plans/pynference/reports/SN.md` esiste già, è una ripresa: vai a
   2.8.
6. Crea il report con il modello di 2.7 (`Stato: In corso`, commit di
   partenza da `git rev-parse --short HEAD`, i passi della scheda non
   spuntati). Commit `Start the SN report`.
7. Leggi ciò che la scheda elenca in "Leggi", e nient'altro di grande. Il
   piano supera le 1.300 righe: trova gli intervalli con
   `grep -n '^# Parte\|^## \|^### S' docs/plans/pynference-plan.md` e leggi
   solo la Parte 2, la tua scheda e le voci della Parte 3 che la scheda nomina.

## 2.3 Regole durante il lavoro

**Perimetro**

- Modifica solo i file della lista "Puoi modificare" della scheda. Sempre
  permessi: il tuo report, i tuoi job `docs/plans/pynference/jobs/SN-*.sh` con
  i loro risultati, i file sotto `scratch/`. Se serve toccare altro, fermati
  (2.6).
- **Gemma non cambia.** Dopo ogni modifica a codice condiviso (`gguf.py`,
  `tokenizer.py`, `quant/`, `kernels.py`, `parallel.py`, `models/gemma4.py`,
  `engine.py`, `generate.py`) l'ultimo job GPU della sezione esegue anche la
  regressione di Gemma (C10), che deve passare.
- API: si aggiunge, non si rinomina e non si rimuove, salvo dove la scheda lo
  dice (allora aggiorna tutti i chiamanti).
- Prima corretto, poi veloce: le ottimizzazioni stanno nelle sezioni di
  velocità.
- **Non inventare API.** Prima di usare una funzione esistente trovane la
  firma: `grep -n "def nome\|class Nome" -r pynference bench`. La mappa del
  codice è in 3.3.

**Contesto (il tuo budget)**

- Non leggere per intero file sopra 300 righe: `grep -n`, poi
  `sed -n '<a>,<b>p'`. Per i sorgenti di llama.cpp vale sempre.
- Taglia l'output dei comandi (`| tail -n 60`) o mandalo in un log sotto
  `scratch/logs/` e leggine solo le righe utili.
- Scrivi nel report, sezione "Note", ogni scoperta che servirà più avanti
  (nomi, forme, decisioni). Se il contesto viene compattato, rileggi il report
  prima di continuare.

**Comandi lunghi**

- Per il tool Bash usa un timeout di 600000 ms.
- Comandi che possono superare 9 minuti (download, test lunghi su CPU):
  `docs/plans/pynference/tools/bg.sh start NOME scripts/cpu.sh '...'`, poi
  ripeti `docs/plans/pynference/tools/bg.sh wait NOME` finché stampa
  `finished`.

**Test**

- `pytest`. Test su CPU in `tests/cpu/` (sempre eseguibili con
  `scripts/cpu.sh 'python -m pytest -q tests/cpu 2>&1 | tail -n 30'`); test su
  GPU in `tests/gpu/` (solo dentro i job).
- Ogni kernel nuovo: prima il riferimento su CPU (numpy o torch su CPU, o
  `gguf-py` per le quantizzazioni), poi il test GPU contro di esso, con forme
  piccole **e** con le forme reali del modello.
- I test che usano file di modelli li saltano (`pytest.skip`) se il file
  manca.

**Git**

- Un commit per passo, messaggio in inglese all'imperativo, per esempio
  `Add the IQ4_NL dequant kernel`.
- Nessun push, nessun altro branch, niente `--amend`.
- Non committare pesi, `scratch/`, `results/`, `jobs/*.out/`, né file sopra
  i 5 MB.

**Sicurezza**

- Solo i download di D2. I file scaricati vanno in una cartella nuova e vuota
  e non si eseguono (salvo i pacchetti pip di D2).
- Sul PC scrivi solo in `~/pynference`, in `/tmp/pynference-*` e, in S13, in
  `/nvme/gguf/gpt-oss-120b/`. I file di Flash-Next sono in sola lettura.
- Mai `sudo`, `kill`, `pkill`, modifiche di sistema, processi non tuoi.
- Il server di pynference ascolta su `127.0.0.1` nei test; porta 8090 (8000
  è di Strata, 8091 di llama.cpp nei job).

## 2.4 Job GPU

1. Raggruppa in **un** job tutto ciò che il passo deve eseguire su GPU: ogni
   finestra costa il riavvio di Strata (alcuni minuti).
2. Il job è `docs/plans/pynference/jobs/SN-K.sh`, con K = 1, 2, … in ordine
   dentro la sezione; un nome non si riusa mai. Parti dal modello di
   Appendice B. La riga `# timeout: M` dà i minuti (massimo 120).
3. Il job gira sul PC, in `~/pynference`, nel venv, con Strata fermo e le GPU
   libere. `$PYI_JOB_OUT` (= `results/jobs/SN-K/`) è una cartella vuota per
   i file da riportare sul Mac (finiscono in `jobs/SN-K.out/`). Stampa righe
   brevi e un riepilogo finale: il log oltre ~3000 righe viene tagliato.
4. llama.cpp nei job solo tramite `docs/plans/pynference/tools/llama_ref.sh`
   (Appendice C).
5. Poi: nel report `- Stato: In attesa GPU (job SN-K)` e una riga nella
   tabella "Job GPU"; commit `Request GPU job SN-K`; chiudi il turno con una
   riga. Non aspettare: il job lo esegue lo script.
6. Alla ripresa (2.8) leggi `jobs/SN-K.exit` e il log con `grep`/`tail`;
   rimetti `Stato: In corso`; annota l'esito nella tabella; committa il log
   con il passo.
7. Un job non deve: terminare processi che non ha avviato, toccare Strata,
   scrivere fuori da `~/pynference` e `/tmp`, superare il suo timeout.
8. Lo stesso errore in 3 job di fila per lo stesso passo: fermati (2.6).

## 2.5 Tentativi e debug

- Se lo stesso test fallisce ancora dopo 3 tentativi di correzione: fermati.
- Kernel GPU diverso dal riferimento: riduci le forme (una riga, un blocco,
  una testa), confronta elemento per elemento e annota il primo indice che
  diverge prima di cambiare codice.
- Modello diverso dai golden: trova il primo layer che diverge. Strumenti:
  il riferimento su CPU della scheda (pochi layer, pesi dequantizzati con
  `gguf-py`) e i tensori intermedi di llama.cpp (`llama_ref.sh dump`, nomi
  come `attn_norm-0`, `ffn_out-0`: sono i nomi `cb(...)` del sorgente del
  modello).

## 2.6 Quando fermarsi (`Stato: Bloccata`)

Scrivi il motivo nel report (e i dettagli in "Problemi aperti"), fai commit e
fermati quando:

- un controllo di 2.2 fallisce;
- serve un file non elencato, un download non previsto o una decisione che il
  piano non prende;
- un fatto o un contratto del piano risulta sbagliato e correggerlo
  cambierebbe altre sezioni (correzioni locali: annotale in "Deviazioni" e
  prosegui);
- scatta la regola dei 3 tentativi (2.4, 2.5);
- la regressione di Gemma fallisce e non capisci perché.

## 2.7 Modello del report

```markdown
# SN — <titolo della scheda>

- Stato: In corso | In attesa GPU (job SN-K) | Completata | Bloccata (<motivo>)
- Commit di partenza: <hash> · Commit finale: <hash>

## Passi
- [x] 1. <titolo> — commit <hash>
- [ ] 2. <titolo>

## Job GPU
| Job | Scopo | Exit | Esito |
| --- | --- | --- | --- |

## Test
<comandi esatti, esito, numeri; per i kernel l'errore massimo e la banda>

## Deviazioni dal piano
<cosa, perché, impatto sulle sezioni successive; "Nessuna" se non ce ne sono>

## Valori misurati o fissati

## Note per le sezioni successive

## Problemi aperti
```

## 2.8 Ripresa di una sezione

1. Esegui i passi 1 e 3 di 2.2.
2. Leggi il report della sezione e `git log --oneline -15`.
3. Se il report dice `In attesa GPU (job NOME)`:
   - se `docs/plans/pynference/jobs/NOME.exit` esiste, il job è stato
     eseguito: leggi exit code e log, rimetti `Stato: In corso` e prosegui
     il passo;
   - se non esiste, il job non è ancora stato eseguito: chiudi il turno con
     una riga ("In attesa GPU: job NOME").
4. Se `git status` mostra modifiche non committate: i risultati dei job
   (`jobs/NOME.log`, `jobs/NOME.exit`) sono attesi, tienili. Altre modifiche:
   non scartarle da solo, elencale e fermati (2.6), salvo l'eccezione del
   prompt automatico.
5. Se il report dice `Bloccata`, il problema è stato risolto: rimetti
   `Stato: In corso` e riprendi dal passo bloccato.
6. Riparti dal primo passo non spuntato. Non rifare i passi spuntati.

## 2.9 Fine sezione

1. Tutti i passi spuntati. Test su CPU verdi:
   `scripts/cpu.sh 'python -m pytest -q tests/cpu 2>&1 | tail -n 20'`.
2. Se la sezione ha toccato codice GPU o condiviso: l'ultimo job della
   sezione ha eseguito `python -m pytest -q tests/gpu` e la regressione di
   Gemma (C10), entrambi verdi.
3. Report: `Stato: Completata`, commit finale, test, deviazioni, valori,
   note.
4. Commit `Complete the SN report`; `git status --porcelain` vuoto.
5. Scrivi un riepilogo di 3–5 righe e "Prossima sezione: S<N+1>". Non
   iniziarla.

---

# Parte 3 — Riferimento tecnico

Fatti, convenzioni e contratti. Le sezioni leggono solo le voci che la scheda
nomina.

## 3.1 Macchine e misure

| Cosa | Valore |
| --- | --- |
| PC | Ryzen 9 5950X (16 core, SMT spento, AVX2), 107 GB DDR4-2133 (XMP non parte), MSI MEG X570 ACE, Ubuntu 26.10, kernel 7.3 |
| GPU | 2× Intel Arc Pro B70 (BMG-G31, 31,89 GiB, 256 EU), PCIe 4.0 x8 ciascuna, collegate solo attraverso l'host |
| Disco | `/nvme`, 1,8 TB (1,3 TB liberi a novembre 2026) |
| Software | Python 3.13.13 (venv `~/pynference/.venv`), torch 2.15.0+xpu, triton-xpu 3.9.0, oneCCL 2022.1.2 (backend `xccl`), Level Zero 1.32, compute-runtime 26.27 |
| Mac | Mac Studio, repository in `/Volumes/Home/Alvise/pynference`, `python3` di sistema (senza torch) |

Misure (`docs/phase0a.md`, `docs/phase0b.md`):

| Misura | Valore |
| --- | --- |
| RAM → una GPU (memoria pinned) / → entrambe | 14,4 / 28,7 GB/s |
| Lettura della memoria della GPU | 568 GB/s |
| GEMV Q8_0 Triton (forme di Gemma) | 554–592 GB/s |
| All-reduce xccl fino a 256 KB, ordinato sullo stream | ~36 µs |
| Lancio di un'operazione: normale / da grafo | 5,3 / 0,74 µs |
| Gemma-4-31B Q8_0: decode torch / kernel fusi / XPUGraph | 10,8 / 21,6 / 30,3 token/s |
| Gemma-4-31B Q8_0: prefill di 1964 token | 786 token/s |

## 3.2 Esecuzione sul PC

- `scripts/remote.sh [COMANDO]`: rsync del repository verso `ubuntu:~/pynference`
  (esclusi `.git`, `.venv`, `results/`, `scratch/`, `jobs/*.out/`), poi COMANDO
  nel venv **con le GPU visibili**. Lo usano solo l'utente e `gpu-window.sh`.
- `scripts/cpu.sh 'COMANDO'`: come sopra, con `ZE_AFFINITY_MASK=99` e
  `ONEAPI_DEVICE_SELECTOR=opencl:cpu` (GPU nascoste). L'agente usa solo
  questo.
- Il venv, all'attivazione, aggiunge i file di sviluppo di Level Zero
  (`CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`): servono a Triton e a oneCCL.
  Senza venv attivo né Triton né xccl funzionano.
- Due GPU: `torchrun --nproc_per_node=2 -m pynference.generate ...`; un
  processo per GPU, rank 0 su `xpu:0`.
- llama.cpp: `/opt/llama.cpp` (build 12360, commit `2d8a32502`), solo nei job
  e solo tramite `tools/llama_ref.sh`, che aggiunge
  `LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.2/lib`: senza, il runtime oneAPI
  vede le GPU solo via OpenCL e llama.cpp si blocca sulle copie tra GPU.
- Testo per la perplexity: `/usr/share/common-licenses/GPL-3` (7563 token con
  il tokenizer di Gemma).
- Triton: i kernel `@triton.jit` devono stare in un file `.py` (non in
  `python -c` o stdin); la prima chiamata compila (secondi); l'autotune
  sceglie una configurazione per forma.
- oneCCL: all-reduce grandi su buffer sempre diversi finiscono con
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Tutte le collettive passano dai buffer
  persistenti di `TP` (`parallel.py`).
- XPUGraph: la cattura include gli all-reduce di xccl. Prima della cattura
  ogni kernel va eseguito almeno una volta (compilazione e autotune); gli
  input variabili stanno in tensori sul dispositivo aggiornati prima di ogni
  replay; le uscite sono tensori statici sovrascritti dal replay successivo.

## 3.3 Mappa del codice (commit di base)

| File (righe) | Contenuto e API |
| --- | --- |
| `pynference/gguf.py` (179) | `GGML_TYPES` nome → (id, elementi per blocco, byte per blocco); `TensorInfo` (`name`, `shape` in ordine ggml con `shape[0]` contiguo, `type`, `nbytes`, `n_rows`, `row_bytes`); `GGUFFile(path)` (anche file divisi): `.kv`, `.tensors`, `.arch`, `.get(key)` (cerca prima `<arch>.key`), `.raw(name)` → uint8, `.rows(name, r0, r1)` → `[r1-r0, row_bytes]` uint8; `python -m pynference.gguf FILE` stampa i metadati |
| `pynference/tokenizer.py` (175) | `Tokenizer(g)` solo per `tokenizer.ggml.model == "gemma4"`: `.encode(text, add_bos=None, parse_special=True)`, `.decode(ids)`, `.token_bytes(id)`, `.eog_ids`, `.bos_id`; `StreamDecoder(tok).push(id) -> str`; `chat_prompt(messages, thinking=False)` (formato Gemma 4, senza BOS) |
| `pynference/quant.py` (148) | `q8_0_to_device(raw, n, k, device) -> (qs int8 [n,k], d fp16 [n,k/32])`; `QLinear(qs, d)`: `from_q8_0(raw, n, k, device)`, `__call__(x [M,K]) -> [M,N] fp32` (M = 1 GEMV Triton, M > 1 dequant bf16 + `F.linear`), `dequant()`, `rows_dequant(idx)`, `nbytes`; kernel `_q8_gemv` (autotune), `_q8_dequant`; `python -m pynference.quant` = selftest |
| `pynference/kernels.py` (273) | Kernel fusi del decode: `rms_norm(x, w, eps)`, `add_norm(x, a, w1, eps, scale=None, w2=None)` → `x + rmsnorm(a)·w1` (·scale) e rmsnorm successiva, `norm_rope(x, w, eps, cos, sin, pos, out=None, slot=None)` (RMSNorm per testa + RoPE NEOX, scrittura diretta nella KV cache), `gelu_mul(gu)`, `attn_decode(q, k_cache, v_cache, n_len, n_split, split)` (GQA, scala 1, split sulle chiavi + combinazione); `python -m pynference.kernels` = selftest |
| `pynference/parallel.py` (77) | `TP()`: `.world`, `.rank`, `.device`, `.is_main`, `.split(n)`, `.all_reduce(t)` (in place), `.all_gather(t)` (concatena l'ultima dimensione), `.barrier()`, `.close()`; buffer persistenti per le collettive |
| `pynference/models/__init__.py` (10) | `load_model(g, tp, **kw)`: sceglie la classe da `g.arch` |
| `pynference/models/gemma4.py` (404) | `Gemma4Config.from_gguf`; `rms_norm`, `rope_neox`, `rope_table(n_rot, base, factors, n_pos, device)`; `Gemma4(g, tp, max_ctx=8192)`: `forward(tokens, chunk=512, all_logits=False)`, `decode(token)` (kernel fusi; XPUGraph se `use_graph`), `reset()`, `kv_bytes()`, `n_past`, `fast_decode`, `use_graph`, `load_seconds` |
| `pynference/generate.py` (131) | CLI di generazione: `Sampler(temp, top_k, top_p, seed)`, `top_probs`, `main()` (opzioni `-m -p --chat --system -n --temp --top-k --top-p --seed --ctx --decode --n-probs`) |
| `bench/compare_probs.py` (138) | `fetch` (da llama-server), `ours` (`--decode`, `--tokenwise`), `diff`; prompt Gemma in `PROMPTS` |
| `bench/ppl.py` (68) | perplexity come `llama-perplexity` (`-c 512 --chunks 8`) |
| `bench/decode_profile.py` (76) | ms/token del decode (`--mode torch|fused|graph`, `--no-profile`) |
| `scripts/verify_gemma4.sh` (43) | confronto completo con llama.cpp per Gemma (fase 0b) |

## 3.4 Tensor parallel: convenzioni

- Due processi, rank r su `xpu:r`; `tp.split(n)` = `[r·n/2, (r+1)·n/2)`.
- Q, K, V: righe divise per teste e unite in una sola matrice per rank
  (`[q_r; k_r; v_r]`). Proiezione d'uscita: divisa per colonne d'ingresso,
  poi `all_reduce`.
- FFN: righe di gate e up divise e unite (`[gate_r; up_r]`); down divisa per
  colonne d'ingresso, poi `all_reduce`.
- Embedding: righe del vocabolario divise; lookup con maschera, poi
  `all_reduce`. Testa d'uscita: righe del vocabolario divise, poi
  `all_gather` dei logit; ogni rank ha gli stessi logit fp32 completi.
- Bias: nei layer divisi per righe (q, k, v, gate, up) il bias si divide
  allo stesso modo; nei layer divisi per colonne (o, down) il bias lo somma
  solo il rank 0, prima dell'`all_reduce`.
- Un taglio lungo K deve cadere su un multiplo del blocco del tipo (32, 64
  per Q2_0, 256 per K-quant e IQ a 256). Se non cade, per i pesi densi
  fermati (2.6); per gli esperti vale EP (C5).
- Norme e tensori piccoli: replicati su ogni rank.
- Campionamento: ogni rank campiona lo stesso token dagli stessi logit con lo
  stesso seme (nessun messaggio in più). I comandi (prompt, parametri)
  arrivano dal rank 0 (C4).

## 3.5 Riferimenti

`docs/plans/pynference/tools/fetch_ref.sh` scarica in `scratch/ref/llama.cpp/`
i sorgenti di llama.cpp al commit del binario (`2d8a32502`). Dove cercare:

| Cosa | Dove |
| --- | --- |
| Grafo di un modello; nomi dei tensori intermedi (`cb(cur, "nome", il)`) | `src/models/<arch>.cpp` (`gemma4`, `openai-moe`, `qwen35`, `qwen4exp`, `delta-net-base`) |
| Iperparametri, pattern della finestra, tipo di RoPE (`llama_model_rope_type`), `get_rope_params` | `src/llama-model.cpp`, `src/llama-hparams.cpp`, `src/llama-hparams.h` |
| Attention (`build_attn_mha`: scala, sinks, maschere), MoE (`build_moe_ffn`: gating, pesi, attivazioni), `build_ffn` | `src/llama-graph.cpp` |
| Formule di RoPE e YaRN (`rope_yarn`, `ggml_rope_cache_init`), Gated DeltaNet (`gated_delta_net`), convoluzione (`ssm_conv`) | `ggml/src/ggml-cpu/ops.cpp`; `ggml_rope_yarn_corr_dims` in `ggml/src/ggml.c` |
| Formati di quantizzazione: strutture, griglie, tabelle | `ggml/src/ggml-common.h`; `dequantize_row_*` in `ggml/src/ggml-quants.c`; numpy in `gguf-py/gguf/quants.py` |
| Tokenizer: regex di pre-tokenizzazione per tipo (`LLAMA_VOCAB_PRE_TYPE_*`), BPE, token di fine generazione | `src/llama-vocab.cpp`, `src/unicode.cpp` |
| Template chat | chiave GGUF `tokenizer.chat_template`; `models/templates/*.jinja` |
| Metodo della perplexity | `tools/perplexity/perplexity.cpp`, funzione `perplexity` |
| Dump dei tensori intermedi | `examples/eval-callback/eval-callback.cpp` |

Riferimento numerico per le quantizzazioni: `gguf-py` installato nel venv
(S9): `from gguf.quants import dequantize` e
`gguf.GGMLQuantizationType`; `dequantize(raw_uint8, qtype)` restituisce
float32 con la forma logica.

## 3.6 Contratti condivisi

**C1 — File e cartelle.**

| Cosa | Dove |
| --- | --- |
| Quantizzazioni (S9–S12) | `pynference/quant/` (pacchetto; `quant.py` diventa `quant/q8_0.py`, `from pynference.quant import QLinear` continua a funzionare) |
| Motore, campionamento, chat, CLI, server (S7, S8, S14) | `pynference/engine.py`, `sampling.py`, `chat.py`, `cli.py`, `server.py` |
| Esperti e MoE (S15) | `pynference/moe.py` |
| Gated DeltaNet (S20) | `pynference/deltanet.py` |
| Modelli | `pynference/models/gemma4.py`, `gptoss.py`, `qwen35.py`, `qwen4exp.py` |
| Test | `tests/cpu/`, `tests/gpu/`, golden in `tests/golden/<modello>/` |
| Script di confronto | `bench/compare_probs.py`, `bench/llama_golden.py` (S6), `bench/ppl.py`, `bench/decode_profile.py` |
| Regressione | `scripts/regress.sh` (S6) |
| Solo locale (fuori git) | `scratch/` sul Mac; `results/` sul PC |

**C2 — Pesi quantizzati.** In `pynference/quant/`:

```python
TYPES: dict[str, QType]                        # QType(name, ggml_id, block, block_bytes)
def row_bytes(qtype: str, k: int) -> int
def check_split(qtype: str, k: int) -> None    # ValueError se k non è multiplo del blocco
def make_qweight(raw: np.ndarray, qtype: str, n: int, k: int, device) -> "QWeight"
                                               # raw: [n, row_bytes] uint8 nel layout GGUF
class QWeight:                                 # QLinear (Q8_0) è una QWeight
    qtype: str; N: int; K: int; nbytes: int
    def __call__(self, x: torch.Tensor) -> torch.Tensor        # [M, K] → [M, N] float32
    def dequant(self, out=None) -> torch.Tensor                # [N, K] bf16
    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor  # [len(idx), K] float32
```

M = 1: GEMV Triton con x in float32; M > 1: dequantizzazione in un buffer
bf16 condiviso e `F.linear` in bf16. Ogni tipo ha anche una **GEMV
indicizzata** nel layout nativo GGUF, per gli esperti (C5):

```python
def gemv_indexed(qtype, w_stack, x, slots, out) -> None
# w_stack: [S, N, row_bytes] uint8 (layout GGUF), x: [E, K] float32,
# slots: [E] int32 sul dispositivo, out: [E, N] float32;  out[e] = W[slots[e]] @ x[e]
```

F16/BF16: GEMV Triton che legge pesi a 16 bit; F32: `torch`. Per i pesi densi
un tipo può usare un layout ripacchettato (Q8_0 lo fa); gli esperti restano
nel layout GGUF, così si copiano dalla RAM byte per byte.

**C3 — Interfaccia dei modelli.** Ogni classe di `models/` espone:

```python
cfg; tp; max_ctx: int; n_past: int; load_seconds: float
fast_decode: bool; use_graph: bool
def forward(self, tokens: list[int], chunk: int = 512, all_logits: bool = False) -> torch.Tensor
    # aggiunge i token al contesto; logit float32 [n_vocab] dell'ultimo, o [T, n_vocab]
def decode(self, token: int) -> torch.Tensor   # [n_vocab]; con use_graph è un buffer statico
def reset(self) -> None                        # contesto vuoto, senza riallocare
def kv_bytes(self) -> int
def memory_report(self) -> dict                # byte per GPU: pesi, esperti, cache, scratch (da S15)
```

I logit sono uguali su tutti i rank, con il soft-cap già applicato. Il
contesto si può solo allungare: cache a finestra (anello) e stati ricorrenti
non si possono riportare indietro.

**C4 — Canale di controllo e sessioni.**

- `TP.broadcast_obj(obj)`: il rank 0 manda un oggetto Python (pickle) a tutti
  gli altri, su un gruppo `gloo` creato da `TP` quando `world > 1`; con un
  solo processo restituisce `obj`.
- `engine.Session(model, tok)`: `tokens` (il contesto del modello),
  `prefill(ids) -> logits` (se `ids` inizia con `tokens` aggiunge solo il
  resto, altrimenti `reset` e tutto da capo), `generate(ids, max_tokens,
  sampling, stop_ids, stop_strings=(), on_token=None) -> list[int]`.
- `engine.Engine`: sul rank 0 `run(cmd)` manda il comando e lo esegue; sugli
  altri rank `follow()` riceve ed esegue gli stessi comandi finché arriva
  `{"op": "stop"}`. Comandi: `{"op": "generate", "ids", "max_tokens",
  "sampling": {"temp", "top_k", "top_p", "min_p", "seed"}, "stop_ids"}`,
  `{"op": "reset"}`, `{"op": "stop"}`.
- `sampling.sample(logits, temp, top_k, top_p, min_p, generator) -> int`, su
  CPU in float32; `temp <= 0` = greedy. Deterministico dato il seme.

**C5 — Esperti (MoE).**

- **Expert parallel**: l'esperto `e` vive sul rank `e % world`, intero (gate,
  up, down). Le parti dense restano in TP. Ogni rank calcola i suoi esperti
  tra quelli scelti; la somma tra i rank è l'`all_reduce` dopo il MoE.
- Il router (pesi piccoli) è replicato: ogni rank calcola gli stessi logit e
  gli stessi top-k.
- `moe.ExpertStore(g, tp, budget_bytes)`: per ogni layer e proiezione un
  buffer residente `[S_res, N, row_bytes]` nel layout GGUF, più `S_swap` slot
  di scambio; gli esperti non residenti stanno in memoria pinned sull'host.
  `resolve(layer, ids) -> slots` (tensore int32 sul dispositivo) copia sul
  GPU, su uno stream laterale, gli esperti mancanti; `stats()` dà hit, miss e
  byte copiati.
- Residenza al caricamento: si riempie il budget per GPU (D8, meno parti
  dense, cache e scratch) esperto per esperto, layer per layer; il resto in
  RAM. RAM pinned: al massimo 80 GB in tutto (Strata è fermo quando giriamo).
- Decode: router sulla GPU → top-k sull'host (una sincronizzazione per layer
  MoE) → `resolve` → GEMV indicizzate (C2) → attivazione → somma pesata →
  `all_reduce`. Prefill: token raggruppati per esperto.

**C6 — Job GPU.** Formato in 2.4 e Appendice B. Nome `SN-K`, file
`docs/plans/pynference/jobs/SN-K.sh`; risultati `SN-K.log`, `SN-K.exit`,
`SN-K.out/`.

**C7 — Golden.** In `tests/golden/<modello>/` (`gemma4`, `gpt-oss-120b`,
`qwen35-27b`, `qwen4exp`):

| File | Contenuto | Fonte |
| --- | --- | --- |
| `facts.json` | iperparametri, tensori, tipi, dimensioni | `GGUFFile` |
| `tokens.json` | `[{"text", "ids"}]` per i casi di `tests/golden/tokenizer_cases.json` | llama-server `/tokenize` |
| `template.json` | `[{"name", "messages", "kwargs", "prompt"}]` | llama-server `/apply-template` |
| `ref_probs.json` | formato di `compare_probs.py` (id del prompt, 24 passi greedy con le top-10 probabilità) | llama-server `/completion` |
| `ref_ppl.txt` | riga "Final estimate" e valori per blocco | `llama_ref.sh ppl` |
| `ours_baseline.json`, `ours_ppl.txt` | i nostri risultati alla chiusura della sezione di verifica | pynference |

`tests/golden/prompts.json` (S6) contiene i prompt comuni:
`[{"name", "messages"}]` per i prompt chat e `[{"name", "text"}]` per quelli
grezzi.

**C8 — Tolleranze contro llama.cpp** (stesso GGUF):

- tokenizer: id identici su tutti i casi;
- template: stringhe identiche;
- perplexity (8 blocchi da 512 di GPL-3, batch 512): scarto ≤ 1%;
- greedy: per ogni prompt si confrontano i passi fino alla prima divergenza;
  una divergenza al passo i è ammessa solo se nel riferimento la differenza
  tra la prima e la seconda probabilità a quel passo è < 0,15 (quasi
  pareggio); almeno 3 prompt su 4 identici per tutti i 24 passi (o fino a
  fine generazione);
- KL parziale media sulle top-10 (`compare_probs.py diff`) ≤ 1e-2 per ogni
  prompt, sui passi confrontati.

**C9 — Velocità (obiettivi, non bloccanti).** Misurati con
`bench/decode_profile.py --mode graph` (contesto ~60 token, `--ctx 8192`) e
con un prompt di ~2000 token:

| Modello | Decode | Prefill |
| --- | --- | --- |
| Gemma-4-31B Q8_0 | ≥ 28 token/s (oggi 30,3) | ≥ 700 token/s |
| GPT-OSS-120B MXFP4 | ≥ 25 token/s con esperti residenti; con miss, riportare il tasso | ≥ 300 token/s |
| Qwen3.8-27B Q8_0 | ≥ 25 token/s | ≥ 500 token/s |
| Flash-Next IQ4_XS | ≥ 15 token/s | ≥ 200 token/s |

**C10 — Regressione di Gemma.** `scripts/regress.sh gemma4` (S6), solo dentro
un job: confronto greedy con `tests/golden/gemma4/ours_baseline.json` (token
identici, |Δp| massimo ≤ 1e-3 sulle top-10) e perplexity con
`ours_ppl.txt` (scarto ≤ 0,1%); stampa `PASS` o `FAIL` ed esce con 0 o 1.
Le sezioni di verifica dei modelli aggiungono a `regress.sh` il proprio
modello; S28 li esegue tutti.

## 3.7 Formati di quantizzazione

| Tipo | id | Elementi / byte per blocco | Struttura (vedi `ggml-common.h`) | Nei nostri modelli |
| --- | --- | --- | --- | --- |
| F32, F16, BF16 | 0, 1, 30 | 1 / 4, 2, 2 | valori | norme (F32); Flash-Next 0,04 GiB BF16 |
| Q8_0 | 8 | 32 / 34 | `d` fp16 + 32 int8; x = d·q | Gemma, Qwen-27B, Flash-Next 1,3 GiB |
| Q4_0 | 2 | 32 / 18 | `d` + 16 byte di nibble; x = d·(q−8) | Qwen3.8-27B-Q4_0 (solo test) |
| Q4_1 | 3 | 32 / 20 | `d`, `m` + nibble; x = d·q + m | solo test |
| Q5_0 | 6 | 32 / 22 | `d`, `qh[4]` (quinto bit), `qs[16]`; x = d·(q−16) | Flash-Next 17,7 GiB |
| Q5_1 | 7 | 32 / 24 | `d`, `m`, `qh`, `qs`; x = d·q + m | solo test |
| IQ4_NL | 20 | 32 / 18 | `d` + nibble; x = d·`kvalues_iq4nl`[q] | Flash-Next 32,5 GiB |
| Q2_K … Q6_K | 10–14 | 256 / 84, 110, 144, 176, 210 | super-blocchi con scale a 4–8 bit per sotto-blocco | Qwen3.8-27B-UD-Q4_K_XL (test), Flash-Next Q6_K 2,2 GiB |
| IQ4_XS | 23 | 256 / 136 | `d`, `scales_h`, `scales_l[4]`, `qs[128]`; 8 sotto-blocchi da 32 con scala a 6 bit; `kvalues_iq4nl` | Flash-Next 14,4 GiB |
| MXFP4 | 39 | 32 / 17 | `e` (E8M0) + nibble E2M1 (`kvalues_mxfp4`) | GPT-OSS (esperti) |
| Q2_0 | 42 | 64 / 18 | `d` + 64 valori a 2 bit | Flash-Next 0,44 GiB |
| IQ3_XXS | 18 | 256 / 98 | `d`, `qs` con indici in `iq3xxs_grid` e segni | Flash-Next 16,8 GiB |
| IQ2_S | 22 | 256 / 82 | `d`, `qs`, `qh`, `scales`; `iq2s_grid` e segni | Flash-Next 1,5 GiB |

Le formule esatte sono in `dequantize_row_<tipo>` di `ggml-quants.c`; il
riferimento numerico è `gguf-py` (S9). Q1_0, NVFP4, F8_E4M3, TQ1_0 e TQ2_0
esistono in `gguf.py` ma non servono.

## 3.8 GPT-OSS-120B (da confermare in S13)

File `ggml-org/gpt-oss-120b-GGUF` → `gpt-oss-120b-MXFP4.gguf`, 59,03 GiB.
Riferimento `src/models/openai-moe.cpp` (arch `LLM_ARCH_OPENAI_MOE`).

| Parametro | Valore atteso |
| --- | --- |
| hidden / layer | 2880 / 36 |
| teste | 64 query, 8 KV, dimensione 64 |
| finestra | 128, un layer sì e uno no (`load_swa_pattern(ml, 2)`: S13 annota quali) |
| RoPE | NEOX, base 150000, YaRN fattore 32 su contesto originale 4096 |
| esperti | 128, top-4, FFN dell'esperto 2880, pesi MXFP4 con bias |
| vocabolario | 201088 (o200k), testa d'uscita separata (`output.weight`) |
| norme | RMSNorm, eps dal GGUF |

Per layer: `attn_norm` → Q, K, V con bias → RoPE → attention con scala
1/√64 e **sinks** (un logit per testa, `blk.N.attn_sinks.weight`, che entra
solo nel denominatore della softmax: p_i = e^{s_i−m} / (Σ_j e^{s_j−m} +
e^{sink−m}), m = max(max_j s_j, sink)) → proiezione d'uscita con bias →
residuo → `attn_post_norm` → router (W·x + b) → i 4 logit più alti → pesi =
softmax di quei 4 → per ogni esperto: gate = W_g·x + b_g, up = W_u·x + b_u,
g = min(gate, 7), u = clamp(up, −7, 7), h = g·σ(1,702·g)·(u + 1), out =
W_d·h + b_d → somma pesata → residuo. Infine `output_norm` e `output`, senza
soft-cap. Formato chat "harmony" (canali `analysis` e `final`) dal template
del GGUF.

## 3.9 Qwen3.8-27B (dal GGUF, 2026-11-15)

File `/nvme/gguf/Qwen3.8-27B-Q8_0.gguf`, 27,04 GiB, solo Q8_0 e F32; arch
`qwen35`. Riferimento `src/models/qwen35.cpp`, `delta-net-base.cpp` e
`gated_delta_net`, `ssm_conv` in `ggml-cpu/ops.cpp`.

| Parametro | Valore |
| --- | --- |
| layer | 65 = 64 + 1 layer MTP (`blk.64`, `nextn.*`): si ignora (D4) |
| hidden / FFN densa | 5120 / 17408 (SwiGLU: `ffn_gate`, `ffn_up`, `ffn_down`) |
| attention | un layer ogni 4 (`full_attention_interval` = 4), 24 teste query, 4 KV, dimensione 256, `attn_gate`, norme su Q e K |
| RoPE | 64 dimensioni su 256, base 1e7, sezioni M-RoPE [11, 11, 10, 0] (testo: stessa posizione in tutte) |
| Gated DeltaNet (gli altri layer) | conv causale kernel 4; stato 128×128 per testa; 16 teste chiave (`ssm.group_count`), 48 teste valore (`ssm.time_step_rank`), interno 6144 = 48 × 128 |
| tensori DeltaNet | `attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_alpha`, `ssm_beta`, `ssm_norm`, `ssm_out` |
| norme | eps 1e-6; `post_attention_norm` prima della FFN |
| tokenizer | `gpt2`, pre-tokenizer `qwen35` |
| contesto | 262144 |

TP: teste chiave 16 → 8 per rank, teste valore 48 → 24 per rank (la testa
valore j usa la testa chiave j/3: da verificare in S19), i canali della
convoluzione seguono i gruppi q, k, v; `ssm_out` per colonne + `all_reduce`.
Attention: 24 → 12 query, 4 → 2 KV per rank. FFN 17408 → 8704 = 34 × 256.
Velocità massima stimata: ~27 ms di pesi + ~4,6 ms di all-reduce per token.

## 3.10 Qwen3.8-Flash-Next (dal GGUF, 2026-11-15)

File `/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/` (4 parti), 87,23 GiB,
arch `qwen4exp`, 1224 tensori. Riferimento `src/models/qwen4exp.cpp` (1505
righe: solo con `grep`/`sed`; S23 ne fa la mappa).

| Parametro | Valore |
| --- | --- |
| layer / hidden | 48 / 2560 |
| attention | un layer ogni 4 (blk.3, 7, …, 47), tutti compressi: nessun layer di attention normale (S23); 24 teste query, 2 KV, dimensione 256; RoPE 64 dimensioni, base 1e7, sezioni [11, 11, 10, 0] |
| attention compressa | `attention.compress_ratios` (48 valori, 4 su tutti i 12 layer di attention) |
| indexer | 4 teste, chiave 128, top-k 2048 |
| DeltaNet | come Qwen3.8-27B (conv 4, stato 128, 16 teste chiave, 48 valore, interno 6144) |
| MoE | 512 esperti, top-10, FFN dell'esperto 640, esperto condiviso 640 con gate (`ffn_gate_inp_shexp`) |
| hyper-connections | 4 flussi, rango basso 320 (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`) |
| embedding n-gram (PLE) | una tabella globale `per_layer_token_embd.weight` (160 × 320001536, IQ4_NL, 26,82 GiB), usata solo dal layer 1 (`ple.layers = [1]`, S23); n-gram 3, 8 teste per n-gram, conv 4, `ple_key`, `ple_value`, `ple_conv1d`, norme `ple_norm_*`; costanti di hash in `ple.layer_multipliers`, `ple.head_offsets`, `ple.head_vocab_sizes` |
| tokenizer | `gpt2`, pre-tokenizer `qwen35` |
| dimensioni | esperti instradati 56,24 GiB; resto 30,99 GiB (in gran parte la tabella PLE: S23 misura) |
| tipi (GiB) | IQ4_NL 32,5; Q5_0 17,7; IQ3_XXS 16,8; IQ4_XS 14,4; Q6_K 2,2; IQ2_S 1,5; Q8_0 1,3; Q2_0 0,44; F32 0,29; BF16 0,04 |

Memoria: tabella PLE in RAM pinned (pochi lookup per token), parti dense in
VRAM (TP), esperti in EP (C5), residenti finché il budget lo consente. La
FFN dell'esperto (640) non è multiplo di 256: niente TP dentro gli esperti.

---

# Parte 4 — Le sezioni

Ogni scheda ha la stessa forma: obiettivo, requisiti, cosa leggere, cosa puoi
modificare, passi, criterio di "fatto". Il report è sempre
`docs/plans/pynference/reports/SN.md`. "Una sessione per passo" vuol dire:
dopo il commit di ogni passo aggiorna il report (note per il passo
successivo) e chiudi il turno con una riga; il passo successivo parte con il
contesto vuoto.

### S1 — Ambiente del PC e misure dell'hardware

Completata prima del piano (fase 0a): venv con PyTorch XPU, Triton e xccl;
correzione dei file di Level Zero; misure di banda RAM→GPU, memoria della
GPU, all-reduce, costo dei lanci. Report: `docs/phase0a.md`.

### S2 — Lettore GGUF e tokenizer di Gemma 4

Completata prima del piano (fase 0b): `pynference/gguf.py`,
`pynference/tokenizer.py` (id identici a `llama-tokenize`). Report:
`docs/phase0b.md`.

### S3 — Kernel Q8_0 e tensor parallel con xccl

Completata prima del piano (fase 0b): `pynference/quant.py` (GEMV Q8_0 a
554–592 GB/s), `pynference/parallel.py` (all-reduce e all-gather su buffer
persistenti). Report: `docs/phase0b.md`.

### S4 — Gemma-4-31B e verifica con llama.cpp

Completata prima del piano (fase 0b): `pynference/models/gemma4.py`,
`pynference/generate.py`, `bench/ppl.py`, `bench/compare_probs.py`,
`scripts/verify_gemma4.sh`. Perplexity a +0,26% da llama.cpp, 81/81 token
greedy identici. Report: `docs/phase0b.md`.

### S5 — Decode veloce: kernel fusi e XPUGraph

Completata prima del piano (fase 0b): `pynference/kernels.py`, percorso
`decode` di Gemma con cattura XPUGraph (all-reduce inclusi): 30,3 token/s.
Report: `docs/phase0b.md`.

### S6 — Banco di prova e prima finestra GPU ⛔

**Obiettivo.** Struttura dei test con pytest, golden di Gemma in
`tests/golden/gemma4/`, regressione `scripts/regress.sh`, e le prime due
finestre GPU per collaudare il meccanismo.

**Requisiti.** Job GPU: 2. Download: `pytest`, `jinja2`, `regex` (D2).

**Leggi.** Parte 2.4; 3.2; 3.3; C1, C7, C8, C10; Appendici A–C;
`bench/compare_probs.py` (138 righe).

**Puoi modificare.** `tests/**`, `bench/compare_probs.py`,
`bench/llama_golden.py` (nuovo), `scripts/regress.sh` (nuovo),
`requirements.txt`, `.gitignore`, `pytest.ini` (nuovo).

**Passi.**

1. Pacchetti nel venv del PC:
   `scripts/cpu.sh 'python -m pip install -q pytest jinja2 regex && python -m pip show pytest jinja2 regex | grep -E "^(Name|Version)"'`.
   Aggiungili a `requirements.txt`. `pytest.ini` con `testpaths = tests`.
   `.gitignore` ha già `scratch/`, `results/`, `.pytest_cache/` e i file
   `jobs/*.out/`: aggiungi solo ciò che manca.
2. Prompt comuni e casi del tokenizer: `tests/golden/prompts.json` con i 4
   prompt di oggi di `compare_probs.py` (`chat-it`, `chat-en` come
   `messages`; `code`, `raw` come `text`) e
   `tests/golden/tokenizer_cases.json` (almeno 12 stringhe: spazi multipli,
   a capo multipli, tab, accenti, CJK, emoji, codice, numeri, token speciali
   del modello scritti con l'escape di 2.1).
3. `bench/llama_golden.py --url URL --out DIR`: con un llama-server già
   avviato produce `tokens.json` (`/tokenize` con `add_special` e
   `parse_special`, per ogni caso), `template.json` (`/apply-template` per
   ogni prompt chat, kwargs vuoti) e `ref_probs.json` (formato attuale di
   `compare_probs.py`: per ogni prompt id ottenuti da llama-server, 24 passi
   greedy, `n_probs` 10). Estendi `compare_probs.py`: `ours --ref FILE` usa
   gli id dei prompt del file di riferimento (indipendente dal nostro
   tokenizer); `diff --strict` applica C8 (regola del quasi pareggio) ed esce
   con 1 se non è rispettata. Senza le opzioni nuove il comportamento resta
   quello di oggi. Test su CPU di `diff` con JSON sintetici
   (`tests/cpu/test_compare_probs.py`).
4. Test su CPU: `tests/cpu/test_gguf.py` (Gemma: arch, 60 layer, 833
   tensori, tipi; `skip` se il file manca). Job **S6-1** (prima finestra):
   selftest `python -m pynference.quant` e `python -m pynference.kernels`;
   llama-server su Gemma (`llama_ref.sh server-start`), `llama_golden.py`,
   `server-stop`; `llama_ref.sh ppl`; `compare_probs.py ours --ref` (grafo)
   e `bench/ppl.py`; `bench/decode_profile.py --mode graph --no-profile`.
   Tutto in `$PYI_JOB_OUT`. Timeout 45.
5. Copia i risultati da `jobs/S6-1.out/` in `tests/golden/gemma4/`
   (`tokens.json`, `template.json`, `ref_probs.json`, `ref_ppl.txt`,
   `ours_baseline.json`, `ours_ppl.txt`). Test su CPU: tokenizer contro
   `tokens.json`; `chat_prompt` contro `template.json` (dopo
   `encode`, gli id coincidono); `ours_baseline.json` rispetta C8 contro
   `ref_probs.json`.
6. `scripts/regress.sh` (C10) e `tests/gpu/test_kernels.py`,
   `tests/gpu/test_quant_q8.py` (i selftest come test pytest con soglie).
   Job **S6-2**: `python -m pytest -q tests/gpu` e
   `scripts/regress.sh gemma4`, entrambi verdi.
7. Report: durata dei due job, esiti, valori (perplexity, token/s).

**Fatto quando.** Test su CPU verdi; S6-2 verde; golden di Gemma in git.

### S7 — Motore di generazione e chat da terminale

**Obiettivo.** `engine.py`, `sampling.py`, `chat.py`, `cli.py` (C4):
sessioni con riuso del prefisso, campionamento, template chat dal GGUF, chat
interattiva su due GPU. `generate.py` resta compatibile.

**Requisiti.** Job GPU: 1.

**Leggi.** 3.3 (righe di `generate.py`, `tokenizer.py`, `parallel.py`); C3,
C4; `pynference/generate.py`.

**Puoi modificare.** `pynference/engine.py`, `sampling.py`, `chat.py`,
`cli.py` (nuovi), `pynference/parallel.py` (solo `broadcast_obj`),
`pynference/generate.py`, `tests/**`, `README.md` (sezione d'uso).

**Passi.**

1. `sampling.py`: `Sampler` e `sample` (temp, top-k, top-p, min-p, seme);
   `generate.py` lo importa. Test su CPU: greedy, determinismo con il seme,
   maschera di top-p e min-p.
2. `chat.py`: `render(g_or_template, messages, add_generation_prompt=True,
   **kwargs) -> str` con `jinja2.sandbox.ImmutableSandboxedEnvironment`
   (`trim_blocks` e `lstrip_blocks` come Hugging Face; funzioni
   `raise_exception`, `strftime_now`; filtro `tojson`). Il testo prodotto
   può contenere il BOS: la tokenizzazione del testo del template usa
   `add_bos=False`. Test su CPU contro `tests/golden/gemma4/template.json`.
3. `TP.broadcast_obj` (C4) e test con un solo processo.
4. `engine.py`: `Session` ed `Engine` (C4). `tests/cpu/fake_model.py`: un
   modello finto che rispetta C3 (logit deterministici dal contesto); test su
   CPU di riuso del prefisso, stop, `max_tokens`, conteggi.
5. `cli.py`: `python -m pynference.cli chat -m MODELLO [--system T]
   [--script FILE] [--ctx N]` (il rank 0 legge l'input; comandi `/reset`,
   `/system`, `/exit`; `--script` legge i turni utente da un file, per i
   test) e `generate` (stesse opzioni di `generate.py`). Dopo ogni risposta
   una riga di statistiche (token del prompt nuovi, token/s).
6. Job **S7-1**: chat con `--script` di 3 turni su Gemma (temp 0): il secondo
   e il terzo turno elaborano solo i token nuovi; `generate.py` dà lo stesso
   testo greedy di prima per il prompt di `compare_probs` `chat-it`;
   `pytest tests/gpu`; `regress.sh gemma4`.

**Fatto quando.** Chat su due GPU funzionante, riuso del prefisso
verificato, regressione verde.

### S8 — Server compatibile OpenAI

**Obiettivo.** `pynference/server.py` (D3): `torchrun --nproc_per_node=2 -m
pynference.server -m MODELLO --host 127.0.0.1 --port 8090`.

**Requisiti.** Job GPU: 1.

**Leggi.** C4; D3; le API di `engine.py` e `chat.py` (con `grep -n "def "`).

**Puoi modificare.** `pynference/server.py` (nuovo), `pynference/engine.py`,
`pynference/chat.py`, `tests/**`, `README.md`.

**Passi.**

1. HTTP con la libreria standard (`ThreadingHTTPServer`) sul rank 0;
   `/health`, `/v1/models`, `/v1/chat/completions` con `messages`,
   `max_tokens` (o `max_completion_tokens`), `temperature`, `top_p`, `top_k`,
   `min_p`, `seed`, `stop`, `stream`. Risposta come OpenAI (`id`, `object`,
   `created`, `model`, `choices[].message`, `finish_reason`, `usage`);
   streaming SSE con `delta` e `data: [DONE]`. Una richiesta alla volta (le
   altre aspettano). Errori 400 e 404 in JSON.
2. Test su CPU con `fake_model.py` e un client `urllib` su una porta casuale:
   risposta normale e in streaming (stesso testo), `stop`, `max_tokens`,
   errori.
3. Due rank: il rank 1 segue i comandi dell'`Engine`; chiusura pulita con
   SIGINT/SIGTERM sul rank 0.
4. Job **S8-1**: server su Gemma in background (attesa di `/health`),
   `tests/gpu/test_server_live.py` contro di esso (risposta greedy uguale a
   `cli generate`, streaming uguale alla risposta normale, `usage`
   corretto), arresto del server; `regress.sh gemma4`.

**Fatto quando.** Test verdi; il README spiega come usarlo dal Mac
(`--host 0.0.0.0` a tuo rischio).

### S9 — Quantizzazioni 1: registro, riferimento, blocchi da 32

**Obiettivo.** Pacchetto `pynference/quant/` (C2) con il riferimento
`gguf-py`; tipi F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL (dequant,
GEMV, GEMV indicizzata) e il layout nativo di Q8_0 per gli esperti.

**Requisiti.** Job GPU: 1. Download: `gguf` (D2).

**Leggi.** C2; 3.7; `pynference/quant.py`; in `scratch/ref`
(`tools/fetch_ref.sh`): in `ggml-common.h` le strutture `block_q4_0` …
`block_iq4_nl` e `kvalues_iq4nl`; in `ggml-quants.c` le funzioni
`dequantize_row_q4_0` … `dequantize_row_iq4_nl` (con `grep -n`).

**Puoi modificare.** `pynference/quant/**` (nuovo; `quant.py` si sposta in
`quant/q8_0.py`), `pynference/models/gemma4.py` (solo gli import),
`tests/**`.

**Passi.**

1. `scripts/cpu.sh 'python -m pip install -q "gguf @ git+https://github.com/ggml-org/llama.cpp@2d8a32502909ca065c41c4cf5f2fb1c427c047fc#subdirectory=gguf-py"'`.
   Con uno script in `scratch/` elenca per ogni tipo di 3.7 se
   `gguf.quants` ha la dequantizzazione e la quantizzazione: tabella nel
   report.
2. Pacchetto: `quant/__init__.py` (riesporta `QLinear`, `q8_0_to_device`),
   `quant/q8_0.py`. Nessun cambiamento di comportamento; test di import su
   CPU.
3. `quant/types.py` (`TYPES`, `row_bytes`, `check_split`), `quant/ref.py`
   (dequant numpy con `gguf-py`), `quant/testing.py` (blocchi casuali validi
   per tipo: byte casuali con le scale sostituite da fp16 in [1e-3, 1e-1]),
   `make_qweight`. Test su CPU.
4. Kernel Triton nel layout nativo, per Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL e
   Q8_0: dequant (con uscita bf16 o, per i test, fp32), GEMV, GEMV
   indicizzata. F16/BF16: GEMV Triton.
5. Job **S9-1**: `tests/gpu/test_quant_block32.py`: dequant contro
   `gguf-py` su blocchi casuali e su tensori reali (Q4_0 da
   `/nvme/gguf/Qwen3.8-27B-Q4_0.gguf`, Q5_0 e IQ4_NL da Flash-Next, prime
   256 righe; Flash-Next in sola lettura); GEMV contro fp32 (errore relativo
   ≤ 1e-4); GEMV indicizzata contro un ciclo; banda su 8192×5376.
   `pytest tests/gpu`; `regress.sh gemma4`.
6. Report: tabella tipo | errore dequant | errore GEMV | GB/s.

**Fatto quando.** Tutti i tipi passano; regressione verde.

### S10 — Quantizzazioni 2: K-quant

**Obiettivo.** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K con dequant, GEMV e GEMV
indicizzata (C2).

**Requisiti.** Job GPU: 1.

**Leggi.** C2; 3.7; le API di `quant/` (con `grep -n "def "`); in
`scratch/ref` le strutture `block_q2_K` … `block_q6_K` e
`dequantize_row_q2_K` … `dequantize_row_q6_K`.

**Puoi modificare.** `pynference/quant/**`, `tests/**`.

**Passi.**

1. Con uno script su CPU trova quali K-quant contengono
   `/nvme/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` e Flash-Next (nomi di tensori da
   usare nei test); annotali.
2. Q4_K e Q5_K (scale e minimi a 6 bit su 8 sotto-blocchi).
3. Q6_K.
4. Q2_K e Q3_K.
5. Job **S10-1**: come S9-1 per questi tipi; `regress.sh gemma4`.

**Fatto quando.** Tutti i tipi passano; regressione verde.

### S11 — Quantizzazioni 3: MXFP4, IQ4_XS, Q2_0

**Obiettivo.** I tre tipi con dequant, GEMV e GEMV indicizzata.

**Requisiti.** Job GPU: 1.

**Leggi.** C2; 3.7; in `scratch/ref` `block_mxfp4`, `kvalues_mxfp4`,
`block_iq4_xs`, `block_q2_0` e le loro `dequantize_row_*`.

**Puoi modificare.** `pynference/quant/**`, `tests/**`.

**Passi.**

1. MXFP4 (scala E8M0, valori E2M1): attenzione alla conversione della scala
   esattamente come in `ggml-quants.c`.
2. IQ4_XS (scale a 6 bit divise in `scales_h` e `scales_l`).
3. Q2_0. Se `gguf-py` non lo supporta (S9, passo 1), scrivi il riferimento
   numpy da `dequantize_row_q2_0` riga per riga e annotalo nel report.
4. Job **S11-1**: come S9-1 (tensori reali: IQ4_XS e Q2_0 da Flash-Next;
   MXFP4 su blocchi casuali, su tensori reali in S13); `regress.sh gemma4`.

**Fatto quando.** Tutti i tipi passano; regressione verde.

### S12 — Quantizzazioni 4: IQ3_XXS, IQ2_S

**Obiettivo.** I due tipi a griglia, con le tabelle di `ggml-common.h`.

**Requisiti.** Job GPU: 1.

**Leggi.** C2; 3.7; in `scratch/ref` `block_iq3_xxs`, `block_iq2_s`,
`iq3xxs_grid`, `iq2s_grid`, `ksigns_iq2xs`, `kmask_iq2xs` e le loro
`dequantize_row_*` (solo gli intervalli con `sed -n`).

**Puoi modificare.** `pynference/quant/**` (le tabelle in
`quant/iq_grids.py`, generate da uno script in `scratch/` che legge
`ggml-common.h`: non copiarle a mano), `tests/**`.

**Passi.**

1. Generatore delle tabelle e `quant/iq_grids.py`; test su CPU: lunghezze e
   alcuni valori controllati a mano contro il sorgente.
2. IQ3_XXS.
3. IQ2_S.
4. Job **S12-1**: come S9-1 (tensori reali da Flash-Next); `regress.sh
   gemma4`. Nel report la tabella completa di tutti i tipi di 3.7.

**Fatto quando.** Tutti i tipi di 3.7 hanno dequant, GEMV e GEMV indicizzata
verificati.

### S13 — GPT-OSS-120B: download, fatti, golden

**Obiettivo.** Il file del modello verificato, i fatti di 3.8 confermati e i
golden di llama.cpp in `tests/golden/gpt-oss-120b/`.

**Requisiti.** Download 59 GiB (D2). Job GPU: 1.

**Leggi.** 3.8; C7, C8; Appendice C.

**Puoi modificare.** `tests/golden/gpt-oss-120b/**`, `tests/cpu/**`,
`bench/llama_golden.py`.

**Passi.**

1. Download in background:
   `docs/plans/pynference/tools/bg.sh start S13-dl scripts/cpu.sh 'mkdir -p /nvme/gguf/gpt-oss-120b && cd /nvme/gguf/gpt-oss-120b && curl -sSL -C - --retry 5 -o gpt-oss-120b-MXFP4.gguf https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf'`,
   poi `bg.sh wait S13-dl` finché finisce. SHA-256 (`sha256sum`, in
   background) contro `lfs.oid` dell'API
   `https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF/tree/main`.
2. Fatti con uno script su CPU (`GGUFFile`): arch, tutti gli iperparametri,
   tensori di `blk.0` e `blk.1` con forme e tipi, dimensioni (esperti /
   resto), tokenizer, token speciali, layer con finestra. Nel report la
   tabella "fatto | 3.8 | osservato | esito" e `facts.json`. Differenze:
   "Correzioni a 3.8" nel report (valgono per le sezioni successive);
   Bloccata solo se cambia l'architettura.
3. Job **S13-1** (timeout 90): llama-server con `--n-cpu-moe X` (X: il minimo
   numero di layer con esperti sulla CPU perché il resto stia in 2 × 29 GB,
   calcolato dai fatti), `llama_golden.py` (casi del tokenizer, template dei
   prompt con i kwargs di default, 24 passi greedy), `server-stop`,
   `llama_ref.sh ppl` con lo stesso `--n-cpu-moe`.
4. Golden in `tests/golden/gpt-oss-120b/`; test su CPU dei fatti (`skip` se
   il file manca).

**Fatto quando.** File verificato, fatti confermati o corretti, golden in git.

### S14 — Tokenizer BPE byte-level e template chat

**Obiettivo.** Tokenizer `gpt2` (BPE byte-level) per GPT-OSS (pre-tokenizer
`gpt-4o`) e Qwen (`qwen35`); `load_tokenizer(g)`; template e separazione del
ragionamento per GPT-OSS e Qwen in CLI e server; golden di Qwen3.8-27B.

**Requisiti.** Job GPU: 1 (golden di Qwen3.8-27B, che entra nelle GPU).

**Leggi.** C7; in `scratch/ref/src/llama-vocab.cpp`: i blocchi
`regex_exprs` di `LLAMA_VOCAB_PRE_TYPE_GPT4O` e `..._QWEN35` (con `grep -n`),
`llm_tokenizer_bpe_session::tokenize`, le regole dei token di fine
generazione (`special_eog_ids`); in `src/unicode.cpp` la mappa byte → unicode.

**Puoi modificare.** `pynference/tokenizer.py` (aggiungi `BPETokenizer` e
`load_tokenizer`; `Tokenizer` di Gemma resta), `pynference/chat.py`,
`pynference/cli.py`, `pynference/server.py`, `tests/**`,
`tests/golden/qwen35-27b/**`.

**Passi.**

1. Mappa nel report: regex (come stringhe Python raw), gestione dei token
   speciali, `add_bos`, token di fine generazione per i due vocabolari.
2. `BPETokenizer` (mappa byte → unicode di GPT-2, pre-tokenizzazione con il
   pacchetto `regex`, merge per rango, token speciali come in Gemma,
   decodifica a byte per `StreamDecoder`); `load_tokenizer(g)` sceglie la
   classe da `tokenizer.ggml.model`.
3. Test su CPU contro `tests/golden/gpt-oss-120b/tokens.json`.
4. Job **S14-1** (timeout 60): golden di `/nvme/gguf/Qwen3.8-27B-Q8_0.gguf`
   come in S13-1 (template con `enable_thinking` vero e falso), più
   `llama_ref.sh ppl`; in `tests/golden/qwen35-27b/`.
5. Test su CPU: id di Qwen; `chat.render` contro i `template.json` dei due
   modelli.
6. Separazione del ragionamento in `chat.py` (`ReasoningSplitter` per
   famiglia: canali di GPT-OSS, tag di Qwen; funziona sullo stream): la CLI
   la mostra con `--show-reasoning`, il server la mette in
   `reasoning_content`. Test su CPU con testi sintetici.

**Fatto quando.** Id e template identici ai golden; test verdi.

### S15 — Esperti tra VRAM e RAM, MoE generico

**Obiettivo.** `pynference/moe.py`: `ExpertStore` (C5) e un layer MoE
generico (router, attivazione `swiglu` o `swiglu_oai`, bias opzionali,
esperto condiviso opzionale), provato con pesi sintetici su due GPU.

**Requisiti.** Job GPU: 1.

**Leggi.** C2 (GEMV indicizzata), C3, C5; 3.1 (banda RAM→GPU); le API di
`quant/` e `parallel.py`.

**Puoi modificare.** `pynference/moe.py` (nuovo), `pynference/quant/**`,
`pynference/kernels.py` (solo kernel nuovi), `bench/moe_check.py` (nuovo),
`tests/**`.

**Passi.**

1. Note di progetto nel report: formula del budget per GPU, layout degli
   slot, stream ed eventi per le copie, cosa succede a un miss.
2. Parte su CPU: proprietà degli esperti (EP), pianificazione della
   residenza, buffer pinned. Test su CPU (funzioni pure, `world` simulato).
3. Parte su GPU: buffer residenti, slot di scambio con LRU, `resolve`,
   copie asincrone, `stats`.
4. MoE: decode (router, top-k, `resolve`, GEMV indicizzate, attivazione,
   somma pesata, `all_reduce`) e prefill (token raggruppati per esperto).
5. Job **S15-1**: `torchrun --nproc_per_node=2 bench/moe_check.py`: layer MoE
   sintetico (32 esperti, top-4, K = 2880, FFN 2880, tipi Q8_0, MXFP4,
   IQ4_XS) contro un riferimento torch su CPU; budget zero (tutto in RAM) =
   budget pieno; banda dei miss. `pytest tests/gpu`; `regress.sh gemma4`.

**Fatto quando.** Risultati uguali al riferimento in tutti i casi di
residenza; misure nel report.

### S16 — GPT-OSS 1: attention con sinks, finestra, YaRN

**Una sessione per passo.**

**Obiettivo.** `pynference/models/gptoss.py`: configurazione, caricamento
delle parti dense in TP, embedding, testa d'uscita, attention con sinks e
finestra, RoPE con YaRN. Il MoE arriva in S17.

**Requisiti.** Job GPU: 1–2.

**Leggi.** Ogni passo dice cosa leggere. Sempre: 3.4, 3.8, C3, il report di
S13 (fatti e correzioni).

**Puoi modificare.** `pynference/models/gptoss.py` (nuovo),
`pynference/models/__init__.py`, `pynference/kernels.py` (sinks opzionali in
`attn_decode`, tabelle RoPE con YaRN), `tests/**`,
`docs/plans/pynference/reports/S16-map.md`.

**Passi.**

1. Mappa `reports/S16-map.md` (al massimo 120 righe), senza codice: per ogni
   punto file:riga e formula. Leggi in `scratch/ref`: il grafo in
   `openai-moe.cpp`; in `llama-graph.cpp` `build_attn_mha` (sinks, scala,
   maschera) e `build_qkv` (bias); in `llama-model.cpp` il tipo di RoPE e i
   parametri YaRN (chiavi GGUF, `yarn_ext_factor`, `attn_factor`,
   `beta_fast`, `beta_slow`); `rope_yarn` in `ggml-cpu/ops.cpp` e
   `ggml_rope_yarn_corr_dims` in `ggml.c`; quali layer hanno la finestra.
2. Configurazione, caricamento (Q, K, V, O con bias; norme; embedding;
   `output`), tabelle RoPE con YaRN. Test su CPU: tabelle contro una
   trascrizione diretta di `rope_yarn` per alcune posizioni.
3. Attention: prefill (torch) con sinks e finestra; decode con
   `attn_decode(..., sinks=...)`. Test su CPU della formula (dati casuali).
4. Riferimento su CPU del layer 0 (solo attention, pesi dequantizzati con
   `gguf-py`) e job **S16-1**: uscita dell'attention del layer 0 su GPU
   contro il riferimento (coseno ≥ 0,9999); `llama_ref.sh dump` di un prompt
   breve con `--n-cpu-moe 36` e confronto dei valori di `attn_out-0` (nomi
   nel sorgente); `pytest tests/gpu`; `regress.sh gemma4`.

**Fatto quando.** Attention del layer 0 uguale al riferimento e a llama.cpp.

### S17 — GPT-OSS 2: MoE, modello completo, verifica ⛔

**Una sessione per passo.**

**Obiettivo.** GPT-OSS-120B completo, con gli esperti in `ExpertStore`, che
rispetta C8.

**Requisiti.** Job GPU: 2–3.

**Leggi.** 3.8; C5, C8; `reports/S16-map.md`; il report di S15 (note).

**Puoi modificare.** `pynference/models/gptoss.py`, `pynference/moe.py`,
`scripts/regress.sh`, `tests/**`, `tests/golden/gpt-oss-120b/**`.

**Passi.**

1. MoE di GPT-OSS: router con bias, top-4, softmax sui 4 logit, esperti
   MXFP4 con bias, `swiglu_oai` (3.8).
2. Modello completo (prefill e decode senza grafo), `memory_report`, tempi di
   caricamento.
3. Job **S17-1**: `compare_probs.py ours --ref
   tests/golden/gpt-oss-120b/ref_probs.json` e `diff --strict`; `bench/ppl.py`;
   se un confronto non torna, dump dei layer (2.5) nello stesso job.
4. Correzioni fino a C8 (job S17-2, …, regola dei 3 tentativi).
5. `regress.sh gpt-oss-120b` (baseline in `ours_baseline.json`,
   `ours_ppl.txt`); nel job finale anche `cli chat --script` di 2 turni e
   `regress.sh gemma4`.
6. Report: numeri di C8, token/s senza grafo, esperti residenti / totali, il
   testo di una risposta, il comando per provare la chat.

**Fatto quando.** C8 rispettato; regressioni verdi.

### S18 — GPT-OSS 3: velocità

**Obiettivo.** Decode di GPT-OSS vicino a C9: kernel fusi, grafo a segmenti
(una sincronizzazione per layer MoE), statistiche degli esperti.

**Requisiti.** Job GPU: 2–3.

**Leggi.** C5, C9; 3.2 (XPUGraph); le API di `kernels.py`, `moe.py`,
`gptoss.py` (con `grep -n "def "`); in `reports/S17.md` le note "Per S18" e
"Determinismo".

**Puoi modificare.** `pynference/models/gptoss.py`, `pynference/moe.py`,
`pynference/kernels.py`, `pynference/quant/**`, `bench/decode_profile.py`,
`tests/**`.

**Passi.**

1. Profilo di partenza (job): `decode_profile.py` con profilo per kernel.
   Nello stesso job, la RAM bloccata dal driver: il nostro processo blocca
   RAM dell'host quasi pari alla VRAM che torch riserva (`GPUActive` in
   `/proc/meminfo` ≈ 40 GiB con ~20 GiB per GPU, S17-6), mentre Strata con
   ~30 GiB di VRAM per GPU ne blocca 7 GiB. È questo che limita gli esperti
   residenti al ~54% (`EXPERT_HOST_RESERVE_BYTES`) e costa ~440 MB di copie
   per GPU a ogni token (2,7–3,1 token/s in S17, llama.cpp 49–54). Misura
   `GPUActive` su allocazioni di prova (un processo senza oneCCL e due con,
   buffer grandi e piccoli); se trovi la causa, alza il budget degli
   esperti prima del passo 2.
2. Kernel fusi per norme, RoPE YaRN e attention con sinks; grafo per i
   segmenti tra due sincronizzazioni.
3. GEMV indicizzata MXFP4: banda e configurazioni.
4. Job finale: velocità, tasso di hit, `regress.sh gpt-oss-120b`,
   `regress.sh gemma4`.

**Fatto quando.** Misure nel report; regressioni verdi (C9 è un obiettivo,
non un blocco).

### S19 — Qwen3.8-27B: fatti, golden, mappa del riferimento

**Una sessione per passo.**

**Obiettivo.** Fatti di 3.9 confermati, golden presenti (S14), mappa del
riferimento per DeltaNet e attention.

**Requisiti.** Job GPU: solo se mancano golden.

**Leggi.** 3.9; C7. Ogni passo dice cosa leggere in `scratch/ref`.

**Puoi modificare.** `tests/golden/qwen35-27b/**`, `tests/cpu/**`,
`docs/plans/pynference/reports/S19-map.md`.

**Passi.**

1. Fatti (script su CPU) contro 3.9: tabella e `facts.json`; quali layer
   sono attention; forme di tutti i tensori di un layer DeltaNet e di uno di
   attention.
2. Mappa A (DeltaNet): in `qwen35.cpp` il layer lineare (proiezioni,
   convoluzione, preparazione di q, k, v, g, beta, norma con gate, uscita);
   `gated_delta_net` e `ssm_conv` in `ggml-cpu/ops.cpp`. Formule con
   file:riga.
3. Mappa B (attention e resto): gate dell'attention, RoPE parziale e
   sezioni, norme, FFN, uscita; il rapporto tra teste valore e teste chiave.
4. Piano TP e memoria (3.4, 3.9): divisione di ogni tensore, allineamenti,
   memoria per GPU. In `S19-map.md` (al massimo 150 righe in tutto).

**Fatto quando.** Mappa completa; fatti confermati; golden presenti.

### S20 — Gated DeltaNet: kernel

**Obiettivo.** `pynference/deltanet.py`: riferimento su CPU e kernel Triton
per convoluzione causale con stato, ricorrenza gated delta (decode di un
token e prefill sequenziale), norma con gate.

**Requisiti.** Job GPU: 1–2.

**Leggi.** `reports/S19-map.md`; 3.9.

**Puoi modificare.** `pynference/deltanet.py` (nuovo), `tests/**`.

**Passi.**

1. Riferimento su CPU in torch, due versioni indipendenti (vettoriale e un
   ciclo per token trascritto da `ops.cpp`) che devono coincidere. Test su
   CPU.
2. Kernel di decode (un token: convoluzione, aggiornamento dello stato,
   uscita).
3. Kernel di prefill (sequenziale sui token, parallelo sulle teste).
4. Job **S20-1**: GPU contro riferimento (casuale e con i pesi reali del
   layer 0 di Qwen3.8-27B, forme per rank); tempi; `pytest tests/gpu`.

**Fatto quando.** Errore relativo dello stato ≤ 1e-4 e uscite con coseno
≥ 0,9999 contro il riferimento.

### S21 — Qwen3.8-27B: modello completo e verifica ⛔

**Una sessione per passo.**

**Obiettivo.** `pynference/models/qwen35.py` completo che rispetta C8.

**Requisiti.** Job GPU: 2–3.

**Leggi.** `reports/S19-map.md`; 3.4, 3.9; C3, C8.

**Puoi modificare.** `pynference/models/qwen35.py` (nuovo),
`pynference/models/__init__.py`, `pynference/deltanet.py`,
`pynference/kernels.py` (solo kernel nuovi), `scripts/regress.sh`,
`tests/**`, `tests/golden/qwen35-27b/**`.

**Passi.**

1. Configurazione e caricamento in TP secondo la mappa (`blk.64` escluso).
2. Layer DeltaNet (prefill e decode, stati nella cache, `reset`).
3. Layer di attention con gate, FFN, uscita.
4. Job **S21-1**: dump di llama.cpp dei layer 0 e 3 contro i nostri;
   `compare_probs.py --ref` e `diff --strict`; `bench/ppl.py`.
5. Correzioni fino a C8; `regress.sh qwen35-27b` con baseline; chat con
   ragionamento attivo e no (`cli chat --script`); `regress.sh gemma4`.

**Fatto quando.** C8 rispettato; regressioni verdi.

### S22 — Qwen3.8-27B: velocità

**Obiettivo.** Decode di Qwen3.8-27B vicino a C9: grafo completo (gli stati
DeltaNet sono buffer statici), kernel fusi.

**Requisiti.** Job GPU: 2–3.

**Leggi.** C9; 3.2; le API di `qwen35.py`, `deltanet.py`, `kernels.py`.

**Puoi modificare.** `pynference/models/qwen35.py`, `pynference/deltanet.py`,
`pynference/kernels.py`, `bench/decode_profile.py`, `tests/**`.

**Passi.**

1. Profilo di partenza (job).
2. Decode con kernel fusi e cattura XPUGraph.
3. Job finale: velocità, `regress.sh qwen35-27b`, `regress.sh gemma4`.

**Fatto quando.** Misure nel report; regressioni verdi.

### S23 — Flash-Next: fatti, golden, mappa del riferimento

**Una sessione per passo.**

**Obiettivo.** Fatti di 3.10, golden di llama.cpp e la mappa di
`qwen4exp.cpp` divisa in tre parti.

**Requisiti.** Job GPU: 1 (timeout 120).

**Leggi.** 3.10; C7. Ogni passo dice cosa leggere in `scratch/ref`.

**Puoi modificare.** `tests/golden/qwen4exp/**`, `tests/cpu/**`,
`docs/plans/pynference/reports/S23-map-a.md`, `S23-map-b.md`, `S23-map-c.md`.

**Passi.**

1. Fatti (script su CPU): iperparametri, tensori per tipo di layer, tipi,
   dimensioni per gruppo (esperti, tabella PLE, resto); vocabolario confrontato
   con Qwen3.8-27B. Tabella e `facts.json`.
2. Mappa A (`S23-map-a.md`, ≤ 120 righe): struttura dei layer, quali sono
   DeltaNet, attention, attention compressa; flusso delle hyper-connections.
3. Mappa B (`S23-map-b.md`): attention compressa e indexer.
4. Mappa C (`S23-map-c.md`): PLE (hash degli n-gram, lookup, convoluzione,
   chiavi e valori), MoE con esperto condiviso, uscita con hyper-connections.
   In ogni mappa i nomi `cb(...)` utili per i dump.
5. Job **S23-1**: llama-server con `--n-cpu-moe 48` (tutti gli esperti sulla
   CPU: la RAM basta con Strata fermo), `llama_golden.py` (template con
   ragionamento attivo e no), `llama_ref.sh ppl`, dump del layer 0 e del
   primo layer di attention compressa su un prompt breve. Golden in
   `tests/golden/qwen4exp/`.

**Fatto quando.** Fatti, golden e tre mappe in git.

### S24 — Flash-Next 1: hyper-connections ed embedding n-gram

**Una sessione per passo.**

**Obiettivo.** Moduli per le hyper-connections e per il PLE, con
riferimento su CPU e versione GPU, confrontati con i dump di llama.cpp.

**Requisiti.** Job GPU: 1–2.

**Leggi.** `reports/S23-map-a.md`, `S23-map-c.md`; 3.10; C2.

**Puoi modificare.** `pynference/models/qwen4exp.py` (nuovo),
`pynference/kernels.py` (solo kernel nuovi), `tests/**`.

**Passi.**

1. Piano dei moduli (nel report): tensori, forme per rank, cosa è
   replicato.
2. PLE (solo il layer 1, S23): hash degli n-gram, lookup dalla tabella in
   RAM pinned (righe dequantizzate su CPU con `quant/ref.py` o su GPU dopo
   la copia), convoluzione, chiavi e valori. Riferimento su CPU e GPU.
3. Hyper-connections (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`). Riferimento su
   CPU e GPU.
4. Job: moduli contro i dump di S23-1: hyper-connections sul layer 0, PLE
   sul layer 1 (`ple_*-1`).

**Fatto quando.** Moduli uguali ai dump (coseno ≥ 0,999).

### S25 — Flash-Next 2: attention compressa e indexer

**Una sessione per passo.**

**Obiettivo.** Attention compressa con indexer (top-k 2048, rapporto 4),
prefill e decode, confrontata con i dump.

**Requisiti.** Job GPU: 1–2.

**Leggi.** `reports/S23-map-b.md`; 3.10.

**Puoi modificare.** `pynference/models/qwen4exp.py`,
`pynference/kernels.py` (solo kernel nuovi), `tests/**`.

**Passi.**

1. Riferimento su CPU (indexer, selezione, compressione, attention).
2. Versione GPU (prefill con torch, decode con un kernel).
3. Job: contro i dump del primo layer compresso; contesti più corti e più
   lunghi di 2048 token (top-k 2048 = 512 blocchi da 4: fino a 2048 token
   l'indexer tiene tutti i blocchi, oltre seleziona; `S23-map-b.md`).

**Fatto quando.** Uguale ai dump; i due regimi di lunghezza provati.

### S26 — Flash-Next 3: MoE, modello completo, verifica ⛔

**Una sessione per passo.**

**Obiettivo.** Flash-Next completo, con esperti in `ExpertStore` (tipi misti),
PLE in RAM, che rispetta C8.

**Requisiti.** Job GPU: 3–4.

**Leggi.** `reports/S23-map-a.md`, `S23-map-c.md`; 3.10; C5, C8; in
`reports/S18.md` il passo 1 e le sonde dei job S18-1 e S18-2.

**Puoi modificare.** `pynference/models/qwen4exp.py`,
`pynference/models/__init__.py`, `pynference/moe.py`,
`pynference/models/gptoss.py` (solo il budget degli esperti, passo 1),
`scripts/regress.sh`, `tests/**`, `tests/golden/qwen4exp/**`.

**Passi.**

1. RAM bloccata dal driver, prima del piano di memoria (job **S26-1**). Ogni
   allocazione torch su XPU blocca RAM dell'host quasi pari ai byte
   riservati (`GPUActive` in `/proc/meminfo`; S17-6 e sonde di S18-1 e
   S18-2, con e senza oneCCL, blocchi grandi e piccoli). Strata, motore SYCL
   senza torch, tiene invece ~60 GiB di VRAM con 7,2 GiB di `GPUActive`
   (misura del 2026-11-17): il limite è nel percorso di torch (allocatore o
   runtime SYCL del suo pacchetto), non nel driver. Flash-Next pesa 88 GB:
   con lo specchio 1:1 non entra fra 64 GB di VRAM e 62 GB di RAM; su
   GPT-OSS lo specchio tiene gli esperti residenti al ~54%. Trova un modo di
   allocare i buffer di `ExpertStore` che non blocchi RAM, misurando
   `GPUActive` a ogni prova, e applicalo allo store. Su GPT-OSS alza il
   budget degli esperti (`EXPERT_HOST_RESERVE_BYTES` in `gptoss.py`) e
   verifica `regress.sh gpt-oss-120b` e la velocità
   (`decode_profile.py --mode graph --no-profile`; in S18 4,2–4,6 token/s).
   Sul PC non c'è un compilatore SYCL di sistema e i download sono solo
   quelli della D2: se serve un modulo nativo o un pacchetto nuovo, fermati
   (2.6).
2. MoE: 512 esperti, top-10, esperto condiviso con gate; piano di memoria
   (budget degli esperti residenti) nel report, con l'esito del passo 1.
3. Modello completo (DeltaNet di S20, attention compressa su tutti i 12
   layer di attention, HC, PLE sul layer 1).
4. Job **S26-2**: `compare_probs.py --ref`, `diff --strict`, `bench/ppl.py`.
5. Correzioni fino a C8; `regress.sh qwen4exp` con baseline; chat con
   ragionamento; `regress.sh gemma4`.

**Fatto quando.** C8 rispettato; regressioni verdi.

### S27 — Flash-Next 4: velocità

**Obiettivo.** Decode di Flash-Next vicino a C9.

**Requisiti.** Job GPU: 2–3.

**Leggi.** C5, C9; le API di `qwen4exp.py`, `moe.py`, `kernels.py`.

**Puoi modificare.** `pynference/models/qwen4exp.py`, `pynference/moe.py`,
`pynference/kernels.py`, `pynference/quant/**`, `pynference/deltanet.py`,
`bench/decode_profile.py`, `tests/**`.

**Passi.**

1. Profilo di partenza (job): tempo per componente, hit degli esperti.
2. Grafo a segmenti, kernel fusi, prefetch degli esperti dove utile.
3. Job finale: velocità, `regress.sh qwen4exp`, `regress.sh gemma4`.

**Fatto quando.** Misure nel report; regressioni verdi.

### S28 — Validazione finale, benchmark, documentazione ⛔

**Obiettivo.** Tutti i modelli verificati insieme, benchmark e
documentazione d'uso.

**Requisiti.** Job GPU: 2.

**Leggi.** C8, C9, C10; i report di S17, S21, S26 ("Valori").

**Puoi modificare.** `README.md`, `docs/models.md` (nuovo),
`docs/plans/pynference/reports/S28.md`, `bench/**`, `scripts/regress.sh`,
`tests/**`.

**Passi.**

1. Job: `regress.sh` per tutti e quattro i modelli; `pytest tests/gpu`.
2. Job: benchmark per modello (decode con 512 e 8192 token di contesto,
   prefill di 2000 token, memoria per GPU) e una prova del server per
   modello.
3. `docs/models.md`: per modello file, memoria, velocità, opzioni
   consigliate; README aggiornato (comandi, modelli supportati).
4. Report con la tabella finale e l'elenco delle prove per il controllo ⛔.

**Fatto quando.** Tutto verde e documentato; il merge in `main` lo decidi tu.

---

## Appendice A — Comandi utili

```bash
# test su CPU (GPU nascoste)
scripts/cpu.sh 'python -m pytest -q tests/cpu 2>&1 | tail -n 20'
# leggere un GGUF sul PC
scripts/cpu.sh 'python -m pynference.gguf /nvme/gguf/Qwen3.8-27B-Q8_0.gguf | head -n 60'
# script Python di servizio: scrivilo in scratch/ (non va sul PC) e passalo su stdin
scripts/cpu.sh 'python - ARGOMENTI' < scratch/script.py
# comandi lunghi
docs/plans/pynference/tools/bg.sh start NOME scripts/cpu.sh 'COMANDO'
docs/plans/pynference/tools/bg.sh wait NOME
# sorgenti di riferimento
docs/plans/pynference/tools/fetch_ref.sh
grep -n 'build_attn_mha' scratch/ref/llama.cpp/src/llama-graph.cpp
```

Nei job (Appendice B), su due GPU:

```bash
torchrun --nproc_per_node=2 -m pynference.generate -m MODELLO --chat -p "..." -n 64 --temp 0
torchrun --nproc_per_node=2 bench/ppl.py -m MODELLO -c 512 --chunks 8
torchrun --nproc_per_node=2 bench/compare_probs.py ours --ref tests/golden/M/ref_probs.json -m MODELLO -o $PYI_JOB_OUT/ours.json
python bench/compare_probs.py diff --strict tests/golden/M/ref_probs.json $PYI_JOB_OUT/ours.json
torchrun --nproc_per_node=2 bench/decode_profile.py -m MODELLO --mode graph --no-profile
```

`torchrun` stampa avvisi `CCL_WARN` e `W111…`: filtrali con
`grep -v -E "CCL_WARN|^W1"`.

## Appendice B — Modello di job GPU

```bash
#!/usr/bin/env bash
# job: S9-1
# timeout: 30
# scopo: kernel dei tipi a blocchi da 32 contro gguf-py; regressione di Gemma
set -uo pipefail
echo "== start $(date +%T)"
python -m pytest -q tests/gpu/test_quant_block32.py 2>&1 | tail -n 30
echo "== quant exit ${PIPESTATUS[0]}"
scripts/regress.sh gemma4 2>&1 | grep -v -E "CCL_WARN|^W1" | tail -n 20
echo "== regress exit ${PIPESTATUS[0]}"
echo "== end $(date +%T)"
```

- Gira in `~/pynference` sul PC, venv attivo, GPU libere. Variabili:
  `PYI_JOB` (nome), `PYI_JOB_OUT` (cartella dei file da riportare).
- `set -uo pipefail` senza `-e`: il job prosegue anche se un passo fallisce,
  e stampa l'exit code di ogni passo (`PIPESTATUS[0]`).
- Output breve: `tail`, `grep`; i file lunghi in `$PYI_JOB_OUT`.

## Appendice C — llama.cpp nei job

`docs/plans/pynference/tools/llama_ref.sh` (vedi le istruzioni in testa al
file):

```bash
T=docs/plans/pynference/tools/llama_ref.sh
M=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf
$T server-start $M 8091                      # aspetta /health
python bench/llama_golden.py --url http://127.0.0.1:8091 --out $PYI_JOB_OUT
$T server-stop 8091
$T ppl $M $PYI_JOB_OUT/ref_ppl.log           # stampa la riga "Final estimate"
$T dump $M "The capital of France is" $PYI_JOB_OUT/dump.txt
grep -n "attn_out-0\|ffn_out-0" $PYI_JOB_OUT/dump.txt | head
```

Per i modelli che non entrano nelle GPU aggiungi gli argomenti di llama.cpp
in coda, per esempio `--n-cpu-moe 12` (esperti di 12 layer sulla CPU). Il
server ascolta solo su `127.0.0.1:8091` ed è marcato `--alias pynference-ref`.
