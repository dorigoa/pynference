#!/usr/bin/env bash
# job: S23-1
# timeout: 120
# scopo: golden Flash-Next (qwen4exp): llama-server con --n-cpu-moe 48
#        (tutti gli esperti su CPU), llama_golden.py + varianti
#        enable_thinking (thinking-0 ri-renderizzata con un server
#        --reasoning off, lezione S14-2), ppl con lo stesso --n-cpu-moe,
#        dump di un prompt breve (un passaggio copre i layer 0, 1-PLE e
#        3-QSA). Nessun test GPU: S23 non tocca codice.
set -uo pipefail
echo "== start $(date +%T)"
T=docs/plans/pynference/tools/llama_ref.sh
M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MOE=(--n-cpu-moe 48)

# 1) normal server: tokens.json, template.json, ref_probs.json
$T server-start $M 8091 "${MOE[@]}"
echo "== server-start exit $? ($(date +%T))"
grep -i -m 8 "VRAM\|buffer size\|model size\|offload" "${PYI_JOB_OUT}/llama-server-8091.log" | tail -n 8 || true
python bench/llama_golden.py --url http://127.0.0.1:8091 --out "$PYI_JOB_OUT"
echo "== golden exit $? ($(date +%T))"
python - "$PYI_JOB_OUT" <<'PYEOF'
"""Append enable_thinking true/false variants to template.json (as S14-1)."""
import json
import os
import sys
import urllib.request

out, url = sys.argv[1], "http://127.0.0.1:8091"
path = os.path.join(out, "template.json")
rows = json.load(open(path, encoding="utf-8"))


def apply(messages, kwargs):
    payload = json.dumps({"messages": messages, "kwargs": kwargs}).encode()
    req = urllib.request.Request(url + "/apply-template", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/plain"})
    body = urllib.request.urlopen(req, timeout=600).read()
    try:  # some builds answer JSON even without the text/plain header
        j = json.loads(body)
        if isinstance(j, dict) and "prompt" in j:
            return j["prompt"]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return body.decode()


extra = []
for r in list(rows):
    for think in (True, False):
        kw = dict(r["kwargs"])
        kw["enable_thinking"] = think
        extra.append({"name": f"{r['name']}-thinking-{int(think)}",
                      "messages": r["messages"], "kwargs": kw,
                      "prompt": apply(r["messages"], kw)})
rows.extend(extra)
text = json.dumps(rows, ensure_ascii=False, indent=1).replace("<", chr(92) + "u003c")
assert "<" not in text
open(path, "w", encoding="utf-8").write(text + "\n")
print(f"template.json: +{len(extra)} enable_thinking variants", flush=True)
PYEOF
echo "== template-thinking exit $? ($(date +%T))"
$T server-stop 8091
echo "== server-stop exit $? ($(date +%T))"

# 2) --reasoning off server: re-render thinking-0 variants (kwargs are
#    ignored by /apply-template; the server default is the only way)
$T server-start $M 8091 --reasoning off "${MOE[@]}"
echo "== server-start-off exit $? ($(date +%T))"
python - "$PYI_JOB_OUT" <<'PYEOF'
"""Rewrite the -thinking-0 template rows with a --reasoning-off server."""
import json
import os
import sys
import urllib.request

out = sys.argv[1]
url = "http://127.0.0.1:8091"
path = os.path.join(out, "template.json")
rows = json.load(open(path, encoding="utf-8"))


def apply(messages, kwargs):
    payload = json.dumps({"messages": messages, "kwargs": kwargs}).encode()
    req = urllib.request.Request(url + "/apply-template", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/plain"})
    body = urllib.request.urlopen(req, timeout=600).read()
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "prompt" in j:
            return j["prompt"]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return body.decode()


by_name = {r["name"]: r for r in rows}
n_fixed = 0
for r in rows:
    if r["name"].endswith("-thinking-0"):
        base = by_name[r["name"].removesuffix("-thinking-0")]
        new_prompt = apply(r["messages"], {"enable_thinking": False})
        assert new_prompt != base["prompt"], f"{r['name']}: still identical to base"
        r["prompt"] = new_prompt
        n_fixed += 1
assert n_fixed == 2, f"expected 2 thinking-0 rows, fixed {n_fixed}"
# sanity: thinking-1 variants must still equal their base (default server on)
for r in rows:
    if r["name"].endswith("-thinking-1"):
        base = by_name[r["name"].removesuffix("-thinking-1")]
        assert r["prompt"] == base["prompt"], f"{r['name']} != base"
text = json.dumps(rows, ensure_ascii=False, indent=1).replace("<", chr(92) + "u003c")
assert "<" not in text
open(path, "w", encoding="utf-8").write(text + "\n")
print(f"template.json: {n_fixed} thinking-0 variants re-rendered", flush=True)
PYEOF
echo "== template-fix exit $? ($(date +%T))"
$T server-stop 8091
echo "== server-stop exit $? ($(date +%T))"

# 3) ppl on GPL-3, 8x512, same offload
$T ppl $M "$PYI_JOB_OUT/ref_ppl.log" "${MOE[@]}"
echo "== ppl exit $? ($(date +%T))"
grep "Final estimate" "$PYI_JOB_OUT/ref_ppl.log" > "$PYI_JOB_OUT/ref_ppl.txt" || echo "no Final estimate line"

# 4) dump of a short prompt: one eval covers layer 0 (deltanet), layer 1
#    (the only PLE layer, for S24) and layer 3 (first QSA layer)
$T dump $M "The capital of France is" "$PYI_JOB_OUT/dump.txt" "${MOE[@]}"
echo "== dump exit $? ($(date +%T))"
for n in "hc_norm-0" "hc_mixed-0" "linear_attn_out-0" "ffn_out-0" \
         "ple_gate-1" "ple_conv_out-1" "indexer_score-3" "kqv_out-3" \
         "attn_output-3" "result_output"; do
  if grep -a -q -m 1 -- "$n = " "$PYI_JOB_OUT/dump.txt"; then echo "dump has $n"; else echo "MISSING $n"; fi
done
ls -l "$PYI_JOB_OUT"
echo "== end $(date +%T)"
