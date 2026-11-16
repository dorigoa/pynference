#!/usr/bin/env bash
# job: S14-2
# timeout: 60
# scopo: correggere le varianti thinking-0 del golden Qwen3.8-27B.
#   S14-1 passava "kwargs" a /apply-template, che llama-server ignora
#   (server-context.cpp: post_apply_template passa il body a
#   oaicompat_chat_params_parse; l'unica via sono i default del server).
#   Qui il server parte con --reasoning off e i due prompt base vengono
#   ri-renderizzati con thinking disattivato; il template.json completo
#   (base, thinking-1, thinking-0 corretti) finisce in $PYI_JOB_OUT.
set -uo pipefail
echo "== start $(date +%T)"
T=docs/plans/pynference/tools/llama_ref.sh
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf
$T server-start $M 8091 --reasoning off
echo "== server-start exit $?"
grep -i -m 4 'thinking\|reasoning' "${PYI_JOB_OUT}/llama-server-8091.log" || true
python - "$PYI_JOB_OUT" <<'PYEOF'
"""Regenerate enable_thinking=false variants with a --reasoning-off server."""
import json
import os
import sys
import urllib.request

out = sys.argv[1]
url = "http://127.0.0.1:8091"
rows = json.load(open("tests/golden/qwen35-27b/template.json", encoding="utf-8"))


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
# sanity: thinking-1 variants must still equal their base (server default on)
for r in rows:
    if r["name"].endswith("-thinking-1"):
        base = by_name[r["name"].removesuffix("-thinking-1")]
        assert r["prompt"] == base["prompt"], f"{r['name']} != base"
text = json.dumps(rows, ensure_ascii=False, indent=1).replace("<", chr(92) + "u003c")
assert "<" not in text
open(os.path.join(out, "template.json"), "w", encoding="utf-8").write(text + "\n")
print(f"template.json: {n_fixed} thinking-0 variants re-rendered", flush=True)
PYEOF
echo "== fix exit $?"
$T server-stop 8091
echo "== server-stop exit $?"
ls -l "$PYI_JOB_OUT"
echo "== end $(date +%T)"
