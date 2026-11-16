#!/usr/bin/env bash
# job: S14-1
# timeout: 60
# scopo: golden Qwen3.8-27B (tokens, template con enable_thinking true/false,
#        ref_probs) + ppl, come S13-1 ma senza argomenti di offload
set -uo pipefail
echo "== start $(date +%T)"
T=docs/plans/pynference/tools/llama_ref.sh
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf
$T server-start $M 8091
echo "== server-start exit $?"
grep -i -m 8 "VRAM\|buffer size\|model size\|offloaded" "${PYI_JOB_OUT}/llama-server-8091.log" | tail -n 8 || true
python bench/llama_golden.py --url http://127.0.0.1:8091 --out "$PYI_JOB_OUT"
echo "== golden exit $?"
python - "$PYI_JOB_OUT" <<'PYEOF'
"""Append enable_thinking true/false variants to template.json (S14-1)."""
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
echo "== template-thinking exit $?"
$T server-stop 8091
echo "== server-stop exit $?"
$T ppl $M "$PYI_JOB_OUT/ref_ppl.log"
echo "== ppl exit $?"
grep "Final estimate" "$PYI_JOB_OUT/ref_ppl.log" > "$PYI_JOB_OUT/ref_ppl.txt" || echo "no Final estimate line"
ls -l "$PYI_JOB_OUT"
echo "== end $(date +%T)"
