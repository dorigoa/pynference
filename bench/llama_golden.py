"""Golden reference files from a running llama-server (GPU jobs only, S6).

  python bench/llama_golden.py --url http://127.0.0.1:8091 --out DIR

Writes into DIR:
  tokens.json     one {"text", "ids"} entry per case in tests/golden/tokenizer_cases.json,
                  ids from /tokenize with add_special and parse_special;
  template.json   one {"name", "messages", "kwargs", "prompt"} entry per chat prompt
                  in tests/golden/prompts.json, from /apply-template with empty kwargs;
  ref_probs.json  compare_probs.py format: for every prompt (chat prompts after
                  /apply-template, raw ones as-is), token ids from llama-server's
                  /tokenize, then 24 greedy steps with n_probs 10 via /completion.

Requires a llama-server started with docs/plans/pynference/tools/llama_ref.sh server-start.
Special-token "<" characters are written as < escapes (plan 2.1), so the
files are valid JSON and safe to handle.
"""
import argparse
import json
import math
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PREDICT, N_PROBS = 24, 10  # keep in sync with bench/compare_probs.py


def post(url, path, payload, accept=None):
    headers = {"Content-Type": "application/json"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers=headers)
    body = urllib.request.urlopen(req, timeout=600).read()
    if accept == "text/plain":
        try:  # some server builds answer JSON even without the text/plain header
            j = json.loads(body)
            if isinstance(j, dict) and "prompt" in j:
                return j["prompt"]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return body.decode()
    return json.loads(body)


def dump_escaped(obj, path):
    """json.dump, then escape the first char of model special tokens as <."""
    text = json.dumps(obj, ensure_ascii=False, indent=1).replace("<", chr(92) + "u003c")
    assert "<" not in text  # nothing else left unescaped
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def tokenize(url, text):
    r = post(url, "/tokenize", {"content": text, "add_special": True, "parse_special": True})
    return [int(t) for t in r["tokens"]]


def main():
    ap = argparse.ArgumentParser(description="Produce golden reference files from llama-server.")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = json.load(open(os.path.join(ROOT, "tests/golden/tokenizer_cases.json"), encoding="utf-8"))
    prompts = json.load(open(os.path.join(ROOT, "tests/golden/prompts.json"), encoding="utf-8"))

    tokens = [{"text": t, "ids": tokenize(args.url, t)} for t in cases]
    dump_escaped(tokens, os.path.join(args.out, "tokens.json"))
    print(f"tokens.json: {len(tokens)} cases", flush=True)

    template, prompt_str = [], {}
    for p in prompts:
        if "messages" in p:
            s = post(args.url, "/apply-template", {"messages": p["messages"], "kwargs": {}},
                     accept="text/plain")
            template.append({"name": p["name"], "messages": p["messages"], "kwargs": {}, "prompt": s})
        else:
            s = p["text"]
        prompt_str[p["name"]] = s
    dump_escaped(template, os.path.join(args.out, "template.json"))
    print(f"template.json: {len(template)} prompts", flush=True)

    out = []
    for p in prompts:
        ids = tokenize(args.url, prompt_str[p["name"]])
        res = post(args.url, "/completion", {
            "prompt": ids, "n_predict": N_PREDICT, "n_probs": N_PROBS, "temperature": 0.0,
            "cache_prompt": False, "post_sampling_probs": False, "ignore_eos": False})
        steps = []
        for st in res.get("completion_probabilities", []):
            if "top_logprobs" in st:
                top = [{"id": t["id"], "token": t["token"], "p": math.exp(t["logprob"])}
                       for t in st["top_logprobs"]]
                steps.append({"id": st["id"], "top": top})
            else:  # older format
                top = [{"id": t.get("id"), "token": t["tok_str"], "p": t["prob"]} for t in st["probs"]]
                steps.append({"id": top[0]["id"], "top": top})
        out.append({"name": p["name"], "prompt": ids, "steps": steps, "text": res.get("content", "")})
        print(f"ref_probs {p['name']}: {res.get('content', '')[:80]!r}", flush=True)
    dump_escaped(out, os.path.join(args.out, "ref_probs.json"))
    print("ref_probs.json: done", flush=True)


if __name__ == "__main__":
    main()
