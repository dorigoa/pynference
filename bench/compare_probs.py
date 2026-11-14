"""Compare greedy generations and top-token probabilities with llama.cpp.

Both engines get the same token ids (from pynference's tokenizer, which matches
llama.cpp's) and generate greedily; at every step the top-N probabilities of
the full softmax are recorded.

  # reference: a running llama-server
  python bench/compare_probs.py fetch --url http://localhost:8091 -m MODEL.gguf -o results/ref_probs.json
  # pynference
  torchrun --nproc_per_node=2 bench/compare_probs.py ours -m MODEL.gguf -o results/our_probs.json
  # comparison
  python bench/compare_probs.py diff results/ref_probs.json results/our_probs.json
"""
import argparse
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynference.gguf import GGUFFile  # noqa: E402
from pynference.tokenizer import Tokenizer, chat_prompt  # noqa: E402

PROMPTS = [
    ("chat-it", chat_prompt([{"role": "user", "content": "Qual è la capitale d'Italia? Rispondi in una frase."}])),
    ("chat-en", chat_prompt([{"role": "user", "content": "Write a haiku about the sea."}])),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("raw", "The history of the Roman Empire began in 27 BC when"),
]
N_PREDICT, N_PROBS = 24, 10


def prompts_ids(model_path):
    tok = Tokenizer(GGUFFile(model_path))
    return tok, [(name, tok.encode(text)) for name, text in PROMPTS]


def fetch(args):
    _, prompts = prompts_ids(args.model)
    out = []
    for name, ids in prompts:
        req = {"prompt": ids, "n_predict": N_PREDICT, "n_probs": N_PROBS, "temperature": 0.0,
               "cache_prompt": False, "post_sampling_probs": False, "ignore_eos": False}
        r = urllib.request.Request(args.url + "/completion", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(r))
        steps = []
        for st in res.get("completion_probabilities", []):
            if "top_logprobs" in st:
                top = [{"id": t["id"], "token": t["token"], "p": math.exp(t["logprob"])} for t in st["top_logprobs"]]
                steps.append({"id": st["id"], "top": top})
            else:   # older format
                top = [{"id": t.get("id"), "token": t["tok_str"], "p": t["prob"]} for t in st["probs"]]
                steps.append({"id": top[0]["id"], "top": top})
        out.append({"name": name, "prompt": ids, "steps": steps, "text": res.get("content", "")})
        print(f"{name}: {res.get('content', '')[:80]!r}")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


def ours(args):
    import torch
    from pynference.models import load_model
    from pynference.parallel import TP
    tp = TP()
    tok, prompts = prompts_ids(args.model)
    g = GGUFFile(args.model)
    model = load_model(g, tp, max_ctx=1024)
    model.fast_decode = args.decode != "torch"
    model.use_graph = args.decode == "graph"
    out = []
    for name, ids in prompts:
        model.reset()
        if args.tokenwise:      # every prompt token through the fp32 decode path (no bf16 prefill)
            for t in ids:
                logits = model.forward([t])
        else:
            logits = model.forward(ids)
        steps, gen = [], []
        for _ in range(N_PREDICT):
            p = torch.softmax(logits.double(), dim=-1)
            v, i = p.topk(N_PROBS)
            top = [{"id": int(t), "token": tok.tokens[int(t)], "p": float(pv)} for pv, t in zip(v, i)]
            nxt = top[0]["id"]
            steps.append({"id": nxt, "top": top})
            if nxt in tok.eog_ids:
                break
            gen.append(nxt)
            logits = model.forward([nxt])
        out.append({"name": name, "prompt": ids, "steps": steps, "text": tok.decode(gen)})
        if tp.is_main:
            print(f"{name}: {tok.decode(gen)[:80]!r}", flush=True)
    if tp.is_main:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    tp.close()


def diff(args):
    ref, our = json.load(open(args.ref)), json.load(open(args.ours))
    for r, o in zip(ref, our):
        assert r["prompt"] == o["prompt"], r["name"]
        same, max_dp, kl_terms = 0, 0.0, []
        for rs, os_ in zip(r["steps"], o["steps"]):
            rp = {t["id"]: t["p"] for t in rs["top"]}
            op = {t["id"]: t["p"] for t in os_["top"]}
            common = rp.keys() & op.keys()
            max_dp = max([max_dp] + [abs(rp[k] - op[k]) for k in common])
            kl_terms.append(sum(rp[k] * math.log(rp[k] / op[k]) for k in common if rp[k] > 0 and op[k] > 0))
            if rs["id"] != os_["id"]:
                break
            same += 1
        n = min(len(r["steps"]), len(o["steps"]))
        print(f"{r['name']:8s} same greedy tokens {same}/{n}, max |dp| top-{N_PROBS} {max_dp:.4f}, "
              f"mean partial KL {sum(kl_terms) / max(len(kl_terms), 1):.2e}")
        print(f"   ref:  {r['text'][:100]!r}\n   ours: {o['text'][:100]!r}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fetch")
    a.add_argument("--url", default="http://localhost:8091")
    a.add_argument("-m", "--model", required=True)
    a.add_argument("-o", "--out", required=True)
    a = sub.add_parser("ours")
    a.add_argument("-m", "--model", required=True)
    a.add_argument("-o", "--out", required=True)
    a.add_argument("--tokenwise", action="store_true", help="feed the prompt one token at a time (fp32 path)")
    a.add_argument("--decode", choices=["graph", "fused", "torch"], default="graph")
    a = sub.add_parser("diff")
    a.add_argument("ref")
    a.add_argument("ours")
    args = ap.parse_args()
    {"fetch": fetch, "ours": ours, "diff": diff}[args.cmd](args)


if __name__ == "__main__":
    main()
