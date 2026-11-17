"""Worker for tests/gpu/test_gptoss_attn0.py: run under torchrun with two xpu ranks.

Computes the layer-0 attention block (the llama.cpp "attn_out-0", openai-moe.cpp:100-103,
so INCLUDING wo and wo_b) with the real TP engine on two GPUs, and compares it with:
- a CPU reference: gguf-py Q8_0 dequant of the layer-0 attention weights, f32 CPU math,
  an independent transcription of the YaRN/RoPE/sinks/window formulas (S16 map);
- the `llama_ref.sh dump` of the same prompt (eval-callback values of attn_out-0,
  first/last 3 columns of each token row) when GPTOSS_DUMP points at the dump file.
Rank 0 prints "RES ..." lines and "ALL OK"; nonzero exit on failure.
"""
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

os.environ.setdefault("PYI_GPTOSS_SKIP_MOE", "1")   # layer-0 attention only: no expert store

MODEL = os.environ.get("GPTOSS_GGUF", "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf")
DUMP = os.environ.get("GPTOSS_DUMP")
COS_TOL = 0.9999         # plan S16 step 4: GPU vs CPU reference
DUMP_COS_TOL = 0.999     # vs llama.cpp dump (bf16 kq accumulation, 4-decimal print)

from pynference.gguf import GGUFFile          # noqa: E402
from pynference.models.gptoss import GPTOss   # noqa: E402
from pynference.parallel import TP            # noqa: E402
from pynference.quant.ref import ref_dequant  # noqa: E402


def parse_dump(path):
    """eval-callback dump (common/debug.cpp): token ids printed by eval-callback,
    and the sampled values of attn_out-0: {token: ([col,...], [value,...])}."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    ids, ne = None, None
    for i, ln in enumerate(lines):
        m = re.search(r"number of input tokens = (\d+)", ln)
        if m:
            # llama.cpp prefixes every printed id with a log stamp (S16-2: "0.28.208.203 I   976")
            n = int(m.group(1))
            ids = []
            for j in range(n):
                s = re.search(r"(\d+)\s*$", lines[i + 1 + j])
                if not s:
                    raise AssertionError(f"dump: bad token line: {lines[i + 1 + j]!r}")
                ids.append(int(s.group(1)))
        m = re.search(r"attn_out-0\s*=\s*\(f32\).*?\{\s*(\d+), (\d+), (\d+), (\d+)\}\s*$", ln)
        if m and "attn_out-0{" not in ln:
            ne = tuple(int(x) for x in m.groups())
            first = i + 1
    if ids is None or ne is None:
        raise AssertionError("dump: token list or attn_out-0 header not found")
    n0, n1 = ne[0], ne[1]
    rows, marker = [], False
    for ln in lines[first:]:
        if ln.strip().startswith("sum ="):
            break
        if re.fullmatch(r"\s*\.\.\.,\s*", ln):
            marker = True              # i1 truncation marker between row groups
            continue
        m = re.match(r"^\s*\[(.*)\s*\],\s*$", ln)
        if not m:
            continue
        items = [x.strip() for x in m.group(1).split(",")]
        vals, gap = [], "..." in items  # gap = i0 truncation inside this row
        for it in items:
            if it and it != "...":
                vals.append(float(it))
        if not vals:
            continue
        rows.append((gap, vals))
    if any(len(v) != (6 if g else n0) for g, v in rows):
        raise AssertionError(f"dump: unexpected row lengths {[len(v) for _, v in rows]}")
    if len(rows) == n1:
        tmap = list(range(n1))
    elif len(rows) == 6 and n1 > 6 and marker:
        tmap = [0, 1, 2, n1 - 3, n1 - 2, n1 - 1]
    else:
        raise AssertionError(f"dump: got {len(rows)} rows for n1={n1}")
    cols = ([0, 1, 2, n0 - 3, n0 - 2, n0 - 1] if rows[0][0] else list(range(n0)))
    return ids, ne, {t: (cols, v) for t, (_, v) in zip(tmap, rows)}


def reference(g, cfg, ids):
    """CPU f32 reference of the layer-0 attention block, formulas transcribed from
    S16-map.md (weights dequantized with gguf-py)."""
    f32 = np.float32
    p, T = "blk.0.", len(ids)
    n_embd, nq, nkv, D = cfg["n_embd"], cfg["n_head"], cfg["n_head_kv"], cfg["head_dim"]
    deq = lambda name: ref_dequant(g.rows(name, 0, g.tensors[name].shape[1]), "Q8_0")
    f32t = lambda name: np.frombuffer(g.raw(name), dtype=np.float32).reshape(-1)

    x = np.concatenate([ref_dequant(g.rows("token_embd.weight", i, i + 1), "Q8_0") for i in ids])
    h = x * np.reciprocal(np.sqrt((x * x).mean(1, keepdims=True) + f32(cfg["eps"]))) \
        * f32t(p + "attn_norm.weight")
    q = (h @ deq(p + "attn_q.weight").T + f32t(p + "attn_q.bias")).reshape(T, nq, D)
    k = (h @ deq(p + "attn_k.weight").T + f32t(p + "attn_k.bias")).reshape(T, nkv, D)
    v = (h @ deq(p + "attn_v.weight").T + f32t(p + "attn_v.bias")).reshape(T, nkv, D)

    # YaRN angles (rope_yarn in ggml-cpu/ops.cpp:6076-6114, corr_dims in ggml.c:4373-4389)
    base, fac, orig = cfg["rope_base"], cfg["yarn_factor"], cfg["yarn_orig_ctx"]
    n_rot = cfg["n_rot"]
    dim = lambda b: n_rot * np.log(orig / (b * 2 * np.pi)) / (2 * np.log(base))
    c0 = max(0, int(np.floor(dim(cfg["yarn_beta_fast"]))))
    c1 = min(n_rot - 1, int(np.ceil(dim(cfg["yarn_beta_slow"]))))
    kk = np.arange(n_rot // 2, dtype=np.float64)
    ramp = np.clip(1.0 - (kk - c0) / max(0.001, c1 - c0), 0.0, 1.0)
    scale = np.float64(1.0 / fac) * (1.0 - ramp) + ramp          # ext_factor = 1
    mscale = cfg["yarn_attn_factor"] * (1.0 + 0.1 * np.log(fac))
    ang = np.arange(T, dtype=np.float64)[:, None] * (base ** (-2.0 * kk / n_rot) * scale)[None, :]
    cos = (mscale * np.cos(ang)).astype(f32)[:, None, :]
    sin = (mscale * np.sin(ang)).astype(f32)[:, None, :]

    def neox(a):  # pairs (i, i + n_rot/2)
        a1, a2 = a[..., :n_rot // 2], a[..., n_rot // 2:n_rot]
        return np.concatenate([a1 * cos - a2 * sin, a2 * cos + a1 * sin], -1)

    q, k = neox(q), neox(k)
    sinks = f32t(p + "attn_sinks.weight")
    scale_at = f32(1.0 / np.sqrt(D))
    win = cfg["window"] if cfg["is_swa"][0] else 10 ** 9
    G = nq // nkv
    o = np.empty((T, nq, D), dtype=f32)
    for t in range(T):
        vis = np.array([j for j in range(t + 1) if t - j < win])
        for hh in range(nq):
            s = (q[t, hh] @ k[vis, hh // G].T) * scale_at
            m = max(s.max(), sinks[hh])
            e = np.exp(s - m)
            o[t, hh] = (e @ v[vis, hh // G]) / (e.sum() + np.exp(sinks[hh] - m))
    out = o.reshape(T, nq * D) @ deq(p + "attn_output.weight").T + f32t(p + "attn_output.bias")
    return out.astype(np.float32)


def main():
    tp = TP()
    g = GGUFFile(MODEL)
    model = GPTOss(g, tp)
    model.reset()
    c = model.cfg
    if tp.is_main:
        print(f"cfg: n_embd={c.n_embd} heads={c.n_head}/{c.n_head_kv} D={c.head_dim} "
              f"window={c.window} is_swa[0]={c.is_swa[0]}", flush=True)
    assert c.is_swa[0], "layer 0 is expected to be windowed (swa pattern 2)"

    dump = None
    if DUMP:
        ids, ne, dump = parse_dump(DUMP)
        assert ne[0] == c.n_embd, ne
        if tp.is_main:
            print(f"dump: {len(ids)} tokens {ids}, ne={ne}", flush=True)
    else:
        import json
        with open(os.path.join(ROOT, "tests", "golden", "gpt-oss-120b", "tokens.json")) as f:
            ids = json.load(f)[0]["ids"]
    T = len(ids)
    ids_t = torch.tensor(ids, dtype=torch.long, device=tp.device)
    x = model.embed_tokens(ids_t)
    a = model._layer_attn(0, x, torch.arange(T, device=tp.device), 0).float().cpu().numpy()

    ref = reference(g, {"n_embd": c.n_embd, "n_head": c.n_head, "n_head_kv": c.n_head_kv,
                        "head_dim": c.head_dim, "eps": c.eps, "rope_base": c.rope_base,
                        "yarn_factor": c.yarn_factor, "yarn_orig_ctx": c.yarn_orig_ctx,
                        "yarn_beta_fast": c.yarn_beta_fast, "yarn_beta_slow": c.yarn_beta_slow,
                        "yarn_attn_factor": c.yarn_attn_factor, "n_rot": c.n_rot,
                        "window": c.window, "is_swa": c.is_swa}, ids)
    cos = float((a * ref).sum() / (np.linalg.norm(a) * np.linalg.norm(ref)))
    err = torch.tensor([np.abs(a - ref).max() / np.abs(ref).max()],
                       dtype=torch.float32, device=tp.device)
    tp.all_reduce(err)  # both ranks ran the same TP block: max abs rel err, summed
    if tp.is_main:
        print(f"RES cos_attn0={cos:.7f} maxrel_sum_ranks={err.item():.3e} T={T}", flush=True)
    # plan S16-4 threshold: cos >= COS_TOL. The maxrel guard is only a gross-error net:
    # the engine matmul/KV is f16 while this reference is f32, so on real Q8_0 weights
    # maxrel_sum_ranks is ~4.5e-03 (S16-1/-2) even though cos is 0.9999955.
    ok = cos >= COS_TOL and err.item() < 2e-2
    assert ok, f"attn_out-0 vs reference: cos={cos:.7f} tol={COS_TOL} maxrel_sum={err.item():.3e}"

    if dump is not None:
        dv = np.concatenate([v for _, (cols, v) in sorted(dump.items())])
        av = np.concatenate([a[t, cols] for t, (cols, _) in sorted(dump.items())])
        dcos = float((av * dv).sum() / (np.linalg.norm(av) * np.linalg.norm(dv)))
        rel = np.abs(av - dv) / max(1e-6, float(np.abs(dv).max()))
        mx = int(np.argmax(np.abs(av - dv)))
        if tp.is_main:
            print(f"RES dump_cos={dcos:.7f} dump_maxrel={rel.max():.3e} "
                  f"worst ours={av[mx]:.4f} ref={dv[mx]:.4f}", flush=True)
        assert dcos >= DUMP_COS_TOL, f"attn_out-0 vs dump: cos={dcos:.7f} tol={DUMP_COS_TOL}"

    if tp.is_main:
        print("ALL OK", flush=True)
    tp.close()


if __name__ == "__main__":
    main()
