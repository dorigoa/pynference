"""S16 step 4 (CPU part): the layer-0 attention reference transcribed in
tests/gpu/_gptoss_attn0_worker.py, checked on synthetic small data against the
validated gptoss helpers (rms_norm, rope_neox, attn_prefill — tested in step 3
against direct transcriptions of llama.cpp), plus the eval-callback dump parser
against synthetic dumps (first/last 3 columns and full rows, truncated rows).
No model file, no GPU needed."""
import importlib.util
import math
import os

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "tests", "gpu", "_gptoss_attn0_worker.py")

_spec = importlib.util.spec_from_file_location("gptoss_attn0_worker", WORKER)
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)   # main() is guarded: importing is safe

n_embd, nq, nkv, D = 4, 4, 2, 4   # n_rot == head_dim, like the real model
n_rot, T, eps, window = 4, 4, 1e-5, 3
_rng = np.random.default_rng(5)
W = {n: (_rng.standard_normal(s) * 0.3).astype(np.float32)
     for n, s in {"blk.0.attn_q.weight": (nq * D, n_embd),
                  "blk.0.attn_k.weight": (nkv * D, n_embd),
                  "blk.0.attn_v.weight": (nkv * D, n_embd),
                  "blk.0.attn_output.weight": (n_embd, nq * D)}.items()}
B = {n: (_rng.standard_normal(s) * 0.1).astype(np.float32)
     for n, s in {"blk.0.attn_q.bias": nq * D, "blk.0.attn_k.bias": nkv * D,
                  "blk.0.attn_v.bias": nkv * D, "blk.0.attn_norm.weight": n_embd,
                  "blk.0.attn_output.bias": n_embd, "blk.0.attn_sinks.weight": nq}.items()}
EMB = (_rng.standard_normal((7, n_embd)) * 0.5).astype(np.float32)


class _T:
    def __init__(self, shape):
        self.shape = shape          # gguf logical [K, N]: rows() slices N


class _G:
    def __init__(self):
        self.tensors = {n: _T((a.shape[1], a.shape[0]))
                        for n, a in {**W, "token_embd.weight": EMB}.items()}

    def rows(self, name, r0, r1):
        return ("MARK", name, r0, r1)

    def raw(self, name):
        return np.frombuffer(B[name].tobytes(), dtype=np.uint8)


def _ref_deq(raw, _qt):
    arr = EMB if raw[1] == "token_embd.weight" else W[raw[1]]
    return arr[raw[2]:raw[3]]


def test_worker_reference_matches_gptoss_helpers(monkeypatch):
    monkeypatch.setattr(w, "ref_dequant", _ref_deq)
    ids = [6, 0, 3, 1]
    cfg = {"n_embd": n_embd, "n_head": nq, "n_head_kv": nkv, "head_dim": D, "eps": eps,
           "rope_base": 150000.0, "yarn_factor": 32.0, "yarn_orig_ctx": 16,
           "yarn_beta_fast": 32.0, "yarn_beta_slow": 1.0, "yarn_attn_factor": 1.0,
           "n_rot": n_rot, "window": window, "is_swa": [True, False]}
    out = w.reference(_G(), cfg, ids)

    from pynference.models.gptoss import attn_prefill, rms_norm, rope_neox, rope_yarn_table
    x = torch.from_numpy(EMB[ids])
    h = rms_norm(x, torch.from_numpy(B["blk.0.attn_norm.weight"]), eps)
    q = (h @ torch.from_numpy(W["blk.0.attn_q.weight"]).T
         + torch.from_numpy(B["blk.0.attn_q.bias"])).view(T, nq, D)
    k = (h @ torch.from_numpy(W["blk.0.attn_k.weight"]).T
         + torch.from_numpy(B["blk.0.attn_k.bias"])).view(T, nkv, D)
    v = (h @ torch.from_numpy(W["blk.0.attn_v.weight"]).T
         + torch.from_numpy(B["blk.0.attn_v.bias"])).view(T, nkv, D)
    cos, sin = rope_yarn_table(n_rot, 150000.0, 32.0, 16, 32.0, 1.0, T, torch.device("cpu"))
    q, k = rope_neox(q, cos, sin), rope_neox(k, cos, sin)
    pos = torch.arange(T)
    a = attn_prefill(q, k.transpose(0, 1), v.transpose(0, 1), pos, pos, window,
                     1.0 / math.sqrt(D), torch.from_numpy(B["blk.0.attn_sinks.weight"]))
    ref = (a @ torch.from_numpy(W["blk.0.attn_output.weight"]).T
           + torch.from_numpy(B["blk.0.attn_output.bias"])).numpy()

    c = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref)))
    rel = float(np.abs(out - ref).max() / np.abs(ref).max())
    assert c > 0.999999 and rel < 1e-4, (c, rel)


def _dump_text(ne0, ne1, ids, gap_rows, stamp=False):
    """Synthesize the eval-callback block for attn_out-0 (common/debug.cpp):
    header + printed matrix; gap_rows=True -> first/last 3 values per row;
    stamp=True -> llama.cpp log stamps on the token lines (real dumps, S16-2:
    "0.28.208.200 I number of input tokens = 5", "0.28.208.203 I   976")."""
    val = lambda t, i: 0.05 * t + 0.001 * i
    ts = list(range(ne1)) if ne1 <= 6 else [0, 1, 2, ne1 - 3, ne1 - 2, ne1 - 1]
    p = "0.28.208.200 I " if stamp else "0 [main] "
    L = [f"{p}number of input tokens = {ne1}"]
    L += [(f"0.28.208.203 I {i:6d}" if stamp else f"  {i}") for i in ids]
    L += ["0 [main] common_debug_cb_eval:               attn_out-0 = (f32)"
          "      ADD(x{1, 1, 1, 1}, ) = {%d, %d, 1, 1}" % (ne0, ne1)]
    L += ["    [", "        ["]
    for t in ts:
        if ne1 > 6 and t == ne1 - 3:
            L.append("          ..., ")      # i1 truncation marker
        cols = ([0, 1, 2, ne0 - 3, ne0 - 2, ne0 - 1] if gap_rows else list(range(ne0)))
        body = ", ".join(f"{val(t, i):12.4f}" for i in (cols[:3] if gap_rows else cols))
        if gap_rows:
            body += ",   ..., " + ", ".join(f"{val(t, i):12.4f}" for i in cols[3:])
        L.append("            [" + body + "  ],")
    L += ["        ],", "    ]", "    sum = 0.000000"]
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("ne1,gap_rows,stamp",
                         [(5, True, False), (9, True, False), (5, False, False),
                          (5, True, True), (9, True, True)])
def test_parse_dump_synthetic(tmp_path, ne1, gap_rows, stamp):
    ids = [1000 + t for t in range(ne1)]
    path = tmp_path / "dump.txt"
    path.write_text(_dump_text(2880, ne1, ids, gap_rows, stamp))
    got_ids, ne, rows = w.parse_dump(str(path))
    assert got_ids == ids and ne == (2880, ne1, 1, 1)
    want = list(range(ne1)) if ne1 <= 6 else [0, 1, 2, ne1 - 3, ne1 - 2, ne1 - 1]
    assert sorted(rows) == want
    for t in want:
        cols, vv = rows[t]
        if gap_rows:
            assert cols == [0, 1, 2, 2877, 2878, 2879]
        assert vv == pytest.approx([round(0.05 * t + 0.001 * i, 4) for i in cols], abs=1e-4)
