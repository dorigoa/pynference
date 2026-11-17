"""CPU tests of the gpt-oss attention formula (S16 step 3): window + softmax sinks.

The Triton kernels run here under the Triton interpreter, on CPU tensors (the same
trick as test_quant_decode_interp.py): the env var must be set before any Triton
import, and no test file before this one imports Triton (collection order is
alphabetical, like the S15 TRITON_INTERPRET contract).

Random data only: the weights/model checks are the GPU job of step 4.
"""
import math
import os
import types

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import torch

from pynference import kernels
from pynference.models.gptoss import GPTOss, attn_prefill


def _ref_prefill(q, kc, vc, kp, pos, window, scale, sinks):
    """straightforward numpy transcription of S16 map sections 3 and 4, per (query, head)"""
    T, Hq, D = q.shape
    Hkv = kc.shape[0]
    G = Hq // Hkv
    out = np.zeros((T, Hq, D), dtype=np.float64)
    for i in range(T):
        for h in range(Hq):
            kv = h // G
            sc = kc[kv] @ q[i, h] * scale
            vis = (kp >= 0) & (pos[i] - kp >= 0) & (pos[i] - kp < window)
            wp = np.where(vis, sc, -np.inf)
            m = max(wp.max(), sinks[h])
            p = np.exp(wp - m)
            den = p.sum() + np.exp(sinks[h] - m)
            out[i, h] = (p[:, None] * vc[kv]).sum(0) / den
    return out


def test_attn_prefill_sinks_window():
    rng = np.random.default_rng(3)
    T, Hq, Hkv, D, W = 7, 6, 2, 8, 5
    q = rng.standard_normal((T, Hq, D))
    kc = rng.standard_normal((Hkv, 10, D)) * 0.5
    vc = rng.standard_normal((Hkv, 10, D))
    kp = np.array([-1, 0, 1, 2, 3, 4, 5, 6, -1, 8])      # empty slots + window edge
    pos = np.arange(2, 2 + T)                             # queries 2..8
    scale = 1.0 / np.sqrt(D)
    sinks = rng.standard_normal(Hq) * 2.0
    got = attn_prefill(torch.tensor(q, dtype=torch.float32),
                       torch.tensor(kc, dtype=torch.float16),
                       torch.tensor(vc, dtype=torch.float16),
                       torch.tensor(kp), torch.tensor(pos), W, scale,
                       torch.tensor(sinks, dtype=torch.float32)).double().numpy()
    # the reference must see the same fp16-rounded cache the implementation got
    _ref = _ref_prefill(q, kc.astype(np.float16).astype(np.float64),
                        vc.astype(np.float16).astype(np.float64), kp, pos, W, scale, sinks)
    np.testing.assert_allclose(got, _ref.reshape(T, -1), rtol=0, atol=2e-5)
    # the window really bites: with W=5 a query sees at most 5 keys
    vis = [(pos[i] - kp >= 0) & (pos[i] - kp < W) & (kp >= 0) for i in range(T)]
    assert min(int(v.sum()) for v in vis) <= W


def test_attn_prefill_no_sinks_dense_matches_softmax():
    torch.manual_seed(0)
    T, Hq, Hkv, D = 5, 4, 2, 16
    q = torch.randn(T, Hq, D)
    kc = (torch.randn(Hkv, T, D) * 0.5).half()
    vc = torch.randn(Hkv, T, D).half()
    pos = torch.arange(T)
    got = attn_prefill(q, kc, vc, pos, pos, 0, 1.0, None)
    G = Hq // Hkv
    ref = torch.zeros(T, Hq * D)
    for kv in range(Hkv):
        s = (q[:, kv * G:(kv + 1) * G].transpose(0, 1).float()
             @ kc[kv].float().transpose(0, 1))            # [G, T, T]
        mask = pos[None, :] <= pos[:, None]
        p = s.masked_fill(~mask[None], float("-inf")).softmax(-1)
        o = p @ vc[kv].float()                            # [G, T, D]
        ref[:, kv * G * D:(kv + 1) * G * D] = o.transpose(0, 1).reshape(T, G * D)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=0)


def test_attn_decode_sinks_matches_prefill():
    torch.manual_seed(1)
    Hq, Hkv, D, S, n = 4, 2, 16, 64, 23
    q = torch.randn(Hq, D)
    kc = (torch.randn(Hkv, S, D) * 0.5).half()
    vc = torch.randn(Hkv, S, D)
    sinks = torch.randn(Hq) * 2.0
    o = kernels.attn_decode(q, kc, vc, torch.tensor([n]), 1, 64, sinks=sinks)
    # same keys as one prefill row: keys 0..n-1 at positions 0..n-1, query at n-1, dense, scale 1.0
    kp = torch.full((S,), -1)
    kp[:n] = torch.arange(n)
    pre = attn_prefill(q.view(1, Hq, D), kc, vc, kp, torch.tensor([n - 1]), 0, 1.0, sinks)
    torch.testing.assert_close(o.reshape(-1), pre[0], atol=2e-4, rtol=0)


def test_attn_decode_ring_matches_prefill_window():
    """decode steps through the ring cache == prefill of the same history (window layer)"""
    torch.manual_seed(2)
    Hq, Hkv, D, W, N = 4, 2, 16, 8, 20
    q = torch.randn(N, Hq, D)
    kc = torch.zeros(Hkv, W, D)
    vc = torch.zeros(Hkv, W, D)
    sinks = torch.randn(Hq)
    new_k = torch.randn(N, Hkv, D) * 0.5
    new_v = torch.randn(N, Hkv, D)
    for p in range(N):
        kc[:, p % W] = new_k[p]
        vc[:, p % W] = new_v[p]
        n_len = min(p + 1, W)
        o = kernels.attn_decode(q[p], kc, vc, torch.tensor([n_len]), 1, W, sinks=sinks)
        # slot j holds the latest position <= p that is == j (mod W), else a negative one
        kp = torch.tensor([(p - ((p - j) % W)) for j in range(W)])
        pre = attn_prefill(q[p:p + 1], kc, vc, kp, torch.tensor([p]), W, 1.0, sinks)
        torch.testing.assert_close(o.reshape(-1), pre[0], atol=2e-4, rtol=0)


def test_model_decode_attention_matches_prefill_scale():
    """C8 guard (S17): the real GPTOss._attention_decode must score with self._scale.

    kernels.attn_decode scores at scale 1.0 by design (gemma4 convention), so the
    model has to hand it q * _scale; the two kernel-vs-prefill tests above run both
    sides at 1.0 and cannot see a lost scale in the model wiring. Here the model's
    own decode method runs (on a stub with just the attributes it reads) against
    attn_prefill with the model's own scale, same cache, same history, same sinks."""
    torch.manual_seed(4)
    Hq, Hkv, D, S, n = 4, 2, 16, 48, 23
    scale = 1.0 / math.sqrt(D)
    sinks = torch.randn(Hq) * 2.0
    kc = (torch.randn(Hkv, S, D) * 0.5).half()
    vc = torch.randn(Hkv, S, D)
    kc[:, n - 1:] = 0.0                       # only slots 0..n-2 hold history
    vc[:, n - 1:] = 0.0
    q = torch.randn(1, Hq, D)
    k = torch.randn(1, Hkv, D)
    v = torch.randn(1, Hkv, D)
    stub = types.SimpleNamespace(
        cfg=types.SimpleNamespace(window=8),
        layers={0: types.SimpleNamespace(swa=False, sinks=sinks)},
        cache={0: (kc, vc)},
        _split_full=S, _nsp_full=1, _scale=scale,
    )
    pos = torch.tensor([n - 1])
    got = GPTOss._attention_decode(stub, 0, q, k, v, pos)
    # the identical history through the prefill path the model uses for the prompt:
    # keys 0..n-1 (index_copy_ above stored our k/v into slot n-1 in place), query n-1
    kp = torch.full((S,), -1)
    kp[:n] = torch.arange(n)
    pre = attn_prefill(q, kc, vc, kp, pos, 0, scale, sinks)
    torch.testing.assert_close(got.reshape(-1), pre[0], atol=2e-4, rtol=0)
