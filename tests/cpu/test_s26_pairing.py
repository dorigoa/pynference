"""QSA._attend GQA pairing convention (S26 step 5).

The S26-11 GPU run diverged first at attn_output-3 (cos 0.243); the cpu2
emulation (s26_nodes.py) showed the rank-local Q/K/V/gate parts bit-identical
to the TP1 slices and ONLY the attention output wrong. Cause: the GQA
pairing here is CONTIGUOUS GROUPING — q-head h pairs with kv-head
h // (n_head/n_head_kv) — torch's SDPA `enable_gqa` (repeat_interleave)
matches it, and so does llama.cpp: the S25 blk.3 dump comparison (attn nodes
cos 0.9989 at TP1 with 24 q / 2 kv) verifies this convention, while the
modular h % n_head_kv mapping would mis-feed half the heads and cannot reach
that cos. (Contrary to plan 3.10's note; the dump is ground truth.)

Consequence for TP: head_parity_rows() must NOT split QSA heads — parity
scatters each kv group's contiguous q-head block across ranks while each
rank only holds kv-heads j % world, so rank-local attention feeds half its
heads the wrong kv. QSA splits contiguously instead (rank r -> heads
[r*n_head_l, (r+1)*n_head_l), kv-head r), which is what these tests guard:
the TP-local parts a TP2 rank must produce are contiguous slices of TP1's.
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from pynference.models.qwen4exp import QSA  # noqa: E402


def _kv_feed(n_head, n_head_kv, T=5, hd=256):
    """Run _attend with each kv head's k/v constant-filled with 100+j, so
    output head h's value reveals which kv head fed it."""
    qsa = QSA.__new__(QSA)
    qsa.n_head, qsa.n_head_kv, qsa.head_dim = n_head, n_head_kv, hd
    qsa.kq_scale = 1.0 / math.sqrt(hd)
    q = (torch.arange(T * n_head * hd, dtype=torch.float32).view(T, n_head, hd)
         / 100.0 - 3.0)
    k = torch.zeros(T, n_head_kv, hd)
    v = torch.zeros(T, n_head_kv, hd)
    for j in range(n_head_kv):
        k[:, j, 0] = 100 + j
        v[:, j, 0] = 100 + j
    out = qsa._attend(q, k, v, torch.zeros(T, T))
    return [int(round(float(out[0, h, 0]))) - 100 for h in range(n_head)]


def test_gqa_pairing_is_contiguous_grouping():
    rep = 12
    assert _kv_feed(24, 2) == [h // rep for h in range(24)]
    # a case where the conventions differ mid-tensor (world=4-style 8q/2kv):
    # contiguous groups, not the modular h % n_head_kv mapping.
    assert _kv_feed(8, 2) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_gqa_pairing_rank_local_kv1():
    # the TP2 rank shape (12 q fed by 1 kv): every local head sees the
    # rank's single kv head — consistent only under the contiguous split.
    assert _kv_feed(12, 1) == [0] * 12
