"""CPU unit tests for pynference/models/qwen35.py (S21 step 1): config from
GGUF metadata (facts.json values) and the DeltaNet head-gather index
computation of the S19 TP plan. No model file, no GPU.

TRITON_INTERPRET is set before the imports (S20 lesson: pytest collects
alphabetically, so any test file that can transitively import Triton must
enable the interpreter itself)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np

from pynference.models.qwen35 import (Qwen35Config, head_parity_rows,
                                      merge_runs, qkv_parity_runs)


class _T:
    def __init__(self, shape, type="Q8_0", nbytes=0):
        self.shape, self.type, self.nbytes = shape, type, nbytes


class _G:
    """Stands in for GGUFFile with the qwen35 keys of tests/golden/qwen35-27b/facts.json."""
    arch = "qwen35"

    def __init__(self):
        self.kv = {
            "qwen35.block_count": 65,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.embedding_length": 5120,
            "qwen35.feed_forward_length": 17408,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.rope.dimension_count": 64,
            "qwen35.rope.freq_base": 10000000.0,
            "qwen35.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
            "qwen35.context_length": 262144,
            "qwen35.full_attention_interval": 4,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.group_count": 16,
            "qwen35.ssm.time_step_rank": 48,
            "qwen35.ssm.inner_size": 6144,
        }
        self.tensors = {"token_embd.weight": _T((5120, 248320))}

    def get(self, key, default=None):
        return self.kv.get(f"{self.arch}.{key}", self.kv.get(key, default))


def _cfg():
    return Qwen35Config.from_gguf(_G())


def test_config_from_gguf():
    c = _cfg()
    assert c.n_layer == 64                      # blk.64 (MTP) excluded, D4
    assert len(c.is_attn) == 64 and sum(c.is_attn) == 16
    assert c.is_attn[3] and c.is_attn[63]       # attention at il = 3, 7, ..., 63
    assert not c.is_attn[0] and not c.is_attn[62]
    assert (c.n_embd, c.ffn, c.head_dim, c.n_rot) == (5120, 17408, 256, 64)
    assert (c.n_head, c.n_head_kv) == (24, 4)
    assert (c.n_key_heads, c.n_value_heads, c.ssm_state, c.conv_kernel) == (16, 48, 128, 4)
    assert c.n_vocab == 248320 and c.n_ctx_train == 262144
    assert c.rope_base == 1e7
    assert abs(c.eps - 1e-6) < 1e-12


def test_head_parity_partition():
    for world in (1, 2):
        for rank in range(world):
            rows = head_parity_rows(48, 128, world, rank)
            idx = np.concatenate([np.arange(a, a + l) for a, l in rows])
            assert len(idx) == 48 * 128 // world
            assert (idx // 128 % world == rank).all()      # only parity-rank heads
            assert np.unique(idx).size == idx.size          # no duplicates
    # world=1 merges everything back to one contiguous run per region
    assert head_parity_rows(48, 128, 1, 0) == [(0, 48 * 128)]
    assert merge_runs([[0, 4], [4, 4], [12, 2]]) == [(0, 8), (12, 2)]


def _expand(runs):
    return np.concatenate([np.arange(a, a + l) for a, l in runs])


def test_qkv_parity_runs():
    for world in (1, 2):
        for rank in range(world):
            qkv, vruns, ids, vrows = qkv_parity_runs(16, 48, 128, world, rank)
            idx = _expand(qkv)
            n_qk, n_v = 16 * 128 // world, 48 * 128 // world
            assert idx.size == 2 * n_qk + n_v               # per-rank conv channels
            q = idx[idx < 2 * 16 * 128]                     # q|k regions
            qh = set(q[q < 16 * 128] // 128)
            kh = set(q[q >= 16 * 128] // 128 - 16)
            assert qh == kh == set(range(rank, 16, world))  # key heads once, parity rank
            v = idx[idx >= 2 * 16 * 128]
            assert list(ids) == sorted(ids) and ids.size == 48 // world
            assert list(_expand(vruns)) == [c for j in ids for c in range(j * 128, j * 128 + 128)]
            assert np.array_equal((v - 2 * 16 * 128) // 128, np.repeat(ids, 128))  # gate/conv = value heads
            assert list(vrows) == [(int(j), 1) for j in ids]
    # world=1: the qkv gather is the identity (single contiguous run)
    assert qkv_parity_runs(16, 48, 128, 1, 0)[0] == [(0, 10240)]


def test_local_key_rule_matches_global_mod():
    # inside a rank the kernel uses h % n_key_l; check it reproduces the
    # global value-head -> key-head rule j % H_k (S19 map A.12 + "Piano TP")
    H_k, world = 16, 2
    for rank in range(world):
        ids = np.arange(rank, 48, world)                    # global value head per local h
        for h, j in enumerate(ids):
            gk = j % H_k                                    # global key head
            assert gk % world == rank                       # stays on this rank...
            assert gk // world == h % (H_k // world)        # ...and equals the kernel rule h % n_key_l
