"""S17 step 2 (CPU): the GPTOss C3 driver (forward / decode / memory_report) on a
tiny stub model — real GPTOss methods, fake tensors and fake MoE/attention-decode
(no GGUF file, no GPU). Checks the wiring: residuals, norms, positions, KV ring,
chunked prefill vs per-token decode, logits head, context limits.
The Triton attention-decode kernel is replaced with attn_prefill (already
validated in S16) using the same ring semantics as _attention."""
import math

import pytest
import torch

from pynference.models import gptoss
from pynference.models.gptoss import (GPTOss, GPTOSSConfig, _Layer, attn_prefill,
                                      parse_meminfo, rope_yarn_table)

n_embd, nq, nkv, D = 8, 4, 2, 4        # n_rot == head_dim, like the real model
n_vocab, window, eps, max_ctx = 11, 3, 1e-5, 12
n_layer = 3


class TP1:
    world, rank, is_main = 1, 0, True
    device = torch.device("cpu")

    @staticmethod
    def split(n):
        return 0, n

    @staticmethod
    def all_reduce(t):
        return t

    @staticmethod
    def all_gather(t):
        return t


class _Embed:
    def __init__(self, emb):
        self.emb = emb

    def rows_dequant(self, idx):
        return self.emb[idx]


class _Store:
    def memory_bytes(self):
        return 12345

    def pinned_bytes(self):
        return 6789


class _MoE:
    def __init__(self, W):
        self.W = W

    def prefill(self, x):
        return x @ self.W.T

    # S18 graph-piece decode API (mirrors pynference.moe.MoE): the driver
    # reads _logits on the host, then runs decode_body per resolve batch,
    # accumulating _d["y"]. One "expert" carrying the whole [k] output keeps
    # the stub equivalent to its prefill twin (x @ W.T).
    def _logits(self, x):
        return torch.zeros(1, 1, dtype=torch.float32, device=x.device)

    def _init_decode_bufs(self, dev):
        self._d = {"y": torch.zeros(self.W.shape[0], dtype=torch.float32, device=dev)}

    def decode_route(self, logits):
        return [0], [1.0]

    def decode_fill(self, ids, w):
        return 1

    def decode_fill_batch(self, bi):
        pass

    def decode_body(self, x):
        self._d["y"] += (x @ self.W.T).reshape(-1)
        return self._d["y"]


def _fake_attention_decode(self, il, q, k, v, pos):
    """T=1 attention on the same cache as _attention, via attn_prefill."""
    L, c = self.layers[il], self.cfg
    kc, vc = self.cache[il]
    p = int(pos)
    if L.swa:
        W = c.window
        kc[:, p % W] = k[0].to(kc.dtype)
        vc[:, p % W] = v[0].to(vc.dtype)
        j = torch.arange(W)
        kp = p - torch.remainder(p - j, W)   # slot j = latest written position == j (mod W)
        kcache, vcache, win = kc, vc, W
    else:
        kc[:, p] = k[0].to(kc.dtype)
        vc[:, p] = v[0].to(vc.dtype)
        kcache, vcache = kc[:, :p + 1], vc[:, :p + 1]
        kp, win = torch.arange(p + 1), 0
    o = attn_prefill(q, kcache, vcache, kp, pos, win, self._scale, L.sinks)
    return o.view(q.shape[1], D)


def _stub(seed=7):
    torch.manual_seed(seed)
    m = GPTOss.__new__(GPTOss)
    m.cfg = GPTOSSConfig(
        n_layer=n_layer, n_embd=n_embd, n_head=nq, n_head_kv=nkv, head_dim=D,
        n_rot=D, window=window, is_swa=[il % 2 == 0 for il in range(n_layer)],
        rope_base=150000.0, yarn_factor=32.0, yarn_orig_ctx=16, yarn_beta_fast=32.0,
        yarn_beta_slow=1.0, yarn_ext_factor=1.0, yarn_attn_factor=1.0, eps=eps,
        n_vocab=n_vocab, n_ctx_train=64)
    m.tp, m.max_ctx, m.kv_dtype = TP1(), max_ctx, torch.float32
    m.n_head_l, m.n_kv_l = nq, nkv
    m._scale = 1.0 / math.sqrt(D)
    m._split_swa, m._nsp_swa = window, 1
    m._split_full, m._nsp_full = max_ctx, 1
    m.fast_decode, m.use_graph = True, False
    m.v0, m.v1 = 0, n_vocab
    m.embed = _Embed(torch.randn(n_vocab, n_embd))
    Wout = torch.randn(n_vocab, n_embd)
    m.lm_head = lambda x: x @ Wout.T
    m.output_norm = torch.rand(n_embd) + 0.5
    m.cos, m.sin = rope_yarn_table(D, 150000.0, 32.0, 16, 32.0, 1.0, max_ctx, m.tp.device)
    m.layers = []
    for il in range(n_layer):
        L = _Layer()
        L.swa = m.cfg.is_swa[il]
        Wq, Wk, Wv = (torch.randn(n * D, n_embd) for n in (nq, nkv, nkv))
        L.qkv = lambda h, W=torch.cat([Wq, Wk, Wv]): h @ W.T
        L.qkv_bias = torch.cat([torch.randn(n * D) * 0.1 for n in (nq, nkv, nkv)])
        Wo = torch.randn(n_embd, nq * D)
        L.o = lambda a, W=Wo: a @ W.T
        L.o_bias = torch.randn(n_embd) * 0.1
        L.sinks = torch.randn(nq)
        L.attn_norm = torch.rand(n_embd) + 0.5
        L.post_attn_norm = torch.rand(n_embd) + 0.5
        m.layers.append(L)
    m.cache = None
    m.reset()
    m.store = _Store()
    m.moe = [_MoE(torch.randn(n_embd, n_embd) * 0.1) for _ in range(n_layer)]
    m._weights_bytes, m._scratch_bytes = 1000, 2000
    m._diag_n = 0
    return m


def test_forward_decode_equivalence(monkeypatch):
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub()
    ids = [5, 0, 9, 3, 11, 7, 2]
    m.reset()
    ref = m.forward(ids)
    assert ref.shape == (n_vocab,) and ref.dtype == torch.float32
    m.reset()
    assert m.n_past == 0
    last = None
    for t in ids:
        last = m.decode(t)
        assert last.shape == (n_vocab,)
    assert m.n_past == len(ids)
    assert torch.allclose(ref, last, atol=1e-4), (ref - last).abs().max()


def test_chunked_all_logits_matches_decode(monkeypatch):
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=11)
    ids = [4, 10, 1, 6, 0, 8, 3, 9]
    m.reset()
    la = m.forward(ids, chunk=2, all_logits=True)
    assert la.shape == (len(ids), n_vocab)
    m.reset()
    for i, t in enumerate(ids):
        assert torch.allclose(la[i], m.decode(t), atol=1e-4)
    m.reset()
    assert torch.allclose(m.forward(ids), la[-1], atol=1e-4)


def test_forward_single_token_uses_decode(monkeypatch):
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=13)
    m.reset()
    m.decode(6)
    ref = m.decode(2)
    m.reset()
    m.forward([6])
    got = m.forward([2])                     # fast path -> decode
    assert torch.allclose(ref, got, atol=0)


def test_context_full(monkeypatch):
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=17)
    m.reset()
    with pytest.raises(ValueError, match="context full"):
        m.forward(list(range(13)))
    m.reset()
    m.forward(list(range(max_ctx)))
    with pytest.raises(ValueError, match="context full"):
        m.decode(1)


def test_reset_reuses_cache_and_clears_history(monkeypatch):
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=19)
    m.reset()
    cache = m.cache
    a = m.forward([2, 7, 1])
    m.reset()
    assert m.n_past == 0 and m.cache is cache
    b = m.forward([2, 7, 1])
    assert torch.allclose(a, b, atol=0)


def test_memory_report_keys():
    m = _stub(seed=23)
    r = m.memory_report()
    assert r["weights"] == 1000 and r["experts"] == 12345 and r["scratch"] == 2000
    assert r["cache"] == m.kv_bytes() and r["pinned_host"] == 6789
    assert r["device_total"] == 1000 + 12345 + m.kv_bytes() + 2000
    swa_layers = sum(1 for L in m.layers if L.swa)
    want = (swa_layers * window + (n_layer - swa_layers) * max_ctx) * nkv * D * 4 * 2
    assert r["cache"] == want


def test_dense_prefill_is_fp32_from_q8_0_bytes():
    # S17-9: the prompt-path q/k/v/o run in fp32 rebuilt from the Q8_0 int8
    # values and fp16 scales (QLinear's own M > 1 path is bf16: C8 near-ties)
    import numpy as np
    from types import SimpleNamespace
    from pynference.models.gptoss import dense_prefill
    from pynference.quant.ref import ref_dequant
    from pynference.quant.testing import make_test_blocks
    n, k = 48, 96
    raw = make_test_blocks("Q8_0", n, k, seed=7)
    blocks = torch.from_numpy(np.array(raw, copy=True)).view(n, k // 32, 34)
    lin = SimpleNamespace(qs=blocks[:, :, 2:].contiguous().view(torch.int8).view(n, k),
                          d=blocks[:, :, :2].contiguous().view(torch.float16).view(n, k // 32))
    x = torch.randn(5, k, generator=torch.Generator().manual_seed(1))
    w = torch.from_numpy(ref_dequant(raw, "Q8_0").reshape(n, k))
    torch.testing.assert_close(dense_prefill(lin, x), x @ w.T, rtol=1e-6, atol=1e-6)
    assert not torch.equal(dense_prefill(lin, x),
                           (x.bfloat16() @ w.bfloat16().T).float())  # the old bf16 path
    assert torch.equal(dense_prefill(lambda t: t * 2, x), x * 2)  # stubs: called as is


def test_expert_store_ladder_retries_on_oom(monkeypatch):
    """S17-2 OOM fix: each OutOfMemoryError drops the budget 10% and retries."""
    calls = []

    def fake_store(self, groups, tp, budget):
        calls.append(budget)
        if len(calls) < 3:
            raise torch.OutOfMemoryError("simulated store OOM")
        return f"store@{budget}"

    monkeypatch.setattr(GPTOss, "_expert_store", fake_store)
    m = _stub(seed=29)
    got = m._expert_store_ladder(None, m.tp, 20_000_000_000)
    assert calls == [20_000_000_000, 18_000_000_000, 16_200_000_000]
    assert got == f"store@{16_200_000_000}"


def test_expert_store_ladder_gives_up_below_floor(monkeypatch):
    def always_oom(self, groups, tp, budget):
        raise torch.OutOfMemoryError("no room at all")

    monkeypatch.setattr(GPTOss, "_expert_store", always_oom)
    m = _stub(seed=31)
    with pytest.raises(torch.OutOfMemoryError):
        m._expert_store_ladder(None, m.tp, 9e9)


DENSE, KV, SCRATCH = 1_400_000_000, 150_000_000, 1 << 30
PLAN0 = int(30.0e9) - DENSE - KV - SCRATCH     # rank 0, D8_BUDGET_GB[0] = 30.0


def test_expert_budget_is_plan_when_gpus_hidden(monkeypatch):
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 0)
    m = _stub(seed=41)
    assert m._expert_budget(DENSE, KV, SCRATCH) == PLAN0


def test_expert_budget_capped_by_device_total(monkeypatch):
    """Jobs S17-2/3: store dies at ~25.5 GiB held though the probe fits 31
    GiB -> cap the store at total - EXPERT_GPU_OVERHEAD_BYTES - the rest."""
    total = 34_236_000_000                     # 31.89 GiB, the real B70
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(gptoss, "read_mem_total_bytes", lambda: -1)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.xpu, "mem_get_info",
                        lambda dev: (total - 2_000_000_000, total), raising=False)
    m = _stub(seed=43)
    want = total - GPTOss.EXPERT_GPU_OVERHEAD_BYTES - DENSE - KV - SCRATCH
    assert want < PLAN0
    assert m._expert_budget(DENSE, KV, SCRATCH) == want


def test_expert_budget_plan_when_total_is_smaller_than_overhead_cap(monkeypatch):
    """Plenty of room: the D8 plan budget still wins over the cap."""
    total = 48_000_000_000
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(gptoss, "read_mem_total_bytes", lambda: -1)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.xpu, "mem_get_info", lambda dev: (total, total), raising=False)
    m = _stub(seed=47)
    assert m._expert_budget(DENSE, KV, SCRATCH) == PLAN0


def test_expert_budget_falls_back_to_d8_without_mem_get_info(monkeypatch):
    def boom(dev):
        raise AttributeError("no mem_get_info")

    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(gptoss, "read_mem_total_bytes", lambda: -1)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.xpu, "mem_get_info", boom, raising=False)
    m = _stub(seed=53)
    want = int(30.0e9) - GPTOss.EXPERT_GPU_OVERHEAD_BYTES - DENSE - KV - SCRATCH
    assert m._expert_budget(DENSE, KV, SCRATCH) == want


def test_expert_budget_capped_by_host_ram():
    """Job S17-6: GPUActive ~= the sum of the ranks' torch reservations, so
    every store byte also costs host RAM: cap the total store across the
    world at (MemTotal - EXPERT_HOST_RESERVE_BYTES) - dense - kv - scratch."""
    m = _stub(seed=59)
    m.tp.world = 2
    host_total = 60 << 30
    want = ((host_total - GPTOss.EXPERT_HOST_RESERVE_BYTES) // 2
            - DENSE - KV - SCRATCH)
    assert want < PLAN0
    assert m._expert_budget(DENSE, KV, SCRATCH, host_total=host_total) == want


def test_expert_budget_host_cap_when_gpus_visible(monkeypatch):
    """With GPUs visible the cap reads /proc/meminfo; a huge device total
    must not override the host cap (the S17-6 machine: 31 GiB VRAM/GPU,
    ~60 GiB RAM, two ranks)."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(gptoss, "read_mem_total_bytes", lambda: 50 << 30)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.xpu, "mem_get_info",
                        lambda dev: (40_000_000_000, 40_000_000_000), raising=False)
    m = _stub(seed=61)
    want = ((50 << 30) - GPTOss.EXPERT_HOST_RESERVE_BYTES) // 1 - DENSE - KV - SCRATCH
    gpu_cap = 40_000_000_000 - GPTOss.EXPERT_GPU_OVERHEAD_BYTES - DENSE - KV - SCRATCH
    assert want < PLAN0 and want < gpu_cap
    assert m._expert_budget(DENSE, KV, SCRATCH) == want


# ---- PYI_MEMDIAG diagnostics (job S17-6, host-OOM diagnosis) ----

MEMINFO = """MemTotal:       64000000 kB
MemFree:         2000000 kB
MemAvailable:   33554432 kB
GPUActive:      60775472 kB
"""
VMSTAT = """nr_free_pages 500000
nr_gpu_active 17877696
"""


def test_parse_meminfo_reads_mem_available_and_gpu_active():
    assert parse_meminfo(MEMINFO) == (33554432, 60775472)


def test_parse_meminfo_gpu_active_fallback_pages_to_kb():
    """Kernel 7.3 exposes GPUActive in meminfo; older dumps have nr_gpu_active
    (4 KiB pages) only in /proc/vmstat -> same kB unit."""
    assert parse_meminfo(MEMINFO.replace("GPUActive:      60775472 kB\n", ""))[1] == -1
    assert parse_meminfo(VMSTAT) == (-1, 17877696 * 4)


def _decode8(m, first):
    for t in range(8):                               # 8 < max_ctx, one ring pass
        m.decode((t + first) % n_vocab)


def test_memdiag_off_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PYI_MEMDIAG", raising=False)
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=61)
    m._memdiag("load")
    _decode8(m, 0)
    assert list(tmp_path.iterdir()) == []            # no env: writes nothing


def test_memdiag_samples_load_and_every_8_decode_tokens(tmp_path, monkeypatch):
    out = tmp_path / "memdiag.log"
    monkeypatch.setenv("PYI_MEMDIAG", str(out))
    monkeypatch.setattr(GPTOss, "_attention_decode", _fake_attention_decode)
    m = _stub(seed=67)
    m._memdiag("load")
    _decode8(m, 0)
    m.reset()                                        # clears n_past, keeps _diag_n
    _decode8(m, 8)
    lines = out.read_text().splitlines()
    phases = [ln.split()[3] for ln in lines]
    assert phases == ["load", "decode8", "decode16"]
    assert "rank0" in lines[0] and "mem_avail=" in lines[0]
    assert "store_pinned=%.2fGiB" % (6789 / 2**30) in lines[0]
    assert "torch_alloc=0.00GiB" in lines[0]         # CPU stub, no xpu device
