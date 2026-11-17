"""GPT-OSS dense models (GGUF arch "gpt-oss", e.g. gpt-oss-120b), tensor-parallel.

Follows llama.cpp src/models/openai-moe.cpp (S16 map: reports/S16-map.md):
- attention Q/K/V with biases added after the projection, no q/k norms;
- attention with sinks (one logit per Q head in the softmax denominator),
  scale 1/sqrt(head_dim), sliding window 128 on even layers (pattern 2);
- RoPE NEOX with YaRN: freq_scale 1/32 over ctx_orig 4096, beta 32/1,
  ext_factor 1, net mscale 1 + 0.1*ln(32) baked into the cos/sin table,
  applied identically on windowed and full layers (openai-moe.cpp:12);
- output projection with bias (added by rank 0 before the all-reduce);
- MoE (plan 3.8): router with bias, top-4, softmax over the 4 selected
  logits, 128 fused MXFP4 experts with per-expert biases, activation
  swiglu_oai (alpha 1.702, limit 7.0) — served by pynference.moe (C5).

Tensor split over `tp.world` GPUs: attention heads split (Q rows [q_r; k_r; v_r],
sinks sliced by Q head), the output projection split by input columns with
all-reduce, the vocabulary split for the embedding lookup and the output head;
norms replicated.
"""
from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..gguf import GGUFFile
from ..parallel import TP
# QLinear (Triton) is imported inside the loading helpers: importing Triton at
# module load would break the TRITON_INTERPRET contract of tests/cpu/test_quant_decode_interp.py
# (see the S15 report).


@dataclass
class GPTOSSConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    window: int
    is_swa: list[bool]
    rope_base: float
    yarn_factor: float
    yarn_orig_ctx: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_ext_factor: float
    yarn_attn_factor: float
    eps: float
    n_vocab: int
    n_ctx_train: int

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "GPTOSSConfig":
        n_layer = int(g.get("block_count"))
        rope_type = str(g.get("rope.scaling.type", "none")).lower()
        if rope_type != "yarn":
            raise NotImplementedError(f"gpt-oss: rope scaling {rope_type!r} not supported (yarn only)")
        factor = float(g.get("rope.scaling.factor"))
        if factor <= 1.0:
            raise ValueError(f"gpt-oss: rope scaling factor {factor} <= 1")
        kv = g.get("attention.head_count_kv")
        kv = [int(x) for x in kv] if hasattr(kv, "__len__") else [int(kv)] * n_layer
        if any(h != kv[0] for h in kv):
            raise NotImplementedError("gpt-oss: per-layer head_count_kv not supported")
        # llama.cpp load_swa_pattern(ml, 2) with dense_first=false: is_swa[il] = il % 2 < 1
        return cls(
            n_layer=n_layer,
            n_embd=int(g.get("embedding_length")),
            n_head=int(g.get("attention.head_count")),
            n_head_kv=kv[0],
            head_dim=int(g.get("attention.key_length")),
            n_rot=int(g.get("rope.dimension_count", g.get("attention.key_length"))),
            window=int(g.get("attention.sliding_window")),
            is_swa=[il % 2 == 0 for il in range(n_layer)],
            rope_base=float(g.get("rope.freq_base")),
            yarn_factor=factor,
            yarn_orig_ctx=int(g.get("rope.scaling.original_context_length")),
            yarn_beta_fast=float(g.get("rope.scaling.yarn_beta_fast", 32.0)),
            yarn_beta_slow=float(g.get("rope.scaling.yarn_beta_slow", 1.0)),
            # llama-context.cpp:265-267: ext_factor unset -> 1.0 for YARN; the attn_factor
            # chain (295/303/306) nets to the GGUF value, absent here -> 1.0 (S16 map §5).
            yarn_ext_factor=1.0,
            yarn_attn_factor=float(g.get("rope.scaling.attn_factor", 1.0) or 1.0),
            eps=float(g.get("attention.layer_norm_rms_epsilon")),
            n_vocab=int(g.tensors["token_embd.weight"].shape[1]),
            n_ctx_train=int(g.get("context_length")),
        )


def rope_yarn_dims(n_rot: int, base: float, orig_ctx: int, beta_fast: float, beta_slow: float) -> tuple[int, int]:
    """ggml_rope_yarn_corr_dims (ggml.c:4373-4389) with truncate = true (the rope_yarn call
    in ggml-cpu/ops.cpp passes the truncated dims): floor of the beta_fast dim, ceil of the
    beta_slow dim, clamped to [0, n_rot - 1]. For gpt-oss: (8, 18)."""
    def corr_dim(beta: float) -> float:
        return n_rot * math.log(orig_ctx / (beta * 2 * math.pi)) / (2 * math.log(base))
    return max(0, math.floor(corr_dim(beta_fast))), min(n_rot - 1, math.ceil(corr_dim(beta_slow)))


def rope_yarn_table(n_rot: int, base: float, factor: float, orig_ctx: int,
                    beta_fast: float, beta_slow: float, n_pos: int, device,
                    ext_factor: float = 1.0, attn_factor: float = 1.0):
    """cos/sin tables [n_pos, n_rot/2] for NEOX RoPE with YaRN, as computed by
    ggml_rope_cache_init -> rope_yarn (ggml-cpu/ops.cpp:6076-6114, 6268-6271):
    freq_scale = 1/factor; for pair k (dims k, k + n_rot/2) with corr_dims [c0, c1]:
    ramp = clamp01(1 - (k - c0)/(c1 - c0)), theta = theta_raw*((1 - ramp*ext)/factor + ramp*ext);
    mscale = attn_factor*(1 + 0.1*ln(factor)) multiplies cos/sin (ops.cpp:6095-6096)."""
    c0, c1 = rope_yarn_dims(n_rot, base, orig_ctx, beta_fast, beta_slow)
    freq_scale = 1.0 / factor
    k = np.arange(n_rot // 2, dtype=np.float64)
    inv = base ** (-2.0 * k / n_rot)                      # theta_scale^k, ops.cpp:6268
    ramp = 1.0 - np.clip((k - c0) / max(0.001, c1 - c0), 0.0, 1.0)
    mix = ramp * ext_factor
    scale = freq_scale * (1.0 - mix) + mix                # theta = theta_raw * scale
    mscale = attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale))
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * (inv * scale)[None, :]
    return (torch.from_numpy((mscale * np.cos(ang)).astype(np.float32)).to(device),
            torch.from_numpy((mscale * np.sin(ang)).astype(np.float32)).to(device))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, H, D]; cos/sin [T, R/2]. NEOX: rotates pairs (i, i + R/2) of the first R dims."""
    half = cos.shape[-1]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    c, s = cos[:, None, :], sin[:, None, :]
    rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return rot if 2 * half == x.shape[-1] else torch.cat([rot, x[..., 2 * half:]], dim=-1)


def attn_prefill(q, kcache, vcache, kp, pos, window, scale, sinks=None):
    """One attention op over cached keys (torch, prefill and reference path).

    q [T, Hq, D] float32; kcache/vcache [Hkv, S, D] already containing the current tokens,
    with key positions kp [S] (negative = empty slot); pos [T] query positions.
    window 0 = dense (causal only), > 0 = sliding window: key visible iff 0 <= pos - kp < window
    (llama-graph.cpp:583-604 calls is_masked_swa with p1 = the query's own position).
    scale multiplies the scores (gpt-oss: 1/sqrt(head_dim)); sinks [Hq] or None enter only the
    softmax denominator, per query head h = kv*(Hq//Hkv) + g (S16 map S3).
    Returns [T, Hq*D] float32."""
    T, Hq, D = q.shape
    Hkv, S, _ = kcache.shape
    G = Hq // Hkv
    qh = q.view(T, Hkv, G, D).permute(1, 2, 0, 3).reshape(Hkv, G * T, D)
    s = (qh @ kcache.float().transpose(1, 2)).view(Hkv, G, T, S) * scale
    d = pos[:, None] - kp[None, :]
    vis = (kp[None, :] >= 0) & (d >= 0)
    if window:
        vis &= d < window
    s = s.masked_fill(~vis[None, None], float("-inf"))
    if sinks is None:
        m = s.amax(-1, keepdim=True)
        p = (s - m).exp()
        den = p.sum(-1, keepdim=True)
    else:
        sk = sinks.view(Hkv, G, 1, 1)
        m = torch.maximum(s.amax(-1, keepdim=True), sk)
        p = (s - m).exp()
        den = p.sum(-1, keepdim=True) + (sk - m).exp()
    o = (torch.matmul(p.view(Hkv, G * T, S), vcache.float())
         .view(Hkv, G, T, D) / den).permute(2, 0, 1, 3).reshape(T, Hq * D)
    return o


# ------------------------------------------------------------- MoE (C5, plan 3.8)

# Per-GPU VRAM budget in decimal GB, plan D8 (GPU 0 also drives the desktop).
# The expert store gets what is left after dense weights, KV cache and scratch.
D8_BUDGET_GB = (30.0, 30.9)


def moe_expert_groups(g: GGUFFile):
    """One ExpertGroup per (layer, projection) for the fused MXFP4 experts.

    GGUF layout: blk.N.ffn_{gate,up,down}_exps.weight is [k, n, E] in ggml
    order (k = 2880 contiguous inputs, n = 2880 outputs, E experts), so global
    expert e occupies rows [e*n, (e+1)*n), i.e. e * slot_bytes bytes flat.
    """
    from ..moe import ExpertGroup  # lazy import: no Triton at module load (header note)
    groups = []
    for il in range(int(g.get("block_count"))):
        for proj in ("gate", "up", "down"):
            name = f"blk.{il}.ffn_{proj}_exps.weight"
            t = g.tensors[name]
            if t.type != "MXFP4":
                raise NotImplementedError(
                    f"{name}: {t.type} experts not supported (MXFP4 only)")
            k, n, e = t.shape
            groups.append(ExpertGroup(proj, il, e, n, k, t.type,
                                      lambda ex, t=name, n=n: g.rows(t, ex * n, (ex + 1) * n)))
    return groups


def _f32_reshaped(g: GGUFFile, name: str, shape) -> torch.Tensor:
    t = g.tensors[name]
    assert t.type == "F32", (name, t.type)
    return torch.from_numpy(np.array(g.raw(name).view(np.float32))).reshape(shape)


def moe_router(g: GGUFFile, il: int):
    """Router f32: ffn_gate_inp.weight [k, E] (ggml order) -> [E, k]; bias [E]."""
    k, e = g.tensors[f"blk.{il}.ffn_gate_inp.weight"].shape
    return (_f32_reshaped(g, f"blk.{il}.ffn_gate_inp.weight", (e, k)),
            _f32_reshaped(g, f"blk.{il}.ffn_gate_inp.bias", (e,)))


def moe_expert_bias(g: GGUFFile, il: int):
    """Per-expert f32 biases ffn_{gate,up,down}_exps.bias [n, E] -> (E, n) each,
    ordered (gate, up, down) as MoE expects."""
    out = []
    for proj in ("gate", "up", "down"):
        name = f"blk.{il}.ffn_{proj}_exps.bias"
        n, e = g.tensors[name].shape
        out.append(_f32_reshaped(g, name, (e, n)))
    return tuple(out)


def parse_meminfo(text: str) -> tuple[int, int]:
    """(MemAvailable, GPUActive) in kB from a /proc/meminfo dump (-1 if absent).
    GPUActive is a kernel 7.3 field: system RAM held for GPU processes by the
    driver (S17 host-OOM diagnosis, job S17-6). Fallback: nr_gpu_active in
    /proc/vmstat, counted in 4 KiB pages."""
    avail = gpuact = -1
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if sep == ":":
            fields = rest.split()               # meminfo style: "key: value kB"
        else:                                   # vmstat style: "key value"
            key, *fields = key.split()
        if not key or not fields:
            continue
        if key == "MemAvailable":
            avail = int(fields[0])
        elif key == "GPUActive":
            gpuact = int(fields[0])
        elif key == "nr_gpu_active" and gpuact < 0:
            gpuact = int(fields[0]) * 4
    return avail, gpuact


def read_mem_total_bytes() -> int:
    """MemTotal in bytes from /proc/meminfo, -1 if unavailable (S17 host
    cap: the driver holds system RAM for the GPU, see
    GPTOss.EXPERT_HOST_RESERVE_BYTES)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return -1
    return -1


class _Layer:
    pass


class GPTOss:
    def __init__(self, g: GGUFFile, tp: TP, max_ctx: int = 8192, kv_dtype=torch.float16):
        self.g, self.tp, self.cfg = g, tp, GPTOSSConfig.from_gguf(g)
        self.max_ctx, self.kv_dtype = max_ctx, kv_dtype
        c, dev = self.cfg, tp.device
        if c.n_head % tp.world or c.n_head_kv % tp.world:
            raise ValueError(f"heads not divisible by {tp.world} GPUs")
        self.n_head_l = c.n_head // tp.world
        self.n_kv_l = c.n_head_kv // tp.world
        self._scale = 1.0 / math.sqrt(c.head_dim)          # build_attn scale 1/sqrt(n_rot)
        self._split_swa = c.window
        self._nsp_swa = (c.window + self._split_swa - 1) // self._split_swa
        self._split_full = max(256, ((max_ctx + 63) // 64 + 15) // 16 * 16)
        self._nsp_full = (max_ctx + self._split_full - 1) // self._split_full
        self.fast_decode, self.use_graph = True, False   # C3; the graph comes in S18
        self._dense_bytes = 0   # non-expert GPU bytes, for the expert budget (D8)
        self._diag_n = 0        # decode-token counter for PYI_MEMDIAG sampling
        t0 = time.time()

        self.v0, self.v1 = tp.split(c.n_vocab)
        self.embed = self._qlinear("token_embd.weight", rows=(self.v0, self.v1))
        self.lm_head = self._qlinear("output.weight", rows=(self.v0, self.v1))
        self.output_norm = self._f32("output_norm.weight")
        self.cos, self.sin = rope_yarn_table(
            c.n_rot, c.rope_base, c.yarn_factor, c.yarn_orig_ctx,
            c.yarn_beta_fast, c.yarn_beta_slow, max_ctx, dev,
            ext_factor=c.yarn_ext_factor, attn_factor=c.yarn_attn_factor)

        self.layers = []
        for il in range(c.n_layer):
            self.layers.append(self._load_layer(il))
            if tp.is_main and (il + 1) % 10 == 0:
                print(f"  loaded {il + 1}/{c.n_layer} layers ({time.time() - t0:.0f}s)", flush=True)
        # the expert budget needs the KV cache allocated, so size the MoE after reset()
        self.reset()
        if os.environ.get("PYI_GPTOSS_SKIP_MOE"):
            # attention-only tests (tests/gpu/_gptoss_attn0_worker): skip the
            # expert store entirely (its VRAM + pinned RAM does not fit there)
            self.store, self.moe = None, None
            self._weights_bytes, self._scratch_bytes = self._dense_bytes, 0
        else:
            self.store, self.moe = self._load_moe()
        torch.xpu.synchronize(dev)
        self.load_seconds = time.time() - t0
        if tp.is_main:
            print(f"  gpt-oss: load done in {self.load_seconds:.1f}s", flush=True)
        self._memdiag("load")

    # ---- loading ----
    def _f32(self, name: str) -> torch.Tensor:
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        self._dense_bytes += t.nbytes
        return torch.from_numpy(np.array(self.g.raw(name).view(np.float32))).to(self.tp.device)

    def _check_q8(self, name: str):
        if self.g.tensors[name].type != "Q8_0":
            raise NotImplementedError(f"{name}: {self.g.tensors[name].type} weights not supported yet (Q8_0 only)")

    def _qlinear(self, name: str, rows=None, cols=None):
        """Q8_0 [K, N] weight; rows=(r0, r1) output rows, cols=(c0, c1) input columns."""
        from ..quant import QLinear
        self._check_q8(name)
        K, N = self.g.tensors[name].shape
        r0, r1 = rows if rows else (0, N)
        raw = self.g.rows(name, r0, r1)
        if cols:
            raw = raw[:, cols[0] // 32 * 34: cols[1] // 32 * 34]
        self._dense_bytes += raw.nbytes
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _stack_rows(self, names, row_ranges):
        from ..quant import QLinear
        parts = []
        for name, (r0, r1) in zip(names, row_ranges):
            self._check_q8(name)
            parts.append(self.g.rows(name, r0, r1))
        raw = np.concatenate(parts, axis=0)
        self._dense_bytes += raw.nbytes
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _load_layer(self, il: int) -> _Layer:
        c, tp = self.cfg, self.tp
        p = f"blk.{il}."
        L = _Layer()
        L.swa = c.is_swa[il]
        qr = tp.split(c.n_head * c.head_dim)
        kr = tp.split(c.n_head_kv * c.head_dim)
        hr = tp.split(c.n_head)
        L.qkv = self._stack_rows([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"],
                                 [qr, kr, kr])
        # biases are added after the projection (llama-graph.cpp:1924-1953), split like the rows
        L.qkv_bias = torch.cat([self._f32(p + "attn_q.bias")[qr[0]:qr[1]],
                                self._f32(p + "attn_k.bias")[kr[0]:kr[1]],
                                self._f32(p + "attn_v.bias")[kr[0]:kr[1]]])
        L.o = self._qlinear(p + "attn_output.weight", cols=qr)
        # column-split bias: rank 0 adds the full [n_embd] bias before the all-reduce (plan 3.4)
        L.o_bias = self._f32(p + "attn_output.bias") if tp.is_main else None
        # sinks are per Q head (S16 map §7): keep this rank's slice of heads
        L.sinks = self._f32(p + "attn_sinks.weight")[hr[0]:hr[1]]
        L.attn_norm = self._f32(p + "attn_norm.weight")
        L.post_attn_norm = self._f32(p + "post_attention_norm.weight")
        return L

    def _load_moe(self):
        """ExpertStore + one MoE per layer (C5, plan 3.8): router with bias,
        top-4, softmax over the 4 selected logits, MXFP4 experts with
        per-expert biases, swiglu_oai (alpha 1.702, limit 7.0).

        Expert budget per GPU (D8): the per-rank VRAM budget minus dense
        weights, KV cache and a scratch allowance (bf16 dequant buffers for
        the prefill path + chunk-sized attention transients)."""
        from ..moe import MoE   # lazy import (module header note)
        c, tp, dev = self.cfg, self.tp, self.tp.device
        groups = moe_expert_groups(self.g)
        n_exp = int(self.g.get("expert_count", groups[0].n_experts))
        top_k = int(self.g.get("expert_used_count", 4))
        if n_exp != groups[0].n_experts or top_k < 1:
            raise ValueError(f"gpt-oss: expert_count {n_exp} / expert_used_count {top_k}")
        router, biases = [], []
        for il in range(c.n_layer):
            w, b = moe_router(self.g, il)
            eb = moe_expert_bias(self.g, il)
            router.append((w.to(dev), b.to(dev)))
            biases.append(tuple(t.to(dev) for t in eb))
        tiny = sum(t.numel() * 4 for il in range(c.n_layer)
                   for t in (*router[il], *biases[il]))       # router + expert biases
        cos_sin = 2 * self.max_ctx * (c.n_rot // 2) * 4        # rope table
        dense = self._dense_bytes + tiny + cos_sin
        kv = self.kv_bytes()
        scratch = 1 << 30   # bf16 dequant scratch (~50 MB) + prefill transients
        budget = self._expert_budget(dense, kv, scratch)
        self._weights_bytes, self._scratch_bytes = dense, scratch   # for memory_report (C3)
        if tp.is_main:
            print(f"  MoE: {n_exp} experts x {c.n_layer} layers, budget "
                  f"{budget / 1e9:.2f} GB (dense {dense / 1e9:.2f}, kv {kv / 1e9:.2f}, "
                  f"scratch {scratch / 1e9:.2f})", flush=True)
        store = self._expert_store_ladder(groups, tp, budget)
        moe = [MoE(store, il, top_k=top_k, router_w=router[il][0],
                   router_bias=router[il][1], activation="swiglu_oai",
                   routing="logits", expert_bias=biases[il])
               for il in range(c.n_layer)]
        if tp.is_main:
            tot = c.n_layer * len(store.local_ids[0])
            res = sum(len(m.resident) for m in store.maps.values())
            swp = sum(m.S - m.R for m in store.maps.values())
            print(f"  MoE: {res} of {tot} local experts resident, {swp} swap slots, "
                  f"store {store.memory_bytes() / 1e9:.2f} GB, "
                  f"pinned {store.pinned_bytes() / 1e9:.2f} GB", flush=True)
        return store, moe

    # Expert-store budget ladder (S17, job S17-2): plan D8 is a capacity
    # measured on an idle GPU (S1); with dense weights loaded, the Xe page
    # tables and hundreds of store buffers the usable bytes are a few GB
    # lower, and an OutOfMemoryError inside the store build is not
    # recoverable per-buffer. So: drop the budget a step and rebuild — same
    # residency plan, fewer resident experts, more pinned RAM (C5-checked).
    EXPERT_BUDGET_MIN_GB = 8.0

    # Driver-capacity cap (S17, jobs S17-2..4): plan D8 is an idle-GPU
    # capacity (S1); a single-process probe fits 31.5 GiB even with the real
    # buffer pattern (S17-4 probe2), but with dense weights, CCL and two
    # ranks torch runs out of device memory at a ~24.3 GB store (S17-4 ppl,
    # S17-2) and ranks died at ~25.5 GiB held (S17-3 SIGKILL). So cap the
    # store at device total minus a measured overhead, not at what D8 hopes
    # for; 11 GiB leaves the store ~20 GB, ~2 GiB below the death zone.
    EXPERT_GPU_OVERHEAD_BYTES = 11 << 30

    # Host-RAM cap (S17, job S17-6 diagnosis): the xe driver holds system
    # RAM for the GPU ~1:1 — memdiag at load end showed GPUActive 40.24 GiB
    # ~= the two ranks' torch reservations with store_pinned 0, and prefill
    # added another +12.7 GiB before the kernel's global OOM killer fired on
    # this ~60 GiB machine. On such a host the host, not the GPU, is the
    # binding constraint: keep the total store across all ranks inside
    # MemTotal - EXPERT_HOST_RESERVE_BYTES, the reserve covering driver
    # anon state, heaps, page tables, prefill/decode transients and margin.
    # 24 GiB leaves a ~16 GB store per rank here (~55% experts resident).
    EXPERT_HOST_RESERVE_BYTES = 24 << 30

    def _expert_budget(self, dense: int, kv: int, scratch: int,
                       host_total: int | None = None) -> int:
        """Store budget: min of the D8 plan budget, the host-RAM cap
        ((MemTotal - EXPERT_HOST_RESERVE_BYTES) // world - dense - kv -
        scratch) and (device total - EXPERT_GPU_OVERHEAD_BYTES - dense - kv
        - scratch). host_total=None reads /proc/meminfo only when GPUs are
        visible (CPU tests hide them and stay on the pure plan)."""
        plan = max(int(D8_BUDGET_GB[self.tp.rank] * 1e9) - dense - kv - scratch, 0)
        if host_total is None and torch.xpu.device_count() > 0:
            host_total = read_mem_total_bytes()
        if host_total and host_total > 0:
            host_cap = max((host_total - self.EXPERT_HOST_RESERVE_BYTES)
                           // self.tp.world - dense - kv - scratch, 0)
            plan = min(plan, host_cap)
        if torch.xpu.device_count() == 0:      # CPU tests hide the GPUs
            return plan
        gc.collect()
        torch.xpu.empty_cache()   # hand dense-load cache blocks back first
        try:
            total = torch.xpu.mem_get_info(self.tp.device)[1]
        except (AttributeError, RuntimeError):
            total = int(D8_BUDGET_GB[self.tp.rank] * 1e9)
        cap = max(total - self.EXPERT_GPU_OVERHEAD_BYTES - dense - kv - scratch, 0)
        return min(plan, cap)

    def _expert_store(self, groups, tp, budget):
        from ..moe import ExpertStore   # lazy import (module header note)
        return ExpertStore(groups, tp, budget, verbose=True)

    def _expert_store_ladder(self, groups, tp, budget):
        b, floor = budget, int(self.EXPERT_BUDGET_MIN_GB * 1e9)
        while b >= floor:
            try:
                return self._expert_store(groups, tp, b)
            except torch.OutOfMemoryError:
                gc.collect()
                if torch.device(tp.device).type == "xpu":
                    torch.xpu.empty_cache()
                nb = int(b * 0.9)
                if nb < floor:
                    raise
                if tp.is_main:
                    print(f"  MoE: store OOM at {b / 1e9:.2f} GB, "
                          f"retry {nb / 1e9:.2f} GB", flush=True)
                b = nb
        raise RuntimeError(f"expert budget {budget / 1e9:.2f} GB below floor")

    # ---- KV cache ----
    def reset(self):
        """Allocate (or clear) the KV cache: a ring buffer of `window` slots for sliding
        layers (position p in slot p % window), max_ctx slots for full-attention layers."""
        c, dev = self.cfg, self.tp.device
        self.n_past = 0
        if getattr(self, "cache", None):
            return
        self.cache = []
        for L in self.layers:
            n = c.window if L.swa else self.max_ctx
            k = torch.zeros(self.n_kv_l, n, c.head_dim, dtype=self.kv_dtype, device=dev)
            self.cache.append((k, torch.zeros_like(k)))

    def kv_bytes(self) -> int:
        return sum(k.numel() * k.element_size() * 2 for k, _ in self.cache)

    # ---- prompt (T > 1) path ----
    def _attention(self, il: int, q, k, v, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        """q [T, Hq, D]; k, v [T, Hkv, D], already roped. Stores k/v in the cache and runs
        attn_prefill (sinks + window + scale). Returns [T, Hq*D] float32."""
        L = self.layers[il]
        kc, vc = self.cache[il]
        T = q.shape[0]
        if L.swa:
            W = self.cfg.window
            # slot j holds the latest position < pos0 that is == j (mod W), if any
            j = torch.arange(W, device=q.device)
            kp = (pos0 - 1) - torch.remainder(pos0 - 1 - j, W)
            kcache = torch.cat([kc, k.transpose(0, 1).to(kc.dtype)], dim=1)
            vcache = torch.cat([vc, v.transpose(0, 1).to(vc.dtype)], dim=1)
            kp = torch.cat([kp, pos])
            n = min(T, W)
            slots = pos[T - n:] % W
            kc[:, slots] = k[T - n:].transpose(0, 1).to(kc.dtype)
            vc[:, slots] = v[T - n:].transpose(0, 1).to(vc.dtype)
            window = W
        else:
            kc[:, pos0:pos0 + T] = k.transpose(0, 1).to(kc.dtype)
            vc[:, pos0:pos0 + T] = v.transpose(0, 1).to(vc.dtype)
            kcache, vcache = kc[:, :pos0 + T], vc[:, :pos0 + T]
            kp = torch.arange(pos0 + T, device=q.device)
            window = 0
        return attn_prefill(q, kcache, vcache, kp, pos, window, self._scale, L.sinks)

    # ---- decode (one token) path ----
    def _attention_decode(self, il: int, q, k, v, pos: torch.Tensor) -> torch.Tensor:
        """q [1, Hq, D]; k, v [1, Hkv, D], already roped; pos int tensor [1] = absolute
        position. Stores k/v in the cache and runs kernels.attn_decode with this layer's
        sinks. Returns [Hq, D] float32 (gpt-oss has no q/k/v norms)."""
        from .. import kernels as K   # Triton at call time only: see the module header note
        c, L = self.cfg, self.layers[il]
        kc, vc = self.cache[il]
        Hq, D = q.shape[1], q.shape[2]
        if L.swa:
            slot = torch.remainder(pos, c.window)
            n_len = torch.clamp(pos + 1, max=c.window)
            split, nsp = self._split_swa, self._nsp_swa
        else:
            slot, n_len = pos, pos + 1
            split, nsp = self._split_full, self._nsp_full
        kc.index_copy_(1, slot, k.transpose(0, 1).to(kc.dtype))
        vc.index_copy_(1, slot, v.transpose(0, 1).to(vc.dtype))
        return K.attn_decode(q.reshape(Hq, D), kc, vc, n_len, nsp, split, sinks=L.sinks)

    # ---- full attention block = the llama.cpp "attn_out" (openai-moe.cpp:100-103) ----
    def _layer_attn(self, il: int, x: torch.Tensor, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        """RMSNorm, QKV + biases after the projection, RoPE, attention with sinks/window,
        O + bias (rank 0), all-reduce. x [T, n_embd] float32 -> [T, n_embd] float32."""
        c, L, tp = self.cfg, self.layers[il], self.tp
        T, Hq, Hkv, D = x.shape[0], self.n_head_l, self.n_kv_l, c.head_dim
        h = rms_norm(x, L.attn_norm, c.eps)
        qkv = L.qkv(h)
        b = L.qkv_bias
        qb = b[:Hq * D].view(Hq, D)
        kb = b[Hq * D:Hq * D + Hkv * D].view(Hkv, D)
        vb = b[Hq * D + Hkv * D:].view(Hkv, D)
        q = qkv[:, :Hq * D].view(T, Hq, D) + qb
        k = qkv[:, Hq * D:(Hq + Hkv) * D].view(T, Hkv, D) + kb
        v = qkv[:, (Hq + Hkv) * D:].view(T, Hkv, D) + vb
        cos, sin = self.cos[pos], self.sin[pos]
        q = rope_neox(q, cos, sin)
        k = rope_neox(k, cos, sin)
        a = self._attention(il, q, k, v, pos, pos0)
        o = L.o(a)
        if L.o_bias is not None:
            o = o + L.o_bias
        return tp.all_reduce(o)

    # ---- full attention block, decode (T == 1) ----
    def _layer_attn_decode(self, il: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Decode twin of _layer_attn: x [1, n_embd] -> [1, n_embd] float32."""
        c, L, tp = self.cfg, self.layers[il], self.tp
        Hq, Hkv, D = self.n_head_l, self.n_kv_l, c.head_dim
        h = rms_norm(x, L.attn_norm, c.eps)
        qkv = L.qkv(h)
        b = L.qkv_bias
        q = qkv[:, :Hq * D].view(1, Hq, D) + b[:Hq * D].view(Hq, D)
        k = qkv[:, Hq * D:(Hq + Hkv) * D].view(1, Hkv, D) + b[Hq * D:(Hq + Hkv) * D].view(Hkv, D)
        v = qkv[:, (Hq + Hkv) * D:].view(1, Hkv, D) + b[(Hq + Hkv) * D:].view(Hkv, D)
        cos, sin = self.cos[pos], self.sin[pos]
        q = rope_neox(q, cos, sin)
        k = rope_neox(k, cos, sin)
        a = self._attention_decode(il, q, k, v, pos)   # [Hq, D]
        o = L.o(a.view(1, Hq * D))
        if L.o_bias is not None:
            o = o + L.o_bias
        return tp.all_reduce(o)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embedding rows, [T] -> [T, n_embd] float32 (no sqrt scaling, openai-moe.cpp)."""
        local = (ids >= self.v0) & (ids < self.v1)
        rows = self.embed.rows_dequant((ids - self.v0).clamp(0, self.v1 - self.v0 - 1))
        return self.tp.all_reduce(rows * local[:, None])

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + vocab-split output head gathered: [M, n_vocab] float32
        on every rank. gpt-oss has no soft-cap (plan 3.8)."""
        x = rms_norm(x, self.output_norm, self.cfg.eps)
        return self.tp.all_gather(self.lm_head(x))

    # ---- C3 driver ----
    @torch.inference_mode()
    def forward(self, tokens: list[int], chunk: int = 512, all_logits: bool = False) -> torch.Tensor:
        """Append tokens to the context; return the logits of the last token
        ([n_vocab]) or of every token ([len(tokens), n_vocab]).

        Layer (openai-moe.cpp, plan 3.8): x += attn_block(x);
        x += MoE(rms_norm(x, post_attn_norm)); the MoE sums local experts and
        all-reduces (C5)."""
        c, dev = self.cfg, self.tp.device
        if self.n_past + len(tokens) > self.max_ctx:
            raise ValueError(f"context full ({self.n_past} + {len(tokens)} > {self.max_ctx})")
        if len(tokens) == 1 and not all_logits and self.fast_decode:
            return self.decode(tokens[0])
        outs = []
        for s in range(0, len(tokens), chunk):
            part = tokens[s:s + chunk]
            pos0 = self.n_past
            ids = torch.tensor(part, dtype=torch.long, device=dev)
            pos = torch.arange(pos0, pos0 + len(part), dtype=torch.long, device=dev)
            x = self.embed_tokens(ids)
            for il in range(c.n_layer):
                x = x + self._layer_attn(il, x, pos, pos0)
                h = rms_norm(x, self.layers[il].post_attn_norm, c.eps)
                x = x + self.moe[il].prefill(h)
            self.n_past += len(part)
            if all_logits:
                outs.append(self.logits(x))
        self._memdiag("prefill")
        if all_logits:
            return torch.cat(outs)
        return self.logits(x[-1:])[0]

    @torch.inference_mode()
    def decode(self, token: int) -> torch.Tensor:
        """One generated token: logits [n_vocab] float32 on every rank. No XPUGraph
        yet (fast_decode uses the fused Triton attention kernel; the graph comes in S18)."""
        c, dev = self.cfg, self.tp.device
        if self.n_past >= self.max_ctx:
            raise ValueError(f"context full ({self.n_past} >= {self.max_ctx})")
        pos = torch.tensor([self.n_past], dtype=torch.long, device=dev)
        x = self.embed_tokens(torch.tensor([token], dtype=torch.long, device=dev))
        for il in range(c.n_layer):
            x = x + self._layer_attn_decode(il, x, pos)
            h = rms_norm(x, self.layers[il].post_attn_norm, c.eps)
            x = x + self.moe[il].decode(h).view(1, -1)   # MoE.decode -> [k] summed
        self.n_past += 1
        self._diag_n += 1
        if self._diag_n % 8 == 0:
            self._memdiag(f"decode{self._diag_n}")
        return self.logits(x)[0]

    def memory_report(self) -> dict:
        """Bytes on this GPU (C3): dense weights (incl. router/biases and the rope
        table), resident + swap experts, KV cache, scratch allowance; plus the
        pinned host bytes held so far (HostPool pins rows on demand, S17)."""
        return {"weights": self._weights_bytes,
                "experts": self.store.memory_bytes(),
                "cache": self.kv_bytes(),
                "scratch": self._scratch_bytes,
                "pinned_host": self.store.pinned_bytes(),
                "device_total": (self._weights_bytes + self.store.memory_bytes()
                                 + self.kv_bytes() + self._scratch_bytes)}

    # ---- host/driver memory diagnostics (job S17-6) ----
    def _memdiag(self, phase: str) -> None:
        """If PYI_MEMDIAG=<file> is set, append one line per sampling point:
        host MemAvailable and kernel GPUActive (/proc/meminfo, fallback
        nr_gpu_active in /proc/vmstat: RAM held by the xe driver for GPU
        processes), this rank's torch xpu allocated/reserved and the store's
        pinned host bytes (pool allocated_bytes). Distinguishes HostPool-pinned
        growth from torch-cache growth in the host OOM (S17, "Problemi aperti").
        Called at load, after prefill and every 8 decode tokens. Never raises:
        diagnostics must not change the run."""
        path = os.environ.get("PYI_MEMDIAG")
        if not path:
            return
        avail = gpuact = -1
        try:
            with open("/proc/meminfo") as f:
                avail, gpuact = parse_meminfo(f.read())
        except OSError:
            pass
        if gpuact < 0:
            try:
                with open("/proc/vmstat") as f:
                    _, gpuact = parse_meminfo(f.read())
            except OSError:
                pass
        try:
            if torch.device(self.tp.device).type == "xpu":
                alloc = torch.xpu.memory_allocated(self.tp.device)
                resv = torch.xpu.memory_reserved(self.tp.device)
            else:
                alloc = resv = 0
        except (AttributeError, RuntimeError):
            alloc = resv = -1
        store = getattr(self, "store", None)
        pinned = store.pinned_bytes() if store is not None else 0
        try:
            with open(path, "a") as f:
                f.write(f"{time.time():.3f} rank{self.tp.rank} pid{os.getpid()} {phase} "
                        f"mem_avail={avail / 2**20:.2f}GiB gpu_active={gpuact / 2**20:.2f}GiB "
                        f"torch_alloc={alloc / 2**30:.2f}GiB torch_resv={resv / 2**30:.2f}GiB "
                        f"store_pinned={pinned / 2**30:.2f}GiB\n")
        except OSError:
            pass
