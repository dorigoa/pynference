"""GPU tests for pynference.kernels: the selftest as pytest with thresholds.

Shapes are the real Gemma-4-31B ones plus small cases. Run only inside a GPU
job (2.4); skipped when the GPUs are hidden (e.g. under scripts/cpu.sh).
Thresholds are loose margins over the values recorded in job S6-1 (report S6).
"""
import numpy as np
import pytest

import torch

import triton

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.kernels import add_norm, attn_decode, bias_rope, gelu_mul, norm_rope, rms_norm, swiglu_oai
from pynference.models.gemma4 import rope_neox, rope_table

DEV = torch.device("xpu", 0)
EPS = 1e-6
# S6-1 observed: elementwise kernels <= 1e-5, norm_rope->fp16 cache ~1e-3,
# attn_decode <= 1.43e-06. Margins: 2 orders of magnitude (fp32) / fp16 rounding.
ATOL_FP32 = 1e-2
ATOL_FP16_CACHE = 2e-2
ATOL_ATTN = 1e-4


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def ref_rms(x, w):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y * w if w is not None else y


def test_rms_norm():
    x = torch.randn(3, 5376, device=DEV) * 3
    w = torch.randn(5376, device=DEV)
    assert (rms_norm(x, w, EPS) - ref_rms(x, w)).abs().max().item() < ATOL_FP32


def test_add_norm():
    x = torch.randn(3, 5376, device=DEV) * 3
    w = torch.randn(5376, device=DEV)
    a = torch.randn_like(x)
    w2 = torch.randn(5376, device=DEV)
    s = torch.tensor([0.7], device=DEV)
    xo, ho = add_norm(x, a, w, EPS, scale=s, w2=w2)
    rx = (x + ref_rms(a, w)) * s
    assert (xo - rx).abs().max().item() < ATOL_FP32
    assert (ho - ref_rms(rx, w2)).abs().max().item() < ATOL_FP32


def _rope_case():
    T, H, D = 3, 4, 512
    qkv = torch.randn(T, 2 * H * D, device=DEV)
    xq = qkv[:, H * D:].view(T, H, D)  # strided view, like q/k inside qkv
    wq = torch.randn(D, device=DEV)
    ff = np.array([1.0] * 64 + [1e30] * 192, dtype=np.float32)
    cos, sin = rope_table(512, 1e6, ff, 64, DEV)
    pos = torch.tensor([5, 17, 40], device=DEV)
    return xq, wq, cos, sin, pos


def test_norm_rope():
    xq, wq, cos, sin, pos = _rope_case()
    ref = rope_neox(ref_rms(xq, wq), cos[pos], sin[pos])
    assert (norm_rope(xq, wq, EPS, cos, sin, pos) - ref).abs().max().item() < ATOL_FP32


def test_norm_rope_to_kv_cache():
    xq, wq, cos, sin, pos = _rope_case()
    ref = rope_neox(ref_rms(xq, wq), cos[pos], sin[pos])
    H, D = ref.shape[1], ref.shape[2]
    cache = torch.zeros(H, 64, D, dtype=torch.float16, device=DEV)
    slot = torch.tensor([9, 3, 60], device=DEV)
    norm_rope(xq, wq, EPS, cos, sin, pos, out=cache, slot=slot)
    got = cache[:, slot].transpose(0, 1).float()
    assert (got - ref).abs().max().item() < ATOL_FP16_CACHE  # values stored in fp16


def test_norm_rope_no_rope():
    xq, wq, cos, sin, pos = _rope_case()
    assert (norm_rope(xq, None, EPS) - ref_rms(xq, None)).abs().max().item() < ATOL_FP32


def test_gelu_mul():
    import torch.nn.functional as F
    gu = torch.randn(2, 2 * 10752, device=DEV) * 3
    ref = F.gelu(gu[:, :10752], approximate="tanh") * gu[:, 10752:]
    assert (gelu_mul(gu) - ref).abs().max().item() < ATOL_FP32


@pytest.mark.parametrize("Hq,Hkv,D,S,n", [(16, 8, 256, 1024, 1024), (16, 8, 256, 1024, 37),
                                          (16, 2, 512, 8192, 3000)])
def test_attn_decode(Hq, Hkv, D, S, n):
    q = torch.randn(Hq, D, device=DEV)
    kc = (torch.randn(Hkv, S, D, device=DEV) * 0.1).half()
    vc = torch.randn(Hkv, S, D, device=DEV).half()
    G = Hq // Hkv
    sc = torch.einsum("hgd,hsd->hgs", q.view(Hkv, G, D), kc[:, :n].float())
    ref = torch.einsum("hgs,hsd->hgd", sc.softmax(-1), vc[:, :n].float()).reshape(Hq, D)
    split = 256
    o = attn_decode(q, kc, vc, torch.tensor([n], device=DEV, dtype=torch.int32),
                    triton.cdiv(S, split), split)
    assert (o - ref).abs().max().item() < ATOL_ATTN


def _bias_rope_case(T, H, D):
    qkv = torch.randn(T, H * D, device=DEV) * 2
    x = qkv.view(T, H, D)
    b = torch.randn(H, D, device=DEV)
    cos, sin = rope_table(D, 500000.0, None, 64, DEV)   # pos rows read by value
    pos = torch.tensor([5, 17, 40][:T], device=DEV)
    return x, b, cos, sin, pos


@pytest.mark.parametrize("T,H,D", [(3, 4, 64), (1, 32, 64)])   # small + gpt-oss rank slice
def test_bias_rope(T, H, D):
    scale = 0.125                                             # 1/sqrt(64), gpt-oss score scale
    x, b, cos, sin, pos = _bias_rope_case(T, H, D)
    ref = rope_neox(x + b, cos[pos], sin[pos]) * scale
    assert (bias_rope(x, b, cos, sin, pos, scale) - ref).abs().max().item() < ATOL_FP32


@pytest.mark.parametrize("T,H,D", [(3, 4, 64), (1, 32, 64)])
def test_bias_rope_to_kv_cache(T, H, D):
    x, b, _, _, _ = _bias_rope_case(T, H, D)
    cache = torch.zeros(H, 64, D, dtype=torch.float16, device=DEV)
    slot = torch.tensor([9, 3, 60][:T], device=DEV)           # distinct: last write wins
    bias_rope(x, b, None, None, None, 1.0, out=cache, slot=slot)
    got = cache[:, slot].transpose(0, 1).float()
    assert (got - (x + b)).abs().max().item() < ATOL_FP16_CACHE


@pytest.mark.parametrize("M,N", [(2, 100), (4, 2880)])        # small + gpt-oss expert rows
def test_swiglu_oai(M, N):
    from pynference.moe import swiglu_oai as ref_fn
    g = torch.randn(M, N, device=DEV) * 4
    u = torch.randn(M, N, device=DEV) * 4
    out = torch.empty(M, N, dtype=torch.float32, device=DEV)
    assert (swiglu_oai(g, u, out=out) - ref_fn(g, u)).abs().max().item() < ATOL_FP32
