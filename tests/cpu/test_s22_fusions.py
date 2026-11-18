"""CPU unit tests for the S22 fused decode kernels (S22 step 2).

Each new kernel is checked against the torch chain it replaces in
`Qwen35._decode_body`: `kernels.res_norm` (raw residual + next RMSNorm),
`kernels.silu_mul_gu` (merged gate|up FFN rows), `kernels.sigmoid_gate`
(interleaved attention output gate), `deltanet.gates` (merged alpha|beta)
and `deltanet.norm_gated_decode` (per-head gated RMSNorm of one token),
under TRITON_INTERPRET. Small shapes with non-power-of-two trailing sizes
exercise the masks, and `z` is passed as a row-strided view to mirror the
real caller slicing the [1, H_v*S] `gate` GEMV output.

TRITON_INTERPRET is set before the imports (collection-order contract)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import torch.nn.functional as F

from pynference import deltanet, kernels as K
from pynference.models.gptoss import rms_norm

EPS = 1e-6


def test_silu_mul_gu_matches_torch_chain():
    torch.manual_seed(0)
    M, Ff = 3, 5                       # non-power-of-two F exercises the mask
    gu = torch.randn(M, 2 * Ff)
    ref = F.silu(gu[:, :Ff]) * gu[:, Ff:]
    torch.testing.assert_close(K.silu_mul_gu(gu), ref, rtol=1e-6, atol=1e-7)


def test_sigmoid_gate_reads_interleaved_head_rows():
    """gate = rows D..2D of each 2*D head block of the raw `attn_q` GEMV."""
    torch.manual_seed(1)
    M, Hq, D = 2, 3, 4
    a, gu = torch.randn(M, Hq * D), torch.randn(M, 2 * Hq * D)
    gate = gu.view(M, Hq, 2, D)[:, :, 1, :].reshape(M, Hq * D)
    torch.testing.assert_close(K.sigmoid_gate(a, gu, D),
                               a * torch.sigmoid(gate), rtol=1e-6, atol=1e-7)


def test_res_norm_is_sum_then_rmsnorm():
    torch.manual_seed(2)
    M, N = 2, 20
    x, a, w2 = torch.randn(M, N), torch.randn(M, N), torch.randn(N)
    xo, ho = K.res_norm(x, a, w2, EPS)
    ref = x + a
    torch.testing.assert_close(xo, ref, rtol=0, atol=0)   # same fp32 add
    torch.testing.assert_close(ho, rms_norm(ref, w2, EPS), rtol=1e-5, atol=1e-6)


def test_delta_gates_match_torch_chain():
    torch.manual_seed(3)
    H = 4
    a_log = -torch.rand(H) * 2 - 0.1          # -exp(A_log) < 0 (Mappa A.4)
    dt_bias = torch.randn(H)
    alpha = torch.randn(H)
    alpha[2] = 25.0 - dt_bias[2]              # x > 20: F.softplus passthrough
    beta_raw = torch.randn(H)
    g, beta = deltanet.gates(torch.cat([alpha, beta_raw]), a_log, dt_bias)
    torch.testing.assert_close(g, a_log * F.softplus(alpha + dt_bias),
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(beta, torch.sigmoid(beta_raw), rtol=1e-6, atol=1e-7)


def test_norm_gated_decode_matches_norm_gated():
    torch.manual_seed(4)
    Hv, S, W = 3, 5, 9                        # z rows stride over a wider GEMV
    y, w = torch.randn(Hv, S), torch.randn(S)
    z = torch.randn(Hv, W)[:, :S]             # row stride W, like the `gate` slice
    torch.testing.assert_close(
        deltanet.norm_gated_decode(y, z, w, EPS),
        deltanet.norm_gated(y, z.contiguous(), w, EPS), rtol=1e-5, atol=1e-6)
