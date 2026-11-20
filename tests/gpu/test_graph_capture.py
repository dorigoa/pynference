"""GPU capture tests for the S27 segmented decode graphs.

Job S27-6 crashed capturing A[il] (QSA layers): Level Zero refused the
bool advanced-indexing kernel of `vis[nb]` in `Qwen4Attention._decode_static`
with UR_RESULT_ERROR_UNSUPPORTED_FEATURE (44), while `topk`, int64 gathers,
`index_copy_` and `where` in the same body captured fine. These tests keep
the exact operation pattern of the selection block and the Triton QSA
attention kernel capturable, and check that a replay reproduces the eager
result bit for bit.

Run only inside a GPU job (plan 2.4); skipped when the GPUs are hidden
(e.g. under scripts/cpu.sh). See report S27.
"""
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference import kernels

DEV = torch.device("xpu", 0)


def _captured(fn):
    """Warm up `fn` on a side stream (kernels autotuned once), then capture
    it into a fresh XPUGraph — the same pattern as `_capture()`."""
    s = torch.xpu.Stream(DEV)
    s.wait_stream(torch.xpu.current_stream(DEV))
    with torch.xpu.stream(s):
        fn()
    torch.xpu.current_stream(DEV).wait_stream(s)
    torch.xpu.synchronize(DEV)
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        out = fn()
    torch.xpu.synchronize(DEV)
    return g, out


def _select_inputs(pool_cap, cp, cp_dim, k, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = pool_cap * cp - cp // 2                        # mid-stream, partial block
    keys = torch.randn(pool_cap * cp + cp, cp_dim, generator=g).to(DEV)
    pos = torch.arange(pool_cap * cp + cp, dtype=torch.int64, device=DEV)
    pool = torch.zeros(pool_cap, cp_dim, device=DEV)
    score = torch.randn(pool_cap, generator=g).to(DEV)
    return dict(keys=keys, pos=pos, pool=pool, score=score,
                pos0=torch.tensor(n - 1, dtype=torch.int64, device=DEV),
                n=torch.tensor(n, dtype=torch.int64, device=DEV),
                cp=cp, cp_dim=cp_dim, k=k)


def _select_body(keys, pos, pool, score, pos0, n, cp, cp_dim, k):
    """The `_decode_static` selection block of segment A[il] (S27): the
    in-place pool row update, topk, and the cell-id build with the gather
    `vis` reconstruction (no bool advanced indexing) — capture-friendly
    ops only, exactly as shipped in qwen4exp.py."""
    dev = score.device
    pool_cap = pool.shape[0]
    n_new = n + 1
    nb = torch.div(n, cp, rounding_mode="floor")
    cells = nb * cp + torch.arange(cp, device=dev)
    cid = torch.where(cells <= n, cells, torch.zeros_like(cells))
    pool.index_copy_(0, nb.view(1), (keys[cid].sum(0) * (1.0 / cp)).view(1, -1))
    rows = torch.arange(pool_cap, device=dev) < torch.div(
        n_new, cp, rounding_mode="floor").clamp(max=pool_cap)
    last = pos[torch.arange(pool_cap, device=dev) * cp + (cp - 1)]
    vis = (last <= pos0) & rows
    score = torch.where(rows, torch.where(vis, score, score.new_full((), float("-inf"))),
                        score.new_full((), float("-inf")))
    top = score.topk(min(pool_cap, k)).indices
    bcells = top[:, None] * cp + torch.arange(cp, device=dev)
    ok = ((last.gather(0, top) <= pos0) &
          (top < torch.div(n_new, cp, rounding_mode="floor").clamp(max=pool_cap)))[:, None] \
        & (bcells < n_new)
    head = torch.where(ok, bcells, torch.full_like(bcells, -1)).reshape(-1)
    tail = nb * cp + torch.arange(cp, device=dev)
    npool = torch.div(n_new, cp, rounding_mode="floor").clamp(max=pool_cap)
    nb_vis = (last.gather(0, nb.view(1)) <= pos0) & (nb < npool)
    ok_tail = (tail < n_new) & (~nb_vis) & (pos[torch.minimum(tail, n)] <= pos0)
    idx = torch.cat([head, torch.where(ok_tail, tail, torch.full_like(tail, -1))]) \
        .to(torch.int32)
    return idx


def test_capture_qsa_select_block_replays_eager():
    """The whole selection block — index_copy_, topk, gather-based vis
    reconstruction, where/reshape/cat, int32 cast — captures into an
    XPUGraph, and the captured output tensor + the in-place pool row after
    a replay equal the eager run on identical inputs, bit for bit."""
    args = _select_inputs(pool_cap=256, cp=4, cp_dim=16, k=32)
    pool_ref = torch.zeros_like(args["pool"])
    idx_ref = _select_body(**dict(args, pool=pool_ref))   # eager reference
    g, out = _captured(lambda: _select_body(**args))
    args["pool"].zero_()                                  # undo the warmup
    g.replay()
    torch.xpu.synchronize(DEV)
    assert torch.equal(out, idx_ref)
    assert torch.equal(args["pool"], pool_ref)


def test_capture_qsa_attn_decode_kernel():
    """The Triton `qsa_attn_decode` kernel (called from `_decode_static`
    with int32 cell ids that include -1 padding) captures and replays."""
    torch.manual_seed(1)
    Hq, Hkv, D, S, M = 8, 2, 64, 1024, 96
    q = torch.randn(Hq, D, device=DEV) * 0.5
    kc = torch.randn(S, Hkv, D, device=DEV)
    vc = torch.randn(S, Hkv, D, device=DEV)
    idx = torch.randint(0, S, (M - 8,), device=DEV, dtype=torch.int32)
    idx = torch.cat([idx, torch.full((8,), -1, dtype=torch.int32, device=DEV)])
    scale = 1.0 / D**0.5
    out_eager = kernels.qsa_attn_decode(q, kc, vc, idx, scale)
    out = torch.empty_like(out_eager)
    g, _ = _captured(lambda: kernels.qsa_attn_decode(q, kc, vc, idx, scale, out=out))
    out.zero_()
    g.replay()
    torch.xpu.synchronize(DEV)
    torch.testing.assert_close(out, out_eager, rtol=0, atol=0)
