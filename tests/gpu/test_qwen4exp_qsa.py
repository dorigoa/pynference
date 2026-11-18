"""GPU tests for S25 step 2: QSA prefill on xpu (torch) and decode (kernel).

Run only inside a GPU job (2.4); skipped when the GPUs are hidden (e.g.
under scripts/cpu.sh). Random synthetic weights (the dump-anchored test on
blk.3 is the S25 step-3 job). Coverage:

- prefill: the device-agnostic `QSA.forward` on xpu matches the CPU
  reference — selection regime on small dims, and the real head shapes
  (24/2 GQA, head_dim 256, rope 64) at a shorter context;
- kernel: `qsa_attn_decode` (gather over the indexer keep-set, -1 padding
  slots skipped) vs a torch gather+softmax reference, small and real
  shapes, f32 and f16 caches;
- decode end-to-end: `QSA.decode` on xpu (kernel attention) matches
  `QSA.decode` on CPU (sdpa attention) step by step through the selection
  regime, across block boundaries.

Bars: keep-set (indexer_sel finite pattern) exact on every token whose
top-k cut is not within 50x the measured cross-device score drift of a tie
(such near-tie tokens are printed and capped, never silently ignored);
outputs cosine >= 0.9999 and relative error <= 1e-4 on the non-ambiguous
rows (a 1-token decode step that is entirely near-tie has no comparable
row and is only counted against the near-tie budget); pool keys and scores
cosine >= 0.9999, relative <= 5e-3. Cross-device
torch-vs-torch can never be bit-exact (different matmul reduction orders;
and the f16 raw-key cache amplifies any fp32 drift into a 1-ulp jump of a
cache row), so exact node matches are not asserted here — the dump-anchored
CPU test is the bar for exactness (job S25-1 lesson, report S25). TF32-
style reduced matmul precision is disabled so the drift stays small.
"""
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

torch.backends.cuda.matmul.allow_tf32 = False   # the xpu plugin honors these
torch.backends.cudnn.allow_tf32 = False

from pynference import kernels
from pynference.models.qwen4exp import QSA

DEV = torch.device("xpu", 0)
EPS = 1e-6
TOL_COS = 0.9999
TOL_REL = 1e-4
CROSS_REL = 5e-3        # torch-node bars across devices (see module docstring)


def _f64(x):
    return x.detach().cpu().double().ravel()


def _cos(a, b):
    a, b = _f64(a), _f64(b)
    return float(a @ b / (a.norm() * b.norm()))


def _rel(a, b):
    a, b = _f64(a), _f64(b)
    return float((a - b).norm() / b.norm())


def _lw(w):
    return lambda x: x @ w.T


def qsa_pair(*, C, n_head, n_head_kv, head_dim, idx_heads, idx_dim,
             idx_top_k, compress, rope_n_rot, seed=0, gscale=0.05):
    """Two QSA instances (CPU + xpu) sharing one set of synthetic weights."""
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s) * gscale  # noqa: E731
    w = dict(q=g(2 * head_dim * n_head, C), k=g(head_dim * n_head_kv, C),
             v=g(head_dim * n_head_kv, C), o=g(C, head_dim * n_head),
             iq=g(idx_heads * idx_dim, C), ik=g(idx_dim, C),
             gk=torch.ones(head_dim) * 0.5, gq=torch.ones(head_dim) * 0.5,
             gik=torch.ones(idx_dim) * 0.5, giq=torch.ones(idx_dim) * 0.5)

    def build(dev):
        m = {k: (t.to(dev) if dev is not None else t) for k, t in w.items()}
        return QSA(_lw(m["q"]), _lw(m["k"]), _lw(m["v"]), _lw(m["o"]),
                   m["gk"], m["gq"], _lw(m["iq"]), _lw(m["ik"]),
                   m["gik"], m["giq"],
                   n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim,
                   idx_heads=idx_heads, idx_dim=idx_dim, idx_top_k=idx_top_k,
                   compress=compress, rope_n_rot=rope_n_rot, rope_base=1e4,
                   eps=EPS)

    return build(None), build(DEV)


def _amb_rows(score, n_top_pool, margin):
    """[T] bool from score [T, n_pool]: tokens whose top-k block cut sits
    within `margin` of a tie, so cross-device drift can legitimately pick a
    different (equally good) block. No selection (n_pool <= n_top_pool) or
    a cut that lands on invisible (-inf) blocks -> never ambiguous."""
    T, n_pool = score.shape
    k = min(n_pool, n_top_pool)
    if k >= n_pool:
        return torch.zeros(T, dtype=torch.bool)
    s = score.sort(1, descending=True).values
    nvis = torch.isfinite(score).sum(1)
    gap = s[:, k - 1] - s[:, k]
    gap = torch.where(torch.isfinite(s[:, k]), gap,
                      torch.full_like(gap, float("inf")))
    return (nvis > k) & (gap < margin)


def _check_cross(pc, pg, out_c, out_g, n_top_pool, tag):
    """Shared CPU-vs-xpu comparison of one QSA call's parts + output."""
    sc_c, sc_g = pc["indexer_score"], pg["indexer_score"]
    fin = torch.isfinite(sc_c)
    assert torch.equal(fin, torch.isfinite(sc_g)), f"score visibility {tag}"
    assert _cos(sc_c[fin], sc_g[fin]) >= TOL_COS and \
        _rel(sc_c[fin], sc_g[fin]) <= CROSS_REL, f"score {tag}"
    assert _cos(pc["indexer_k"], pg["indexer_k"]) >= TOL_COS and \
        _rel(pc["indexer_k"], pg["indexer_k"]) <= CROSS_REL, f"pool {tag}"
    drift = (sc_c[fin] - sc_g[fin]).abs().max().item()
    amb = _amb_rows(sc_c, n_top_pool, 50 * drift + 1e-9)
    kc, kg = torch.isfinite(pc["indexer_sel"]), torch.isfinite(pg["indexer_sel"])
    assert torch.equal(kc[~amb], kg[~amb]), f"keep-set {tag}"
    if (~amb).any():   # a 1-token decode step can be near-tie entirely:
        assert _cos(out_c[~amb], out_g[~amb]) >= TOL_COS and \
            _rel(out_c[~amb], out_g[~amb]) <= TOL_REL, f"out {tag}"
        # no comparable row left (job S25-2: empty slice -> NaN assert); the
        # caller caps how often that may happen (near-tie budget)
    return kc, int(amb.sum()), drift


SMALL = dict(C=128, n_head=4, n_head_kv=2, head_dim=64, idx_heads=2,
             idx_dim=32, idx_top_k=8, compress=4, rope_n_rot=16)
REAL = dict(C=2560, n_head=24, n_head_kv=2, head_dim=256, idx_heads=4,
            idx_dim=128, idx_top_k=2048, compress=4, rope_n_rot=64)


def _check_prefill(qsa_c, qsa_g, C, T):
    """Returns the kept-cell count of the CPU reference's indexer_sel."""
    torch.manual_seed(7)
    cur = torch.randn(T, C) * 0.05
    pos = torch.arange(T)
    out_c, _, pc = qsa_c.forward(cur, pos, qsa_c.zero_state())
    out_g, _, pg = qsa_g.forward(cur.to(DEV), pos.to(DEV), qsa_g.zero_state(DEV))
    pg = {k: v.cpu() for k, v in pg.items()}
    kc, n_amb, drift = _check_cross(pc, pg, out_c, out_g.cpu(),
                                    qsa_c.n_top_pool, f"prefill T={T}")
    print(f"drift prefill T={T}: {drift:.2e} near-tie {n_amb}/{T}")
    assert n_amb <= T // 4, "selection near-tie across devices too often"
    return int(kc.sum())


def test_prefill_selection_regime():
    # 16 blocks, top-2: the keep-set is selection-bound, not causal-bound
    # (all-causal would keep 64*65/2 = 2080 cells, not ~11 per token)
    kept = _check_prefill(*qsa_pair(seed=1, **SMALL), C=SMALL["C"], T=64)
    assert 64 * (2 * SMALL["compress"] + SMALL["compress"] - 1) >= kept


def test_prefill_real_head_shapes():
    # real 24/2 GQA, head_dim 256, rope 64 (n_pool 31 < 512: no selection,
    # this pins the numerical path at the production head geometry)
    kept = _check_prefill(*qsa_pair(seed=2, **REAL), C=REAL["C"], T=126)
    assert kept == 126 * 127 // 2  # causal-bound, nothing dropped


@pytest.mark.parametrize("Hq,Hkv,D,S,M", [(4, 2, 32, 512, 40),
                                          (24, 2, 256, 4096, 2054)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_qsa_attn_decode_kernel(Hq, Hkv, D, S, M, dtype):
    torch.manual_seed(3)
    q = (torch.randn(Hq, D, device=DEV) * 0.5)
    kc = torch.randn(S, Hkv, D, device=DEV, dtype=dtype)
    vc = torch.randn(S, Hkv, D, device=DEV, dtype=dtype)
    keep = torch.randperm(S, device=DEV)[: M - 7].sort().values  # pad slots at end
    idx = torch.full((M,), -1, dtype=torch.int32, device=DEV)
    idx[: keep.shape[0]] = keep.to(torch.int32)
    scale = D ** -0.5
    o = kernels.qsa_attn_decode(q, kc, vc, idx, scale)
    sel = keep.long()
    g = Hq // Hkv
    ref = torch.stack([(kc[sel, h // g].float() @ (q[h] * scale)).softmax(0)
                       @ vc[sel, h // g].float() for h in range(Hq)])
    assert _cos(o, ref) >= TOL_COS and _rel(o, ref) <= TOL_REL


def test_decode_matches_cpu():
    # incremental decode, xpu (Triton attention) vs CPU (sdpa attention):
    # identical keep-sets off near-ties, pooled keys within cross-device
    # bars, outputs within bars, across block boundaries and the full
    # selection regime (N 5..21)
    qsa_c, qsa_g = qsa_pair(seed=4, **SMALL)
    torch.manual_seed(5)
    T0, T1 = 5, 21
    cur = torch.randn(T0, SMALL["C"]) * 0.05
    _, st_c, _ = qsa_c.forward(cur, torch.arange(T0), qsa_c.zero_state())
    _, st_g, _ = qsa_g.forward(cur.to(DEV), torch.arange(T0).to(DEV),
                               qsa_g.zero_state(DEV))
    n = n_amb = 0
    for i in range(T0, T1):
        step = torch.randn(1, SMALL["C"]) * 0.05
        pos = torch.tensor([i])
        of, st_c, pf = qsa_c.decode(step, pos, st_c)
        og, st_g, pg = qsa_g.decode(step.to(DEV), pos.to(DEV), st_g)
        og, pg = og.cpu(), {k: v.cpu() for k, v in pg.items()}
        n += 1
        kc, a, drift = _check_cross(pf, pg, of, og, qsa_c.n_top_pool,
                                    f"decode N={n}")
        if a:
            n_amb += 1
            print(f"drift decode N={n}: {drift:.2e} near-tie "
                  "(keep-set/output skipped, counted)")
    assert n_amb <= (T1 - T0) // 2, "selection near-tie across devices too often"
