"""S27 step 4 (CPU half): the pieces the segmented XPUGraph decode is built
from, each pinned against its append/eager twin so a graph-mode failure on
GPU localizes to a piece, not to the replay:

  * `QSA` static state (`zero_state(capacity=...)`): `_decode_static` (via
    `decode`) and the `forward`-on-static + `absorb` fold must reproduce the
    append path — outputs, pool rows, final buffers;
  * `deltanet.conv_prefill`/`delta_prefill` with `state_out=` (in-place twin
    of the replacement-style state update the graph replays);
  * `deltanet.norm_sigmoid_decode` (fused gate tail) == the batched torch
    tail `_delta_block` uses, with the [nv, S] reshape convention;
  * `hc_combine_` == `hc_combine`; PLE `inplace_state=True` == default.

The Triton branches of the fused kernels only run on GPU (job S27-6); the
CPU twins above are the ones these tests exercise.
"""
import pytest
import torch

from pynference import deltanet
from pynference.models.qwen4exp import PLE, QSA, hc_combine, hc_combine_

EPS = 1e-6


def _synth_qsa(seed, C=128, HD=64, NH=4, NKV=2, ROT=16, IH=2, ID=32, CP=4,
               TOPK=8):
    torch.manual_seed(seed)
    lw = lambda w: (lambda x: x @ w.T)  # noqa: E731
    g = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
    qsa = QSA(lw(g(2 * HD * NH, C)), lw(g(HD * NKV, C)), lw(g(HD * NKV, C)),
              lw(g(C, HD * NH)), torch.ones(HD) * 0.5, torch.ones(HD) * 0.5,
              lw(g(IH * ID, C)), lw(g(ID, C)),
              torch.ones(ID) * 0.5, torch.ones(ID) * 0.5,
              n_head=NH, n_head_kv=NKV, head_dim=HD,
              idx_heads=IH, idx_dim=ID, idx_top_k=TOPK, compress=CP,
              rope_n_rot=ROT, rope_base=1e4, eps=EPS)
    return qsa, g, C, CP


def test_qsa_static_decode_matches_append():
    """Prefill -> absorb -> static decodes -> a mid-stream chunked forward ->
    more static decodes, all against the append path. T1=21 with idx_top_k=8
    forces the real selection regime (blocks dropped), and the chunk starts
    inside a partial pool block so absorb's clamp-recompute is exercised.

    The `top-k` cut is undefined ON TIES (same note as `_keep_cells`: ggml's
    order is undefined there too). The static path must top-k the fixed
    pool-capacity vector to stay graph-shape-stable, so a tie across the cut
    could resolve differently between the paths in a way neither owns; seed
    20 has no tied visible scores in this loop (asserted below, so a fixture
    change that reintroduces ties fails loudly instead of silently)."""
    qsa, g, C, CP = _synth_qsa(20)
    T0, CHUNK, T1, CAP = 5, 9, 21, 24

    cur = g(T0, C)
    out_a, st_a, _ = qsa.forward(cur, torch.arange(T0), qsa.zero_state())
    st_s = qsa.zero_state(capacity=CAP)
    out_s, new_s, _ = qsa.forward(cur, torch.arange(T0), st_s)
    qsa.absorb(st_s, new_s)
    assert torch.allclose(out_a, out_s, atol=1e-6), "prefill through static prev"
    assert int(st_s["n"]) == T0
    n = T0

    while n < T1:
        if n == CHUNK:  # prefill-style resume: T=3 forward on the static state
            step3, pos3 = g(3, C), torch.arange(CHUNK, CHUNK + 3)
            out3, st_a, _ = qsa.forward(step3, pos3, st_a)
            out4, new4 = qsa.forward(step3, pos3, st_s)[:2]
            qsa.absorb(st_s, new4)
            assert torch.allclose(out3, out4, atol=1e-5, rtol=1e-4), "chunk at N=9"
            n += 3
        step, pos = g(1, C), torch.tensor([n])
        oa, st_a, pa = qsa.decode(step, pos, st_a)
        ob, st_b, parts_b = qsa.decode(step, pos, st_s)
        vis_scores = [s for s in pa["indexer_score"][0].tolist()
                      if s != float("-inf")]
        assert len(vis_scores) == len(set(vis_scores)), f"tied scores at N={n + 1}"
        assert st_b is st_s and parts_b is None, "static decode must be in-place"
        assert torch.allclose(oa, ob, atol=1e-5, rtol=1e-4), f"out at N={n + 1}"
        # the pool the two paths carry must agree row for row (the append
        # path's parts.indexer_k is the full pool; static carries it in state)
        n_pool = (n + 1 + CP - 1) // CP
        assert torch.allclose(st_s["pool"][:n_pool], pa["indexer_k"],
                              atol=1e-6, rtol=1e-5), f"pool at N={n + 1}"
        n += 1

    assert n == T1 and int(st_s["n"]) == T1
    assert torch.equal(st_s["idx_pos"][:n], st_a["idx_pos"])
    assert torch.allclose(st_s["idx_keys"][:n], st_a["idx_keys"], atol=1e-7)
    assert torch.allclose(st_s["kv_k"][:n], st_a["kv_k"], atol=1e-7)
    assert torch.allclose(st_s["kv_v"][:n], st_a["kv_v"], atol=1e-7)
    # the sentinel tail really stays invisible after all those steps
    assert (st_s["idx_pos"][n:] == 1 << 60).all()


def test_qsa_absorb_over_capacity_raises():
    qsa, g, C, CP = _synth_qsa(3)
    _, st_a, _ = qsa.forward(g(20, C), torch.arange(20), qsa.zero_state())
    with pytest.raises(AssertionError, match="over capacity"):
        qsa.absorb(qsa.zero_state(capacity=16), st_a)


def test_delta_state_out_parity():
    """`state_out=` writes the new conv/ssm state into the caller's buffers
    (fixed addresses for the graph) and must be bit-identical to the
    replacement-style return, at the T the graph runs (1) and at T=3."""
    torch.manual_seed(4)
    S, NK, NV, K = 32, 3, 2, 4
    L_qkv = (2 * NK + NV) * S
    g = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
    for T in (1, 3):
        qkv, conv0, w = g(T, L_qkv), g(K - 1, L_qkv), g(L_qkv, K)
        conv_orig = conv0.clone()
        qkv1, c1 = deltanet.conv_prefill(qkv, conv0, w)
        conv2 = conv0.clone()
        qkv2, got = deltanet.conv_prefill(qkv, conv0, w, state_out=conv2)
        assert torch.equal(qkv1, qkv2) and torch.equal(c1, conv2)
        assert got is conv2  # returns the caller's buffer, not a new one
        assert torch.equal(conv0, conv_orig)  # non-inplace read-only
        # the graph's pattern: state_out *is* the state buffer (it is read
        # into `ins` before being overwritten, so in-place is safe)
        qkv3, got3 = deltanet.conv_prefill(qkv, conv0, w, state_out=conv0)
        assert got3 is conv0 and torch.equal(conv0, c1) and torch.equal(qkv3, qkv1)

        q = deltanet.l2norm_qk(qkv2[:, :NK * S].view(T, NK, S), EPS)
        k = deltanet.l2norm_qk(qkv2[:, NK * S:2 * NK * S].view(T, NK, S), EPS)
        v = qkv2[:, 2 * NK * S:].view(T, NV, S)
        gate, beta = g(T, NV) - 2.0, torch.sigmoid(g(T, NV))
        state0 = g(NV, S, S) * 0.2
        state_orig = state0.clone()
        out1, m1 = deltanet.delta_prefill(q, k, v, gate, beta, state0)
        st2 = state0.clone()
        out2, got = deltanet.delta_prefill(q, k, v, gate, beta, state0,
                                           state_out=st2)
        assert torch.equal(out1, out2) and torch.equal(m1, st2)
        assert got is st2
        assert torch.equal(state0, state_orig)  # read-only for non-inplace too
        out3, got3 = deltanet.delta_prefill(q, k, v, gate, beta, state0,
                                            state_out=state0)
        assert got3 is state0 and torch.equal(state0, m1) and torch.equal(out3, out1)


def test_norm_sigmoid_decode_twin():
    """The fused decode tail (CPU twin) equals the batched torch tail
    `_delta_block` uses, at the graph's [nv, S] reshape convention."""
    torch.manual_seed(5)
    NV, S = 2, 32
    g = lambda *s: torch.randn(*s) * 0.3  # noqa: E731
    out, z, w = g(1, NV, S), g(1, NV, S), g(NV, S)
    y = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + EPS) * w
    y = y * torch.sigmoid(z)
    got = deltanet.norm_sigmoid_decode(out[0], z[0], w, EPS)
    assert got.shape == (NV, S)
    assert torch.equal(got, y[0])


def test_hc_combine_inplace_parity():
    torch.manual_seed(6)
    T, HC, C = 1, 4, 64
    g = lambda *s: torch.randn(*s) * 0.3  # noqa: E731
    res, blk, inj = g(T, HC * C), g(T, C), g(T, HC)
    ref = hc_combine(res.clone(), blk, inj)
    assert torch.equal(hc_combine_(res, blk, inj), ref)
    assert torch.equal(res, ref)  # written in place, address kept


def test_ple_inplace_state_parity():
    torch.manual_seed(7)
    C, HC, NGRAM = 32, 2, 3
    lw = lambda w: (lambda x: x @ w.T)  # noqa: E731
    g = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
    ple = PLE(lw(g(HC * C, C)), lw(g(C, C)),
              g(HC, C), g(HC, C), g(HC, C), g(HC * C, 4),
              n_embd=C, ngram=NGRAM, eps=EPS)
    res, emb = g(1, HC * C), g(1, C)
    state0 = ple.zero_state()
    r1, s1, _ = ple.forward(res.clone(), emb, state0)
    r2, s2, _ = ple.forward(res.clone(), emb, state0, inplace_state=True)
    assert torch.equal(r1, r2)
    assert torch.equal(s1, s2)
    assert s2 is state0 and s2.data_ptr() == state0.data_ptr()
