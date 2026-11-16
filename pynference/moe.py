"""Experts between VRAM and RAM: ExpertStore and a generic MoE layer (C5).

Expert parallel (C5): global expert `e` lives on rank `e % world`, whole
(gate, up, down move together, so one slot index serves every projection of
a layer). The router is tiny and replicated, so each rank computes the same
top-k and filters to its own experts; the cross-rank sum is one `all_reduce`
after the layer.

Each (layer, projection) has one resident GPU buffer `[S, N, row_bytes]` of
raw GGUF bytes (C2 layout, copied from RAM byte for byte) plus a few LRU
swap slots; non-resident experts live in pinned host memory and are never
duplicated — weights are read-only, so eviction needs no write-back.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .quant import gemv_indexed, row_bytes

PINNED_LIMIT_BYTES = 80 * (1 << 30)  # C5: at most 80 GB of pinned RAM total


# ------------------------------------------------------------- expert ownership

def expert_owner(e: int, world: int) -> int:
    """Rank owning global expert e (expert parallel, C5)."""
    return e % world


def local_experts(n_experts: int, rank: int, world: int) -> list[int]:
    """Global ids of the experts this rank owns, in ascending order."""
    return [e for e in range(n_experts) if e % world == rank]


# ------------------------------------------------------------- residency plan

@dataclass
class ExpertGroup:
    """One (layer, projection) family of expert weights in the GGUF layout.

    `raw_fn(e)` returns the GGUF bytes of global expert `e` as a
    [n * row_bytes(qtype, k)] uint8 array (contiguous, row-major).
    """

    name: str
    layer: int
    n_experts: int
    n: int
    k: int
    qtype: str
    raw_fn: Callable[[int], np.ndarray]

    @property
    def slot_bytes(self) -> int:
        return self.n * row_bytes(self.qtype, self.k)

    def expert_bytes(self, e: int) -> np.ndarray:
        # raw_fn may return [n, row_bytes] (e.g. quant.testing.make_test_blocks):
        # flatten so HostPool's [rows, slot_bytes] view stays 2D.
        raw = np.ascontiguousarray(self.raw_fn(e), dtype=np.uint8).reshape(-1)
        if raw.size != self.slot_bytes:
            raise ValueError(
                f"{self.name} expert {e}: got {raw.size} bytes, "
                f"expected {self.slot_bytes} for {self.qtype} {self.n}x{self.k}")
        return raw


def plan_residency(triple_bytes: list[int], n_local: int, budget_bytes: int,
                   swap_slots: int) -> tuple[list[int], list[int]]:
    """Per-layer resident and swap slot counts (C5 load-time plan).

    `triple_bytes[l]` is the cost of one local expert of layer l summed over
    that layer's projections (the three projections always move together).
    Swap slots are guaranteed first for every layer that can still miss, so
    `budget_bytes = 0` means "nothing resident, everything in RAM" and still
    runs. The rest of the budget is filled expert by expert, layer by layer
    (expert index ascending, layer ascending), so all layers stay evenly
    covered; a fully resident layer hands its swap slots back.
    """
    if budget_bytes >= sum(n_local * t for t in triple_bytes):
        return [n_local] * len(triple_bytes), [0] * len(triple_bytes)
    R = [0] * len(triple_bytes)
    X = [min(swap_slots, n_local) for _ in triple_bytes]
    available = budget_bytes - sum(x * t for x, t in zip(X, triple_bytes))
    for i in range(n_local):
        for l, t in enumerate(triple_bytes):
            if R[l] >= n_local or t > available:
                continue
            R[l] += 1
            available -= t
            if R[l] == n_local and X[l]:
                available += X[l] * t
                X[l] = 0
    return R, X


# ------------------------------------------------------------- slot bookkeeping

class SlotMap:
    """Host-side expert -> slot map with LRU swap for one layer (C5).

    Slots `[0, R)` hold the resident experts forever; slots `[R, S)` are the
    LRU swap area. `resolve` returns one slot per requested id (valid for
    every projection of the layer) plus the (expert, slot) pairs to copy.
    Only host bookkeeping: the ids arrive on the host anyway, because decode
    does its top-k there (C5), and no device state is read back.
    """

    def __init__(self, resident_ids: list[int], swap_slots: int):
        self.R = len(resident_ids)
        self.S = self.R + swap_slots
        self.expert = list(resident_ids) + [-1] * swap_slots
        self.slot = {e: i for i, e in enumerate(resident_ids)}
        self.resident = set(resident_ids)
        self.ts = [0] * self.S          # last use, monotonically assigned
        self._now = 1

    def resolve(self, ids: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        """-> (slots for each id, misses as (expert id, slot to copy into))."""
        slots: list[int] = []
        misses: list[tuple[int, int]] = []
        fresh: set[int] = set()         # victims claimed earlier in this call
        for e in ids:
            s = self.slot.get(e)
            if s is not None:
                if s >= self.R:  # a swap hit refreshes its LRU rank
                    self.ts[s] = self._now
                    self._now += 1
                slots.append(s)
                continue
            cands = [s for s in range(self.R, self.S) if s not in fresh]
            if not cands:
                raise ValueError(
                    f"cannot load {len(fresh) + 1} new experts at once: "
                    f"only {self.S - self.R} swap slots")
            v = min(cands, key=lambda s: self.ts[s])
            if self.expert[v] >= 0:
                del self.slot[self.expert[v]]
            self.expert[v] = e
            self.slot[e] = v
            self.ts[v] = self._now
            self._now += 1
            fresh.add(v)
            slots.append(v)
            misses.append((e, v))
        return slots, misses


# ------------------------------------------------------------- host (pinned) pool

def _pin(t: torch.Tensor) -> torch.Tensor:
    """Page-lock a CPU tensor. The xpu pinned allocator needs a visible
    device, so with GPUs hidden (tests on CPU) the buffer stays pageable;
    the pinned path runs in GPU jobs."""
    if torch.xpu.device_count() > 0:
        return t.pin_memory()
    return t


class HostPool:
    """One pinned [rows, slot_bytes] block holding given experts of a group."""

    def __init__(self, group: ExpertGroup, ids: list[int], pin: bool = True):
        self.group = group
        self.ids = list(ids)
        self.row = {e: i for i, e in enumerate(self.ids)}
        t = torch.empty(len(self.ids) * group.slot_bytes, dtype=torch.uint8)
        self.buf = _pin(t) if pin else t

    @property
    def nbytes(self) -> int:
        return self.buf.numel()

    def fill(self) -> None:
        """Load every held expert from the source into its pinned row."""
        if self.ids:
            data = np.stack([self.group.expert_bytes(e) for e in self.ids])
            self.buf.view(len(self.ids), self.group.slot_bytes).copy_(
                torch.from_numpy(data))


# ------------------------------------------------------------- routing / activation

def route_weights(logits: torch.Tensor, top_k: int) -> tuple[list[int], list[float]]:
    """Decode routing (C5): softmax over all experts, top-k, renormalize.

    Returns host lists (ids ascending by score, weights summing to 1).
    """
    p = torch.softmax(logits.detach().float().reshape(-1), 0)
    w, idx = torch.topk(p, top_k)
    w = w / w.sum()
    return idx.tolist(), w.tolist()


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702,
               limit: float = 7.0010) -> torch.Tensor:
    """GPT-OSS activation (model card / ggml swiglu_oai): both inputs clamped
    to +-limit, gate * sigmoid(alpha * gate) * (up + limit). To be re-checked
    against gpt-oss golden in S17 (no llama.cpp source available for S15)."""
    gate = gate.clamp(-limit, limit)
    up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(alpha * gate) * (up + limit)


ACTIVATIONS = {"swiglu": swiglu, "swiglu_oai": swiglu_oai}


# ------------------------------------------------------------- prefill grouping

def group_by_expert(ids: list[list[int]], weights: list[list[float]],
                    local: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Prefill token grouping (C5): token rows gathered per local expert.

    `ids[t]` / `weights[t]` are the top-k global expert ids and their
    renormalized weights for token row t. Returns {expert: (rows int64,
    weights float32)} iterating in ascending expert order.
    """
    acc: dict[int, tuple[list[int], list[float]]] = {}
    for t, (row, wrow) in enumerate(zip(ids, weights)):
        for e, w in zip(row, wrow):
            if e in local:
                acc.setdefault(int(e), ([], []))[0].append(t)
                acc[int(e)][1].append(float(w))
    return {e: (np.asarray(rows, dtype=np.int64), np.asarray(w, dtype=np.float32))
            for e, (rows, w) in sorted(acc.items())}


# ------------------------------------------------------------- expert store (GPU)

class ExpertStore:
    """Resident expert buffers + LRU swap on one GPU (C5).

    `g` carries `.groups: list[ExpertGroup]`, one per (layer, projection).
    Each group gets a `[S, N, row_bytes]` uint8 buffer in the native GGUF
    layout, copied from RAM byte for byte: slots `[0, R)` hold the resident
    local experts, slots `[R, S)` are the LRU swap area. Slot numbering is
    shared across a layer's projections (triples move together), so
    `resolve(layer, ids)` yields one int32 tensor that `gemv_indexed` (C2)
    can feed to every projection of the layer. Non-resident local experts
    live in pinned host memory (`HostPool`). `resolve` copies misses in on
    a side stream and makes the current stream wait on the copy event, so
    kernels launched after it see fresh bytes; `stats()` reports hits,
    misses and copied bytes.
    """

    def __init__(self, g, tp, budget_bytes: int, swap_slots: int = 2):
        groups = list(getattr(g, "groups", g))
        if not groups:
            raise ValueError("ExpertStore: no expert groups")
        self.tp = tp
        self.layers: dict[int, dict[str, ExpertGroup]] = {}
        for gr in groups:
            lay = self.layers.setdefault(gr.layer, {})
            if gr.name in lay:
                raise ValueError(f"duplicate expert group layer={gr.layer} name={gr.name}")
            lay[gr.name] = gr

        self.local_ids: dict[int, list[int]] = {}
        self.triple_bytes: dict[int, int] = {}
        triples = []
        layers_sorted = sorted(self.layers)
        n_locals = set()
        for layer in layers_sorted:
            lay = self.layers[layer]
            n_experts = {gr.n_experts for gr in lay.values()}
            if len(n_experts) != 1:
                raise ValueError(f"layer {layer}: projections disagree on n_experts")
            loc = local_experts(n_experts.pop(), tp.rank, tp.world)
            self.local_ids[layer] = loc
            n_locals.add(len(loc))
            self.triple_bytes[layer] = sum(gr.slot_bytes for gr in lay.values())
            triples.append(self.triple_bytes[layer])
        if len(n_locals) != 1:
            raise ValueError("ExpertStore: all layers must have the same local expert count")

        R, X = plan_residency(triples, n_locals.pop(), budget_bytes, swap_slots)
        self.bufs: dict[int, dict[str, torch.Tensor]] = {}
        self.pools: dict[tuple[int, str], HostPool] = {}
        self.maps: dict[int, SlotMap] = {}
        pinned = 0
        for i, layer in enumerate(layers_sorted):
            resident = self.local_ids[layer][:R[i]]
            rest = [e for e in self.local_ids[layer][R[i]:]]
            self.maps[layer] = SlotMap(resident, X[i])
            bufs = {}
            for name, gr in self.layers[layer].items():
                rb = row_bytes(gr.qtype, gr.k)
                buf = torch.empty(self.maps[layer].S, gr.n, rb,
                                  dtype=torch.uint8, device=tp.device)
                if resident:
                    rows = np.stack([gr.expert_bytes(e) for e in resident])
                    buf[:R[i]].copy_(torch.from_numpy(rows).view(R[i], gr.n, rb))
                bufs[name] = buf
                pool = HostPool(gr, rest)
                pool.fill()
                pinned += pool.nbytes
                self.pools[layer, name] = pool
            self.bufs[layer] = bufs
        if pinned > PINNED_LIMIT_BYTES:
            raise ValueError(f"pinned RAM {pinned} B exceeds the C5 limit {PINNED_LIMIT_BYTES}")
        self._pinned = pinned
        self._side = torch.xpu.Stream(tp.device)
        self._ev = torch.xpu.Event()
        self.hits = self.misses = 0
        self.bytes_copied = 0

    def resolve(self, layer: int, ids: list[int]) -> torch.Tensor:
        """Device int32 slots for local expert ids (valid for every
        projection of the layer). Missing experts are copied in first; the
        ids arrive on the host because decode does its top-k there (C5)."""
        local = self.local_ids[layer]
        for e in ids:
            if e not in local:
                raise ValueError(f"expert {e} is not owned by rank {self.tp.rank}")
        slots, misses = self.maps[layer].resolve(list(ids))
        self.hits += len(ids) - len(misses)
        if misses:
            self.misses += len(misses)
            self.bytes_copied += self.triple_bytes[layer] * len(misses)
            self._copy_misses(layer, misses)
        return torch.tensor(slots, dtype=torch.int32, device=self.tp.device)

    def _copy_misses(self, layer: int, misses: list[tuple[int, int]]) -> None:
        cur = torch.xpu.current_stream()
        # the side stream must not overwrite a slot a queued kernel may read
        self._side.wait_stream(cur)
        with torch.xpu.stream(self._side):
            for e, slot in misses:
                for name, buf in self.bufs[layer].items():
                    pool = self.pools[layer, name]
                    src = pool.buf.view(-1, pool.group.slot_bytes)[pool.row[e]]
                    buf[slot].reshape(-1).copy_(src, non_blocking=True)
            self._ev.record(self._side)
        cur.wait_event(self._ev)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes_copied}

    def reset_stats(self) -> None:
        self.hits = self.misses = 0
        self.bytes_copied = 0

    def memory_bytes(self) -> int:
        """Expert bytes on this GPU (resident + swap), for memory_report (C3)."""
        return sum(b.numel() for lay in self.bufs.values() for b in lay.values())

    def pinned_bytes(self) -> int:
        return self._pinned


# ------------------------------------------------------------- MoE layer (C5)

class MoE:
    """Generic MoE layer over an ExpertStore (C5): router, top-k, resolve,
    indexed GEMVs, activation, weighted sum, all_reduce.

    The layer consumes the expert groups named "gate" [ffn, k], "up"
    [ffn, k] and "down" [k, ffn] of one store layer. `router_w` [E, k]
    (and the optional `router_bias` [E]) are float32 and replicated: every
    rank computes the same top-k, filters to its own experts (expert
    parallel, C5) and the cross-rank sum is one `tp.all_reduce` at the end.
    `shared`, when given, is a triple (gate, up, down) of `QWeight`s for a
    dense FFN every rank computes on every token; it is fully replicated
    (`make_qweight` does not shard), so it is added after the `all_reduce`
    — adding it before would multiply it by the world size.

    Decode (M = 1): the top-k lands on the host with one sync per layer
    (C5), so the layer can batch local experts for `resolve`: residents in
    one batch with up to `X` swap misses, further misses in blocks of `X`
    (the resolve constraint, see SlotMap). Prefill: one logits transfer,
    then tokens are grouped per expert and each expert's slot is dequantized
    once into a reusable bf16 scratch (the M>1 path of C2) before `F.linear`
    on its token group.
    """

    def __init__(self, store: ExpertStore, layer: int, *, top_k: int,
                 router_w: torch.Tensor, router_bias: torch.Tensor | None = None,
                 activation: str = "swiglu", alpha: float = 1.702,
                 limit: float = 7.0010, shared=None):
        self.store = store
        self.tp = store.tp
        self.layer = layer
        groups = store.layers[layer]
        missing = {"gate", "up", "down"} - set(groups)
        if missing:
            raise ValueError(f"MoE layer {layer}: missing expert groups {sorted(missing)}")
        self.gate, self.up, self.down = groups["gate"], groups["up"], groups["down"]
        self.k, self.ffn = self.gate.k, self.gate.n
        if (self.up.n, self.up.k) != (self.ffn, self.k) or \
           (self.down.n, self.down.k) != (self.k, self.ffn):
            raise ValueError("MoE: gate/up must be [ffn, k] and down [k, ffn]")
        self.n_experts = self.gate.n_experts
        self.top_k = top_k
        if router_w.shape != (self.n_experts, self.k) or router_w.dtype != torch.float32:
            raise ValueError(f"router_w must be float32 [{self.n_experts}, {self.k}]")
        if router_bias is not None and router_bias.shape != (self.n_experts,):
            raise ValueError(f"router_bias must be [{self.n_experts}]")
        self.router_w = router_w
        self.router_bias = router_bias
        act = ACTIVATIONS[activation]
        self.act = functools.partial(act, alpha=alpha, limit=limit) \
            if activation == "swiglu_oai" else act
        self.shared = tuple(shared) if shared is not None else None
        self.local = set(store.local_ids[layer])
        self._scratch: dict[tuple[str, int, int], torch.Tensor] = {}

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, k] f32 -> [T, E] router logits."""
        l = x @ self.router_w.T
        if self.router_bias is not None:
            l = l + self.router_bias
        return l

    def _shared_out(self, x: torch.Tensor) -> torch.Tensor:
        gate, up, down = self.shared
        return down(self.act(gate(x), up(x)))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """One token: x [1, k] (or [k]) f32 -> [k] f32 summed over ranks."""
        x = x.reshape(1, self.k)
        dev = x.device
        ids, w = route_weights(self._logits(x)[0], self.top_k)  # one host sync (C5)
        y = torch.zeros(self.k, dtype=torch.float32, device=dev)
        m = self.store.maps[self.layer]
        X = m.S - m.R  # swap slots: fresh misses allowed per resolve
        sel = [(e, wt) for e, wt in zip(ids, w) if e in self.local]
        res = [p for p in sel if p[0] in m.resident]
        mis = [p for p in sel if p[0] not in m.resident]
        step = max(X, 1)
        batches = [res + mis[i:i + step] for i in range(0, len(mis), step)] or [res]
        for b in batches:
            if not b:
                continue
            slots = self.store.resolve(self.layer, [e for e, _ in b])
            L = len(b)
            xrep = x.repeat(L, 1)  # same token for each expert, [L, k]
            g = torch.empty(L, self.ffn, dtype=torch.float32, device=dev)
            u = torch.empty(L, self.ffn, dtype=torch.float32, device=dev)
            gemv_indexed(self.gate.qtype, self.store.bufs[self.layer]["gate"], xrep, slots, g)
            gemv_indexed(self.up.qtype, self.store.bufs[self.layer]["up"], xrep, slots, u)
            a = self.act(g, u).contiguous()
            d = torch.empty(L, self.k, dtype=torch.float32, device=dev)
            gemv_indexed(self.down.qtype, self.store.bufs[self.layer]["down"], a, slots, d)
            wts = torch.tensor([wt for _, wt in b], dtype=torch.float32, device=dev)
            y += (d * wts[:, None]).sum(0)
        y = self.tp.all_reduce(y)
        if self.shared is not None:  # replicated: add once, after the reduce
            y = y + self._shared_out(x)[0]
        return y

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Token chunk: x [T, k] f32 -> [T, k] f32 summed over ranks."""
        x = x.reshape(-1, self.k)
        dev = x.device
        p = torch.softmax(self._logits(x), 1)
        w, idx = torch.topk(p, self.top_k, dim=1)
        w = w / w.sum(1, keepdim=True)
        ids_h, w_h = idx.cpu().tolist(), w.float().cpu().tolist()  # one sync
        y = torch.zeros_like(x)
        for e, (rows, wts) in group_by_expert(ids_h, w_h, self.local).items():
            slot = self.store.resolve(self.layer, [e]).item()
            r = torch.from_numpy(rows).to(dev)
            xt = x[r]
            a = self.act(self._gemm(self.gate, slot, xt),
                         self._gemm(self.up, slot, xt))
            d = self._gemm(self.down, slot, a)
            y.index_add_(0, r, d * torch.from_numpy(wts).to(dev)[:, None])
        y = self.tp.all_reduce(y)
        if self.shared is not None:  # replicated: add once, after the reduce
            y = y + self._shared_out(x)
        return y

    def _gemm(self, group: ExpertGroup, slot: int, x: torch.Tensor) -> torch.Tensor:
        """Prefill M>1 path (C2): dequant one slot into bf16 scratch, F.linear."""
        from .quant import kernels
        w = self._scratch.get((group.name, group.n, group.k))
        if w is None:
            w = torch.empty(group.n, group.k, dtype=torch.bfloat16, device=x.device)
            self._scratch[(group.name, group.n, group.k)] = w
        kernels.dequant_native(group.qtype, self.store.bufs[self.layer][group.name][slot], w)
        return F.linear(x.bfloat16(), w).float()
