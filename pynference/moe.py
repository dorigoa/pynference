"""Experts between VRAM and RAM: ExpertStore and a generic MoE layer (C5).

Expert parallel (C5): global expert `e` lives on rank `e % world`, whole
(gate, up, down move together, so one slot index serves every projection of
a layer). The router is tiny and replicated, so each rank computes the same
top-k and filters to its own experts; the cross-rank sum is one `all_reduce`
after the layer.

Each (layer, projection) has one resident GPU buffer `[S, N, row_bytes]` of
raw GGUF bytes (C2 layout, copied from RAM byte for byte) plus a few LRU
swap slots; non-resident experts are read straight from the (read-only,
reclaimable) GGUF at swap time and staged to the GPU through a small pinned
ring (`ring_slots` slot-sized buffers, S17 job S17-6), so page-locked host
RAM stays tens of MB per rank instead of the whole swap set — weights are
read-only, so eviction needs no write-back. On a roomy host (S27-3, PC with
121.5 GiB RAM) `host_pin_bytes` lets the store instead page-lock the whole
non-resident set at load and serve misses straight from it, so a miss never
reads the disk.
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


# ---------------------------------------------------------- allocator setting
#
# S26 (jobs S26-1..3): the default torch-XPU allocator mirrors host RAM ~1:1
# for every device allocation (S26-1: GPUActive ~= reserved and MemAvailable
# ~= -allocated at 8 and 24 GiB, with every knob the xe stack exposes). The
# generic allocator setting `expandable_segments:True` removes the mirror
# (S26-2: 24 GiB reserved costs ~0.2 GiB of host, H2D and kernels fine) —
# but oneCCL cannot register buffers that live in expandable segments: with
# the setting on for the whole process the run SIGABRTs under torchrun
# (S26-3 `global`), a buffer newly allocated while the setting is on and
# then handed to a collective aborts too (`setterkeep`), while toggling it
# on around a private buffer and back off before the next collective passes
# cleanly (`setter`, exit 0 both ranks). So ExpertStore turns the setting on
# only around its resident+swap buffers — which no collective ever touches,
# collectives see only the ordinary-memory outputs of the kernels that read
# them — and off again before anything else can allocate.

def _expandable_segments_supported() -> bool:
    """True if torch exposes the generic allocator-settings setter (torch
    2.15+xpu does) and XPU devices are visible."""
    return (hasattr(torch._C, "_accelerator_setAllocatorSettings")
            and torch.xpu.device_count() > 0)


def _set_expandable_segments(on: bool) -> bool:
    """Toggle `expandable_segments` on the default allocator and drop cached
    blocks, so later allocations actually use the new policy (empty_cache
    matters both ways: a block cached under the other policy could be reused
    for the wrong buffer — expandable segments must not reach a collective).
    Returns True on success."""
    try:
        torch._C._accelerator_setAllocatorSettings(
            "expandable_segments:True" if on else "expandable_segments:False")
        torch.xpu.empty_cache()
    except (AttributeError, RuntimeError):
        return False
    return True


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
    """One pinned [rows, slot_bytes] block holding given experts of a group.

    `lazy=True` defers the pinned allocation to the first expert the store
    actually swaps in and loads rows one at a time (S17, job S17-4: eager
    pools cost ~7 GB per rank of non-reclaimable pinned RAM already at load
    time — on a 62 GB host that is what fed the kernel's global OOM killer
    during the gpt-oss load; a lazy pool only pins what inference touches).

    Since job S17-6 the store pins pools only when the host allows it
    (`ExpertStore(..., host_pin_bytes=...)`, S27-3): with the new 121.5 GiB
    PC RAM the whole non-resident expert set fits page-locked, and swap
    misses then copy straight from the pinned row — no GGUF read, no ring
    staging. Without that allowance the store leaves pools lazy and stages
    misses through its bounded pinned ring, so the driver never holds a
    whole pool's worth of locked RAM. Rows load once and stay cached
    (weights are read-only), so a loaded row is a host cache entry.
    """

    def __init__(self, group: ExpertGroup, ids: list[int], pin: bool = True,
                 lazy: bool = False):
        self.group = group
        self.ids = list(ids)
        self.row = {e: i for i, e in enumerate(self.ids)}
        self._pin = pin
        self._loaded = bytearray(len(self.ids))
        self.buf = None if lazy else self._new_buf()

    def _new_buf(self) -> torch.Tensor:
        # zeros, not empty: lazy rows stay visibly unloaded until row_bytes()
        # touches them (and the memset runs host-side only on first miss)
        t = torch.zeros(len(self.ids) * self.group.slot_bytes, dtype=torch.uint8)
        return _pin(t) if self._pin else t

    @property
    def nbytes(self) -> int:
        """Planned pinned bytes: worst case, every held expert pinned."""
        return len(self.ids) * self.group.slot_bytes

    @property
    def allocated_bytes(self) -> int:
        """Pinned bytes allocated so far (0 while a lazy pool is untouched)."""
        return 0 if self.buf is None else self.buf.numel()

    def _ensure(self) -> None:
        if self.buf is None:
            self.buf = self._new_buf()

    def fill(self) -> None:
        """Load every held expert from the source into its pinned row."""
        self._ensure()
        if self.ids:
            data = np.stack([self.group.expert_bytes(e) for e in self.ids])
            self.buf.view(len(self.ids), self.group.slot_bytes).copy_(
                torch.from_numpy(data))
            self._loaded[:] = b"\x01" * len(self.ids)

    def _row(self, e: int) -> torch.Tensor:
        self._ensure()
        return self.buf.view(-1, self.group.slot_bytes)[self.row[e]]

    def row_bytes(self, e: int) -> torch.Tensor:
        """Host row of expert e; loads from the source on first touch and
        stays cached after (S27-3: with the host cache on, the store copies
        swap misses from these rows, so reloading would be wasted work)."""
        r = self.row[e]
        if not self._loaded[r]:
            self._row(e).copy_(torch.from_numpy(self.group.expert_bytes(e)))
            self._loaded[r] = 1
        return self._row(e)

    def cached_row(self, e: int):
        """Loaded row for expert e, or None (S27-3 host-cache lookup)."""
        if self.buf is not None and self._loaded[self.row[e]]:
            return self._row(e)
        return None


# ------------------------------------------------------------- routing / activation

def route_weights(logits: torch.Tensor, top_k: int) -> tuple[list[int], list[float]]:
    """Decode routing (C5): softmax over all experts, top-k, renormalize.

    Returns host lists (ids ascending by score, weights summing to 1).
    Mathematically equivalent to `topk_weights(..., "logits")`, but GPT-OSS
    (C8) runs the exact llama.cpp op sequence, so use that routing there.
    """
    p = torch.softmax(logits.detach().float().reshape(-1), 0)
    w, idx = torch.topk(p, top_k)
    w = w / w.sum()
    return idx.tolist(), w.tolist()


ROUTINGS = ("renorm", "logits", "softmax")


def topk_weights(logits: torch.Tensor, top_k: int,
                 routing: str = "renorm") -> tuple[torch.Tensor, torch.Tensor]:
    """Router logits [..., E] -> (weights [..., k], ids [..., k]).

    "renorm": softmax over all experts, top-k, renormalize (C5 default).
    "logits": GPT-OSS exact form (llama-graph.cpp:2295-2420, gating
    SOFTMAX_WEIGHT): top-k on the raw logits, softmax over just those k,
    no renormalization.
    "softmax": Flash-Next / Qwen3-Next exact form (S23 map C,
    llama-graph.cpp MOE_NORM_TOPK): softmax over all experts, top-k of the
    probabilities, divide by their sum clamped to the f16 min normal
    (6.103515625e-5 = 2**-14, ggml used f16 weights). With E = 512 and
    k = 10 the sum is >= 10/512 in the worst case, so the clamp is inert
    and this matches "renorm" bit-for-bit in practice; it is kept separate
    to stay faithful to the reference graph.
    """
    if routing == "logits":
        top, idx = torch.topk(logits, top_k, dim=-1)
        return torch.softmax(top, -1), idx
    if routing not in ROUTINGS:
        raise ValueError(f"unknown routing {routing!r}, expected one of {ROUTINGS}")
    p = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(p, top_k, dim=-1)
    if routing == "softmax":
        return w / w.sum(-1, keepdim=True).clamp_min(2.0 ** -14), idx
    return w / w.sum(-1, keepdim=True), idx


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702,
               limit: float = 7.0) -> torch.Tensor:
    """GPT-OSS activation, exactly as ggml swiglu_oai_f32
    (ggml/src/ggml-cpu/ops.cpp:3388-3450, verified in S17): gate is clipped
    only on the high side (min(gate, limit)), up is clamped to +-limit, and
    the result is gate * sigmoid(alpha * gate) * (up + 1). llama.cpp passes
    alpha = 1.702f, limit = 7.0f (llama-graph.cpp:2143-2149)."""
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


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
    are read from the read-only GGUF at swap time and staged through a small
    pinned ring of `ring_slots` slot-sized buffers (S17, job S17-6: the xe
    driver holds system RAM for page-locked and device memory, so pinning
    whole pools on the first miss fed the host OOM killer just like eager
    pinning at load did; the ring keeps locked host RAM at `ring_slots *
    slot_bytes`). `resolve` copies misses in on
    a side stream and makes the current stream wait on the copy event, so
    kernels launched after it see fresh bytes; `stats()` reports hits,
    misses and copied bytes.

    `host_pin_bytes` (S27-3): when the host has that much RAM to spare, the
    store fills every non-resident pool at load (`pinned_planned_bytes() <=
    host_pin_bytes`) and swap misses copy straight from the pinned rows —
    the GGUF is read once at load, never during decode. Otherwise the pools
    stay lazy and misses go through the ring, as before.
    """

    def __init__(self, g, tp, budget_bytes: int, swap_slots: int = 2,
                 verbose: bool = False, ring_slots: int = 4,
                 host_pin_bytes: int = 0):
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
        # S26: allocate the big buffers without the host-RAM mirror (see the
        # allocator-setting note above). Anything allocated after this window
        # — slot-index tensors, CCL buffers, everything else in the process —
        # is back under the ordinary policy, which is what oneCCL needs.
        self.no_mirror = False
        if str(tp.device).startswith("xpu") and _expandable_segments_supported():
            self.no_mirror = _set_expandable_segments(True)
            if verbose:
                print("  store: expandable_segments "
                      f"{'on (no host-RAM mirror, S26)' if self.no_mirror else 'OFF (fallback: mirrored allocations)'}",
                      flush=True)
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
                if verbose:
                    m = self.maps[layer]
                    print(f"  store: layer {layer} {name}: {m.R} resident + "
                          f"{m.S - m.R} swap slots -> "
                          f"{m.S * gr.n * rb / 1e6:.0f} MB", flush=True)
                buf = torch.empty(self.maps[layer].S, gr.n, rb,
                                  dtype=torch.uint8, device=tp.device)
                if resident:
                    rows = np.stack([gr.expert_bytes(e) for e in resident])
                    buf[:R[i]].copy_(torch.from_numpy(rows).view(R[i], gr.n, rb))
                bufs[name] = buf
                pool = HostPool(gr, rest, lazy=True)
                pinned += pool.nbytes
                self.pools[layer, name] = pool
            self.bufs[layer] = bufs
        if self.no_mirror and not _set_expandable_segments(False):
            # Leaving the setting on would hand the next collective an
            # expandable buffer it cannot register (S26-3 `setterkeep`
            # SIGABRT): fail loudly instead of crashing in CCL later.
            raise RuntimeError(
                "ExpertStore: could not restore the allocator setting")
        if pinned > PINNED_LIMIT_BYTES:
            raise ValueError(f"pinned RAM {pinned} B exceeds the C5 limit {PINNED_LIMIT_BYTES}")
        self._pinned = pinned
        # S27-3 host cache: with the new RAM (PC has 121.5 GiB since
        # 2026-11-19) the whole non-resident set fits page-locked on a
        # roomy host, and a swap miss never touches the disk. Fill once,
        # here: rows are read-only and never rewritten (weights are
        # immutable), so they are safe non-blocking H2D sources forever.
        self.host_cache = False
        if host_pin_bytes > 0 and 0 < pinned <= host_pin_bytes:
            for pool in self.pools.values():
                pool.fill()
            self.host_cache = True
            if verbose:
                print(f"  store: host cache ON, {pinned / 1e9:.2f} GB of "
                      f"non-resident experts pinned "
                      f"(allowance {host_pin_bytes / 1e9:.2f} GB)", flush=True)
        elif host_pin_bytes > 0 and verbose:
            print(f"  store: host cache OFF, {pinned / 1e9:.2f} GB > "
                  f"allowance {host_pin_bytes / 1e9:.2f} GB (ring staging)",
                  flush=True)
        self._side = torch.xpu.Stream(tp.device)
        self._ev = torch.xpu.Event()
        # Pinned staging ring (S17, job S17-6): swap misses fill these few
        # page-locked slot-sized buffers from the GGUF and copy them to the
        # GPU; a buffer is refilled only after the event of its own H2D copy,
        # so it never races a queued kernel. Allocated on the first miss.
        self._ring_slots = max(ring_slots, 1)
        self._ring_bytes = max(gr.slot_bytes for gr in groups)
        self._ring = None
        self._ring_ev = None
        self._ring_i = 0
        self.hits = self.misses = 0
        self.bytes_copied = 0

    def resolve_host(self, layer: int, ids: list[int]) -> list[int]:
        """Same as resolve(), but the slots stay on the host (S18): the
        segmented-graph glue writes them into its own pinned staging buffer
        and one async H2D feeds the static device tensor the graph reads,
        instead of a fresh device tensor per layer."""
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
        return slots

    def resolve(self, layer: int, ids: list[int]) -> torch.Tensor:
        """Device int32 slots for local expert ids (valid for every
        projection of the layer). Missing experts are copied in first; the
        ids arrive on the host because decode does its top-k there (C5)."""
        return torch.tensor(self.resolve_host(layer, ids),
                            dtype=torch.int32, device=self.tp.device)

    def _ensure_ring(self) -> None:
        """Allocate the pinned staging ring on the first swap miss (so
        pinned_bytes() is 0 while the store has only served residents)."""
        if self._ring is None:
            self._ring = [
                _pin(torch.zeros(self._ring_bytes, dtype=torch.uint8))
                for _ in range(self._ring_slots)]
            self._ring_ev = [None] * self._ring_slots

    def _copy_misses(self, layer: int, misses: list[tuple[int, int]]) -> None:
        cur = torch.xpu.current_stream()
        # the side stream must not overwrite a slot a queued kernel may read
        self._side.wait_stream(cur)
        if self.host_cache:
            # S27-3: rows are pinned by the load-time fill and never
            # rewritten, so the pool row is a direct non-blocking H2D
            # source — no ring staging, no GGUF read during decode.
            with torch.xpu.stream(self._side):
                for e, slot in misses:
                    for name, buf in self.bufs[layer].items():
                        row = self.pools[layer, name].row_bytes(e)
                        buf[slot].reshape(-1).copy_(row, non_blocking=True)
                self._ev.record(self._side)
            cur.wait_event(self._ev)
            return
        self._ensure_ring()
        with torch.xpu.stream(self._side):
            for e, slot in misses:
                for name, buf in self.bufs[layer].items():
                    gr = self.layers[layer][name]
                    r = self._ring_i
                    self._ring_i = (r + 1) % self._ring_slots
                    if self._ring_ev[r] is not None:
                        # this stage buffer is still feeding an earlier copy
                        self._ring_ev[r].synchronize()
                    stage = self._ring[r][:gr.slot_bytes]
                    stage.copy_(torch.from_numpy(gr.expert_bytes(e)))
                    buf[slot].reshape(-1).copy_(stage, non_blocking=True)
                    ev = self._ring_ev[r] = torch.xpu.Event()
                    ev.record(self._side)
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
        """Host bytes page-locked so far: the staging ring (the store's only
        pinned memory without a host cache, job S17-6) plus the pool buffers
        filled for the S27-3 host cache (or touched through the row API)."""
        ring = 0 if self._ring is None else sum(b.numel() for b in self._ring)
        return ring + sum(p.allocated_bytes for p in self.pools.values())

    def pinned_planned_bytes(self) -> int:
        """Worst-case pinned bytes: every held expert pinned (C5 check)."""
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
    — adding it before would multiply it by the world size. `shared_gate`,
    with it, is a float32 [k] vector: the shared output is scaled per token
    by `sigmoid(x @ shared_gate)` (Flash-Next ffn_gate_inp_shexp, S23 map C).

    `routing` selects the weight rule of `topk_weights` ("renorm" default,
    "logits" for GPT-OSS). `expert_bias`, when given, is a triple (gate, up,
    down) of float32 [E, n] tensors (n = ffn, ffn, k): a per-expert bias
    added to each projection output (GPT-OSS ffn_*_exps.bias); it is small
    enough to replicate [E, n] on every rank and index by global expert id.

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
                 limit: float = 7.0, shared=None, shared_gate=None,
                 routing: str = "renorm", expert_bias=None):
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
        if routing not in ROUTINGS:
            raise ValueError(f"unknown routing {routing!r}, expected one of {ROUTINGS}")
        self.routing = routing
        if expert_bias is None:
            self.eb = None
        else:
            eb = tuple(expert_bias)
            want = ((self.n_experts, self.ffn), (self.n_experts, self.ffn),
                    (self.n_experts, self.k))
            if len(eb) != 3 or any(t.shape != w or t.dtype != torch.float32
                                   for t, w in zip(eb, want)):
                raise ValueError(f"expert_bias must be three float32 tensors "
                                 f"{[list(w) for w in want]}")
            self.eb = eb
        act = ACTIVATIONS[activation]
        self.act = functools.partial(act, alpha=alpha, limit=limit) \
            if activation == "swiglu_oai" else act
        # kernels.swiglu_oai is a fused implementation of that one activation
        # (S18): decode_body may use it on GPU only when self.act is it too
        self._fused_act = activation == "swiglu_oai"
        self.alpha, self.limit = alpha, limit
        self.shared = tuple(shared) if shared is not None else None
        if shared_gate is not None:
            if self.shared is None:
                raise ValueError("shared_gate needs a shared expert triple")
            if shared_gate.shape != (self.k,) or shared_gate.dtype != torch.float32:
                raise ValueError(f"shared_gate must be float32 [{self.k}]")
        self.shared_gate = shared_gate
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
        y = down(self.act(gate(x), up(x)))
        if self.shared_gate is not None:
            # x [T, k] f32 -> gate scalar per token, applied in f32
            y = y * torch.sigmoid(x.float() @ self.shared_gate)[:, None]
        return y

    # ---- graph-friendly decode pieces (S18) ----
    def _init_decode_bufs(self, dev) -> None:
        """Static decode buffers for the segmented graph: fixed-shape rows
        (always top_k expert rows; padded rows read slot 0 with weight 0, so
        they cost bandwidth but add nothing) at fixed addresses, so each layer
        segment can be captured once and replayed. y accumulates across the
        resolve batches of one token; the reduce and the residual live in the
        neighbouring segments (see decode_body)."""
        if getattr(self, "_d", None) is not None:
            return
        Lm = self.top_k
        f32 = torch.float32

        def host(t):  # pinned on GPU so one async H2D feeds the device twins
            return t.pin_memory() if torch.device(dev).type == "xpu" else t

        self._d = {
            "xrep": torch.zeros(Lm, self.k, dtype=f32, device=dev),
            "g": torch.zeros(Lm, self.ffn, dtype=f32, device=dev),
            "u": torch.zeros(Lm, self.ffn, dtype=f32, device=dev),
            "a": torch.zeros(Lm, self.ffn, dtype=f32, device=dev),
            "d": torch.zeros(Lm, self.k, dtype=f32, device=dev),
            "y": torch.zeros(self.k, dtype=f32, device=dev),
            "slots": torch.zeros(Lm, dtype=torch.int32, device=dev),
            "wts": torch.zeros(Lm, dtype=f32, device=dev),
            "ei": torch.zeros(Lm, dtype=torch.long, device=dev),
        }
        self._dh = {n: host(torch.zeros(Lm, dtype=self._d[n].dtype))
                    for n in ("slots", "wts", "ei")}
        self._batches: list[list[tuple[int, float]]] = [[]]

    def decode_route(self, logits: torch.Tensor) -> tuple[list[int], list[float]]:
        """The one host sync per MoE layer (C5): top-k on the device, chosen
        experts on the host, so resolve can run there."""
        w, idx = topk_weights(logits, self.top_k, self.routing)
        return idx[0].tolist(), w[0].tolist()

    def decode_fill(self, ids: list[int], w: list[float]) -> int:
        """Routing -> resolve batches, host side. Residents ride once with the
        first X misses, further misses in blocks of X (S17-8 fix). Returns the
        number of decode_body runs the driver must do, always >= 1: an empty
        batch runs the body on zero weights and adds nothing."""
        m = self.store.maps[self.layer]
        X = m.S - m.R  # swap slots: fresh misses allowed per resolve
        sel = [(e, wt) for e, wt in zip(ids, w) if e in self.local]
        res = [p for p in sel if p[0] in m.resident]
        mis = [p for p in sel if p[0] not in m.resident]
        step = max(X, 1)
        batches = [b for b in ([res + mis[:step]] +
                               [mis[i:i + step] for i in range(step, len(mis), step)]) if b]
        self._batches = batches or [[]]
        return len(self._batches)

    def decode_fill_batch(self, bi: int) -> None:
        """Resolve batch bi (misses staged on the side stream, S17 ring), then
        fill the static top_k-row slots/weights/expert-id tensors through one
        pinned staging copy each; the captured segment reads only those."""
        b = self._batches[bi]
        slots = self.store.resolve_host(self.layer, [e for e, _ in b])
        L = len(b)
        self._dh["slots"][:L].copy_(torch.tensor(slots, dtype=torch.int32))
        self._dh["slots"][L:].zero_()
        self._dh["wts"][:L].copy_(torch.tensor([wt for _, wt in b], dtype=torch.float32))
        self._dh["wts"][L:].zero_()
        self._dh["ei"][:L].copy_(torch.tensor([e for e, _ in b], dtype=torch.long))
        self._dh["ei"][L:].zero_()
        for n in ("slots", "wts", "ei"):
            self._d[n].copy_(self._dh[n], non_blocking=True)

    def decode_body(self, x: torch.Tensor) -> torch.Tensor:
        """One resolve batch over the static buffers: indexed GEMVs (gate, up,
        down) + per-expert biases + activation (fused swiglu_oai kernel on
        GPU), weighted-summed into y. No
        zeroing (the driver zeroes once per layer so several batches
        accumulate) and no all_reduce/residual — those live in the neighbour
        segments, so every rank replays the same collective count whatever its
        batch count is (S18 graph, C5). Returns y [k]."""
        d = self._d
        d["xrep"].copy_(x.reshape(1, self.k).expand(self.top_k, -1))
        bufs = self.store.bufs[self.layer]
        gemv_indexed(self.gate.qtype, bufs["gate"], d["xrep"], d["slots"], d["g"])
        gemv_indexed(self.up.qtype, bufs["up"], d["xrep"], d["slots"], d["u"])
        if self.eb is not None:  # per-expert bias, indexed by global id
            d["g"] += self.eb[0][d["ei"]]
            d["u"] += self.eb[1][d["ei"]]
        if self._fused_act and torch.device(x.device).type == "xpu":
            from . import kernels as K   # Triton at call time only: header note
            K.swiglu_oai(d["g"], d["u"], out=d["a"],
                         alpha=self.alpha, limit=self.limit)
        else:
            d["a"].copy_(self.act(d["g"], d["u"]))
        gemv_indexed(self.down.qtype, bufs["down"], d["a"], d["slots"], d["d"])
        if self.eb is not None:
            d["d"] += self.eb[2][d["ei"]]
        d["y"] += (d["d"] * d["wts"][:, None]).sum(0)
        return d["y"]

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """One token: x [1, k] (or [k]) f32 -> [k] f32 summed over ranks.
        Eager driver over the graph pieces (S18): the same bodies, no capture."""
        x = x.reshape(1, self.k)
        self._init_decode_bufs(x.device)
        ids, w = self.decode_route(self._logits(x))
        nb = self.decode_fill(ids, w)
        self._d["y"].zero_()
        for bi in range(nb):
            self.decode_fill_batch(bi)
            self.decode_body(x)
        y = self.tp.all_reduce(self._d["y"])
        if self.shared is not None:  # replicated: add once, after the reduce
            y = y + self._shared_out(x)[0]
        return y

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Token chunk: x [T, k] f32 -> [T, k] f32 summed over ranks."""
        x = x.reshape(-1, self.k)
        dev = x.device
        w, idx = topk_weights(self._logits(x), self.top_k, self.routing)
        ids_h, w_h = idx.cpu().tolist(), w.float().cpu().tolist()  # one sync
        y = torch.zeros_like(x)
        for e, (rows, wts) in group_by_expert(ids_h, w_h, self.local).items():
            slot = self.store.resolve(self.layer, [e]).item()
            r = torch.from_numpy(rows).to(dev)
            xt = x[r]
            g, u = self._gemm(self.gate, slot, xt), self._gemm(self.up, slot, xt)
            if self.eb is not None:  # per-expert bias (global id e)
                g = g + self.eb[0][e]
                u = u + self.eb[1][e]
            a = self.act(g, u)
            d = self._gemm(self.down, slot, a)
            if self.eb is not None:
                d = d + self.eb[2][e]
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
