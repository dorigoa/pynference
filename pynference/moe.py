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

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .quant import row_bytes

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
        raw = np.ascontiguousarray(self.raw_fn(e), dtype=np.uint8)
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
            v = min((s for s in range(self.R, self.S) if s not in fresh),
                    key=lambda s: self.ts[s])
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
