"""Token sampling on the CPU in float32 (contract C4).

`sample` never touches the GPU, so every rank of a tensor-parallel run samples
the same token from the same logits with the same seed: no extra messages.
`temp <= 0` is greedy; given the seed the stream is deterministic.

The filtering pipeline matches what `pynference.generate` has always done
(softmax over the top-k slice, then the top-p nucleus), with min-p added:
tokens whose probability drops below `min_p` times the top probability are
removed before the final renormalisation.
"""
from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temp: float, top_k: int, top_p: float,
           min_p: float, generator: torch.Generator | None = None) -> int:
    """One token id from `logits`; `temp <= 0` = greedy. Deterministic given `generator`."""
    logits = logits.float().cpu().view(-1)
    if temp <= 0:
        return int(logits.argmax())
    if 0 < top_k < logits.numel():
        vals, idx = logits.topk(top_k)
    else:
        vals, idx = logits.sort(descending=True)
    probs = torch.softmax(vals / temp, dim=-1)
    p_max = float(probs[0])
    if top_p < 1.0:
        keep = (probs.cumsum(0) - probs) < top_p
        probs = probs * keep
    if min_p > 0.0:
        probs = probs * (probs >= min_p * p_max)
    total = float(probs.sum())
    if total <= 0.0:              # everything filtered out (e.g. top_p = 0): keep the top token
        return int(idx[0])
    probs = probs / total
    return int(idx[torch.multinomial(probs, 1, generator=generator)])


class Sampler:
    """Stateful wrapper: keeps the RNG so a whole generation is one seed stream."""

    def __init__(self, temp: float, top_k: int, top_p: float, seed: int, min_p: float = 0.0):
        self.temp, self.top_k, self.top_p, self.min_p = temp, top_k, top_p, min_p
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, logits: torch.Tensor) -> int:
        return sample(logits, self.temp, self.top_k, self.top_p, self.min_p, self.gen)
