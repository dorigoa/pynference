"""CPU tests for pynference.sampling (contract C4): greedy, seed determinism,
top-p and min-p masks."""
import pytest
import torch

from pynference.sampling import Sampler, sample

VOCAB = 8


def logits_at(step: int) -> torch.Tensor:
    """A varying but fixed logit vector, so a seeded run is reproducible."""
    gen = torch.Generator().manual_seed(1000 + step)
    return torch.randn(VOCAB, generator=gen)


def run_seeded(seed: int, n: int, **kw) -> list[int]:
    sampler = Sampler(kw.pop("temp", 1.0), kw.pop("top_k", 0), kw.pop("top_p", 1.0), seed,
                      **kw)
    return [sampler(logits_at(i)) for i in range(n)]


def test_greedy_is_argmax():
    x = torch.tensor([0.1, -2.0, 3.5, 0.0, -1.0, 4.2, -0.5, 2.0])
    assert sample(x, 0.0, 0, 1.0, 0.0, None) == 5
    assert sample(x, -1.0, 0, 1.0, 0.0, None) == 5       # temp <= 0 = greedy
    sampler = Sampler(0.0, 40, 0.95, 1234)               # temp 0 ignores the seed
    assert [sampler(logits_at(i)) for i in range(5)] == [int(logits_at(i).argmax()) for i in range(5)]


def test_deterministic_with_seed():
    a = run_seeded(42, 30, temp=0.8, top_k=4, top_p=0.9)
    b = run_seeded(42, 30, temp=0.8, top_k=4, top_p=0.9)
    assert a == b
    assert len(set(a)) > 1                                # really sampling, not greedy
    c = run_seeded(7, 30, temp=0.8, top_k=4, top_p=0.9)
    assert a != c


def test_top_p_mask():
    # probs ~ [0.9526, 0.0442, 0.0008, ...]: top_p 0.5 keeps only the first token
    x = torch.tensor([3.0, 0.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0])
    for i in range(50):
        assert sample(x, 1.0, 0, 0.5, 0.0, torch.Generator().manual_seed(i)) == 0
    # with top_p 0.99 the nucleus is {0, 1}: token 1 becomes reachable, token 2 never drawn
    drawn = [sample(x, 1.0, 0, 0.99, 0.0, torch.Generator().manual_seed(i)) for i in range(300)]
    assert set(drawn) <= {0, 1}
    assert 1 in drawn


def test_min_p_mask():
    # softmax ~ [0.646, 0.237, 0.087, 0.032]; min_p 0.3 cuts below 0.3*0.646 = 0.194
    x = torch.tensor([2.0, 1.0, -0.35, -1.35, -7.0, -7.0, -7.0, -7.0])
    drawn = [sample(x, 1.0, 0, 1.0, 0.3, torch.Generator().manual_seed(i)) for i in range(200)]
    assert set(drawn) <= {0, 1}
    # min_p 0 keeps them reachable (the third token has ~8.7%)
    drawn = [sample(x, 1.0, 0, 1.0, 0.0, torch.Generator().manual_seed(i)) for i in range(200)]
    assert 2 in drawn


def test_generate_reexports_sampler():
    from pynference.generate import Sampler as GS
    assert GS is Sampler
