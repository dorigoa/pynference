"""CPU tests for pynference.engine (contract C4) over the fake C3 model:
prefix reuse, stop, max_tokens, counts. The greedy stream from [1, 2, 3] is
7, 2, 4, 8, 4, 8, ... ((sum % 12) + 1)."""
import pytest
from fake_model import GREEDY, FakeModel, FakeTok

from pynference.engine import Engine, Session


def make() -> tuple[FakeModel, Session]:
    m = FakeModel()
    return m, Session(m, FakeTok())


def test_prefill_reuses_prefix():
    m, s = make()
    s.prefill([1, 2, 3])
    assert m.forward_sizes == [3]
    s.prefill([1, 2, 3, 4])                       # extends: only the tail
    assert m.forward_sizes == [3, 1]
    assert s.tokens == [1, 2, 3, 4] and m.ctx == s.tokens
    s.prefill([9, 9])                             # different prompt: reset + all
    assert m.forward_sizes == [3, 1, 2]
    assert m.ctx == [9, 9] and s.tokens == [9, 9]


def test_generate_max_tokens_and_counts():
    m, s = make()
    out = s.generate([1, 2, 3], 5, dict(GREEDY), stop_ids={0})
    assert out == [7, 2, 4, 8, 4]
    assert s.tokens == [1, 2, 3, 7, 2, 4, 8, 4] == m.ctx
    assert m.forward_sizes == [3, 1, 1, 1, 1, 1]
    assert s.last_stop is None


def test_generate_second_turn_feeds_only_the_new_tokens():
    m, s = make()
    s.generate([1, 2, 3], 2, dict(GREEDY), stop_ids={0})          # 7, 2
    ids = [1, 2, 3, 7, 2, 0]                                      # + eog + new turn
    out = s.generate(ids, 1, dict(GREEDY), stop_ids={0})
    assert out == [4]                                             # sum(ids) % 12 + 1 = 4
    assert m.forward_sizes == [3, 1, 1, 1, 1]                     # prefill: only the new id


def test_generate_stops_on_stop_id():
    m, s = make()
    out = s.generate([1, 2, 3], 9, dict(GREEDY), stop_ids={4})
    assert out == [7, 2]
    assert s.last_stop == 4
    assert s.tokens == [1, 2, 3, 7, 2] == m.ctx


def test_generate_stop_strings():
    m, s = make()
    out = s.generate([1, 2, 3], 9, dict(GREEDY), stop_ids={0}, stop_strings=("4 8",))
    assert out == [7, 2]                                          # cut before the match
    assert s.tokens == [1, 2, 3, 7, 2, 4] == m.ctx                # 4 was fed before the match


def test_on_token_sees_every_generated_token():
    m, s = make()
    seen: list[tuple[int, str]] = []
    out = s.generate([1, 2, 3], 3, dict(GREEDY), stop_ids={0},
                     on_token=lambda i, t: seen.append((i, t)))
    assert out == [7, 2, 4]
    assert seen == [(7, "7"), (2, "2"), (4, "4")]


def test_engine_run_and_reset_single_process():
    m = FakeModel()
    e = Engine(m, FakeTok())                                      # tp=None: no broadcast
    out = e.run({"op": "generate", "ids": [1, 2, 3], "max_tokens": 2,
                 "sampling": dict(GREEDY), "stop_ids": [0]})
    assert out == [7, 2]
    e.run({"op": "reset"})
    assert e.session.tokens == [] and m.ctx == []
    out = e.run({"op": "generate", "ids": [1, 2, 3], "max_tokens": 1,
                 "sampling": dict(GREEDY), "stop_ids": [0]})
    assert out == [7]
    assert m.forward_sizes == [3, 1, 1, 3, 1]                     # after reset: all again


def test_engine_unknown_op():
    e = Engine(FakeModel(), FakeTok())
    with pytest.raises(ValueError, match="unknown"):
        e.run({"op": "teleport"})
