"""CPU tests for TP.broadcast_obj (contract C4), single-process path."""
from pynference.parallel import TP


def make_tp(world: int, rank: int) -> TP:
    tp = TP.__new__(TP)          # skip __init__: no GPU and no process group here
    tp.world, tp.rank = world, rank
    return tp


def test_broadcast_obj_single_process():
    tp = make_tp(1, 0)
    obj = {"op": "generate", "ids": [1, 2, 3],
           "sampling": {"temp": 0.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "seed": 1234},
           "stop_ids": [1, 106]}
    assert tp.broadcast_obj(obj) is obj
    assert tp.broadcast_obj(None) is None


def test_broadcast_obj_returns_object_not_copy_on_main():
    tp = make_tp(1, 0)
    cmd = ["a", 1]
    assert tp.broadcast_obj(cmd) == cmd
