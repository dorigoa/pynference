"""CPU tests for bench/compare_probs.py diff and its --strict mode (contract C8)."""
import copy
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("compare_probs_mod", ROOT / "bench" / "compare_probs.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

NAMES = ("p0", "p1", "p2", "p3")


def step(tid, probs):
    return {"id": tid, "top": [{"id": t, "token": "x", "p": p} for t, p in probs]}


def ident_steps(ids):
    return [step(t, [(t, 0.6), (t + 100, 0.4)]) for t in ids]


def make(name, ids, steps):
    return {"name": name, "prompt": ids, "steps": steps, "text": "t"}


def suite(ref_steps, our_steps):
    """Four prompts, identical unless the given prompt index overrides ours."""
    ref, our = [], []
    for i, name in enumerate(NAMES):
        ids = [10 + i, 20 + i]
        ref.append(make(name, ids, ref_steps[i]))
        our.append(make(name, ids, our_steps[i]))
    return ref, our


def tail_near_tie(prefix_ids):
    """Ref/ours steps sharing a prefix, then one diverging step that is a near-tie."""
    ref = ident_steps(prefix_ids) + [step(5, [(5, 0.5), (9, 0.4)])]      # gap 0.10 < 0.15
    our = ident_steps(prefix_ids) + [step(9, [(9, 0.48), (5, 0.42)])]
    return ref, our


def test_all_identical_passes():
    steps = [ident_steps([7, 8]) for _ in NAMES]
    ref, our = suite(steps, steps)
    assert cp.strict_problems(ref, our) == []


def test_one_near_tie_divergence_passes():
    steps = [ident_steps([7, 8]) for _ in NAMES]
    r1, o1 = tail_near_tie([7, 8])
    ref, our = suite(steps, steps)
    ref[1]["steps"], our[1]["steps"] = r1, o1
    assert cp.strict_problems(ref, our) == []


def test_non_near_tie_divergence_fails():
    steps = [ident_steps([7, 8]) for _ in NAMES]
    ref, our = suite(steps, steps)
    ref[2]["steps"] = ident_steps([7, 8]) + [step(5, [(5, 0.9), (9, 0.05)])]  # gap 0.85
    our[2]["steps"] = ident_steps([7, 8]) + [step(9, [(9, 0.5), (5, 0.4)])]
    problems = cp.strict_problems(ref, our)
    assert any("not a near-tie" in p for p in problems)


def test_two_divergences_fail_even_as_near_ties():
    steps = [ident_steps([7, 8]) for _ in NAMES]
    ref, our = suite(steps, steps)
    for i in (0, 3):
        r, o = tail_near_tie([7, 8])
        ref[i]["steps"], our[i]["steps"] = r, o
    problems = cp.strict_problems(ref, our)
    assert any("greedy-identical prompts" in p for p in problems)
    assert not any("not a near-tie" in p for p in problems)


def test_kl_violation_fails():
    ref_steps = [ident_steps([7, 8]) for _ in NAMES]
    our_steps = copy.deepcopy(ref_steps)
    our_steps[1][0]["top"] = [{"id": 7, "token": "x", "p": 0.75}, {"id": 107, "token": "x", "p": 0.25}]
    ref, our = suite(ref_steps, our_steps)
    problems = cp.strict_problems(ref, our)
    assert any("mean partial KL" in p for p in problems)


def test_diff_strict_cli_exit_codes(tmp_path, monkeypatch, capsys):
    good = [ident_steps([7, 8]) for _ in NAMES]
    ref, our = suite(good, good)
    rfp, ofp = tmp_path / "r.json", tmp_path / "o.json"
    rfp.write_text(json.dumps(ref))
    ofp.write_text(json.dumps(our))
    monkeypatch.setattr(sys, "argv", ["x", "diff", "--strict", str(rfp), str(ofp)])
    cp.main()  # must not raise SystemExit
    assert "C8 strict: PASS" in capsys.readouterr().out

    bad_ours = [list(s) for s in good]
    bad_ours[0][0] = step(99, [(99, 0.95), (7, 0.03)])  # greedy id differs, big ref gap
    bad_ref, bad_our = suite(good, bad_ours)
    rfp.write_text(json.dumps(bad_ref))
    ofp.write_text(json.dumps(bad_our))
    monkeypatch.setattr(sys, "argv", ["x", "diff", "--strict", str(rfp), str(ofp)])
    with pytest.raises(SystemExit) as ei:
        cp.main()
    assert ei.value.code == 1
    assert "C8 strict: FAIL" in capsys.readouterr().out


def test_diff_without_strict_never_exits(tmp_path, monkeypatch, capsys):
    good = [ident_steps([7, 8]) for _ in NAMES]
    bad_ours = [list(s) for s in good]
    bad_ours[0][0] = step(99, [(99, 0.95), (7, 0.03)])
    ref, our = suite(good, bad_ours)
    rfp, ofp = tmp_path / "r.json", tmp_path / "o.json"
    rfp.write_text(json.dumps(ref))
    ofp.write_text(json.dumps(our))
    monkeypatch.setattr(sys, "argv", ["x", "diff", str(rfp), str(ofp)])
    cp.main()  # today's behavior: only prints
    out = capsys.readouterr().out
    assert "same greedy tokens" in out and "C8" not in out
