"""CPU tests: GPT-OSS-120B facts (S13) checked against the GGUF in git's facts.json.

The tests that open the model are skipped when the file is absent (it lives only
on the PC, /nvme). Goldens (tokens/template/ref_probs/ref_ppl) come from job
S13-1 (llama.cpp llama-server, --n-cpu-moe 3); see report S13. facts.json
escapes the first char of special tokens as \\u003c (plan 2.1), so the file has
no raw chat markers.
"""
import json
import os
import pathlib

import pytest

from pynference.gguf import GGUFFile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "gpt-oss-120b"
MODEL = "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf"

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

FACTS = json.loads((GOLDEN / "facts.json").read_text(encoding="utf-8"))


def load(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def unesc(s):
    """facts.json escapes the first char of special tokens (plan 2.1)."""
    return s.replace("\\u003c", chr(0x3C))


@pytest.fixture(scope="module")
def g():
    f = GGUFFile(MODEL)
    yield f
    f.close()


@needs_model
def test_metadata_matches_facts(g):
    for k, v in FACTS["metadata"].items():
        if isinstance(v, dict):  # long arrays/strings: lengths checked in another test
            continue
        if isinstance(v, list):
            assert [str(x) for x in g.kv[k]] == [str(x) for x in v], k
        else:
            assert g.kv[k] == v, k
    assert g.arch == FACTS["architecture"]


@needs_model
def test_long_metadata_lengths(g):
    m = FACTS["metadata"]
    assert len(g.kv["tokenizer.ggml.tokens"]) == m["tokenizer.ggml.tokens"]["_list_len"]
    assert len(g.kv["tokenizer.ggml.token_type"]) == m["tokenizer.ggml.token_type"]["_array_len"]
    assert len(g.kv["tokenizer.ggml.merges"]) == m["tokenizer.ggml.merges"]["_list_len"]
    tpl = g.kv["tokenizer.chat_template"]
    assert len(tpl) == m["tokenizer.chat_template"]["_string_len"]
    assert "channel" in tpl  # harmony template


@needs_model
def test_blk_and_global_tensors_match_facts(g):
    for blk in ("blk.0", "blk.1"):
        for name, info in FACTS["blocks"][blk].items():
            t = g.tensors[name]
            assert tuple(t.shape) == tuple(info["shape"]), name
            assert t.type == info["type"], name
    for name, info in FACTS["global_tensors"].items():
        t = g.tensors[name]
        assert tuple(t.shape) == tuple(info["shape"]), name
        assert t.type == info["type"], name


@needs_model
def test_tensor_type_counts(g):
    counts = {}
    for t in g.tensors.values():
        counts[t.type] = counts.get(t.type, 0) + 1
    assert counts == FACTS["tensor_types"]
    assert len(g.tensors) == FACTS["n_tensors"]


@needs_model
def test_all_layers_fused_experts_router_sinks(g):
    """Deviation from 3.8: experts are fused per layer, 3-D MXFP4 tensors."""
    n = FACTS["metadata"]["gpt-oss.block_count"]
    for i in range(n):
        for p in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            w = g.tensors[f"blk.{i}.{p}.weight"]
            assert w.type == "MXFP4" and tuple(w.shape) == (2880, 2880, 128), w.name
            b = g.tensors[f"blk.{i}.{p}.bias"]
            assert b.type == "F32" and tuple(b.shape) == (2880, 128), b.name
        rw = g.tensors[f"blk.{i}.ffn_gate_inp.weight"]
        assert rw.type == "F32" and tuple(rw.shape) == (2880, 128)
        s = g.tensors[f"blk.{i}.attn_sinks.weight"]
        assert s.type == "F32" and tuple(s.shape) == (64,)


@needs_model
def test_tokenizer_ids_match_facts(g):
    tk = FACTS["tokenizer"]
    toks = g.kv["tokenizer.ggml.tokens"]
    types = g.kv["tokenizer.ggml.token_type"]
    assert len(toks) == tk["n_tokens"]
    for key, (tid, ttype) in tk["special_tokens"].items():
        assert toks[tid] == unesc(key), key
        assert int(types[tid]) == ttype, key
    assert g.kv["tokenizer.ggml.bos_token_id"] == tk["bos_token_id"]
    assert g.kv["tokenizer.ggml.eos_token_id"] == tk["eos_token_id"]
    assert g.kv["tokenizer.ggml.model"] == tk["model"]
    assert g.kv["tokenizer.ggml.pre"] == tk["pre"]
    assert g.kv.get("tokenizer.ggml.add_bos_token", False) == tk["add_bos_token"]
    # observed: the GGUF carries no eot/eom token ids (harmony specials exist, but
    # their ids are not stored as metadata)
    for key in ("eot_token_id", "eom_token_id", "eot_ids", "eom_ids"):
        assert f"tokenizer.ggml.{key}" not in g.kv


def test_golden_files_shape():
    """Golden files from job S13-1 are in git and have the llama_golden.py shape."""
    tokens = load("tokens.json")
    assert len(tokens) == 14
    assert all(set(c) == {"text", "ids"} and c["ids"] for c in tokens)
    tpl = load("template.json")
    assert [c["name"] for c in tpl] == ["chat-it", "chat-en"]
    assert all(c["kwargs"] == {} and "system" in c["prompt"] for c in tpl)
    rp = load("ref_probs.json")
    assert [c["name"] for c in rp] == ["chat-it", "chat-en", "code", "raw"]
    assert all(len(c["steps"]) == 24 and c["prompt"] for c in rp)


def test_ref_ppl_golden():
    line = (GOLDEN / "ref_ppl.txt").read_text(encoding="utf-8")
    assert "Final estimate: PPL" in line
    ppl = float(line.split("PPL =")[1].split("+/-")[0].strip())
    assert 1.0 < ppl < 5.0  # llama.cpp sanity for this file; used by S17 for C-compare
