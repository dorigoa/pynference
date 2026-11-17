"""CPU tests: Qwen3.8-27B facts (S19) checked against plan 3.9 and the GGUF.

facts.json lives in git (tests/golden/qwen35-27b/) and escapes the first char of
special tokens as \\u003c (plan 2.1), so the file has no raw chat markers. The
tests that open the model are skipped when the file is absent (it lives only on
the PC, /nvme). See report S19 for the extracted table.
"""
import json
import os
import pathlib

import pytest

from pynference.gguf import GGUFFile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen35-27b"
MODEL = "/nvme/gguf/Qwen3.8-27B-Q8_0.gguf"

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


# ---------- facts.json vs plan 3.9 (no model needed) ----------

def test_arch_and_sizes_match_39():
    assert FACTS["architecture"] == "qwen35"
    m = FACTS["metadata"]
    assert m["qwen35.block_count"] == 65  # 64 + 1 MTP layer (blk.64, ignored: D4)
    assert m["qwen35.nextn_predict_layers"] == 1
    assert m["qwen35.embedding_length"] == 5120
    assert m["qwen35.feed_forward_length"] == 17408
    assert m["qwen35.context_length"] == 262144
    assert FACTS["tensor_bytes_gib"] == pytest.approx(27.04)
    assert FACTS["tensor_types"] == {"Q8_0": 506, "F32": 360}


def test_attention_facts_match_39():
    m = FACTS["metadata"]
    assert m["qwen35.attention.head_count"] == 24
    assert m["qwen35.attention.head_count_kv"] == 4
    assert m["qwen35.attention.key_length"] == 256
    assert m["qwen35.attention.value_length"] == 256
    assert m["qwen35.full_attention_interval"] == 4
    assert m["qwen35.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-6)


def test_rope_facts_match_39():
    m = FACTS["metadata"]
    assert m["qwen35.rope.dimension_count"] == 64
    assert m["qwen35.rope.freq_base"] == 1e7
    assert m["qwen35.rope.dimension_sections"] == [11, 11, 10, 0]


def test_ssm_facts_match_39():
    m = FACTS["metadata"]
    assert m["qwen35.ssm.conv_kernel"] == 4
    assert m["qwen35.ssm.group_count"] == 16
    assert m["qwen35.ssm.time_step_rank"] == 48
    assert m["qwen35.ssm.inner_size"] == 6144
    assert m["qwen35.ssm.state_size"] == 128


def test_tokenizer_facts_match_39():
    tk = FACTS["tokenizer"]
    assert tk["model"] == "gpt2"
    assert tk["pre"] == "qwen35"
    assert tk["bos_token_id"] == 248044
    assert tk["eos_token_id"] == 248046
    assert tk["n_tokens"] == 248320


def test_attention_layers_are_every_fourth():
    kinds = FACTS["layer_kinds"]
    assert len(kinds) == 65
    for i in range(64):
        assert kinds[f"blk.{i}"] == ("attention" if (i + 1) % 4 == 0 else "ssm"), i
    assert kinds["blk.64"] == "attention"  # MTP replica layer (ignored: D4)
    assert sum(v == "attention" for v in kinds.values()) == 17
    assert sum(v == "ssm" for v in kinds.values()) == 48


def test_deltanet_layer_shapes_match_39():
    blk = FACTS["blocks"]["blk.0"]
    got = {n[len("blk.0."):]: (tuple(i["shape"]), i["type"]) for n, i in blk.items()}
    exp = {
        "attn_qkv.weight": ((5120, 10240), "Q8_0"),  # q 16*128 + k 16*128 + v 48*128
        "attn_gate.weight": ((5120, 6144), "Q8_0"),  # 48 * 128 gate channels
        "ssm_conv1d.weight": ((4, 10240), "F32"),    # kernel 4 over qkv channels
        "ssm_a": ((48,), "F32"),
        "ssm_dt.bias": ((48,), "F32"),
        "ssm_alpha.weight": ((5120, 48), "Q8_0"),
        "ssm_beta.weight": ((5120, 48), "Q8_0"),
        "ssm_norm.weight": ((128,), "F32"),          # per-head norm over state dim
        "ssm_out.weight": ((6144, 5120), "Q8_0"),
        "attn_norm.weight": ((5120,), "F32"),
        "post_attention_norm.weight": ((5120,), "F32"),
        "ffn_gate.weight": ((5120, 17408), "Q8_0"),
        "ffn_up.weight": ((5120, 17408), "Q8_0"),
        "ffn_down.weight": ((17408, 5120), "Q8_0"),
    }
    assert got == exp
    assert FACTS["ssm_block_tensor_signatures"] == {k: {"shape": list(s), "type": t}
                                                    for k, (s, t) in exp.items()}


def test_attention_layer_shapes_match_39():
    blk = FACTS["blocks"]["blk.3"]
    got = {n[len("blk.3."):]: (tuple(i["shape"]), i["type"]) for n, i in blk.items()}
    exp = {
        # gate folded into attn_q: 24*256 for q + 24*256 for the gate (attn_gate)
        "attn_q.weight": ((5120, 12288), "Q8_0"),
        "attn_k.weight": ((5120, 1024), "Q8_0"),
        "attn_v.weight": ((5120, 1024), "Q8_0"),
        "attn_q_norm.weight": ((256,), "F32"),
        "attn_k_norm.weight": ((256,), "F32"),
        "attn_output.weight": ((6144, 5120), "Q8_0"),
        "attn_norm.weight": ((5120,), "F32"),
        "post_attention_norm.weight": ((5120,), "F32"),
        "ffn_gate.weight": ((5120, 17408), "Q8_0"),
        "ffn_up.weight": ((5120, 17408), "Q8_0"),
        "ffn_down.weight": ((17408, 5120), "Q8_0"),
    }
    assert got == exp
    # the MTP block blk.64 shares the attention tensors and adds blk.64.nextn.*
    mtp = {
        "nextn.eh_proj.weight": ((10240, 5120), "Q8_0"),
        "nextn.enorm.weight": ((5120,), "F32"),
        "nextn.hnorm.weight": ((5120,), "F32"),
        "nextn.shared_head_norm.weight": ((5120,), "F32"),
    }
    exp.update(mtp)
    assert FACTS["attention_block_tensor_signatures"] == {k: {"shape": list(s), "type": t}
                                                          for k, (s, t) in exp.items()}


def test_facts_json_has_no_raw_markers():
    text = (GOLDEN / "facts.json").read_text(encoding="utf-8")
    assert chr(0x3C) not in text
    tk = FACTS["tokenizer"]
    assert len(tk["special_tokens"]) == 33
    assert tk["special_tokens"]["\x3c|im_start|>"] == [248045, 3]


def test_golden_token_files_present():
    tokens = load("tokens.json")
    assert all(set(c) == {"text", "ids"} and c["ids"] for c in tokens)
    tpl = load("template.json")
    assert all({"name", "messages", "kwargs", "prompt"} <= set(c) for c in tpl)
    probs = load("ref_probs.json")
    assert isinstance(probs, list) and probs
    assert "Final estimate" in (GOLDEN / "ref_ppl.txt").read_text(encoding="utf-8")


# ---------- facts.json vs the GGUF (skipped without the file) ----------

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
    assert len(g.kv["tokenizer.ggml.tokens"]) == FACTS["metadata"]["tokenizer.ggml.tokens"]["_list_len"]
    assert len(g.kv["tokenizer.ggml.token_type"]) == FACTS["metadata"]["tokenizer.ggml.token_type"]["_array_len"]
    assert len(g.kv["tokenizer.ggml.merges"]) == FACTS["metadata"]["tokenizer.ggml.merges"]["_list_len"]
    assert len(g.kv["tokenizer.chat_template"]) == FACTS["metadata"]["tokenizer.chat_template"]["_string_len"]


@needs_model
def test_blocks_and_globals_match_facts(g):
    for blk in FACTS["blocks"]:
        for name, info in FACTS["blocks"][blk].items():
            t = g.tensors[name]
            assert tuple(t.shape) == tuple(info["shape"]), name
            assert t.type == info["type"], name
    for name, info in FACTS["global_tensors"].items():
        t = g.tensors[name]
        assert tuple(t.shape) == tuple(info["shape"]), name
        assert t.type == info["type"], name
    assert len(g.tensors) == FACTS["n_tensors"]
    counts = {}
    for t in g.tensors.values():
        counts[t.type] = counts.get(t.type, 0) + 1
    assert counts == FACTS["tensor_types"]


@needs_model
def test_layer_kinds_match_tensor_names(g):
    for i in range(64):
        pre = f"blk.{i}."
        has_q = any(n.startswith(pre + "attn_q.weight") for n in g.tensors)
        has_ssm = any(n.startswith(pre + "ssm_") for n in g.tensors)
        assert has_q != has_ssm, i
        assert (FACTS["layer_kinds"][f"blk.{i}"] == "attention") == has_q, i


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
