"""CPU tests: Qwen3.8-Flash-Next facts (S23) checked against plan 3.10 and the GGUF.

facts.json lives in git (tests/golden/qwen4exp/) and escapes the first char of
special tokens as \\u003c (plan 2.1), so the file has no raw chat markers. The
tests that open the model are skipped when the file is absent (it lives only on
the PC, /nvme). See report S23 for the extracted table.
"""
import json
import os
import pathlib

import pytest

from pynference.gguf import GGUFFile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

FACTS = json.loads((GOLDEN / "facts.json").read_text(encoding="utf-8"))


# ---------- facts.json vs plan 3.10 (no model needed) ----------

def test_arch_and_sizes_match_310():
    assert FACTS["architecture"] == "qwen4exp"
    assert FACTS["n_tensors"] == 1224
    assert FACTS["tensor_bytes_gib"] == pytest.approx(87.23)
    m = FACTS["metadata"]
    assert m["qwen4exp.block_count"] == 48
    assert m["qwen4exp.embedding_length"] == 2560
    assert m["qwen4exp.context_length"] == 262144


def test_type_gib_match_310():
    # plan 3.10 GiB-per-type table
    got = FACTS["tensor_gib_by_type"]
    exp = {"IQ4_NL": 32.5, "Q5_0": 17.7, "IQ3_XXS": 16.8, "IQ4_XS": 14.4, "Q6_K": 2.2,
           "IQ2_S": 1.5, "Q8_0": 1.3, "Q2_0": 0.44, "F32": 0.29, "BF16": 0.04}
    assert set(got) == set(exp)
    for k, v in exp.items():
        assert got[k] == pytest.approx(v, abs=0.06), k


def test_attention_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.attention.head_count"] == 24
    assert m["qwen4exp.attention.head_count_kv"] == 2
    assert m["qwen4exp.attention.key_length"] == 256
    assert m["qwen4exp.attention.value_length"] == 256
    assert m["qwen4exp.full_attention_interval"] == 4
    assert m["qwen4exp.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-6)
    assert m["qwen4exp.rope.dimension_count"] == 64
    assert m["qwen4exp.rope.freq_base"] == 1e7
    assert m["qwen4exp.rope.dimension_sections"] == [11, 11, 10, 0]
    # S23 measures: ALL twelve attention layers are compressed (ratio 4), and
    # every attention layer carries an indexer; there is no plain attention layer.
    ratios = m["qwen4exp.attention.compress_ratios"]
    assert ratios == [0, 0, 0, 4] * 12


def test_indexer_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.attention.indexer.head_count"] == 4
    assert m["qwen4exp.attention.indexer.key_length"] == 128
    assert m["qwen4exp.attention.indexer.top_k"] == 2048


def test_deltanet_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.ssm.conv_kernel"] == 4
    assert m["qwen4exp.ssm.state_size"] == 128
    assert m["qwen4exp.ssm.group_count"] == 16
    assert m["qwen4exp.ssm.time_step_rank"] == 48
    assert m["qwen4exp.ssm.inner_size"] == 6144


def test_moe_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.expert_count"] == 512
    assert m["qwen4exp.expert_used_count"] == 10
    assert m["qwen4exp.expert_feed_forward_length"] == 640
    assert m["qwen4exp.expert_shared_feed_forward_length"] == 640


def test_hyper_connection_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.hyper_connection.count"] == 4
    assert m["qwen4exp.hyper_connection.low_rank"] == 320


def test_ple_facts_match_310():
    m = FACTS["metadata"]
    assert m["qwen4exp.embedding_length_per_layer_input"] == 160
    assert m["qwen4exp.ple.ngram_size"] == 3
    assert m["qwen4exp.ple.heads_per_ngram"] == 8
    assert m["qwen4exp.ple.conv_kernel"] == 4
    assert m["qwen4exp.ple.layer_multipliers"] == [23703573157769, 20109073645365, 8052911324071]
    assert len(m["qwen4exp.ple.head_offsets"]) == 16
    assert len(m["qwen4exp.ple.head_vocab_sizes"]) == 16
    # S23 measures: the PLE path (ple_* tensors) exists on layer 1 only
    # (qwen4exp.ple.layers = [1]); per_layer_token_embd is a single global table.
    assert m["qwen4exp.ple.layers"] == [1]


def test_tokenizer_facts_match_310():
    tk = FACTS["tokenizer"]
    assert tk["model"] == "gpt2"
    assert tk["pre"] == "qwen35"
    assert tk["n_tokens"] == 248320
    assert tk["bos_token_id"] == 248044
    assert tk["eos_token_id"] == 248046


def test_vocab_identical_to_qwen35_27b():
    v = FACTS["vocab_vs_qwen35_27b"]
    assert v["same_tokens"] is True
    assert v["first_diff_index"] is None
    assert v["same_token_types"] is True
    assert v["same_merges"] is True
    assert v["chat_template_equal"] is True
    assert v["n_tokens"] == [248320, 248320]
    assert v["n_merges"] == [247587, 247587]
    # the only tokenizer difference: padding_token_id (248044 here, 248055 on 27B)
    assert v["ids"]["padding_token_id"] == [248044, 248055]
    assert v["ids"]["bos_token_id"] == [248044, 248044]
    assert v["ids"]["eos_token_id"] == [248046, 248046]


def test_layer_kinds():
    kinds = FACTS["layer_kinds"]
    assert len(kinds) == 48
    assert FACTS["kind_counts"] == {"deltanet": 35, "deltanet+ple": 1, "compressed_attention": 12}
    for i in range(48):
        if (i + 1) % 4 == 0:
            assert kinds[f"blk.{i}"] == "compressed_attention", i
        elif i == 1:
            assert kinds[f"blk.{i}"] == "deltanet+ple", i
        else:
            assert kinds[f"blk.{i}"] == "deltanet", i


def test_size_groups():
    # plan 3.10: experts 56.24 GiB, the rest 30.99 GiB mostly the PLE table
    g = FACTS["group_gib"]
    assert g["routed_experts"] == pytest.approx(56.24)
    assert g["ple_table"] == pytest.approx(26.82)
    assert g["rest"] == pytest.approx(4.17)
    assert g["ple_table"] + g["rest"] == pytest.approx(30.99)


def test_deltanet_block_shapes():
    blk = FACTS["blocks"]["blk.0"]
    got = {n[len("blk.0."):]: (tuple(i["shape"]), i["type"]) for n, i in blk.items()}
    assert got == {
        "attn_qkv.weight": ((2560, 10240), "Q6_K"),   # q 16*128 + k 16*128 + v 48*128
        "attn_gate.weight": ((2560, 6144), "Q6_K"),   # 48 * 128 gate channels
        "ssm_conv1d.weight": ((4, 10240), "F32"),
        "ssm_a": ((48,), "F32"),
        "ssm_dt.bias": ((48,), "F32"),
        "ssm_alpha.weight": ((2560, 48), "F32"),
        "ssm_beta.weight": ((2560, 48), "F32"),
        "ssm_norm.weight": ((128,), "F32"),
        "ssm_out.weight": ((6144, 2560), "Q8_0"),
        "hc_attn_norm.weight": ((10240,), "F32"),     # 4 flows * 2560
        "hc_attn_inject.weight": ((10240, 4), "F32"),
        "hc_attn_up.weight": ((320, 10240), "Q8_0"),
        "hc_attn_down.weight": ((10240, 320), "Q8_0"),
        "hc_ffn_norm.weight": ((10240,), "F32"),
        "hc_ffn_inject.weight": ((10240, 4), "F32"),
        "hc_ffn_up.weight": ((320, 10240), "Q8_0"),
        "hc_ffn_down.weight": ((10240, 320), "Q8_0"),
        "ffn_gate_inp.weight": ((2560, 512), "F32"),
        "ffn_gate_inp_shexp.weight": ((2560,), "F32"),
        "ffn_gate_exps.weight": ((2560, 640, 512), "IQ3_XXS"),
        "ffn_up_exps.weight": ((2560, 640, 512), "IQ3_XXS"),
        "ffn_down_exps.weight": ((640, 2560, 512), "Q5_0"),
        "ffn_gate_shexp.weight": ((2560, 640), "Q6_K"),
        "ffn_up_shexp.weight": ((2560, 640), "Q6_K"),
        "ffn_down_shexp.weight": ((640, 2560), "Q8_0"),
    }


def test_ple_block_adds_ple_tensors():
    base = set(FACTS["block_tensor_signatures"]["deltanet"])
    ple = FACTS["block_tensor_signatures"]["deltanet+ple"]
    assert set(FACTS["block_tensor_signatures"]["deltanet"]) <= set(ple)
    extra = {k: (tuple(v["shape"]), v["type"]) for k, v in ple.items() if k not in base}
    assert extra == {
        "ple_key.weight": ((2560, 10240), "Q8_0"),
        "ple_value.weight": ((2560, 2560), "Q8_0"),
        "ple_conv1d.weight": ((4, 10240), "F32"),
        "ple_norm_key.weight": ((10240,), "F32"),
        "ple_norm_query.weight": ((10240,), "F32"),
        "ple_norm_conv.weight": ((10240,), "F32"),
    }


def test_compressed_attention_block_shapes():
    blk = FACTS["blocks"]["blk.3"]
    got = {n[len("blk.3."):]: (tuple(i["shape"]), i["type"]) for n, i in blk.items()}
    assert got["attn_q.weight"] == ((2560, 12288), "Q6_K")  # 24*256 q + 24*256 gate
    assert got["attn_k.weight"] == ((2560, 512), "Q6_K")    # 2 KV heads * 256
    assert got["attn_v.weight"] == ((2560, 512), "Q6_K")
    assert got["attn_output.weight"] == ((6144, 2560), "Q6_K")
    assert got["attn_q_norm.weight"] == ((256,), "F32")
    assert got["attn_k_norm.weight"] == ((256,), "F32")
    assert got["indexer.q_proj.weight"] == ((2560, 512), "BF16")  # 4 heads * 128
    assert got["indexer.k_proj.weight"] == ((2560, 128), "BF16")
    assert got["indexer.q_norm.weight"] == ((128,), "F32")
    assert got["indexer.k_norm.weight"] == ((128,), "F32")
    # no ssm_* and no per-block ffn norms: norms live in the hc_* tensors
    assert not any(k.startswith(("ssm_", "attn_qkv", "ple_")) for k in got)


def test_global_tensors():
    g = FACTS["global_tensors"]
    assert g["token_embd.weight"]["type"] == "IQ4_XS"
    assert tuple(g["token_embd.weight"]["shape"]) == (2560, 248320)
    assert g["output.weight"]["type"] == "Q6_K"
    assert tuple(g["output.weight"]["shape"]) == (2560, 248320)
    # one shared n-gram table, 160 rows x 320,001,536 columns (sum of the 16
    # head vocab sizes, padded); PLE on layer 1 looks rows up in it
    ple = g["per_layer_token_embd.weight"]
    assert tuple(ple["shape"]) == (160, 320001536)
    assert ple["type"] == "IQ4_NL"
    assert ple["gib"] == pytest.approx(26.822)
    assert set(g) == {"token_embd.weight", "output.weight", "per_layer_token_embd.weight",
                      "output_hc_norm.weight", "output_hc_up.weight", "output_hc_down.weight"}


def test_mixed_expert_quant_types_are_recorded():
    # unsloth "UD": the MoE and dense projections mix types per layer
    t = FACTS["per_layer_types"]
    assert set(t) == {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                      "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "attn_gate.weight", "attn_qkv.weight"}
    assert len(t["ffn_gate_exps.weight"]) == 48
    assert set(t["ffn_gate_exps.weight"]) == {"IQ3_XXS", "IQ4_XS", "IQ2_S"}
    assert set(t["ffn_down_exps.weight"]) == {"Q5_0", "IQ4_NL", "Q2_0"}
    assert t["ffn_gate_exps.weight"] == t["ffn_up_exps.weight"]
    # attention layers have no dense qkv/gate: recorded as null
    assert t["attn_qkv.weight"][3] is None
    assert t["attn_qkv.weight"][0] == "Q6_K"


def test_facts_json_has_no_raw_markers():
    text = (GOLDEN / "facts.json").read_text(encoding="utf-8")
    assert chr(0x3C) not in text


# ---------- facts.json vs the GGUF (skipped without the file) ----------

@needs_model
def test_metadata_matches_facts():
    f = GGUFFile(MODEL)
    try:
        for k, v in FACTS["metadata"].items():
            if isinstance(v, dict):  # long arrays/strings: lengths checked below
                continue
            if isinstance(v, list):
                assert [str(x) for x in f.kv[k]] == [str(x) for x in v], k
            else:
                assert f.kv[k] == v, k
        assert f.arch == FACTS["architecture"]
        assert len(f.kv["tokenizer.ggml.tokens"]) == FACTS["metadata"]["tokenizer.ggml.tokens"]["_list_len"]
        assert len(f.kv["tokenizer.chat_template"]) == FACTS["metadata"]["tokenizer.chat_template"]["_string_len"]
    finally:
        f.close()


@needs_model
def test_tensors_match_facts():
    f = GGUFFile(MODEL)
    try:
        counts = {}
        for t in f.tensors.values():
            counts[t.type] = counts.get(t.type, 0) + 1
        assert len(f.tensors) == FACTS["n_tensors"]
        assert counts == FACTS["tensor_types"]
        assert sum(t.nbytes for t in f.tensors.values()) / 2**30 == pytest.approx(FACTS["tensor_bytes_gib"], abs=0.01)
        for name, info in FACTS["global_tensors"].items():
            t = f.tensors[name]
            assert tuple(t.shape) == tuple(info["shape"]), name
            assert t.type == info["type"], name
        for role, row in FACTS["per_layer_types"].items():
            for i, typ in enumerate(row):
                t = f.tensors.get(f"blk.{i}.{role}")
                assert (t.type if t else None) == typ, (i, role)
    finally:
        f.close()
