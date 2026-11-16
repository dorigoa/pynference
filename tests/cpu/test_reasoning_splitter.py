"""CPU tests for `chat.ReasoningSplitter` (S14 step 6) on synthetic streams.

The tags of the two families are spelled with a leading escape so no chat
marker appears literally in this source (plan 2.1): they come from the
module constants anyway. Streams are fed in small chunks, and one test
pushes one char at a time, so every tag is split across `push` boundaries."""
import pytest

from pynference.chat import (
    HARMONY_CHANNEL,
    HARMONY_CLOSE,
    HARMONY_MESSAGE,
    HARMONY_START,
    QWEN_THINK_END,
    ReasoningSplitter,
)


def stream(sp, text, chunk=1):
    r, c = "", ""
    for i in range(0, len(text), chunk):
        a, b = sp.push(text[i:i + chunk])
        r += a
        c += b
    a, b = sp.close()
    return r + a, c + b


def stream_whole(sp, text):
    r1, c1 = sp.push(text)
    r2, c2 = sp.close()
    return r1 + r2, c1 + c2


class TestQwen:
    def test_reasoning_then_answer(self):
        text = "deep thought" + QWEN_THINK_END + "\n\nfinal answer"
        assert stream_whole(ReasoningSplitter("qwen"), text) == (
            "deep thought", "\n\nfinal answer")
        assert stream(ReasoningSplitter("qwen"), text, chunk=3) == (
            "deep thought", "\n\nfinal answer")

    def test_tag_split_one_char_at_a_time(self):
        sp = ReasoningSplitter("qwen")
        assert stream(sp, "a" + QWEN_THINK_END + "b", chunk=1) == ("a", "b")

    def test_stream_cut_mid_tag(self):
        # a stream ending in a partial tag: the held tail is reasoning, not lost
        sp = ReasoningSplitter("qwen")
        assert stream(sp, "think</thi", chunk=2) == ("think</thi", "")

    def test_no_tag_all_reasoning(self):
        # thinking cut by the token cap: no closing tag, all reasoning
        sp = ReasoningSplitter("qwen")
        assert stream_whole(sp, "mid reasoning") == ("mid reasoning", "")

    def test_answer_part_after_tag_streams_out(self):
        sp = ReasoningSplitter("qwen")
        r, c = sp.push("think" + QWEN_THINK_END)
        assert (r, c) == ("think", "")
        r, c = sp.push("now ")
        assert (r, c) == ("", "now ")     # content opens right after the tag
        r, c = sp.push("the answer")
        assert (r, c) == ("", "the answer")


class TestGptOss:
    ANSWER = (HARMONY_CHANNEL + "analysis" + HARMONY_MESSAGE + "let me see"
              + HARMONY_CLOSE + HARMONY_START + "assistant"
              + HARMONY_CHANNEL + "final" + HARMONY_MESSAGE + "Rome."
              + HARMONY_CLOSE)

    def test_analysis_vs_final(self):
        for chunk in (1, 4, len(self.ANSWER)):
            sp = ReasoningSplitter("gpt-oss")
            assert stream(sp, self.ANSWER, chunk=chunk) == ("let me see", "Rome.")

    def test_constrain_and_target_in_descriptor(self):
        text = (HARMONY_CHANNEL + "commentary to=fn" + "\x3c|constrain|>json"
                + HARMONY_MESSAGE + '{"x": 1}' + "\x3c|call|>")
        sp = ReasoningSplitter("gpt-oss")
        assert stream_whole(sp, text) == ("", '{"x": 1}')

    def test_channel_cut_while_reasoning(self):
        # -n cut inside the analysis channel: reasoning flushed, no content
        text = HARMONY_CHANNEL + "analys" + HARMONY_MESSAGE[1:] + "deep"
        # second piece re-forms `<|message|>` across the chunk boundary
        sp = ReasoningSplitter("gpt-oss")
        r, c = sp.push(HARMONY_CHANNEL + "analysis" + "\x3c|me")
        assert (r, c) == ("", "")
        r, c = sp.push("ssage|>deep")
        assert (r, c) == ("deep", "")
        r, c = sp.close()
        assert (r, c) == ("", "")
        assert stream(sp, text, chunk=1)  # smoke: no crash on junk tail

    def test_no_channels_falls_back_to_content(self):
        sp = ReasoningSplitter("gpt-oss")
        assert stream(sp, "plain answer", chunk=5) == ("", "plain answer")

    def test_start_noise_between_channels_not_in_content(self):
        text = (HARMONY_CHANNEL + "analysis" + HARMONY_MESSAGE + "r"
                + HARMONY_START + "assistant"          # channel never closed
                + HARMONY_CHANNEL + "final" + HARMONY_MESSAGE + "a"
                + "\x3c|return|>")
        sp = ReasoningSplitter("gpt-oss")
        assert stream_whole(sp, text) == ("r", "a")


class TestFactories:
    class _Tok:
        def __init__(self, pre):
            if pre is not None:
                self.pre = pre

    def test_from_tokenizer_families(self):
        assert ReasoningSplitter.from_tokenizer(self._Tok("gpt-4o")).family == "gpt-oss"
        assert ReasoningSplitter.from_tokenizer(self._Tok("qwen35")).family == "qwen"

    def test_from_tokenizer_no_splitter(self):
        # Gemma's Tokenizer has no `pre`; an unknown one has no splitter
        assert ReasoningSplitter.from_tokenizer(self._Tok(None)) is None
        assert ReasoningSplitter.from_tokenizer(self._Tok("other")) is None

    def test_unknown_family(self):
        with pytest.raises(ValueError):
            ReasoningSplitter("gemma4")

    def test_split_all_matches_incremental(self):
        text = (HARMONY_CHANNEL + "analysis" + HARMONY_MESSAGE + "why?"
                + HARMONY_CLOSE + HARMONY_START + "assistant"
                + HARMONY_CHANNEL + "final" + HARMONY_MESSAGE + "yes")
        sp = ReasoningSplitter("gpt-oss")
        assert sp.split_all(text) == ("why?", "yes")
