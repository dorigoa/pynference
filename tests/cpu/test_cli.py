"""CPU tests for the CLI chat continuation (`continuation_ids`) against the
real Gemma-4 template and tokenizer (skipped if the model file is absent).

The Gemma template strips thought channels from past assistant messages and
never replays their reasoning, so a re-rendered multi-turn prompt can neither
extend a thinking answer nor keep it: the CLI continues at token level
instead. These tests pin exactly what `continuation_ids` appends."""
import os

import pytest

from pynference.chat import render
from pynference.cli import continuation_ids
from pynference.gguf import GGUFFile
from pynference.tokenizer import Tokenizer

MODEL = "/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf"
TK = {"enable_thinking": True}

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")


@pytest.fixture(scope="module")
def gt():
    g = GGUFFile(MODEL)
    return g, Tokenizer(g)


def user_block(g, turn: str) -> str:
    """The template text for a lone user turn, without the shared head."""
    head = render(g, [], add_generation_prompt=False, **TK)
    block = render(g, [{"role": "user", "content": turn}], add_generation_prompt=True, **TK)
    assert block.startswith(head) and len(block) > len(head)
    return block[len(head):]


def test_truncated_thinking_answer_closes_the_channel(gt):
    g, tok = gt
    turn = "E ieri? Una frase."
    ids = continuation_ids(g, tok, TK, turn, "<|channel>thought\nmi sono...",
                           last_stop=None, last_truncated=True)
    assert ids is not None
    assert tok.decode(ids, special=True) == "\n<channel|><turn|>\n" + user_block(g, turn)


def test_truncated_answer_with_a_closed_channel(gt):
    g, tok = gt
    turn = "Grazie, ciao!"
    ids = continuation_ids(g, tok, TK, turn, "<|channel>thought\nt\n<channel|>Prego!",
                           last_stop=None, last_truncated=True)
    assert ids is not None
    assert tok.decode(ids, special=True) == "<turn|>\n" + user_block(g, turn)


def test_answer_closed_by_turn_stop_id_replays_the_stop(gt):
    g, tok = gt
    stop = tok.token_to_id["<turn|>"]
    turn = "Seconda."
    ids = continuation_ids(g, tok, TK, turn, "Fatto.", last_stop=stop, last_truncated=False)
    assert ids is not None
    assert ids[0] == stop                                   # the stop id belongs to the context
    assert tok.decode(ids[1:], special=True) == "\n" + user_block(g, turn)


def test_no_continuation_without_a_filled_context(gt):
    g, tok = gt
    # stopped on a stop string: no stop id, and not truncated
    assert continuation_ids(g, tok, TK, "x", "y", None, False) is None
    # closed by something that does not end a template turn (EOS)
    assert continuation_ids(g, tok, TK, "x", "y", tok.eos_id, False) is None


def test_continuation_text_round_trips_through_a_reused_context(gt):
    g, tok = gt
    first = render(g, [{"role": "user", "content": "Ciao"}], add_generation_prompt=True, **TK)
    answer = "<|channel>thought\npenso"
    ids = tok.encode(first + answer, add_bos=False)
    cont = continuation_ids(g, tok, TK, "E tu?", answer, None, True)
    assert cont is not None
    # the continuation extends the token-exact context: Session.prefill
    # would feed only these ids (prefix test of test_engine on the real tok)
    assert tok.decode(ids + cont, special=True) == first + answer + "\n<channel|><turn|>\n" \
        + user_block(g, "E tu?")
