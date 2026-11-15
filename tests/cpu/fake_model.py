"""A fake model and tokenizer that satisfy contract C3 on the CPU.

The logits are a deterministic function of the context: the greedy token is
`(sum(context) % 12) + 1`, in 1..12, never the (fake) eog 0. Tests predict the
whole greedy stream and check prefix reuse, stop handling and token counts.
`reset()` keeps `forward_sizes` so tests can still see the call history."""
import torch


class FakeTok:
    bos_id, eos_id = 2, 0
    eog_ids = {0}

    def token_bytes(self, tid: int, special: bool = True) -> bytes:
        return str(tid).encode()

    def decode(self, ids, special: bool = False) -> str:
        return " ".join(str(i) for i in ids)


class FakeModel:
    n_vocab = 16

    def __init__(self):
        self.cfg = {}
        self.tp = None
        self.max_ctx = 4096
        self.n_past = 0
        self.load_seconds = 0.0
        self.fast_decode = False
        self.use_graph = False
        self.forward_sizes: list[int] = []    # tokens fed, one entry per forward
        self.ctx: list[int] = []              # the ids the model consumed

    def forward(self, tokens, chunk: int = 512, all_logits: bool = False) -> torch.Tensor:
        self.forward_sizes.append(len(tokens))
        self.ctx.extend(tokens)
        self.n_past = len(self.ctx)
        return self.logits_for(self.ctx)

    def decode(self, token: int) -> torch.Tensor:
        return self.forward([token])

    def reset(self) -> None:
        self.ctx = []
        self.n_past = 0

    def kv_bytes(self) -> int:
        return 0

    def memory_report(self) -> dict:
        return {}

    @staticmethod
    def logits_for(ctx) -> torch.Tensor:
        target = (sum(ctx) % 12) + 1
        logits = torch.zeros(FakeModel.n_vocab, dtype=torch.float32)
        logits[target] = 2.0
        return logits


GREEDY = {"temp": 0.0, "top_k": 40, "top_p": 0.95, "min_p": 0.0, "seed": 1234}
