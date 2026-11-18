"""Model architectures, selected by the GGUF `general.architecture` key."""
from ..gguf import GGUFFile
from ..parallel import TP


def load_model(g: GGUFFile, tp: TP, **kw):
    if g.arch == "gemma4":
        from .gemma4 import Gemma4
        return Gemma4(g, tp, **kw)
    if g.arch == "gpt-oss":
        from .gptoss import GPTOss
        return GPTOss(g, tp, **kw)
    if g.arch == "qwen35":
        from .qwen35 import Qwen35
        return Qwen35(g, tp, **kw)
    raise NotImplementedError(f"architecture {g.arch!r} not supported yet")
