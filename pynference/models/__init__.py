"""Model architectures, selected by the GGUF `general.architecture` key."""
from ..gguf import GGUFFile
from ..parallel import TP


def load_model(g: GGUFFile, tp: TP, **kw):
    if g.arch == "gemma4":
        from .gemma4 import Gemma4
        return Gemma4(g, tp, **kw)
    raise NotImplementedError(f"architecture {g.arch!r} not supported yet")
