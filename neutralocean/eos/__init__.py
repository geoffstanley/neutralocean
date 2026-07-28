import importlib as _importlib
from .tools import load_eos, make_bsq, vectorize_eos
from .tools import make_eos, make_eos_s_t, make_eos_p  # to be removed

_modules = ["gsw", "gswc", "jmd95", "jmdfwg06", "polyTEOS10bsq", "tools"]

# all local, public functions
__all__ = _modules + [k for (k, v) in locals().items() if callable(v) and not k.startswith("_")]


def __dir__():
    return __all__


# Lazy load of submodules
def __getattr__(name):
    if name in _modules:
        return _importlib.import_module(f"neutralocean.eos.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'neutralocean.eos' has no attribute '{name}'")
