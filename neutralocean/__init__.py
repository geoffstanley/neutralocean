__version__ = "2.4.0"

import importlib as _importlib

# Import from subpackages
from .eos import load_eos, make_bsq, vectorize_eos
from .eos import make_eos, make_eos_s_t, make_eos_p  # to be removed
from .grid import divergence
from .ppinterp import make_pp  # don't import ppval*, valid_range*, ...

# Import from modules
from .bfs import *
from .label import *
from .mixed_layer import *
from .ntp import *
from .stability import *
from .surface import *
from .traj import *

# List of modules not explicitly imported above
modules = ["data", "fzero", "lib", "_vertsolve"]

__all__ = modules + [
    k for (k, v) in locals().items() if not k.startswith("_")
]  # all local, public functions


def __dir__():
    return __all__


# Lazy load of modules.
# Note only `data.py` is lazily loaded; all others get loaded implicitly by the above
# imports.
def __getattr__(name):
    if name in modules:
        return _importlib.import_module(f"neutralocean.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'neutralocean' has no attribute '{name}'")
