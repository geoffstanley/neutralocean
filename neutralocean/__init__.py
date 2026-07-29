__version__ = "2.4.1"

import importlib as _importlib

# Import subpackages
# from . import eos, grid, ppinterp

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
_modules = ["data", "fzero", "lib", "_vertsolve"]

# all local, public functions
# __all__ = _modules + [k for (k, v) in locals().items() if not k.startswith("_")]
__all__ = (
    # Sub-packages
    "eos",
    "grid",
    "ppinterp",
    # Top-level functions
    # -- eos
    "load_eos",
    "make_bsq",
    "vectorize_eos",
    "make_eos",
    "make_eos_s_t",
    "make_eos_p",
    # -- grid
    "divergence",
    # -- ppinterp
    "make_pp",
    # -- bfs
    "bfs_conncomp1",
    "bfs_conncomp1_wet",
    "bfs_conncomp1_wet_perim",
    # -- label
    "veronis",
    "pot_dens_1",
    # -- mixed_layer
    "mld",
    # -- ntp
    "ntp_epsilon_errors",
    "ntp_epsilon_errors_norms",
    # -- stability
    "count_unstable",
    "stabilize_ST",
    "calc_dLRPDdp_fd_1",
    # -- surface
    "potential_surf",
    "anomaly_surf",
    "omega_surf",
    # -- traj
    "ntp_bottle_to_cast",
    "neutral_trajectory",
    # Classes
    # Exceptions
    # Constants
    "__version__",
)


def __dir__():
    return __all__


# Lazy load of modules.
# Note only `data.py` is lazily loaded; all others get loaded implicitly by the above
# imports.
def __getattr__(name):
    if name in _modules:
        return _importlib.import_module(f"neutralocean.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Package 'neutralocean' has no attribute '{name}'")
