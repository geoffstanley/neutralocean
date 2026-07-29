import importlib as _importlib
from .tools import divergence

_modules = ["graph", "rectilinear", "tools", "tripolar", "xgcm"]

# all local, public functions
# __all__ = _modules + [k for (k, v) in locals().items() if callable(v) and not k.startswith("_")]
__all__ = (
    # Submodules
    "graph", # pyright: ignore[reportUnsupportedDunderAll]
    "rectilinear", # pyright: ignore[reportUnsupportedDunderAll]
    "tripolar", # pyright: ignore[reportUnsupportedDunderAll]
    "xgcm", # pyright: ignore[reportUnsupportedDunderAll]
    "tools",
    # Top level functions
    "divergence",
)


def __dir__():
    return __all__


# Lazy load of submodules
def __getattr__(name):
    if name in _modules:
        return _importlib.import_module(f"neutralocean.grid.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Package 'neutralocean.grid' has no attribute '{name}'")
