"""
Piecewise polynomial interpolation in one dimension.

This package separates 1D interpolation into two steps: 
    1. Compute coefficients for a piecewise polynomial interpolant,
    2. Evaluate the interpolant.

If the interpolant will be evaluated many times, then it is fastest to do Step
1 once, then repeat Step 2 as many times as needed. To do this, build the
coefficients using `*coeffs*` methods, such as `linear_coeffs` and
`pchip_coeffs`. Then evaluate the interpolants using `ppval` methods.

If the interpolant will only be evaluated once or a few times, then it will
be faster to combine Steps 1 and 2, building the polynomial (not a piecewise
polynomial) for just the interval containing the evaluation site, using only as
much data around the evaluation site as is required. To do this, use the
`*interp*` methods, such as `linear_interp` and `pchip_interp`. 

The core numerical methods handle just one interpolation problem; that is, the
evaluation site (`x`) is a scalar and the input independent and dependent
data (`X` and `Y`) are 1D arrays. These functions are named with `"_1"`.

Functions that lack this `"_1"` are "universal", accepting an N-dimensional
array for `x` and (N+1)-dimensional arrays for  `X`, and `Y`, so long as they
are mutually broadcastable from the basic 1D problem, thus allowing multiple
1D interpolation problems to be solved with one function call. Effectively, a
low-level for loop is used to wrap the single 1D problem. Operationally, this
is achieved using `numba`'s `guvectorize` decorator.

If the interpolant must be evaluated many times but the dataset is large and 
memory is limited, then the best strategy is to compute the piecewise 
polynomial coefficients for a single 1D interpolation problem (Step 1), then 
evaluate it (Step 2) as many times as needed. This can be achieved by a
`numba.njit` accelerated for loop over each 1D problem, using the `"_1"` 
function variants. For example, `neutralocean.surface._vertsolve` does this
to solve many individual 1D non-linear root finding problems. 

If performing a single 1D interpolation problem and all input data is finite
(no NaN's), the `"_nonan"` variants provide a small speed advantage. 

If there are two dependent variables that share the same dependent variable,
the `"_two"` variants offer a speed advantage over calling the basic
functions twice, since the `"_two"` variants do the work of locating where
the evaluation site `x` lies within the dependent data `X` only once. This
could easily be extended to handle more dependent variables, of course.

The `make_pp` factory function will help select the desired variant of the
`*coeffs*` and `*interp*` functions.
"""

import importlib as _importlib
from .ppval import (
    pval,
    ppval_1,
    ppval_1_two,
    ppval,
    ppval_two,
    ppval_1_nonan,
    ppval_1_nonan_two,
)
from .lib import valid_range_1, valid_range_1_two, valid_range
from .tools import make_pp

modules = ["linear", "pchip", "pplib", "ppval", "tools"]

__all__ = modules + [
    k for (k, v) in locals().items() if callable(v) and not k.startswith("_")
]  # all local, public functions


def __dir__():
    return __all__


# Lazy load of submodules
def __getattr__(name):
    if name in modules:
        return _importlib.import_module(f"neutralocean.ppinterp.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'neutralocean.ppinterp' has no attribute '{name}'"
            )
