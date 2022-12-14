"""
Piecewise polynomial interpolation in one dimension.

This package separates 1D interpolation into two steps: 
    1. Compute coefficients for a piecewise polynomial interpolant,
    2. Evaluate the interpolant.

If the interpolant will be evaluated many times, then it is fastest to do Step
1 once, then repeat Step 2 as many times as needed. To do this, build the
coefficients using `*coeffs*` methods, such as `linear_coeffs` and
`pchip_coeffs`. Then evaluate the interpolants using `ppval` methods.

If the interpolant will only be evaluated a small number of times, then it may
be faster to combine Steps 1 and 2, building an interpolant using only as
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

If memory is limited, then it may best to avoid pre-computing the
coefficients (Step 1) for the entire dataset. If many interpolations are
needed (e.g. if interpolating to many evaluation sites, or if each 1D problem
requires solving a non-linear equation), then one can do Step 1 for a single
1D interpolation problem and then do Step 2 many times, before moving on to
the next 1D problem. To achieve this, one can use a `numba.njit` accelerated
for loop, e.g. as done in `neutralocean.surface._vertsolve`. 

If performing a single 1D interpolation problem and all input data is finite
(no NaN's), the `"_nonan"` variant builds coefficients with a small speed
advantage. 

If there are two dependent variables that share the same dependent variable,
the `"_two"` variants offer a speed advantage over calling the basic
functions twice, since the `"_two"` variants do the work of locating where
the evaluation site `x` lies within the dependent data `X` only once. This
could easily be extended to handle more dependent variables, of course.

The `make_pp` factory function will help select the desired variant of the
`*coeffs*` and `*interp*` functions.
"""

from .ppinterp import ppval_1, ppval, ppval_1_two, ppval_two, ppval_i
from .lib import valid_range_1, valid_range_1_two, valid_range
from .tools import make_pp
