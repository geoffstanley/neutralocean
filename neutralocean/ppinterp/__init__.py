"""
Functions for interpolation in two steps: 
    1. pre-computing coefficients for a Piecewise Polynomial interpolant,
    2. evaluating the interpolant.

This is used to internally by `_vertsolve`, where an interpolant is 
pre-computed for a single vertical cast and then is evaluated many times to 
solve a non-linear equation. 
Functions are provided to perform this interpolation on entire datasets, not
just a single vertical cast, but this may require a large amount of memory to
store the coefficients; thus, for this purpose `interp1d` is often preferable.
"""
from .ppinterp import ppval1, ppval, ppval1_two, ppval_two, ppval_i
from .tools import select_ppc
