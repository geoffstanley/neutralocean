"""
Functions for the particular usage of interpolation required by neutral
surface calculations.
"""

import numpy as np
import numba
from numba import float64, intc


def linear_coefficients(p, y):
    return np.diff(y, axis=-1) / np.diff(p, axis=-1)


# We start with a simple linear search; it's not the bottleneck in what
# we want to do.  We can swap in a binary search later for a small gain.
@numba.njit(intc(float64[:], float64))
def search(p, ptarget):
    if p[0] > ptarget:
        return -1
    for ii in range(1, len(p)):
        if p[ii] > ptarget:
            return ii - 1
    if p[ii] == ptarget:
        return ii - 1
    return -1


@numba.njit(
    float64(
        float64[:], float64[:], float64[:], float64
    )
)
def val(X, Y, Yppc, x):
    ii = search(X, x)
    if ii == -1:
        return np.nan
    dx = x - X[ii]  # dx >= 0
    y = Y[ii] + dx * Yppc[ii]
    return y

@numba.njit(
    numba.typeof((1.0, 1.0))(
        float64[:], float64[:], float64[:], float64[:], float64[:], float64
    )
)
def val2(X, Y, Yppc, Z, Zppc, x):
    ii = search(X, x)
    if ii == -1:
        return (np.nan, np.nan)
    dx = x - X[ii]  # dx >= 0
    y = Y[ii] + dx * Yppc[ii]
    z = Z[ii] + dx * Zppc[ii]
    return (y, z)
