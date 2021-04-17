"""
Functions for the particular usage of interpolation required by neutral
surface calculations.
"""

import numpy as np
import numba
from numba import float64, intc


def linear_coefficients(X, Y):
    return np.diff(Y, axis=-1) / np.diff(X, axis=-1)


@numba.njit(float64(float64[:], float64[:], float64[:], float64))
def val(X, Y, Yppc, x):

    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan

    if x == X[0]:
        return Y[0]

    # i = searchsorted(X,x) is such that:
    #   k = 0                   if x <= X[0]
    #   k = len(X)              if X[-1] < x or np.isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    # Having guaranteed X[0] < x <= X[-1], then
    #   X[i-1] < x <= X[i]  and  1 <= i <= len(X)-1  in all cases,

    # subtract 1 so X[i] < x <= X[i+1]  and  0 <= i <= len(X)-2
    i = np.searchsorted(X, x) - 1

    dx = x - X[i]  # dx > 0 guaranteed
    y = Y[i] + dx * Yppc[i]
    return y


@numba.njit(
    numba.typeof((1.0, 1.0))(
        float64[:], float64[:], float64[:], float64[:], float64[:], float64
    )
)
# def val2(X, Y, Yppc, Z, Zppc, x):
#     ii = search(X, x)
#     if ii == -1:
#         return (np.nan, np.nan)
#     dx = x - X[ii]  # dx >= 0
#     y = Y[ii] + dx * Yppc[ii]
#     z = Z[ii] + dx * Zppc[ii]
#     return (y, z)
def val2(X, Y, Yppc, Z, Zppc, x):

    if np.isnan(x) or x < X[0] or X[-1] < x:
        return (np.nan, np.nan)

    if x == X[0]:
        return (Y[0], Z[0])

    # i = searchsorted(X,x) is such that:
    #   k = 0                   if x <= X[0]
    #   k = len(X)              if X[-1] < x or np.isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    # Having guaranteed X[0] < x <= X[-1], then
    #   X[i-1] < x <= X[i]  and  1 <= i <= len(X)-1  in all cases,

    # subtract 1 so X[i] < x <= X[i+1]  and  0 <= i <= len(X)-2
    i = np.searchsorted(X, x) - 1

    dx = x - X[i]  # dx > 0 guaranteed
    y = Y[i] + dx * Yppc[i]
    z = Z[i] + dx * Zppc[i]
    return (y, z)
