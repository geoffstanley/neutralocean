"""
Functions for the particular usage of interpolation required by neutral
surface calculations.
"""

import numpy as np
import numba
from numba import float64, intc


def linear_coefficients(X, Y):
    return np.diff(Y, axis=-1) / np.diff(X, axis=-1)


# @numba.njit(float64(float64[:], float64[:], float64[:], float64))
@numba.njit
def val_0d(X, Y, Yppc, x):

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


@numba.njit
def val_nd(X, Y, Yppc, x):
    y = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n] = val_0d(X[n], Y[n], Yppc[n], x[n])
    return y


def val(X, Y, Yppc, x):
    nk = X.shape[-1]
    # assert nk == Y.shape[-1], "X and Y must have same size last dimension"
    # assert Y.shape[0:-1] == Yppc.shape[0:-1], "Y and Yppc must have same leading dimensions"
    # assert Yppc.shape[-1] == nk - 1, "Last dimension of Yppc must be one less than that of Y"
    if X.ndim > 1:
        shape = X.shape[0:-1]
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match those of X"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif Y.ndim > 1:
        shape = Y.shape[0:-1]
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match those of Y"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    else:
        shape = x.shape
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match the shape of x"
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match the shape of x"

    X = np.broadcast_to(X, (*shape, nk))
    Y = np.broadcast_to(Y, (*shape, nk))
    Yppc = np.broadcast_to(Yppc, (*shape, nk - 1))
    x = np.broadcast_to(x, shape)

    y = np.empty(shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n] = val_0d(X[n], Y[n], Yppc[n], x[n])
    return y


# @numba.njit(
#     numba.typeof((1.0, 1.0))(
#         float64[:], float64[:], float64[:], float64[:], float64[:], float64
#     )
# )
@numba.njit
def val2_0d(X, Y, Yppc, Z, Zppc, x):

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


@numba.njit
def val2_nd(X, Y, Yppc, Z, Zppc, x):
    y = np.empty(x.shape, dtype=np.float64)
    z = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n], z[n] = val2_0d(X[n], Y[n], Yppc[n], Z[n], Zppc[n], x[n])
    return y, z


def val2(X, Y, Yppc, Z, Zppc, x):
    nk = X.shape[-1]
    # assert nk == Y.shape[-1], "X and Y must have same size last dimension"
    # assert Y.shape[0:-1] == Yppc.shape[0:-1], "Y and Yppc must have same leading dimensions"
    # assert Yppc.shape[-1] == nk - 1, "Last dimension of Yppc must be one less than that of Y"
    # assert Z.shape == Y.shape, "Y and Z must have the same shape"
    # assert Zppc.shape == Yppc.shape, "Yppc and Zppc must have the same shape"
    if X.ndim > 1:
        shape = X.shape[0:-1]
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match those of X"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif Y.ndim > 1:
        shape = Y.shape[0:-1]
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match those of Y"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    else:
        shape = x.shape
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match the shape of x"
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match the shape of x"

    X = np.broadcast_to(X, (*shape, nk))
    Y = np.broadcast_to(Y, (*shape, nk))
    Z = np.broadcast_to(Z, (*shape, nk))
    Yppc = np.broadcast_to(Yppc, (*shape, nk - 1))
    Zppc = np.broadcast_to(Zppc, (*shape, nk - 1))
    x = np.broadcast_to(x, shape)

    return val2_nd(X, Y, Yppc, Z, Zppc, x)
