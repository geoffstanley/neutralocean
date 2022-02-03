"""Core functions to perform interpolation by an arbitrary interpolation kernel
(such as in `.linear` or `.pchip`) in one dimension, with a variety of functions 
to facilitate interpolating with one or two kernels, one or two dependent data
sets, and/or to datasets that are 1D or multi-dimensional.  

The functions in this module are not intended to be called directly (though they
can be).  Rather, they are used by factory functions that return a new function
with the interpolation kernel (the input `f`) enclosed, thereby accelerating the
code.  See `.tools/make_interpolator`.

This subpackage is useful for interpolation when the evaluation site is known,
so that only one or a few interpolations are needed.  However, when 
interpolation must be repeated many times, such as when solving a nonlinear
equation involving the interpolants, the `ppinterp` subpackage is preferred, as 
that subpackage pre-computes the piecewise polynomial coefficients once, and then
interpolation to a given evaluation site is fast.

Note: In theory, many of these functions could be collapsed down to a single 
function.  For example, `_interp_1` and `_interp_1_YZ` could be replaced by
a single function that accepts a tuple as its fourth `Y` parameter; then it 
works like `_interp_1_YZ` over each element of `Y`.  However, as of Numba v0.53,
that approach is considerably *slower* than the approach taken here.
"""

import numpy as np
import numba as nb


@nb.njit
def _interp_1(f, x, X, Y):
    """
    Apply a given kernel of interpolation once.

    Parameters
    ----------
    f : function

        The "kernel" of interpolation: A function that performs a single
        interpolation within a known interval.

        The parameters to `f` are
            x : float
            X : ndarray(float, 1d)
            Y : ndarray(float, 1d)
            i : int
        and the return value of `f` is
            y : float
        which is `Y` as a function of `X` interpolated to the value `x`. Here,
        the subinterval of `X` within which `x` falls is given to this
        function by `i`.  This function will only be called when the
        following is guaranteed:
            (a) `i == 1`  and  `X[0] <= x <= X[1]`, or
            (b) `2 <= i <= len(X) - 1`  and  X[i-1] < x <= X[i]`.

    x : float
        Evaluation site

    X : ndarray(float, 1d)
        The independent data.
        Must be monotonically increasing.
        NaN is treated as +inf, hence NaN's must go after any valid data.
        If X[0] is NaN, it is assumed that all elements of X are NaN.

    Y : ndarray(float, 1d)
        The dependent data, with the same length as `X`.

    Returns
    -------
    y : float
        The value of `Y` interpolated to `X` at `x`.
    """
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan

    # i = searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0] or all(isnan(X))
    #   i = len(X)              if X[-1] < x or isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    #
    # Having guaranteed above that
    # x is not nan, and X[0] is not nan hence not all(isnan(X)),
    # and X[0] <= x <= X[-1],
    # then either
    # (a) i == 0  and x == X[0], or
    # (b) 1 <= i <= len(X)-1  and  X[i-1] < x <= X[i]
    #
    # Next, merge (a) and (b) cases so that
    #   1 <= i <= len(X) - 1
    # is guaranteed, and
    #   X[0 ] <= x <= X[1]  when  i == 1
    #   X[i-1] < x <= X[i]  when  i > 1
    i = max(1, np.searchsorted(X, x))

    # Interpolate within the given interval
    return f(x, X, Y, i)


@nb.njit
def _interp_1_fg(f, g, x, X, Y):
    """As _interp_1 but applies two interpolation kernels."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan
    i = max(1, np.searchsorted(X, x))
    return f(x, X, Y, i), g(x, X, Y, i)


@nb.njit
def _interp_1_YZ(f, x, X, Y, Z):
    """As _interp_1 but applies the interpolation kernel to two dependent data arrays."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan
    i = max(1, np.searchsorted(X, x))
    return f(x, X, Y, i), f(x, X, Z, i)


@nb.njit
def _interp_1_fg_YZ(f, g, x, X, Y, Z):
    """As _interp_1 but applies two interpolation kernels to two dependent data arrays.

    Parameters
    ----------
    f, g : function
        As in `_interp_1`

    x, X : see `_interp_1`

    Y, Z : ndarray(float, 1d)
        As in `_interp_1`

    Returns
    -------
    yf : float
        The value of `Y` interpolated using `f` to `X` at `x`.

    zf : float
        The value of `Z` interpolated using `f` to `X` at `x`.

    yg: float
        The value of `Y` interpolated using `g` to `X` at `x`.

    zg : float
        The value of `Z` interpolated using `g` to `X` at `x`.

    """
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan, np.nan, np.nan
    i = max(1, np.searchsorted(X, x))
    return f(x, X, Y, i), f(x, X, Z, i), g(x, X, Y, i), g(x, X, Z, i)


@nb.njit
def _interp_n(f, x, X, Y):
    """
    As _interp_1 but applies interpolation kernel many times.

    Parameters
    ----------
    f, x, X : see `_interp_1`

    Y: ndarray(float, nd)

        Dependent data.  The last dimension must be the same length as `X`.

    Returns
    -------
    y : ndarray

        The value `y[n]` is `Y[n]` (a 1D array) interpolated to `X` at `x`.
        The shape of `y` is the shape of `Y` less its last dimension.

    Note
    ----
    This function is faster than a `numba.guvectorize`'d version of `_interp_1`
    because `numpy.searchsorted` is only called once, here.
    """
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        y = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
    else:
        y = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        i = max(1, np.searchsorted(X, x))
        for n in np.ndindex(y.shape):
            y[n] = f(x, X, Y[n], i)

    return y


@nb.njit
def _interp_n_YZ(f, x, X, Y, Z):
    """As _interp_n but applies the interpolation kernel to two dependent data ndarrays.
    Assumes Y and Z are the same shape.
    """
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        y = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
        z = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
    else:
        y = np.empty(Y.shape[0:-1], dtype=np.f8)
        z = np.empty(Y.shape[0:-1], dtype=np.f8)
        i = max(1, np.searchsorted(X, x))
        for n in np.ndindex(y.shape):
            y[n] = f(x, X, Y[n], i)
            z[n] = f(x, X, Z[n], i)

    return y, z


@nb.njit
def _interp_n_fg(f, g, x, X, Y):
    """As _interp_n but applies two interpolation kernels."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        yf = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
        yg = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
    else:
        yf = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        yg = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        i = max(1, np.searchsorted(X, x))
        for n in np.ndindex(yf.shape):
            yf[n] = f(x, X, Y[n], i)
            yg[n] = g(x, X, Y[n], i)

    return yf, yg


@nb.njit
def _interp_n_fg_YZ(f, g, x, X, Y, Z):
    """As _interp_n but applies two interpolation kernels to two dependent data ndarrays."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        yf = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
        zf = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
        yg = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
        zg = np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
    else:
        yf = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        zf = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        yg = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        zg = np.empty(Y.shape[0:-1], dtype=Y.dtype)
        i = max(1, np.searchsorted(X, x))
        for n in np.ndindex(yf.shape):
            yf[n] = f(x, X, Y[n], i)
            zf[n] = f(x, X, Z[n], i)
            yg[n] = g(x, X, Y[n], i)
            zg[n] = g(x, X, Z[n], i)

    return yf, zf, yg, zg


# @ft.lru_cache(maxsize=10)
# def make_1_m(ker):
#     @nb.njit
#     def f(x, X, Y):
#         return interp_1_many(ker, x, X, Y)
#         # return interp_1_nd(ker, x, X, Y)

#     return f


# @ft.lru_cache(maxsize=10)
# def make_n_m(ker):
#     @nb.njit
#     def f(x, X, Y):
#         return interp_n_many(ker, x, X, Y)

#     return f


# # Like interp1_nd but Y is a tuple of things like Y from interp1_nd.
# @nb.njit
# def interp_n_many(f, x, X, Y):

#     lenY = len(Y)

#     if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
#         y = [np.full(Y1.shape[0:-1], np.nan, dtype=np.f8) for Y1 in Y]
#     else:

#         y = [np.empty(Y1.shape[0:-1], dtype=np.f8) for Y1 in Y]
#         # Yn = [np.empty(X.shape[-1], dtype=np.f8) for Y1 in Y]
#         # y = np.empty((*x.shape, lenY), dtype=np.f8)
#         # Yn = np.empty((lenY, X.shape[-1]))

#         i = max(1, np.searchsorted(X, x))

#         for n in np.ndindex(y[0].shape):
#             # for i in range(lenY):
#             #    Yn[:, i] = Y[i][n]
#             # Yn = [Y_[n] for Y_ in Y]
#             # Yn = tuple(Y_[n] for Y_ in Y)
#             # yn = interp_1_many(f, x[n], X[n], Yn)
#             # for i in range(lenY):
#             #     y[i][n] = yn[i]
#             # y[n, ...] = interp_1_nd(f, x[n], X, Yn)

#             for m in range(lenY):
#                 y[m][n] = f(x, X, Y[m][n], i)
#                 # y[(*n, m)] = f(x[n], X, Y[m][n], i)

#     return y


# @nb.njit
# def interp_1_nd(f, x, X, Y):
#     if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
#         return np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)
#     i = max(1, np.searchsorted(X, x))
#     y = np.empty(Y.shape[0:-1], dtype=Y.dtype)
#     for n in np.ndindex(y.shape):
#         y[n] = f(x, X, Y[n], i)
#     return y


# @nb.njit
# def interp_1_many(f, x, X, Y):
#     if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
#         return [np.nan for _ in Y]
#     i = max(1, np.searchsorted(X, x))
#     return [f(x, X, Y_, i) for Y_ in Y]


# @nb.njit
# def interp_n_many(x, X, Y, f):

#     yshape = Y[0].shape[0:-1]
#     # Y0 = Y[0]
#     # Y1 = Y[1]
#     y = [np.empty(yshape, dtype=np.f8) for Y_ in Y]
#     xn = x
#     Xn = X
#     # for n in np.ndindex(x.shape):
#     for n in np.ndindex(yshape):
#         # yn = interp_1_YZ(x[n], X[n], Y0[n], Y1[n], f)
#         # yn = interp_1_many(x[n], X[n], Y0[n], Y1[n])
#         # xn = x[n]
#         # Xn = X[n]
#         if np.isnan(xn) or xn < Xn[0] or Xn[-1] < xn or np.isnan(Xn[0]):
#             # if np.isnan(xn) or xn < Xn[0] or Xn[-1] < xn or np.isnan(Xn[0]):
#             # if (
#             #     np.isnan(x[n])
#             #     or x[n] < X[(*n, 0)]
#             #     or X[(*n, -1)] < x[n]
#             #     or np.isnan(X([*n, 0]))
#             # ):
#             # return [np.nan for Y_ in Y]
#             # yn = [np.nan for Y_ in Y]
#             for i in range(2):
#                 y[i][n] = np.nan
#         else:
#             k = max(1, np.searchsorted(Xn, xn))
#             # yn = [ker(xn, Xn, Y_, k) for Y_ in Y]
#             for i in range(2):
#                 # y[i][n] = yn[i]
#                 y[i][n] = f(xn, Xn, Y[i], k)

#     return y


# # 1.65 ms   vs 1.77 ms for interp_1_vec called twice
# @nb.guvectorize(
#     [
#         (
#             f8,
#             f8[:],
#             f8[:],
#             f8[:],
#             int64,
#             f8[:],
#             f8[:],
#         )
#     ],
#     "(),(n),(n),(n),()->(),()",
# )
# def interp_1_vec_YZ(x, X, Y, Z, d, y, z):
#     if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
#         y[0] = np.nan
#         z[0] = np.nan
#     else:
#         i = max(1, np.searchsorted(X, x))
#         if d == 0:
#             y[0] = ker(x, X, Y, i)
#             z[0] = ker(x, X, Z, i)
#         elif d == 1:
#             y[0] = ker1(x, X, Y, i)
#             z[0] = ker1(x, X, Z, i)
#         else:
#             y[0] = 0.0
#             z[0] = 0.0
