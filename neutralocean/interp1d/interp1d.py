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
            
        `x : float`
        
        `X : ndarray(float, 1d)`
        
        `Y : ndarray(float, 1d)`
        
        `i : int`

        and the return value of `f` is
        
        `y : float`

        which is `Y` as a function of `X` interpolated to the value `x`. Here,
        the subinterval of `X` within which `x` falls is given to this
        function by `i`.  This function will only be called when the
        following is guaranteed:
        
        (a) `i == 1`  and  `X[0] <= x <= X[1]`, or

        (b) `2 <= i <= len(X) - 1` and `X[i-1] < x <= X[i]`.

    x : float
        Evaluation site

    X : ndarray(float, 1d)
        The independent data.
        Must be monotonically increasing.
        NaN is treated as `+inf`, hence NaN's must go after any valid data.
        If `X[0]` is NaN, it is assumed that all elements of `X` are NaN.

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

    Notes
    -----
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
