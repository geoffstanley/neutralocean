"""
Functions to evaluate piecewise polynomials, given their coefficients, in one
dimension.
"""


"""
Adapted from 'Piecewise Polynomial Calculus' available on the MATLAB
File Exchange at
https://mathworks.com/matlabcentral/fileexchange/73114-piecewise-polynomial-calculus
"""

import numpy as np
import numba as nb


@nb.njit
def ppval_i(dx, Yppc, i, d=0):
    """
    Evaluate a single interpolant, knowing where the evaluation site lies.

    Provides `i` such that  `X[i] < x <= X[i+1]`
    """

    # Evaluate polynomial, using nested multiplication.
    # E.g. the cubic case is:
    # y = dx^3 * Yppc[i,0] + dx^2 * Yppc[i,1] + dx * Yppc[i,2] + Yppc[i,3]
    #   = dx * (dx * (dx * Yppc[i,0] + Yppc[i,1]) + Yppc[i,2]) + Yppc[i,3]
    y = 0.0
    if d == 0:
        for j in range(0, Yppc.shape[1]):
            y = y * dx + Yppc[i, j]
    else:
        # Evaluate polynomial derivative, using nested multiplication.
        # E.g. the second derivative of the cubic case is:
        # y = 6 * dx * Yppc[i,0] + 2 * Yppc[i,1]
        y = 0.0
        degree = Yppc.shape[1] - 1
        for o in range(0, degree - d + 1):  # o for order
            p = 1.0  # Build integer multiplying the coefficient
            for a in range(degree - o - d + 1, degree - o + 1):
                p *= a
            # p = np.prod(np.arange(degree - o - d + 1, degree - o + 1))  # slow!
            y = y * dx + Yppc[i, o] * p

    return y


@nb.njit
def ppval_1(x, X, Yppc, d=0):
    """
    Evaluate a single piecewise polynomial (PP).

    Parameters
    ----------
    x : float
        Evaluation site
    X : ndarray, 1d
        Independent variable.
    Yppc : ndarray, 2d
        Piecewise Polynomial Coefficients.  First dimension must match `len(X)`.
    d : int, Default 0
        Number of derivatives to take.  If 0, simply evaluate the PP.

    Returns
    -------
    y : float
        The value of the PP (or its `d`'th derivative) at `X = x`.

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
    #   X[0]  <= x <= X[1]  when  i = 1
    #   X[i-1] < x <= X[i]  when  i > 1
    # Then subtract 1 so that
    #   X[0]  <= x <= X[1]   when  i = 0
    #   X[i]  <  x <= X[i+1] when  i > 0
    i = max(0, np.searchsorted(X, x) - 1)

    dx = x - X[i]  # >= 0
    return ppval_i(dx, Yppc, i, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:, :], nb.i8, nb.f8[:])],
    "(),(n),(n,m),()->()",
)
def ppval(x, X, Yppc, d, y):
    """
    Evaluate a given Piecewise Polynomial

    Parameters
    ----------
    x : ndarray

        Sites at which to evaluate the independent variable.

        Must be broadcastable to the shape of `X` or `Y` less their final
        dimension.

    X : ndarray

        Independent variable. Must be broadcastable to the shape of `Y`.

        `X` must be monotonically increasing in its last dimension.
        That is, `X[*i,:]` must be monotonically increasing for any
        `i` tuple indexing all but the last dimension.  NaN's in `X`
        are treated as +Inf.

    Y : ndarray

        Dependent variable.  Must be broadcastable to the shape of `X`.

    Yppc : ndarray

        Piecewise Polynomial Coefficients for the interpolant.
        The shape of `Yppc` less its final dimension must be broadcastable to
        the shape of `X` and/or `Y`.

    Returns
    -------
    y : ndarray

        Value of the piecewise polynomial interpolant (or its `d`'th
        derivative) at `x`.

    Notes
    -----
    A binary search is performed to find the indices of `X` that `x`
    lies between.  As such, nonsense results will arise if `X` is not
    sorted along its last dimension.
    """
    y[0] = ppval_1(x, X, Yppc, d)


@nb.njit
def ppval_1_two(x, X, Yppc, Zppc, d=0):
    """
    Evaluate two piecewise polynomials.

    As `ppval_1` but a second input, `Zppc`, provides a second set of Piecewise
    Polynomial Coefficients.  Correspondingly, a second output, `z`, is returned.
    """
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan
    i = max(0, np.searchsorted(X, x) - 1)
    dx = x - X[i]  # >= 0
    return ppval_i(dx, Yppc, i, d), ppval_i(dx, Zppc, i, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:, :], nb.f8[:, :], nb.i8, nb.f8[:], nb.f8[:])],
    "(),(n),(n,m),(n,m),()->(),()",
)
def ppval_two(x, X, Yppc, Zppc, d, y, z):
    """
    Evaluate two given Piecewise Polynomials

    The inputs and outputs are as for `ppval`, but another input, `Zppc`, gives
    the second piecewise polynomial, for which a second output `z` is given.

    Notes
    -----
    Calling `ppval_two` is faster than calling `ppval` twice, since the former
    calls `np.searchsorted` half as many times.
    """
    y[0], z[0] = ppval_1_two(x, X, Yppc, Zppc, d)
