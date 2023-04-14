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

from .lib import valid_range_1_two


@nb.njit
def pval(x, C, d=0):
    """
    Evaluate a (derivative of a) single polynomial.

    Parameters
    ----------
    x : float
        Evaluation site
    C : array of float
        Polynomial coefficients, starting with the highest order coefficient
        and ending with the coefficient of the constant term.
        The order of the polynomial is `len(C)`.
    d : int, Default 0
        Order of the derivative to evaluate. When `0`, just evaluate the
        polynomial.

    Returns
    -------
    y : float
        The polynomial (or its `d`'th derivative) evaluated at `x`.

    Example
    -------
    If `C == (C2, C1, C0)`, and `d == 0`, then
    `y == x * (x * C2 + C1) + C0`.
    """

    order = len(C)  # order of the polynomial

    # Evaluate polynomial, using nested multiplication.
    # E.g. the cubic case is:
    # y = x^3 * C[0] + x^2 * C[1] + x * C[2] + C[3]
    #   = x * (x * (x * C[0] + C[1]) + C[2]) + C[3]
    y = 0.0
    if d == 0:
        for j in range(0, order):
            y = y * x + C[j]
    else:
        # Evaluate polynomial derivative, using nested multiplication.
        # E.g. the second derivative of the cubic case is:
        # y = 6 * x * C[0] + 2 * C[1]
        y = 0.0
        for o in range(0, order - d):  # o for order
            p = 1.0  # Build integer multiplying the coefficient
            for a in range(order - o - d, order - o):
                p *= a
            # p = np.prod(np.arange(degree - o - d + 1, degree - o + 1))  # slow!
            y = y * x + C[o] * p

    return y


@nb.njit
def ppval_1(x, X, Yppc, d=0):
    """
    Evaluate a single Piecewise Polynomial (PP).

    Parameters
    ----------
    x : float
        Evaluation site
    X : ndarray, 1d
        Independent variable. Must be monotonically increasing.
    Yppc : ndarray, 2d
        Piecewise Polynomial Coefficients.  First dimension must match `len(X)`.
    d : int, Default 0
        Number of derivatives to take.  If 0, simply evaluate the PP.

    Returns
    -------
    y : float
        The value of the PP (or its `d`'th derivative) at `X = x`.

    Notes
    -----
    If `X` and `Y` have NaN's, only the first contiguous block of non-NaN data
    between both `X` and `Y` is used.

    """

    # Find k = index to first valid data site, and
    # K such that K - 1 = index to last valid data site in contiguous range
    #                     of valid data after index k.
    # Note, Yppc[:,-1] = Y, the original data
    k, K = valid_range_1_two(X, Yppc[:, -1])

    # K - 1 == k means there's only one valid data point. Can't interpolate that.
    if K - 1 <= k or np.isnan(x) or x < X[k] or X[K - 1] < x:
        return np.nan

    # i = searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0] or all(isnan(X))
    #   i = len(X)              if X[-1] < x or isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    #
    # So
    # i = searchsorted(X[k:K], x)  is such that:
    #   i = 0              if x <= X[k] or all(isnan(X[k:K]))
    #   i = K-k            if X[K-1] < x or isnan(x)
    #   X[k:K][i-1] < x <= X[k:K][i] otherwise
    #
    # Having guaranteed above that x is not nan, and X[k:K] is all non-NaN, and
    # X[k] <= x <= X[K-1], then
    #   i = (searchsorted(X[k:K],x) + k)
    # is such that
    #   k <= i <= K-1
    # is guaranteed, and
    #   x == X[k]           when i == k,
    #   X[i-1] < x <= X[i]  when k+1 <= i <= K-1.
    #
    # Next, merge i == k into the i == k+1 case:
    #   i = max(k+1, searchsorted(X[k:K], x) + k)
    # is such that
    #   k+1 <= i <= K-1
    # is guaranteed, and
    #   X[k]  <= x <= X[k+1]  when  i == k+1,
    #   X[i-1] < x <= X[i]    when  k+2 <= i <= K-1.
    #
    # Finally, subtract 1:
    #   i = max(k, searchsorted(X[k:K], x) + k - 1)
    # is such that
    #   k <= i <= K-2
    # is guaranteed, and
    #   X[k] <= x <= X[k+1]  when  i == k,
    #   X[i] <  x <= X[i+1]  when  k+1 <= i <= K-2.
    i = max(k, np.searchsorted(X[k:K], x) + k - 1)

    dx = x - X[i]  # >= 0
    return pval(dx, Yppc[i], d)


@nb.njit
def ppval_1_nonan(x, X, Yppc, d=0):
    """As `ppval_1` but knowing there are no NaN's in the inputs X and Yppc"""
    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan

    i = max(0, np.searchsorted(X, x) - 1)

    dx = x - X[i]  # >= 0
    return pval(dx, Yppc[i], d)


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
        `i` tuple indexing all but the last dimension.

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
    If `X` and `Y` have NaN's, only the first contiguous block of non-NaN data
    between both `X` and `Y` is used.

    A binary search is performed to find the indices of `X` that `x`
    lies between.  As such, nonsense results will arise if `X` is not
    sorted along its last dimension.
    """
    y[0] = ppval_1(x, X, Yppc, d)


@nb.njit
def ppinterp_1(f, x, X, Y, d=0):
    """Do a piecewise polynomial interpolation, building the coefficients on the fly."""

    # Find k = index to first valid data site, and
    # K such that K - 1 = index to last valid data site in contiguous range
    #                     of valid data after index k.
    k, K = valid_range_1_two(X, Y)

    # K - 1 == k means there's only one valid data point. Can't interpolate that.
    if K - 1 <= k or np.isnan(x) or x < X[k] or X[K - 1] < x:
        return np.nan

    #   X[k] <= x <= X[k+1]  when  i == k,
    #   X[i] <  x <= X[i+1]  when  k+1 <= i <= K-2.
    i = max(k, np.searchsorted(X[k:K], x) + k - 1)

    Yppc = f(X, Y, i)
    dx = x - X[i]  # >= 0
    return pval(dx, Yppc, d)


@nb.njit
def ppinterp_1_two(f, x, X, Y, Z, d=0):
    """Do two piecewise polynomial interpolations, building the coefficients on the fly."""

    # Find k = index to first valid data site, and
    # K such that K - 1 = index to last valid data site in contiguous range
    #                     of valid data after index k.
    k, K = valid_range_1_two(X, Y)

    # K - 1 == k means there's only one valid data point. Can't interpolate that.
    if K - 1 <= k or np.isnan(x) or x < X[k] or X[K - 1] < x:
        return np.nan, np.nan

    #   X[k] <= x <= X[k+1]  when  i == k,
    #   X[i] <  x <= X[i+1]  when  k+1 <= i <= K-2.
    i = max(k, np.searchsorted(X[k:K], x) + k - 1)

    Yppc = f(X, Y, i)
    Zppc = f(X, Z, i)
    dx = x - X[i]  # >= 0
    return pval(dx, Yppc, d), pval(dx, Zppc, d)


@nb.njit
def ppval_1_two(x, X, Yppc, Zppc, d=0):
    """
    Evaluate two piecewise polynomials.

    As `ppval_1` but a second input, `Zppc`, provides a second set of Piecewise
    Polynomial Coefficients.  Correspondingly, a second output, `z`, is returned.

    `Zppc` must have the same NaN-structure as `Yppc`.
    """

    k, K = valid_range_1_two(X, Yppc[:, -1])

    if K - 1 <= k or np.isnan(x) or x < X[k] or X[K - 1] < x:
        return np.nan, np.nan

    i = max(k, np.searchsorted(X[k:K], x) + k - 1)

    dx = x - X[i]  # >= 0
    return pval(dx, Yppc[i], d), pval(dx, Zppc[i], d)


@nb.njit
def ppval_1_nonan_two(x, X, Yppc, Zppc, d=0):
    """As `ppval_1_two` but knowing there are no NaN's in the inputs X, Yppc, Zppc"""
    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan, np.nan

    i = max(0, np.searchsorted(X, x) - 1)

    dx = x - X[i]  # >= 0
    return pval(dx, Yppc[i], d), pval(dx, Zppc[i], d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:, :], nb.f8[:, :], nb.i8, nb.f8[:], nb.f8[:])],
    "(),(n),(n,m),(n,m),()->(),()",
)
def ppval_two(x, X, Yppc, Zppc, d, y, z):
    """
    Evaluate two given Piecewise Polynomials

    The inputs and outputs are as for `ppval`, but another input, `Zppc`, gives
    the second piecewise polynomial, for which a second output `z` is given.

    `Zppc` must have the same NaN-structure as `Yppc`.

    Notes
    -----
    Calling `ppval_two` is faster than calling `ppval` twice, since the former
    calls `np.searchsorted` half as many times.
    """
    y[0], z[0] = ppval_1_two(x, X, Yppc, Zppc, d)
