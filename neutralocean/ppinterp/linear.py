import numpy as np
import numba as nb

from .lib import diff_1d_samesize
from .ppinterp import ppinterp_1, ppinterp_1_two


def linear_coeffs(X, Y):
    """
    Piecewise Polynomial Coefficients for a linear interpolant

    Parameters
    ----------
    X : ndarray

        Independent variable. Must be same dimensions as `Y`, with any
        dimension possibly a singleton: that is, must be broadcastable
        to the size of `Y`.

        `X` must be monotonically increasing in its last dimension.
        That is, `X[*i,:]` should be monotonically increasing for any
        `i` tuple indexing all but the last dimension.

    Y : ndarray

        Dependent variable.  Must be same dimensions as `X`, with any
        dimension possibly a singleton: that is, must be broadcastable
        to the size of `X`.

    Returns
    -------
    Yppc : ndarray

        Coefficients for a linear interpolant of `Y` to `X`.

    Notes
    -----
    If `X` and `Y` have NaN's, only the first contiguous block of non-NaN data
    between both `X` and `Y` is used.

    Evaluate the piecewise polynomial at `x` as

    >>> y = ppval(x, X, Yppc)

    """
    return _linear_coeffs(X, Y, np.arange(2))


@nb.njit
def linear_coeffs_1(X, Y):
    """
    Coefficients for a single linear interpolant

    Inputs and outputs are as for `linear_coeffs`, but `X` and `Y` both 1D
    arrays of length `n`, and so `Yppc` is a 2D array of size `(n, 2)`.
    """
    C = np.empty((len(Y), 2), dtype=np.float64)
    C[:, 0] = diff_1d_samesize(Y) / diff_1d_samesize(X)
    C[:, 1] = Y
    return C


@nb.guvectorize(
    [(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :])], "(n),(n),(m)->(n,m)"
)
def _linear_coeffs(X, Y, len2array, C):
    C[:, :] = linear_coeffs_1(X, Y)


@nb.njit
def _linear_coeffs_i(X, Y, i):
    """
    Coefficients of a single Piece of a Linear Interpolant

    Parameters
    ----------
    X : 1D array
        Independent data, monotonically increasing

    Y : 1D array
        Dependent data

    i : int
        Select the interval of `X` that contains the eventual evaluation site
        `x`. Specifically,
            if `i == 0` then `X[0] <= x <= X[1]`, or
            if `1 <= i <= len(X) - 1` then `X[i] < x <= X[i+1]`.

        These facts about `i` are assumed true; they are not checked.

    Returns
    -------
    C1, C0 : float
        Piecewise Polynomial Coefficients that can be evaluated at `x`.

    Notes
    -----
    To interpolate `Y` in terms of `X` at evaluation site `x`, simply evaluate
    the piecewise polynomial whose coefficients are `Yppc` at `x` by
        `pval(x - X[i], (C1, C0), 0)`
    """
    C1 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])  # coeff of 1st degree term, x^1
    C0 = Y[i]  # coeff of 0th degree term, x^0

    return C1, C0


@nb.njit
def linear_interp_1(x, X, Y, d=0):
    """Build and evaluate a piecewise polynomial, building only what's needed."""
    return ppinterp_1(_linear_coeffs_i, x, X, Y, d)


@nb.njit
def linear_interp_1_two(x, X, Y, Z, d=0):
    return ppinterp_1_two(_linear_coeffs_i, x, X, Y, Z, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.i8, nb.f8[:])],
    "(),(n),(n),()->()",
)
def linear_interp(x, X, Y, d, y):
    y[0] = linear_interp_1(x, X, Y, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.i8, nb.f8[:], nb.f8[:])],
    "(),(n),(n),(n),()->(),()",
)
def linear_interp_two(x, X, Y, Z, d, y, z):
    y[0], z[0] = linear_interp_1_two(x, X, Y, Z, d)
