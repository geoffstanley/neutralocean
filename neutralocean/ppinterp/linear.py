import numpy as np
import numba as nb

from .lib import diff_1d_samesize
from .ppinterp import ppval_i


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
        `i` tuple indexing all but the last dimension.  NaN's in `X`
        are treated as +Inf.

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
def linear_interp_1(x, X, Y, d=0):
    """Build and evaluate a piecewise polynomial, building (almost) only what's needed."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan

    #   X[0]  <= x <= X[1]   when  i = 0
    #   X[i]  <  x <= X[i+1] when  i > 0
    i = max(0, np.searchsorted(X, x) - 1)

    dx = x - X[i]  # >= 0

    # To evaluate a linear interpolant at x in the range x[i] < x <= X[i+1],
    # we only need 2 data points: (i, i+1).

    Yppc = linear_coeffs_1(X[i : i + 2], Y[i : i + 2])

    return ppval_i(dx, Yppc, 0, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.i8, nb.f8[:])],
    "(),(n),(n),()->()",
)
def linear_interp(x, X, Y, d, y):
    y[0] = linear_interp_1(x, X, Y, d)


@nb.njit
def linear_interp_1_two(x, X, Y, Z, d=0):
    """Build and evaluate two piecewise polynomials, building (almost) only what's needed."""
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan

    #   X[0]  <= x <= X[1]   when  i = 0
    #   X[i]  <  x <= X[i+1] when  i > 0
    i = max(0, np.searchsorted(X, x) - 1)

    dx = x - X[i]  # >= 0

    # To evaluate a linear interpolant at x in the range x[i] < x <= X[i+1],
    # we only need 2 data points: (i, i+1).
    Yppc = linear_coeffs_1(X[i : i + 2], Y[i : i + 2])
    Zppc = linear_coeffs_1(X[i : i + 2], Z[i : i + 2])

    return ppval_i(dx, Yppc, 0, d), ppval_i(dx, Zppc, 0, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.i8, nb.f8[:], nb.f8[:])],
    "(),(n),(n),(n),()->(),()",
)
def linear_interp_two(x, X, Y, Z, d, y, z):
    y[0], z[0] = linear_interp_1_two(x, X, Y, Z, d)
