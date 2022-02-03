import numpy as np
import numba as nb

from .ppinterp import diff_1d_samesize


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
    arrays of length n, and so `Yppc` is a 2D array of size (n, 2).
    """
    C = np.empty((len(Y), 2), dtype=np.float64)
    C[:, 0] = diff_1d_samesize(Y) / diff_1d_samesize(X)
    C[:, 1] = Y
    return C


@nb.guvectorize([(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :])], "(n),(n),(m)->(n,m)")
def _linear_coeffs(X, Y, two, C):
    C[:, :] = linear_coeffs_1(X, Y)
