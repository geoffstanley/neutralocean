import numpy as np
import numba as nb

from .lib import diff_1d_samesize, valid_range_1


def pchip_coeffs(X, Y):
    """
    Piecewise Polynomial Coefficients for a PCHIP
    (Piecewise Cubic Hermite Interpolating Polynomial)

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

        Coefficients for a PCHIP interpolant of `Y` to `X`.

    Notes
    -----

    Evaluate the piecewise polynomial at `x` as

    >>> y = ppval(x, X, Yppc)

    """

    # Suppress the divide by zero and invalid arguments warnings that appear
    # for the `numba.guvectorize`d function, though not for the `numba.njit`ed
    # function.  The PCHIP algorithm is guaranteed to not divide by zero.
    # https://github.com/numba/numba/issues/4793#issuecomment-623296850
    with np.errstate(divide="ignore", invalid="ignore"):
        return _pchip_coeffs(X, Y, np.arange(4))


@nb.njit
def pchip_coeffs_1(X, Y):
    """
    Coefficients for a single PCHIP

    Inputs and outputs are as for `pchip_coeffs`, but `X` and `Y` both 1D
    arrays of length n, and so `Yppc` is a 2D array of size (n, 4).
    """

    # Note, np.diff does not work with nb.njit, here.
    # Also, using our own numba.njit'ed function is faster than np.ediff1d
    h = diff_1d_samesize(X)  # distance between data sites
    δ = diff_1d_samesize(Y) / h  # linear slope between data sites

    nk = X.size  # Number of data points included possible NaN's

    # Find k0 = index to first valid data site, and
    # k1 such that k1 - 1 = index to last valid data site
    k, K = valid_range_1(X + Y)
    # TODO: Remove this, and assume k = 0 and K = len(X)?  Do this if we only
    #       want to use ppinterp inside for loops, within which we trim the
    #       data to only its finite values.  i.e. do this if we're happy with
    #       interp1d for all of our "universal" kind of interpolation.

    d = np.zeros(K)  # slope of interpolant at data sites
    #                  Note elements 0, ... k-1 of d will be unused.
    C = np.full((nk, 4), np.nan)

    if K - k > 2:

        # Calculate PCHIP slopes
        #  Slopes at end points:
        #   Set d[0] and d[k-1] via non-centered, shape-preserving three-point formulae.
        #  Slopes at interior points:
        #   d[i] = weighted average of δ[i-1] and δ[i] when they have the same sign.
        #   d[i] = 0 when δ[i-1] and δ[i] have opposites signs or either is zero.

        d[k] = ((2 * h[k] + h[k + 1]) * δ[k] - h[k] * δ[k + 1]) / (
            h[k] + h[k + 1]
        )
        if np.sign(d[k]) != np.sign(δ[k]):
            d[k] = 0
        elif (np.sign(δ[k]) != np.sign(δ[k + 1])) and (
            abs(d[k]) > abs(3 * δ[k])
        ):
            d[k] = 3 * δ[k]

        for i in range(k + 1, K - 1):
            if np.sign(δ[i - 1]) * np.sign(δ[i]) > 0:
                w1 = h[i - 1] + 2 * h[i]
                w2 = 2 * h[i - 1] + h[i]
                d[i] = (w1 + w2) / (w1 / δ[i - 1] + w2 / δ[i])
            else:
                d[i] = 0

        d[K - 1] = (
            (2 * h[K - 2] + h[K - 3]) * δ[K - 2] - h[K - 2] * δ[K - 3]
        ) / (h[K - 2] + h[K - 3])
        if np.sign(d[K - 1]) != np.sign(δ[K - 2]):
            d[K - 1] = 0
        elif (np.sign(δ[K - 2]) != np.sign(δ[K - 3])) and (
            abs(d[K - 1]) > abs(3 * δ[K - 2])
        ):
            d[K - 1] = 3 * δ[K - 2]

        # Build piecewise cubic Hermite polynomial
        for i in range(k, K - 1):
            dzzdx = (δ[i] - d[i]) / h[i]
            dzdxdx = (d[i + 1] - δ[i]) / h[i]
            C[i, 0] = (dzdxdx - dzzdx) / h[i]
            C[i, 1] = 2 * dzzdx - dzdxdx
            C[i, 2] = d[i]

    elif K - k == 2:
        # Special case: use linear interpolation.

        δ = (Y[k + 1] - Y[k]) / (X[k + 1] - X[k])

        C[k, 0] = 0
        C[k, 1] = 0
        C[k, 2] = δ

    # else:  # K - k == 1
    #     leave coefficients as nans.  Ignore case of x = X[k] while
    #     having X[k+1:] == np.nan and/or Y[k+1:] == np.nan.  That case
    #     cannot be handled here, since setting the coefficients to
    #     be [0, 0, 0, Y[k]], i.e. a constant function, would mean the
    #     interpolant would be Y[k] when evaluated at x > X[k], but we
    #     only want it to be Y[k] at x = X[k] precisely.

    C[:, 3] = Y

    return C


@nb.guvectorize(
    [(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :])], "(n),(n),(m)->(n,m)"
)
def _pchip_coeffs(X, Y, dummy, Yppc):
    Yppc[:, :] = pchip_coeffs_1(X, Y)
