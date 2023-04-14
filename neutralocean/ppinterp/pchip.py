import numpy as np
import numba as nb

from .lib import diff_1d_samesize, valid_range_1_two

from .ppinterp import ppinterp_1, ppinterp_1_two


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
        `i` tuple indexing all but the last dimension.

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
    If `X` and `Y` have NaN's, only the first contiguous block of non-NaN data
    between both `X` and `Y` is used.

    Evaluate the piecewise polynomial at `x` as

    >>> y = ppval(x, X, Yppc)

    """

    # Suppress the divide by zero and invalid arguments warnings that appear
    # for the `numba.guvectorize`d function, though not for the `numba.njit`ed
    # function.  The PCHIP algorithm is guaranteed to not divide by zero.
    # https://github.com/numba/numba/issues/4793#issuecomment-623296850
    with np.errstate(divide="ignore", invalid="ignore"):
        return _pchip_coeffs(X, Y, np.arange(4))


def pchip_interp(x, X, Y, d=0):

    # Suppress the divide by zero and invalid arguments warnings that appear
    # for the `numba.guvectorize`d function, though not for the `numba.njit`ed
    # function.  The PCHIP algorithm is guaranteed to not divide by zero.
    # https://github.com/numba/numba/issues/4793#issuecomment-623296850
    with np.errstate(divide="ignore", invalid="ignore"):
        return _pchip_interp(x, X, Y, d)


def pchip_interp_two(x, X, Y, Z, d=0):

    # Suppress the divide by zero and invalid arguments warnings that appear
    # for the `numba.guvectorize`d function, though not for the `numba.njit`ed
    # function.  The PCHIP algorithm is guaranteed to not divide by zero.
    # https://github.com/numba/numba/issues/4793#issuecomment-623296850
    with np.errstate(divide="ignore", invalid="ignore"):
        return _pchip_interp_two(x, X, Y, Z, d)


@nb.njit
def pchip_coeffs_1(X, Y):
    """
    Coefficients for a single PCHIP when the data may contain NaNs

    Inputs and outputs are as for `pchip_coeffs`, but `X` and `Y` both 1D
    arrays of length `n`, and so `Yppc` is a 2D array of size `(n, 4)`.
    """

    # Find k = index to first valid data site, and
    # K such that K - 1 = index to last valid data site in contiguous range
    #                     of valid data after index k.
    k, K = valid_range_1_two(X, Y)

    return _pchip_coeffs_1(X, Y, k, K)


@nb.njit
def pchip_coeffs_1_nonan(X, Y):
    """
    Coefficients for a single PCHIP when the data has no NaNs

    Inputs and outputs are as for `pchip_coeffs`, but `X` and `Y` both 1D
    arrays of length `n`, and so `Yppc` is a 2D array of size `(n, 4)`.
    """

    return _pchip_coeffs_1(X, Y, 0, X.size)


@nb.njit
def _pchip_coeffs_1(X, Y, k, K):
    """
    Coeffs for a single PCHIP when data is non-NaN on `k:K`
    """

    # Note, np.diff does not work with nb.njit, here.
    # Also, using our own numba.njit'ed function is faster than np.ediff1d
    h = diff_1d_samesize(X)  # distance between data sites
    δ = diff_1d_samesize(Y) / h  # linear slope between data sites

    nk = X.size  # Number of data points included possible NaN's

    d = np.zeros(K)  # slope of interpolant at data sites
    #                  Note elements 0, ... k-1 of d will be unused.
    C = np.full((nk, 4), np.nan)

    if K - k > 2:

        # Calculate PCHIP slopes
        #  Slopes at end points:
        #   Set d[k] and d[K-1] via non-centered, shape-preserving three-point formulae.
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
    #     interpolant would be Y[k] when evaluated 0at x > X[k], but we
    #     only want it to be Y[k] at x = X[k] precisely.

    C[:, 3] = Y

    return C


@nb.njit
def _pchip_coeffs_i(X, Y, i):
    """
    Coefficients of a single Piece of a PCHIP

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
    C3, C2, C1, C0 : float
        Piecewise Polynomial Coefficients that can be evaluated at `x`.

    Notes
    -----
    To interpolate `Y` in terms of `X` at evaluation site `x`, simply evaluate
    the piecewise polynomial whose coefficients are `Yppc` at `x` by
        `pval(x - X[i], (C3, C2, C1, C0))`

    This function is adapted from `_pchip_coeffs_1`
    """

    # Pre-assign sizes for PCHIP variables.
    h = [0.0, 0.0, 0.0]
    δ = [0.0, 0.0, 0.0]
    d = [0.0, 0.0]

    # Check whether x is adjacent to the start or end of this X
    at_start = (i == 0) or np.isnan(X[i - 1] + Y[i - 1])
    at_end = (i == len(X) - 2) or np.isnan(X[i + 2] + Y[i + 2])

    if at_start and at_end:

        # if np.isnan(X[i + 1]) or np.isnan(Y[i + 1]):
        #     # Only one valid data point.  Leave the interpolant as NaN.
        #     d[0], c, b = np.nan, np.nan, np.nan

        # else:

        # ||| X[0] <= x <= X[1] |||   Revert to Linear Interpolation
        # If actually only one non-NaN data point, then d[0] will be NaN, so
        # interpolant will evaluate to NaN.
        d[0] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
        C3, C2 = 0.0, 0.0

    else:
        if at_start:
            #  ||| X[0] <= x <= X[1] < X[2] --->
            h[1] = X[i + 1] - X[i]
            h[2] = X[i + 2] - X[i + 1]
            δ[1] = (Y[i + 1] - Y[i]) / h[1]
            δ[2] = (Y[i + 2] - Y[i + 1]) / h[2]

            #  Noncentered, shape-preserving, three-point formula:
            d[0] = ((2.0 * h[1] + h[2]) * δ[1] - h[1] * δ[2]) / (h[1] + h[2])
            if np.sign(d[0]) != np.sign(δ[1]):
                d[0] = 0.0
            elif (np.sign(δ[1]) != np.sign(δ[2])) and (
                np.abs(d[0]) > np.abs(3.0 * δ[1])
            ):
                d[0] = 3.0 * δ[1]

            # Standard PCHIP formula
            if np.sign(δ[1]) * np.sign(δ[2]) > 0.0:
                w1 = 2.0 * h[2] + h[1]
                w2 = h[2] + 2.0 * h[1]
                d[1] = (w1 + w2) / (w1 / δ[1] + w2 / δ[2])
            else:
                d[1] = 0.0

        elif at_end:
            # <--- X[i-1] < X[i] < x <= X[i+1] |||
            h[0] = X[i] - X[i - 1]
            h[1] = X[i + 1] - X[i]
            δ[0] = (Y[i] - Y[i - 1]) / h[0]
            δ[1] = (Y[i + 1] - Y[i]) / h[1]

            # Standard PCHIP formula
            if np.sign(δ[0]) * np.sign(δ[1]) > 0.0:
                w1 = 2.0 * h[1] + h[0]
                w2 = h[1] + 2.0 * h[0]
                d[0] = (w1 + w2) / (w1 / δ[0] + w2 / δ[1])
            else:
                d[0] = 0.0

            #  Noncentered, shape-preserving, three-point formula:
            d[1] = ((h[0] + 2.0 * h[1]) * δ[1] - h[1] * δ[0]) / (h[0] + h[1])
            if np.sign(d[1]) != np.sign(δ[1]):
                d[1] = 0.0
            elif (np.sign(δ[1]) != np.sign(δ[0])) and (
                np.abs(d[1]) > np.abs(3 * δ[1])
            ):

                d[1] = 3.0 * δ[1]

        else:
            # <--- X[i-1] < X[i] < x <= X[i+1] < X[i+2] --->
            h[0] = X[i] - X[i - 1]  # Way faster to do this
            h[1] = X[i + 1] - X[i]  # than
            h[2] = X[i + 2] - X[i + 1]  # diff(X(i-1:i+3))
            δ[0] = (Y[i] - Y[i - 1]) / h[0]
            δ[1] = (Y[i + 1] - Y[i]) / h[1]
            δ[2] = (Y[i + 2] - Y[i + 1]) / h[2]

            # Standard PCHIP formula
            for j in range(2):
                if np.sign(δ[j]) * np.sign(δ[j + 1]) > 0.0:
                    w1 = 2.0 * h[j + 1] + h[j]
                    w2 = h[j + 1] + 2.0 * h[j]
                    d[j] = (w1 + w2) / (w1 / δ[j] + w2 / δ[j + 1])
                else:
                    d[j] = 0.0

        # Polynomial coefficients for this piece
        dzzdx = (δ[1] - d[0]) / h[1]
        dzdxdx = (d[1] - δ[1]) / h[1]
        C3 = (dzdxdx - dzzdx) / h[1]  # coeff of the 3rd degree term (x^3)
        C2 = 2 * dzzdx - dzdxdx  # coeff of 2nd degree term (x^2)

    # The following code evaluates the `d`'th deriviative of the cubic
    # interpolant at `x`.
    # s = x - X[i]
    # if d == 0:
    #     y = Y[i] + s * (d[0] + s * (C2 + s * C3))
    # elif d == 1:  # first derivative
    #     y = d[0] + s * (2 * C2 + 3 * s * C3)
    # elif d == 2:  # second derivative
    #     y = 2 * C2 + 6 * s * C3
    # elif d == 3:  # third derivative
    #     y = 6 * C3
    # else:
    #     y = 0.0
    # return y

    # Faster to return tuple than build an np.array just to deconstruct it later
    return C3, C2, d[0], Y[i]


@nb.guvectorize(
    [(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :])], "(n),(n),(m)->(n,m)"
)
def _pchip_coeffs(X, Y, len4array, Yppc):
    Yppc[:, :] = pchip_coeffs_1(X, Y)


@nb.njit
def pchip_interp_1(x, X, Y, d=0):
    return ppinterp_1(_pchip_coeffs_i, x, X, Y, d)


@nb.njit
def pchip_interp_1_two(x, X, Y, Z, d=0):
    return ppinterp_1_two(_pchip_coeffs_i, x, X, Y, Z, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.i8, nb.f8[:])],
    "(),(n),(n),()->()",
)
def _pchip_interp(x, X, Y, d, y):
    y[0] = pchip_interp_1(x, X, Y, d)


@nb.guvectorize(
    [(nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.i8, nb.f8[:], nb.f8[:])],
    "(),(n),(n),(n),()->(),()",
)
def _pchip_interp_two(x, X, Y, Z, d, y, z):
    y[0], z[0] = pchip_interp_1_two(x, X, Y, Z, d)
