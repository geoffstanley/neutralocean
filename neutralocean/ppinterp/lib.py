import numpy as np
import numba as nb


@nb.njit
def diff_1d_samesize(x):
    """
    First difference in one dimension; appends nan so output same size as input.

    A simple function like `np.diff` that numba can work with.
    """

    d = np.empty(x.size, dtype=x.dtype)
    for i in range(d.size):
        d[i] = x[i + 1] - x[i]
    d[-1] = np.nan
    return d


@nb.njit
def valid_range_1(X):
    """The index to the first finite value and to the first NaN after the former

    Parameters
    ----------
    X : 1D array
        Input array possibly containing some NaN elements

    Returns
    -------
    k : int
        First `k` such that `X[k]` is finite.
        If `X[i]` is NaN for all `i`, then `k = len(X)`.

    K : int
        If `X[i]` is NaN for all `i > k`, then `K = len(X)`.
        If `X[i]` is finite for all `i > k`, then `K = len(X)`.
        Otherwise, `K` is the first index after `k` such that `X[K]` is NaN.

    Notes
    -----
    `X[i]` is non-NaN for `i = k, ..., K - 1`.
    `K - k` is the size of the first contiguous block of valid values of `X`.
    """
    k = K = len(X)

    # Find k = index to first valid data site
    for i in range(K):
        if np.isfinite(X[i]):
            k = i
            break

    # Find K, such that K-1 = index to last valid data site
    for i in range(k, K):
        if np.isnan(X[i]):
            K = i
            break

    return k, K


@nb.guvectorize([(nb.f8[:], nb.int64[:], nb.int64[:])], "(n)->(),()")
def valid_range(X, k, K):
    """Indices to the first finite value and to the first NaN after the former,
    along last dimension.

    Parameters
    ----------
    X : ndarray
        Input array possibly containing some NaN elements

    Returns
    -------
    k : ndarray of int
        `k[n]` is the smallest index such that `X[n][k[n]]` is finite.
        If `X[i]` is NaN for all `i`, then `k[n] = len(X[n])`.

    K : ndarray of int
        If `X[n][i]` is NaN for all `i > k[n]`, then `K[n] = len(X[n])`.
        If `X[n][i]` is finite for all `i > k[n]`, then `K[n] = len(X[n])`.
        Otherwise, `K[n]` is the smallest index after `k[n]` such that
        `X[n][K[n]]` is NaN.

    Notes
    -----
    `X[n][i]` is non-NaN for `i = k[n], ..., K[n] - 1`.
    `K[n] - k[n]` is the size of the first contiguous block of valid values of `X[n]`.
    """
    k[0], K[0] = valid_range_1(X)


@nb.njit
def valid_range_1_two(X, Y):
    """Indices bounding the first contiguous block of valid data in two 1D arrays

    Parameters
    ----------
    X, Y : 1D array
        Input arrays of the same length, possibly containing some NaN elements

    Returns
    -------
    k : int
        First `k` such that `X[k]` and `Y[k]` are finite.
        If `X[i]` or `Y[i]` is NaN for all `i`, then `k = len(X)`.

    K : int
        If `X[i]` or `Y[i]` is NaN for all `i > k`, then `K = len(X)`.
        If `X[i]` and `Y[i]` are finite for all `i > k`, then `K = len(X)`.
        Otherwise, `K` is the first index after `k` such that `X[K]` or `Y[K]`
        is NaN.

    Notes
    -----
    `X[i]` and `Y[i]` are both non-NaN for `i = k, ..., K - 1`.
    `K - k` is the size of the first contiguous block of valid pairs of values
    of `(X, Y)`.
    """
    k = K = len(X)

    # Find k = index to first valid data site
    for i in range(K):
        if np.isfinite(X[i]) and np.isfinite(Y[i]):
            k = i
            break

    # Find K, such that K-1 = index to last valid data site
    for i in range(k, K):
        if np.isnan(X[i]) or np.isnan(Y[i]):
            K = i
            break

    return k, K
