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
        The index to the first finite value in `X`.
        If `X` is all NaN, then `k = len(X)`.

    K : int
        The index to the first NaN value in `X` after `k`.
        If `X` is all NaN or if `X[k:]` is all finite, then `K = len(X)`.
    """
    k = K = len(X)

    # Find k0 = index to first valid data site
    for i in range(K):
        if np.isfinite(X[i]):
            k = i
            break

    # Find k, such that k-1 = index to last valid data site
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
        `k[n]` is the index to the first finite value in `X[n]`.
        If `X[n]` is all NaN, then `k[n] = X.shape(-1)`.

    K : ndarray of int
        `K[n]` is the index to the first NaN value in `X[n]` after `k[n]`.
        If `X[n]` is all NaN or if `X[n][k:]` is all finite, then `K[n] = X.shape(-1)`.
    """
    k[0], K[0] = valid_range_1(X)
