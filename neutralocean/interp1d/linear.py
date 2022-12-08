"""Kernels for linear interpolation"""
import numba as nb


@nb.njit
def _linterp(x, X, Y, i):
    """
    The "kernel" of linear interpolation.

    Parameters
    ----------
    x : float
        The evaluation site

    X : ndarray(float, 1d)
        The independent data.

    Y : ndarray(float, 1d)
        The dependent data.

    i : int
        The interval of `X` that contains `x`. Specifically,
            (a) `i == 1`  and  `X[0] <= x <= X[1]`, or
            (b) `2 <= i <= len(X) - 1` and `X[i-1] < x <= X[i]`.

        These facts about `i` are assumed true; they are not checked.
        (This function will not be called if `x < X[0]` or `X[-1] < x` or
        `x` is nan or `X` are all nan.)

    Returns
    -------
    y : float
        The value of `Y` linearly interpolated to `X` at `x`.

    """

    # dx = (x - X[i-1]) => 0 guaranteed (and == 0 only if x == X[0])
    return (Y[i] - Y[i - 1]) / (X[i] - X[i - 1]) * (x - X[i - 1]) + Y[i - 1]


@nb.njit
def _linterp1(x, X, Y, i):
    """
    The "kernel" of the 1st derivative of linear interpolation.

    Inputs and outputs analogous to `_linterp`.
    """

    return (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])


# interp_u = make_u(_linterp)
# interp_1 = make_1(_linterp)
# interp_n = make_n(_linterp)
# interp_u_twice = make_u_twice(_linterp)
# interp_1_twice = make_1_twice(_linterp)
# interp_n_twice = make_n_twice(_linterp)


# @nb.njit
# def interp_1_d(x, X, Y, d):
#     if d == 0:
#         return _interp_1(x, X, Y, ker)
#     elif d == 1:
#         return _interp_1(x, X, Y, ker1)
#     else:
#         return 0.0


# @nb.guvectorize(
#     [(float64, float64[:], float64[:], int64, float64[:])],
#     "(),(n),(n),()->()",
# )
# def interp_u_d(x, X, Y, d, y):
#     y[0] = interp_1_d(x, X, Y, d)


# # %timeit -n 100 interp_1_n(500., Z, S, 0)
# # 280 µs ± 23.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# @nb.njit
# def interp_n_d(x, X, Y, d):
#     if d == 0:
#         return _interp_n(x, X, Y, ker)
#     elif d == 1:
#         return _interp_n(x, X, Y, ker1)
#     else:
#         return np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype)


# @nb.njit
# def interp_1_d_twice(x, X, Y, Z, d):
#     if d == 0:
#         return _interp_1_twice(x, X, Y, ker)
#     elif d == 1:
#         return _interp_1_twice(x, X, Y, ker1)
#     else:
#         return 0.0, 0.0


# @nb.guvectorize(
#     [(float64, float64[:], float64[:], float64[:], int64, float64[:], float64[:])],
#     "(),(n),(n),(n)()->(),()",
# )
# def interp_u_d_twice(x, X, Y, d, y, z):
#     y[0], z[0] = interp_1_d_twice(x, X, Y, Y, d)


# @nb.njit
# def interp_n_d_twice(x, X, Y, Z, d):
#     if d == 0:
#         return _interp_n_twice(x, X, Y, Z, ker)
#     elif d == 1:
#         return _interp_n(x, X, Y, Z, ker1)
#     else:
#         return (
#             np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype),
#             np.full(Y.shape[0:-1], np.nan, dtype=Y.dtype),
#         )


# Trying to make one function with different specializations, that behaves
# differently depending upon its input types.
# This is numba's polymorphic dispatching:
# https://numba.pydata.org/numba-doc/latest/developer/dispatching.html
# achieved using numba's generated_jit
# https://numba.pydata.org/numba-doc/dev/user/generated-jit.html
# Unfortunately this does not work because the function signature of interp_d,
# once under `guvectorize`, is utterly different from the function signature
# for interp_n_d or interp_universal...  This error:
#   TypeError: generated implementation <numba._GUFunc 'interp_d'> should be
#   compatible with signature '(x, X, Y, d, y)', but has signature
#   '(*args, **kwargs)'
# @nb.generated_jit(nopython=True)
# def interp_universal(x, X, Y, d, y):
#     if isinstance(x, nb.types.Float) and isinstance(X, nb.types.Array):
#         return interp_n_d
#     else:
#         return interp_d


# @nb.guvectorize(
#     [
#         (float32, float32[:], float32[:], float32[:]),
#         (float64, float64[:], float64[:], float64[:]),
#     ],
#     "(),(n),(n)->()",
# )
# def interp(x, X, Y, y):
#     """
#     A 'universal function' for 1D interpolation along the last dimension.

#     Parameters
#     ----------
#     x : ndarray or float

#         Sites at which to interpolate the independent data.

#         If ndarray, must be the same size as `X` or `Y` less their final
#         dimension, with any dimension possibly a singleton: that is, must be
#         broadcastable to the size of `X` or `Y` less their final dimension.

#     X : ndarray

#         Independent data. Must be same dimensions as `Y`, with any
#         dimension possibly a singleton: that is, must be broadcastable
#         to the size of `Y`.

#         `X` must be monotonically increasing in its last dimension.
#         That is, `X[*i,:]` must be monotonically increasing for any
#         `i` tuple indexing all but the last dimension.  NaN's in `X`
#         are treated as +Inf; hence, NaN's must go after any valid data.
#         (If X[0] is NaN, it is assumed that all elements of X are NaN.)

#     Y : ndarray

#         Dependent data.  Must be same dimensions as `X`, with any
#         dimension possibly a singleton: that is, must be broadcastable
#         to the size of `X`.

#     Returns
#     -------
#     y : ndarray or float

#         Value of the interpolant (or its `d`'th derivative) at `x`.

#         If `x` is float and `X` and `Y` are both 1 dimensional, `y` is float.

#         Otherwise, `y` is ndarray with size matching the largest of the sizes
#         of `x`, the size of `X` less its last dimension, or the size of `Y`
#         less its last dimension.

#     Notes
#     -----
#     A binary search is performed to find the indices of `X` that `x`
#     lies between.  As such, nonsense results will arise if `X` is not
#     sorted along its last dimension.
#     """
#     y[0] = _interp(x, X, Y, ker)
