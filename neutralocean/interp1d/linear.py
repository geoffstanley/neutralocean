import numba

# from .interp import interp1d


# def linterp_1d(x, X, Y, d=0, i=None):
#     """
#     Interpolation in one dimension

#     Parameters
#     ----------
#     x : float
#         The evaluation site

#     X : ndarray(float, 1d)
#         The independent data.

#     Y : ndarray(float, 1d)
#         The dependent data.

#     d : int, Default 0
#         Evaluate the `d`'th derivative of the interpolant.
#         If 0, this simply evaluates the interpolant.

#     i : int, Default None
#         The interval of `X` that contains `x`.  Specifically,
#             (a) `i == 1`  and  `X[0] <= x <= X[1]`, or
#             (b) `2 <= i <= len(X) - 1`  and  X[i-1] < x <= X[i]`.
#         These facts about `i` are assumed true; they are not checked.
#         (This function will not be called if `x < X[0]` or `X[-1] < x` or
#         `x` is nan or `X` are all nan.)
#         If None, the correct value will be determined internally.

#     Returns
#     -------
#     y : float
#         The value (if `d==0`) or the derivative (if `d==1`) of the interpolant
#         for `Y` in terms of `X` evaluated at `x`.
#     """

#     return interp1d(x, X, Y, (linterp_i, linterp_dx_i), d, i)


@numba.njit
def linterp_i(x, X, Y, i):
    """
    The "atom" of linear interpolation.

    Parameters
    ----------
    x : float
        The evaluation site

    X : ndarray(float, 1d)
        The independent data.

    Y : ndarray(float, 1d)
        The dependent data.

    i : int
        The interval of `X` that contains `x`.  Specifically,
            (a) `i == 1`  and  `X[0] <= x <= X[1]`, or
            (b) `2 <= i <= len(X) - 1`  and  X[i-1] < x <= X[i]`.
        These facts about `i` are assumed true; they are not checked.
        (This function will not be called if `x < X[0]` or `X[-1] < x` or
        `x` is nan or `X` are all nan.)

    Returns
    -------
    y : float
        The value of `Y` linearly interpolated to `X` at `x`.

    """

    # dx = (x - X[i-1]) => 0 guaranteed (and == 0 only if x == X[0])
    return Y[i - 1] + (x - X[i - 1]) / (X[i] - X[i - 1]) * (Y[i] - Y[i - 1])


@numba.njit
def linterp_dx_i(x, X, Y, i):
    """
    The "atom" of the 1st derivative of linear interpolation.

    Inputs and outputs analogous to `linterp_i`.
    """

    return (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])


@numba.njit
def linterp_i_d(x, X, Y, i, d):

    if d == 0:
        return linterp_i(x, X, Y, i)
    elif d == 1:
        return linterp_dx_i(x, X, Y, i)
    else:
        return 0.0
