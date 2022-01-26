"""
Functions for linear or PCHIP interpolation of one (or two) dependent variables
in terms of one independent variable, done serially over an arbitrary number
of such interpolation problems.  This is used to interpolate salinity and
temperature in terms of either pressure or depth in each water column. 
"""

import numpy as np
import numba

from .linear import linterp_i


def interp(x, X, Y, fcn=linterp_i):
    """
    A 'universal function' for 1D interpolation along the last dimension.

    Parameters
    ----------
    x : ndarray or float

        Sites at which to interpolate the independent data.

        If ndarray, must be the same size as `X` or `Y` less their final
        dimension, with any dimension possibly a singleton: that is, must be
        broadcastable to the size of `X` or `Y` less their final dimension.

    X : ndarray

        Independent data. Must be same dimensions as `Y`, with any
        dimension possibly a singleton: that is, must be broadcastable
        to the size of `Y`.

        `X` must be monotonically increasing in its last dimension.
        That is, `X[*i,:]` must be monotonically increasing for any
        `i` tuple indexing all but the last dimension.  NaN's in `X`
        are treated as +Inf.

    Y : ndarray

        Dependent data.  Must be same dimensions as `X`, with any
        dimension possibly a singleton: that is, must be broadcastable
        to the size of `X`.

    fcn : function, Default linterp_i

        The "atom" of interpolation: A function that performs a single
        interpolation within a known interval.
        The parameters to `fcn` are
            x : float
            X : ndarray(float, 1d)
            Y : ndarray(float, 1d)
            i : int
        and the return value of `fcn` is
            y : float
        which is `Y` as a function of `X` interpolated to the value `x`. Here,
        the subinterval of `X` within which `x` falls is given to this
        function by `i`.  This function will only be called when the
        following is guaranteed:
            (a) `i == 1`  and  `X[0] <= x <= X[1]`, or
            (b) `2 <= i <= len(X) - 1`  and  X[i-1] < x <= X[i]`.

        For linear interpolation, use `fcn = linterp_i`.
        For pchip interpolation, use `fcn = pchip_i`.
        For the second derivative of the pchip interpolant, use `fcn = pchip_dxx_i`.
        Et cetera.

    Returns
    -------
    y : ndarray or float

        Value of the interpolant (or its `d`'th derivative) at `x`.

        If `x` is float and `X` and `Y` are both 1 dimensional, `y` is float.

        Otherwise, `y` is ndarray with size matching the largest of the sizes
        of `x`, the size of `X` less its last dimension, or the size of `Y`
        less its last dimension.

    Notes
    -----
    A binary search is performed to find the indices of `X` that `x`
    lies between.  As such, nonsense results will arise if `X` is not
    sorted along its last dimension.
    """

    nk = X.shape[-1]

    if isinstance(Y, (tuple, list)):
        Yndim = Y[0].ndim
        Yshape = Y[0].shape
        assert all(
            Y_.shape == Yshape for Y_ in Y
        ), "With Y a tuple, all elemnts must have the same shape"
    else:
        Yndim = Y.ndim
        Yshape = Y.shape

    # assert nk == Y.shape[-1], "X and Y must have same size last dimension"
    if X.ndim > 1:
        shape = X.shape[0:-1]
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match those of X"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif Yndim > 1:
        shape = Yshape[0:-1]
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match those of Y"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif np.isscalar(x):
        shape = ()
    else:
        shape = x.shape
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match the shape of x"
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match the shape of x"

    x = np.broadcast_to(x, shape)
    X = np.broadcast_to(X, (*shape, nk))
    if isinstance(Y, tuple):
        if len(Y) == 2:
            Z = Y[1]
            Y = Y[0]
            Y = np.broadcast_to(Y, (*shape, Yshape[-1]))
            Z = np.broadcast_to(Z, (*shape, Yshape[-1]))

            y, z = interp_n_twice(x, X, Y, Z, fcn)
            if shape == ():
                y, z = y[()], z[()]
            return y, z
        elif len(Y) > 2:
            raise TypeError(
                "If Y is a tuple, its length must be <= 2."
                "Support for longer Y is not yet complete."
            )
        else:  # len(Y) == 1
            # Extract the single element and proceed.
            Y = Y[0]

    Y = np.broadcast_to(Y, (*shape, Yshape[-1]))

    y = interp_n(x, X, Y, fcn)

    if shape == ():
        # Convert 0d array to scalar
        y = y[()]

    return y


@numba.njit
def interp_n(x, X, Y, fcn=linterp_i):
    """
    Repeated interpolation with all inputs' dimensions agreeable
    """
    y = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n] = interp_1(x[n], X[n], Y[n], fcn)

    return y


@numba.njit
def interp_1(x, X, Y, fcn=linterp_i):
    """
    Do one interpolation problem.
    """

    # Check for edge cases
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan

    # i = searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0] or all(isnan(X))
    #   i = len(X)              if X[-1] < x or isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    #
    # Having guaranteed above that
    # x is not nan, and X[0] is not nan hence not all(isnan(X)),
    # and X[0] <= x <= X[-1],
    # then either
    # (a) i == 0  and x == X[0], or
    # (b) 1 <= i <= len(X)-1  and  X[i-1] < x <= X[i]
    i = np.searchsorted(X, x)

    # Next, merge (a) and (b) cases so that
    #   1 <= i <= len(X) - 1
    # is guaranteed, and
    #   X[0 ] <= x <= X[1]  when  i == 1
    #   X[i-1] < x <= X[i]  when  i > 1
    i = max(1, i)

    # Interpolate within the given interval
    return fcn(x, X, Y, i)


# Like interp_n but takes another argument, Z, that is like Y.
@numba.njit
def interp_n_twice(x, X, Y, Z, fcn=linterp_i):
    """
    Repeated interpolation with all inputs' dimensions agreeable, twice.
    """
    y = np.empty(x.shape, dtype=np.float64)
    z = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n], z[n] = interp_1_twice(x[n], X[n], Y[n], Z[n], fcn)

    return y, z


# Like interp_1 but takes another argument, Z, that is like Y.
@numba.njit
def interp_1_twice(x, X, Y, Z, fcn=linterp_i):
    """
    Do one interpolation problem, twice.
    """

    # Check for edge cases
    if np.isnan(x) or x < X[0] or X[-1] < x or np.isnan(X[0]):
        return np.nan, np.nan

    # i = searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0] or all(isnan(X))
    #   i = len(X)              if X[-1] < x or isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    #
    # Having guaranteed above that
    # x is not nan, and X[0] is not nan hence not all(isnan(X)),
    # and X[0] <= x <= X[-1],
    # then either
    # (a) i == 0  and x == X[0], or
    # (b) 1 <= i <= len(X)-1  and  X[i-1] < x <= X[i]
    i = np.searchsorted(X, x)

    # Next, merge (a) and (b) cases so that
    #   1 <= i <= len(X) - 1
    # is guaranteed, and
    #   X[0 ] <= x <= X[1]  when  i == 1
    #   X[i-1] < x <= X[i]  when  i > 1
    i = max(1, i)

    # Interpolate within the given interval
    return fcn(x, X, Y, i), fcn(x, X, Z, i)


# def interp1d(x, X, Y, interp_fns, d=0, i=None):
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

#     interpolant : tuple of functions

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

#     if d < len(interp_fns):
#         fcn = interp_fns[d]
#     else:
#         return 0.0
#     if i is None:
#         return interp_1d(x, X, Y, fcn)
#     else:
#         return fcn(x, X, Y, i)
