"""
Functions for linear or PCHIP interpolation of one (or two) dependent variables
in terms of one independent variable, done serially over an arbitrary number
of such interpolation problems.  This is used to interpolate salinity and
temperature in terms of either pressure or depth in each water column. 
"""

import numpy as np
import numba


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
def pchip_i(x, X, Y, i):
    """
    The "atom" of Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.

    Inputs and outputs analogous to `linterp_i`.
    """

    s, dY, cY, bY = _pchip_coeffs(x, X, Y, i)
    return Y[i - 1] + s * (dY + s * (cY + s * bY))


@numba.njit
def pchip_dx_i(x, X, Y, i):
    """
    The "atom" of the 1st derivative of PCHIP interpolation.

    Inputs and outputs analogous to `linterp_i`.
    """
    s, dY, cY, bY = _pchip_coeffs(x, X, Y, i)
    return dY + s * (2 * cY + 3 * s * bY)


@numba.njit
def pchip_dxx_i(x, X, Y, i):
    """
    The "atom" of the 2nd derivative of PCHIP interpolation.

    Inputs and outputs analogous to `linterp_i`.
    """
    s, _, cY, bY = _pchip_coeffs(x, X, Y, i)
    return 2 * cY + 6 * s * bY


@numba.njit
def pchip_dxxx_i(x, X, Y, i):
    """
    The "atom" of the 3rd derivative of PCHIP interpolation.

    Inputs and outputs analogous to `linterp_i`.
    """
    _, _, _, bY = _pchip_coeffs(x, X, Y, i)
    return 6 * bY


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

            y, z = interp2_nd(x, X, Y, Z, fcn)
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

    y = interp_nd(x, X, Y, fcn)

    if shape == ():
        # Convert 0d array to scalar
        y = y[()]

    return y


@numba.njit
def interp_nd(x, X, Y, fcn=linterp_i):
    """
    Repeated interpolation with all inputs' dimensions agreeable
    """
    y = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n] = interp_1d(x[n], X[n], Y[n], fcn)

    return y


@numba.njit
def interp_1d(x, X, Y, fcn=linterp_i):
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


# Like interp1_nd but takes another argument, Z, that is like Y.
@numba.njit
def interp2_nd(x, X, Y, Z, fcn=linterp_i):
    """
    Repeated interpolation with all inputs' dimensions agreeable, twice.
    """
    y = np.empty(x.shape, dtype=np.float64)
    z = np.empty(x.shape, dtype=np.float64)
    for n in np.ndindex(x.shape):
        y[n], z[n] = interp2_1d(x[n], X[n], Y[n], Z[n], fcn)

    return y, z


# Like interp1_1d but takes another argument, Z, that is like Y.
@numba.njit
def interp2_1d(x, X, Y, Z, fcn=linterp_i):
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


@numba.njit
def _pchip_coeffs(x, X, Y, i):
    """
    Calculate the coefficients of a cubic interpolant

    Parameters
    ----------
    x, X, Y, i :
        As in `pchip_i`

    Returns
    -------
    s : float
        Distance from the evaluation site to the nearest knot in `X` to the left.
    dY, cY, bY : float
        The first, second, and third order coefficients of the cubic interpolant,
        such that the value of the interpolant at `x` is
        `y = Y[i - 1] + s * (dY + s * (cY + s * bY))`

    """

    # Pre-assign sizes for PCHIP variables.
    h = [0.0, 0.0, 0.0]
    DY = [0.0, 0.0, 0.0]
    dY = [0.0, 0.0]

    # Check whether x is adjacent to the start or end of this X
    at_start = i == 1
    at_end = (i == len(X) - 1) or np.isnan(X[i + 1]) or np.isnan(Y[i + 1])

    if at_start and at_end:
        # ||| X[0] <= x <= X[1] |||   Revert to Linear Interpolation
        s = x - X[i - 1]
        dY[0] = (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])
        # leave cY, bY = 0, 0

        # The following code evaluates the cubic interpolant, given `d` that
        # specifies the number of derivatives to take.
        # r = (x - X[i - 1]) / (X[i] - X[i - 1])
        # if d == 0:
        #     y = Y[i - 1] * (1 - r) + Y[i] * r
        # elif d == 1:
        #     y = (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])
        # else:
        #     y = 0.0

    else:
        if at_start:
            #  ||| X[0] <= x <= X[1] < X[2] --->
            h[1] = X[i] - X[i - 1]
            h[2] = X[i + 1] - X[i]
            DY[1] = (Y[i] - Y[i - 1]) / h[1]
            DY[2] = (Y[i + 1] - Y[i]) / h[2]

            #  Noncentered, shape-preserving, three-point formula:
            dY[0] = ((2.0 * h[1] + h[2]) * DY[1] - h[1] * DY[2]) / (h[1] + h[2])
            if np.sign(dY[0]) != np.sign(DY[1]):
                dY[0] = 0.0
            elif (np.sign(DY[1]) != np.sign(DY[2])) and (
                np.abs(dY[0]) > np.abs(3.0 * DY[1])
            ):
                dY[0] = 3.0 * DY[1]

            # Standard PCHIP formula
            if np.sign(DY[1]) * np.sign(DY[2]) > 0.0:
                w1 = 2.0 * h[2] + h[1]
                w2 = h[2] + 2.0 * h[1]
                dY[1] = (w1 + w2) / (w1 / DY[1] + w2 / DY[2])
            else:
                dY[1] = 0.0

        elif at_end:
            # <--- X[i-2] < X[i-1] < x <= X[i] |||
            h[0] = X[i - 1] - X[i - 2]
            h[1] = X[i] - X[i - 1]
            DY[0] = (Y[i - 1] - Y[i - 2]) / h[0]
            DY[1] = (Y[i] - Y[i - 1]) / h[1]

            # Standard PCHIP formula
            if np.sign(DY[0]) * np.sign(DY[1]) > 0.0:
                w1 = 2.0 * h[1] + h[0]
                w2 = h[1] + 2.0 * h[0]
                dY[0] = (w1 + w2) / (w1 / DY[0] + w2 / DY[1])
            else:
                dY[0] = 0.0

            #  Noncentered, shape-preserving, three-point formula:
            dY[1] = ((h[0] + 2.0 * h[1]) * DY[1] - h[1] * DY[0]) / (h[0] + h[1])
            if np.sign(dY[1]) != np.sign(DY[1]):
                dY[1] = 0.0
            elif (np.sign(DY[1]) != np.sign(DY[0])) and (
                np.abs(dY[1]) > np.abs(3 * DY[1])
            ):

                dY[1] = 3.0 * DY[1]

        else:
            # <--- X[i-2] < X[i-1] < x <= X[i] < X[i+1] --->
            h[0] = X[i - 1] - X[i - 2]  # Way faster to do this
            h[1] = X[i] - X[i - 1]  # than
            h[2] = X[i + 1] - X[i]  # diff(X(i-2:i+1))
            DY[0] = (Y[i - 1] - Y[i - 2]) / h[0]
            DY[1] = (Y[i] - Y[i - 1]) / h[1]
            DY[2] = (Y[i + 1] - Y[i]) / h[2]

            # Standard PCHIP formula
            for j in range(2):
                if np.sign(DY[j]) * np.sign(DY[j + 1]) > 0.0:
                    w1 = 2.0 * h[j + 1] + h[j]
                    w2 = h[j + 1] + 2.0 * h[j]
                    dY[j] = (w1 + w2) / (w1 / DY[j] + w2 / DY[j + 1])
                else:
                    dY[j] = 0.0

        # Polynomial coefficients for this piece
        s = x - X[i - 1]
        cY = (3.0 * DY[1] - 2.0 * dY[0] - dY[1]) / h[1]
        bY = (dY[0] - 2.0 * DY[1] + dY[1]) / h[1] ** 2

        # The following code evaluates the cubic interpolant, given `d` that
        # specifies the number of derivatives to take.
        # if d == 0:
        #     y = Y[i - 1] + s * (dY[0] + s * (cY + s * bY))
        # elif d == 1:  # first derivative
        #     y = dY[0] + s * (2 * cY + 3 * s * bY)
        # elif d == 2:  # second derivative
        #     y = 2 * cY + 6 * s * bY
        # elif d == 3:  # third derivative
        #     y = 6 * bY
        # else:
        #     y = 0.0
        # return y

    return s, dY[0], cY, bY
