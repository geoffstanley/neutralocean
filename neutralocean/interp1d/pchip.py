"""Kernels for Piecewise Cubic Hermite Interpolating Polynomials (PCHIPs)"""
import numpy as np
import numba as nb

# from .interp import interp1d


# def pchip_1d(x, X, Y, d=0, i=None):
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

#     return interp1d(x, X, Y, (pchip_i, pchip_dx_i, pchip_dxx_i, pchip_dxxx_i), d, i)


@nb.njit
def _pchip(x, X, Y, i):
    """
    The "kernel" of Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.

    Inputs and outputs analogous to `_linterp`.
    """

    s = x - X[i - 1]
    dY, cY, bY = _pchip_coeffs(X, Y, i)
    return Y[i - 1] + s * (dY + s * (cY + s * bY))


@nb.njit
def _pchip1(x, X, Y, i):
    """
    The "kernel" of the 1st derivative of PCHIP interpolation.

    Inputs and outputs analogous to `_linterp`.
    """
    s = x - X[i - 1]
    dY, cY, bY = _pchip_coeffs(X, Y, i)
    return dY + s * (2 * cY + 3 * s * bY)


@nb.njit
def _pchip2(x, X, Y, i):
    """
    The "kernel" of the 2nd derivative of PCHIP interpolation.

    Inputs and outputs analogous to `_linterp`.
    """
    s = x - X[i - 1]
    _, cY, bY = _pchip_coeffs(X, Y, i)
    return 2 * cY + 6 * s * bY


@nb.njit
def _pchip3(x, X, Y, i):
    """
    The "kernel" of the 3rd derivative of PCHIP interpolation.

    Inputs and outputs analogous to `_linterp`.
    """
    _, _, bY = _pchip_coeffs(X, Y, i)
    return 6 * bY


@nb.njit
def _pchipd(x, X, Y, i, d):

    if d == 0:
        return _pchip(x, X, Y, i)
    elif d == 1:
        return _pchip1(x, X, Y, i)
    elif d == 2:
        return _pchip2(x, X, Y, i)
    elif d == 3:
        return _pchip3(x, X, Y, i)
    else:
        return 0.0


@nb.njit
def _pchip_coeffs(X, Y, i):
    """
    Calculate the coefficients of a cubic interpolant

    Parameters
    ----------
    X, Y, i :
        As in `pchip_i`

    Returns
    -------
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

    return dY[0], cY, bY
