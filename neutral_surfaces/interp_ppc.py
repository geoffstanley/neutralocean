"""
Functions for the particular usage of interpolation required by neutral
surface calculations.
"""

"""
Adapted from 'Piecewise Polynomial Calculus' available on the MATLAB File 
Exchange at https://mathworks.com/matlabcentral/fileexchange/73114-piecewise-polynomial-calculus
This adaptation avoids (essentially) copying the Y array to a slice of Yppc,
simply to save time and memory.  This does require that both Y and Yppc 
together be passed around, as together they define the piecewise polynomial
coefficients. 
"""

import numpy as np
import numba

# from numba import float64, intc


def linear_coeffs(X, Y):
    Yppc = np.diff(Y, axis=-1) / np.diff(X, axis=-1)
    return np.reshape(Yppc, (*Yppc.shape, 1))  # add trailing singleton dim


@numba.njit
def diff_1d(x):
    d = np.empty(x.size - 1, dtype=x.dtype)
    for i in range(d.size):
        d[i] = x[i + 1] - x[i]
    return d


@numba.njit
def pchip_coeffs_0d(X, Y):
    # X and Y are vectors

    # Note, np.diff does not work with numba.njit, here.
    # Also, using diff_1d is faster than np.ediff1d
    h = diff_1d(X)  # distance between data sites
    δ = diff_1d(Y) / h  # slope of linear interpolant between data sites

    K = X.size  # Number of data points included possible NaN's

    # Count the number of consecutive valid (non-NaN) data points.
    k = K
    for i in range(K):
        if np.isnan(X[i]) or np.isnan(Y[i]):
            k = i
            break

    d = np.zeros(k)  # slope of interpolant at data sites
    C = np.full((K - 1, 3), np.nan)

    if k > 2:

        # Calculate PCHIP slopes
        #  Slopes at end points:
        #   Set d[0] and d[k-1] via non-centered, shape-preserving three-point formulae.
        #  Slopes at interior points:
        #   d[i] = weighted average of δ[i-1] and δ[i] when they have the same sign.
        #   d[i] = 0 when δ[i-1] and δ[i] have opposites signs or either is zero.

        d[0] = ((2 * h[0] + h[1]) * δ[0] - h[0] * δ[1]) / (h[0] + h[1])
        if np.sign(d[0]) != np.sign(δ[0]):
            d[0] = 0
        elif (np.sign(δ[0]) != np.sign(δ[1])) and (abs(d[0]) > abs(3 * δ[0])):
            d[0] = 3 * δ[0]

        for i in range(1, k - 1):
            if np.sign(δ[i - 1]) * np.sign(δ[i]) > 0:
                w1 = h[i - 1] + 2 * h[i]
                w2 = 2 * h[i - 1] + h[i]
                d[i] = (w1 + w2) / (w1 / δ[i - 1] + w2 / δ[i])
            else:
                d[i] = 0

        d[k - 1] = ((2 * h[k - 2] + h[k - 3]) * δ[k - 2] - h[k - 2] * δ[k - 3]) / (
            h[k - 2] + h[k - 3]
        )
        if np.sign(d[k - 1]) != np.sign(δ[k - 2]):
            d[k - 1] = 0
        elif (np.sign(δ[k - 2]) != np.sign(δ[k - 3])) and (
            abs(d[k - 1]) > abs(3 * δ[k - 2])
        ):
            d[k - 1] = 3 * δ[k - 2]

        # Build piecewise cubic Hermite polynomial
        for i in range(k - 1):
            dzzdx = (δ[i] - d[i]) / h[i]
            dzdxdx = (d[i + 1] - δ[i]) / h[i]
            C[i, 0] = (dzdxdx - dzzdx) / h[i]
            C[i, 1] = 2 * dzzdx - dzdxdx
            C[i, 2] = d[i]

    elif k == 2:
        # Special case: use linear interpolation.

        δ = (Y[1] - Y[0]) / (X[1] - X[0])

        C[0, 0] = 0
        C[0, 1] = 0
        C[0, 2] = δ

    return C


@numba.njit
def pchip_coeffs_nd(X, Y):

    C = np.full((*Y.shape[0:-1], Y.shape[-1] - 1, 3), np.nan)
    for n in np.ndindex(Y.shape[0:-1]):
        C[n] = pchip_coeffs_0d(X[n], Y[n])

    return C


def pchip_coeffs(X, Y):
    if X.ndim == 1 and X.size == Y.shape[-1]:
        X = np.broadcast_to(X, Y.shape)
    elif Y.ndim == 1 and Y.size == X.shape[-1]:
        Y = np.broadcast_to(Y, X.shape)
    elif X.shape != Y.shape:
        raise ValueError(
            "X and Y must have the same dimensions, or one "
            "of them must be a vector matching the other's last dimension; "
            f"found X's shape {X.shape} and Y's shape {Y.shape}."
        )

    return pchip_coeffs_nd(X, Y)


def val(X, Y, Yppc, x, d=0):
    nk = X.shape[-1]
    # assert nk == Y.shape[-1], "X and Y must have same size last dimension"
    # assert Y.shape[0:-1] == Yppc.shape[0:-1], "Y and Yppc must have same leading dimensions"
    # assert Yppc.shape[-1] == nk - 1, "Last dimension of Yppc must be one less than that of Y"
    if X.ndim > 1:
        shape = X.shape[0:-1]
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match those of X"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif Y.ndim > 1:
        shape = Y.shape[0:-1]
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match those of Y"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    else:
        shape = x.shape
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match the shape of x"
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match the shape of x"

    X = np.broadcast_to(X, (*shape, nk))
    Y = np.broadcast_to(Y, (*shape, Y.shape[-1]))
    x = np.broadcast_to(x, shape)
    Yppc = np.broadcast_to(Yppc, (*shape, nk - 1, Yppc.shape[-1]))

    return val_nd(X, Y, Yppc, x, d)


@numba.njit
def val_nd(X, Y, Yppc, x, d):
    y = np.empty(x.shape, dtype=np.float64)
    if d == 0:
        for n in np.ndindex(x.shape):
            y[n] = val_0d(X[n], Y[n], Yppc[n], x[n])
    else:
        for n in np.ndindex(x.shape):
            y[n] = dval_0d(X[n], Y[n], Yppc[n], x[n], d)
    return y


@numba.njit
def val_0d(X, Y, Yppc, x):
    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan

    if x == X[0]:
        return Y[0]

    # i = searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0]
    #   i = len(X)              if X[-1] < x or np.isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    # Having guaranteed X[0] < x <= X[-1] and x is not nan, then
    #   X[i-1] < x <= X[i]  and  1 <= i <= len(X)-1  in all cases.
    # Subtract 1 so X[i] < x <= X[i+1]  and  0 <= i <= len(X)-2
    i = np.searchsorted(X, x) - 1

    return val_0d_i(X, Y, Yppc, x, i)


@numba.njit
def dval_0d(X, Y, Yppc, x, d):
    # Assumes d is an integer greater than zero.

    # if d == 0:
    #     return val_0d(X, Y, Yppc, x)

    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan

    if x == X[0]:
        p = 1.0  # Build integer multiplying the coefficient
        for a in range(1, d + 1):
            p *= a
        # p = np.prod(np.arange(1, d + 1))  # slow!
        return Yppc[0, -d] * p

    i = np.searchsorted(X, x) - 1

    return dval_0d_i(X, Y, Yppc, x, d, i)


@numba.njit
def val_0d_i(X, Y, Yppc, x, i):
    #  Like val_0d, but provide i such that  X[i] <= x <= X[i+1]

    dx = x - X[i]

    # Evaluate polynomial, using nested multiplication.
    # E.g. the cubic case is:
    # y = dx^3 * Yppc[i,0] + dx^2 * Yppc[i,1] + dx * Yppc[i,2] + Y[i]
    #   = dx * (dx * (dx * Yppc[i,0] + Yppc[i,1]) + Yppc[i,2]) + Y[i]
    y = 0.0
    for o in range(0, Yppc.shape[-1]):
        y = y * dx + Yppc[i, o]
    y = y * dx + Y[i]

    return y


@numba.njit
def dval_0d_i(X, Y, Yppc, x, d, i):
    #  Like dval_0d, but provide i such that  X[i] <= x <= X[i+1]
    # Assumes d is an integer greater than zero.

    # if d == 0:
    #     return val_0d_i(X, Y, Yppc, x, i)

    degree = Yppc.shape[1]

    dx = x - X[i]  # dx > 0 guaranteed

    # Evaluate polynomial derivative, using nested multiplication.
    # E.g. the second derivative of the cubic case is:
    # y = 6 * dx * Yppc[i,0] + 2 * Yppc[i,1]
    y = 0.0
    for o in range(0, degree - d + 1):
        p = 1.0  # Build integer multiplying the coefficient
        for a in range(degree - o - d + 1, degree - o + 1):
            p *= a
        # p = np.prod(np.arange(degree - o - d + 1, degree - o + 1))  # slow!
        y = y * dx + Yppc[i, o] * p

    return y


# Similar routines to evaluate two piecewise polynomials.  These are faster
# than calling the val routines twice, because these only call np.searchsorted
# once.
def val2(X, Y, Yppc, Z, Zppc, x, d=0):
    nk = X.shape[-1]
    # assert nk == Y.shape[-1], "X and Y must have same size last dimension"
    # assert Y.shape[0:-1] == Yppc.shape[0:-1], "Y and Yppc must have same leading dimensions"
    # assert Yppc.shape[-1] == nk - 1, "Last dimension of Yppc must be one less than that of Y"
    # assert Z.shape == Y.shape, "Y and Z must have the same shape"
    # assert Zppc.shape == Yppc.shape, "Yppc and Zppc must have the same shape"
    if X.ndim > 1:
        shape = X.shape[0:-1]
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match those of X"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    elif Y.ndim > 1:
        shape = Y.shape[0:-1]
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match those of Y"
        # assert np.ndim(x) == 0 or len(x) == 1 or x.shape == shape, "x must be scalar or match leading dimensions of X"
    else:
        shape = x.shape
        # assert X.ndim == 1 or X.shape[0:-1] == shape, "X must be 1D or its leading dimensions must match the shape of x"
        # assert Y.ndim == 1 or Y.shape[0:-1] == shape, "Y must be 1D or its leading dimensions must match the shape of x"

    X = np.broadcast_to(X, (*shape, nk))
    Y = np.broadcast_to(Y, (*shape, Y.shape[-1]))
    Z = np.broadcast_to(Z, (*shape, Z.shape[-1]))
    x = np.broadcast_to(x, shape)
    Yppc = np.broadcast_to(Yppc, (*shape, nk - 1, Yppc.shape[-1]))
    Zppc = np.broadcast_to(Zppc, (*shape, nk - 1, Zppc.shape[-1]))

    return val2_nd(X, Y, Yppc, Z, Zppc, x, d)


@numba.njit
def val2_nd(X, Y, Yppc, Z, Zppc, x, d):
    y = np.empty(x.shape, dtype=np.float64)
    z = np.empty(x.shape, dtype=np.float64)
    if d == 0:
        for n in np.ndindex(x.shape):
            y[n], z[n] = val2_0d(X[n], Y[n], Yppc[n], Z[n], Zppc[n], x[n])
    else:
        for n in np.ndindex(x.shape):
            y[n], z[n] = dval2_0d(X[n], Y[n], Yppc[n], Z[n], Zppc[n], x[n], d)
    return y, z


@numba.njit
def val2_0d(X, Y, Yppc, Z, Zppc, x):
    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan, np.nan

    if x == X[0]:
        return Y[0], Z[0]

    i = np.searchsorted(X, x) - 1
    y = val_0d_i(X, Y, Yppc, x, i)
    z = val_0d_i(X, Z, Zppc, x, i)
    return y, z


@numba.njit
def dval2_0d(X, Y, Yppc, Z, Zppc, x, d):

    # if d == 0:
    #     return val2_0d(X, Y, Yppc, Z, Zppc, x)

    if np.isnan(x) or x < X[0] or X[-1] < x:
        return np.nan, np.nan

    if x == X[0]:
        p = 1.0  # Build integer multiplying the coefficient
        for a in range(1, d + 1):
            p *= a
        # p = np.prod(np.arange(1, d + 1))  # slow!
        return Yppc[0, -d] * p, Zppc[0, -d] * p

    i = np.searchsorted(X, x) - 1
    y = dval_0d_i(X, Y, Yppc, x, d, i)
    z = dval_0d_i(X, Z, Zppc, x, d, i)
    return y, z


def deriv(X, Y, Yppc, d=1):
    # Differentiate a piecewise polynomial.
    #
    #
    # YX, YXppc = deriv(X, Y, Yppc, d)
    # returns the coefficients (YX, YXppc) for the piecwise polynomial that is the d'th
    # derivative of the piecewise polynomial whose coefficients are (Y, Yppc).  Both
    # piecewise polynomials have knots given by X.
    #
    #
    # --- Input:
    # X [N, K] or [K], knots of the piecewise polynomials
    # Y [N, K-1], values of the piecewise polynomial at the knots
    # Yppc [N, K-1, o-1], first and higher order coefficients of the piecewise polynomial
    # d,        degree of the derivatives
    #
    #
    # --- Output:
    # YX, [N, K-1],  values of the derivative's piecewise polynomial at the knots
    # YXppc [N, K-1, o-1-d], first and higher order coefficients of the derivative's piecewise polynomial
    #
    #
    # --- Notes:
    # X[n,:] must be monotonically increasing for all n.
    #
    # NaN's in X are treated as +Inf (and as such must come at the end of each row).
    #
    # The dimension `N` can actually be higher-dimensional.

    o = Yppc.shape[-1] + 1  # Order of the piecewise polynomial

    if not (np.isreal(d) and np.mod(d, 1) == 0 and d >= 0):
        raise TypeError("d must be a non-negative integer")

    if d == 0:
        YX = Y.copy()
        YXppc = Yppc.copy()
    elif d < o:  # number of derivatives < order of polynomial

        # Build coefficients that multiply the surviving terms.
        # E.g. o = 4 (quartic polynomial) and d=2 (take second derivative)
        # then coeffs = [12, 6, 2] = [4, 3, 2] * [3, 2, 1]
        coeffs = np.arange(o - d, 0, -1)
        for j in range(1, d):
            coeffs = coeffs * np.arange(o - d + j, j, -1)

        YX = Yppc[..., -d] * coeffs[-1]
        if d == o - 1:
            # output is a piecewise constant polynomial.  But ensure YXppc has
            # trailing dimension that isn't 0
            YXppc = np.zeros((*Yppc.shape[0:-1], 1))
        else:
            YXppc = Yppc[..., 0:-d] * coeffs[0:-1]
    else:  # number of derivatives >= order of polynomial, so annihilates the polynomial

        YX = Y * 0.0  # maintain NaN structure of Y
        YXppc = np.zeros((*Yppc.shape[0:-1], 1))

    return YX, YXppc
