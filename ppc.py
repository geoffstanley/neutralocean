import numpy as np


def linterp(X, Y, *x):
    """
    Julia conversion:  currently requires that Y is bigger than or equal to X.
    That is, X and Y must have the same dimensions, or X must be a 1D array.

    PPC_LINTERP  Linear Interpolant


    C = ppc_linterp(X,Y)
    builds the coefficients C of a piecewise linear function that
    interpolates each column of Y as a function of each column of X.

    y = ppc_linterp(X,Y,x)
    evaluates the piecewise linear function formed from C = ppc_linterp(X,Y)
    at each column of x. That is, y[l,n] interpolates Y[:,n], as a function
    of X[:,n], at x[l,n]. No extrapolation is done: when x[l,n] is out of
    bounds of X[:,n], y[l,n] is NaN.


    --- Input:
    X [K x N], the data sites
    Y [K x N], the data values at the the data sites


    --- Output:
    C [O x K-1 x N], the piecewise polynomial coefficients of order O


    --- Notes:
    X[:,n] must be monotonically increasing for all n.

    NaN's in X are treated as +Inf (and as such must come at the end of each
    column).

    Any input can have a singleton second dimension [N = 1], in which case
    that single value is used for each interpolation problem.

    Any dimension N can actually be higher-dimensional; so long as it has N
    elements.

    Even if L .== 1; x does need a leading singleton dimension.

    If L .== 1, this dimension is squeeze()'d out of y.

    Author    : Geoff Stanley
    Email     : g.stanley@unsw.edu.au
    Email     : geoffstanley@gmail.com
    """

    # Calculate slopes for each part of the piecewise (linear) polynomial
    slope = np.diff(Y, axis=-1) / np.diff(X, axis=-1)

    # Take the starting value for each part of the piecewise (linear) polynomial
    # Auto expand if X is larger than Y, and take NaN structure from slope
    start = Y[..., 0:-1] + np.where(np.isnan(slope), np.nan, 0)

    C = np.concatenate((slope[..., None], start[..., None]), axis=-1)

    if len(x) == 0:
        # Return coefficients of piecewise polynomial
        return C
    else:
        # Evaluate the piecewise polynomial; if requested
        return val(X, C, x[0])


def val(X, C, x):
    """
    Piecewise Polynomial Evaluation
    y = ppc_val(X, C, x)
    evaluates the piecewise polynomials whose coefficients are C and whose
    knots are X, at data sites x.

    --- Input:
    X (K), knots of the piecewise polynomials
    C (N, K-1, O), coefficients of the piecewise polynomial
    x (1), evaluation sites

    --- Output:
    y (N), the piecewise polynomial evaluated at x

    --- Notes:
    X[n] must be monotonically increasing for all n.
    NaN's in X are treated as +Inf (and as such must come at the end of each
    column).
    The dimension N can actually be higher-dimensional.

    --- Acknowledgements:
    This code is adapted from MATLAB's ppval.m
    Author    : Geoff Stanley
    Email     : g.stanley@unsw.edu.au
    Email     : geoffstanley@gmail.com
    """

    if type(x) == float:
        x = np.array(x)

    szC = C.shape
    O = szC[-1]     # Order of the piecewise polynomial
    K = X.shape[-1] # number of knots of the piecewise polynomials
    assert szC[-2] == K-1, "Second last dim of C must be one less than last dim of X"


    if X.ndim == 1 and x.shape == ():
        # Binary search to find i such that:
        # k = 0                 if x <= X[0], or
        # k = K                 if X[K-1] < x or isnan(x) or
        # X[k-1] < x <= X[k]    otherwise
        k = np.searchsorted(X, x)

        if k == 0:
            if x == X[0]:
                y = C[..., k, O-1]
            else: # x < X[0]
                y = np.repeat(np.nan, szC[0:-2])
        elif k == K: # X[K-1] < x or isnan(x)
            y = np.repeat(np.nan, szC[0:-2])
        else:
            # Evaluate this piece of the polynomial
            t = x - X[k-1] # Switch to local coordinates
            y = C[..., k-1, 0]
            for o in range(1, O):
                y = t * y + C[..., k-1, o]

    elif X.ndim == 1 and x.shape == szC[0:-2]:  # k has the same size as x, and as C excluding its last two dimensions
        k = np.searchsorted(X, x)
        y = np.empty(x.shape)
        for i in np.ndindex(x.shape):
            if k[i] == 0:
                if x[i] == X[0]:
                    y[i] = C[(*i, k[i], O-1)]
                else: # x < X[0]
                    y[i] = np.nan
            elif k[i] == K: # X[K-1] < x[i] or isnan(x[i])
                y[i] = np.nan
            else:
                # Evaluate this piece of the polynomial
                t = x[i] - X[k-1] # Switch to local coordinates
                y = C[(*i, k[i]-1, 0)]
                for o in range(1, O):
                    y[i] = t * y[i] + C[(*i, k[i]-1, o)]

    elif X.ndim > 1 and X.shape[0:-1] == szC[0:-2] and x.shape == ():
        y = np.empty(szC[0:-2])
        for i in np.ndindex(szC[0:-2]):
            Xi = X[(*i, slice(None))]
            k = np.searchsorted(Xi, x)
            if k == 0:
                if x == Xi[0]:
                    y[i] = C[(*i, k, O-1)]
                else: # x < X[0]
                    y[i] = np.nan
            elif k == K: # X[K-1] < x[i] or isnan(x[i])
                y[i] = np.nan
            else:
                # Evaluate this piece of the polynomial
                t = x - Xi[k-1] # Switch to local coordinates
                y[i] = C[(*i, k-1, 0)]
                for o in range(1, O):
                    y[i] = t * y[i] + C[(*i, k-1, o)]
    else:
        raise TypeError("This case of X, C, x is not done.")

    return y


def val2(X, C, D, x):
    """
    Piecewise Polynomial Evaluation, Twice
    """

    if type(x) == float:
        x = np.array(x)

    szC = C.shape
    O = szC[-1]     # Order of the piecewise polynomial
    K = X.shape[-1] # number of knots of the piecewise polynomials
    # assert szC[-2] == K-1, "Second last dim of C must be one less than last dim of X"
    # assert X.ndim == 1, "X must be 1D"
    # assert C.ndim == 2, "C must be 2D"
    # assert D.shape == C.shape, "C and D must have the same shape"
    # assert x.shape == (), "x must be scalar"

    # Binary search to find i such that:
    # k = 0                 if x <= X[0], or
    # k = K                 if X[K-1] < x or isnan(x) or
    # X[k-1] < x <= X[k]    otherwise
    k = np.searchsorted(X, x)

    if k == 0:
        if x == X[0]:
            y = C[k, O-1]
            z = D[k, O-1]
        else: # x < X[0]
            y = np.nan
            z = np.nan
    elif k == K: # X[K-1] < x or isnan(x)
        y = np.nan
        z = np.nan
    else:
        # Evaluate this piece of the polynomial
        t = x - X[k-1] # Switch to local coordinates
        y = C[k-1, 0]
        z = D[k-1, 0]
        for o in range(1, O):
            y = t * y + C[k-1, o]
            z = t * z + D[k-1, o]

    return (y,z)

# X = np.array([5, 15, 25, 30])
# Y = np.array([[1, 2, 3, 4], [1, 3, 6, 8]])
# C = linterp(X,Y)
# x = np.array([6, 2])
# y = ppc_val(X, C, x)
