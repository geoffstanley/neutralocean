"""
Functions for finding the zero of a univariate function.
"""

import numpy as np
import numba as nb


eps = np.finfo(np.float64).eps


@nb.njit
def brent_guess(f, x, A, B, t, args=()):
    """
    Find a zero of a function within a given range, starting from a guess

    Parameters
    ----------
    f : function
        Continuous function of a single variable.
    x : float
        initial guess for a root
    A, B : float
        Range within which to search, satisfying `A < B`
    t : float
        Tolerance for convergence.
    args : tuple
        Additional arguments, beyond the optimization argument, to be passed to `f`.
        Pass `()` when `f` is univariate.

    Returns
    -------
    float
        Value of `x` where `f(x) ~ 0`.

    """

    a, b = guess_to_bounds(f, x, A, B, args)
    return brent(f, a, b, t, args)


@nb.njit
def brent(f, a, b, t, args=()):
    """
    Find a zero of a univariate function within a given range

    This is a bracketed root-finding method, so `f(a)` and `f(b)` must differ in
    sign. If they do, a root is guaranteed to be found.

    Parameters
    ----------
    f : function
        Continuous function of a single variable.
    a, b : float
        Range within which to search, satisfying `a < b` and ideally `f(a) * f(b) <= 0`
    t : float
        Tolerance for convergence.
    args : tuple
        Additional arguments, beyond the optimization argument, to be passed to `f`.
        Pass `()` when `f` is univariate.

    Returns
    -------
    float
        Value of `x` where `f(x) ~ 0`.

    Notes
    -----
    `f` should be a `@numba.njit`'ed function (when this function is `njit`'ed).
    """

    # Protection against bad input search range
    if np.isnan(a) or np.isnan(b) or a > b:
        return np.nan

    fa = f(a, *args)
    fb = f(b, *args)

    # Protection against input range that doesn't have a sign change
    if fa * fb > 0:  # DEV note: check if this should be fa * fb >= 0
        return np.nan

    c = a
    fc = fa
    e = b - a
    d = e

    while True:
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol = 2.0 * eps * abs(b) + t
        m = 0.5 * (c - b)

        if abs(m) <= tol or fb == 0.0:
            break

        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if 0.0 < p:
                q = -q
            else:
                p = -p

            s = e
            e = d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                e = m
                d = e

        a = b
        fa = fb

        if tol < abs(d):
            b += d
        elif 0.0 < m:
            b += tol
        else:
            b -= tol

        fb = f(b, *args)

        if (0.0 < fb and 0.0 < fc) or (fb <= 0.0 and fc <= 0.0):
            c = a
            fc = fa
            e = b - a
            d = e

    return b


@nb.njit
def guess_to_bounds(f, x, A, B, args=()):
    """
    Search for a range containing a sign change, expanding geometrically
    outwards from the initial guess.

    This is used as a first step in zero-finding, providing a small search
    range for the Brent algorithm.

    Parameters
    ----------
    f : function
        Continuous function of a single variable
    x : float
        Central point for starting the search
    A, B : float
        Lower and upper bounds, containing `x`, within which to search for a zero.
    args : tuple
        Additional arguments beyond the optimization argument.
        Pass `()` when `f` is univariate.

    Returns
    -------
    a, b : float
        Lower and upper bounds within which `f(x)` changes sign.
    """

    nan = np.nan

    # Check value of f at bounds
    fa = f(A, *args)
    if fa == 0.0:
        return (A, A)

    fb = f(B, *args)
    if fb == 0.0:
        return (B, B)

    x = min(max(x, A), B)

    # initial distance to expand outward from x, in positive and negative directions
    dxp = (B - x) / 50
    dxm = (x - A) / 50

    # Set a = x, except when x is so close to A that machine roundoff makes dxm identically 0
    # which would lead to an infinite loop below.  In this case, set a = A.
    if dxm == 0:
        a = A
    else:
        a = x

    # Similarly, set b = x, except for machine precision problems.
    if dxp == 0:
        b = B
    else:
        b = x

    if a > A:
        fbpos = f(b, *args) > 0.0
    else:  # a == A
        if b == B:
            # So dxm == 0 and dxp == 0.  So A very nearly equals B, but could
            # have A != B due to machine precision problems
            fapos = f(a, *args) > 0.0
            fbpos = f(b, *args) > 0.0
        else:  # b < B
            fapos = f(a, *args) > 0.0

    while True:
        if a > A:
            # Move a left, and test for a sign change
            dxm *= 1.414213562373095
            a = max(x - dxm, A)
            fapos = f(a, *args) > 0.0
            if fapos != fbpos:  # fa and fb have different signs
                return (a, b)
        elif b == B:  # also a == A, so cannot expand anymore
            if fapos != fbpos:  # one last test for sign change
                return (a, b)
            else:  # no sign change found
                return (nan, nan)

        if b < B:
            # Move b right, and test for a sign change
            dxp *= 1.414213562373095
            b = min(x + dxp, B)
            fbpos = f(b, *args) > 0.0
            if fapos != fbpos:  # fa and fb have different signs
                return (a, b)
        elif a == A:  # also b == B, so cannot expand anymore
            if fapos != fbpos:  # one last test for sign change
                return (a, b)
            else:  # no sign change found
                return (nan, nan)
