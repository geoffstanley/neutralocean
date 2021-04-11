"""
Functions for finding the zero of a univariate function.
"""

import numba
import numpy as np


eps = np.finfo(np.float64).eps


@numba.njit
def brent(f, args, a, b, t):
    """
    brent(f, a, b, t)

    Find the zero of a function within a given range using Brent's method.

    Parameters
    ----------
    f : function
        Continuous function of a single variable.
    a, b : float
        Range within which to search.
    t : float
        Tolerance for convergence.

    Returns
    -------
    float
        Value of x where f(x) ~ 0.

    """

    fa = f(a, *args)
    fb = f(b, *args)
    if not fa * fb <= 0:
        return np.nan  # Protection against bad input search range.
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


@numba.njit
def guess_to_bounds(f, args, x, lb, ub):
    """
    guess_to_bounds(f, x, lb, ub)

    Search for a range containing a sign change.

    This is used as a first step in zero-finding, providing a small search
    range for the Brent algorithm.

    Parameters
    ----------
    f : function
        Continuous function of a single variable
    x : float
        Central point for starting the search
    lb, ub : float
        Lower and upper bounds, containing x, within which to search.

    Returns
    -------
    lb, ub : float
        Lower and upper bounds within which f(x) changes sign.

    Notes
    -----
    The search expands geometrically outwards from the guess *x*.
    """

    nan = np.nan

    x = min(max(x, lb), ub)

    # bounds are given
    dxp = (ub - x) / 50
    dxm = (x - lb) / 50

    # Set a = x; except when x is so close to lb that machine roundoff makes dxm identically 0
    # which would lead to an infinite loop below.  In this case; set a = lb.
    if dxm == 0:
        a = lb
    else:
        a = x

    # Similarly; set b = x; except for machine precision problems.
    if dxp == 0:
        b = ub
    else:
        b = x

    fapos = f(a, *args) > 0.0
    fbpos = f(b, *args) > 0.0

    while True:
        if a > lb:
            # Move a left; & test for a sign change
            dxm *= 1.414213562373095
            a = max(x - dxm, lb)
            fapos = f(a, *args) > 0.0
            if fapos != fbpos:  # fa & fb have different signs
                return (a, b)
        elif b == ub:  # also a .== lb; so cannot expand anymore
            if fapos != fbpos:  # one last test for sign change
                return (a, b)
            else:  # no sign change found
                return (nan, nan)

        if b < ub:
            # Move b right; & test for a sign change
            dxp *= 1.414213562373095
            b = min(x + dxp, ub)
            fbpos = f(b, *args) > 0.0
            if fapos != fbpos:  # fa & fb have different signs
                return (a, b)
        elif a == lb:  # also b .== ub; so cannot expand anymore
            if fapos != fbpos:  # one last test for sign change
                return (a, b)
            else:  # no sign change found
                return (nan, nan)
