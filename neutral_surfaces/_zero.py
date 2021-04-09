"""
This is a quick test to see whether brent and guess_to_bounds
can be decorated with "njit", along with the function
supplied to brent.  It includes 2 such functions, one
being quadratic and the other 6th order, so that brent should
have a slightly harder time with the second.

Results, running in ipython:

In [15]: run fzero_testjit.py
(-2.5999999999999988, -0.7737258300203049, -2.500000409065004)

In [16]: %timeit one_root()
169 ns ± 2.45 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

In [17]: %timeit one_root3()
176 ns ± 0.701 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

In [18]: %timeit myfun_univar(1.1)
100 ns ± 1.2 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

In [19]: %timeit myfun_univar3(1.1)
100 ns ± 0.976 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

The last two results are showing mostly the overhead of calling the jitted
function from Python.

The only change I had to make to enable njit to work was to use the nan from
numpy instead of using the Python float() function.  Probably using math.nan
would be identical.

"""

import numba
import numpy as np


eps = np.finfo(np.float64).eps


@numba.njit
def brent(f, a, b, t):

    fa = f(a)
    fb = f(b)
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

        fb = f(b)

        if (0.0 < fb and 0.0 < fc) or (fb <= 0.0 and fc <= 0.0):
            c = a
            fc = fa
            e = b - a
            d = e

    return b


@numba.njit
def guess_to_bounds(f, x, lb, ub):

    # Geometrically expand from the guess x; until a sign change is found

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

    fapos = f(a) > 0.0
    fbpos = f(b) > 0.0

    while True:
        if a > lb:
            # Move a left; & test for a sign change
            dxm *= 1.414213562373095
            a = max(x - dxm, lb)
            fapos = f(a) > 0.0
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
            fbpos = f(b) > 0.0
            if fapos != fbpos:  # fa & fb have different signs
                return (a, b)
        elif a == lb:  # also b .== ub; so cannot expand anymore
            if fapos != fbpos:  # one last test for sign change
                return (a, b)
            else:  # no sign change found
                return (nan, nan)
