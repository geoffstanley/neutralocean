import numpy as np
import numba

from neutral_surfaces._zero import guess_to_bounds, brent
from neutral_surfaces._densjmd95 import rho
from neutral_surfaces._interp import linear_coefficients, linear_eval2


@numba.njit
def density(p, stp_args):
    # might not need this function at all
    s, t = linear_eval2(p, *stp_args)
    return rho(s, t, p)


@numba.njit
def func_zero(p, p_ref, d0, stp_args):
    s, t = linear_eval2(p, *stp_args)
    return rho(s, t, p_ref) - d0


@numba.njit
def solve(func, args, p_start, p_lb, p_ub, tol):
    lb, ub = guess_to_bounds(func, args, p_start, p_lb, p_ub)
    if np.isnan(lb):
        return np.nan, np.nan, np.nan
    p = brent(func, args, lb, ub, tol)
    s, t = linear_eval2(p, *args[2])
    return s, t, p


@numba.njit
def vertsolve1(p_start, p_ref, d0, stp_args, tol):
    args = (p_ref, d0, stp_args)
    pgrid = stp_args[0]
    if np.isnan(p_start):
        p_start = (pgrid[0] + pgrid[-1]) * 0.5
    return solve(func_zero, args, p_start, pgrid[0], pgrid[-1], tol)


@numba.njit
def vertsolve(p_start, p_ref, d0, ngood, stp_args, tol):
    s = np.empty(ngood.shape, dtype=np.float64)
    s.fill(np.nan)
    t = s.copy()
    p = s.copy()

    for n in np.ndindex(p.shape):
        k = ngood[n]
        if k > 1:
            tup = (*n, slice(k))
            # Unfortunately, we can't do the following:
            # stp_args1 = tuple([arg[tup] for arg in stp_args])
            # The tuple() constructor is not supported by numba, and leaving
            # stp_args1 as a list causes problems later.
            a = stp_args
            stp_args1 = (a[0][tup], a[1][tup], a[2][tup], a[3][tup], a[4][tup])
            s[n], t[n], p[n] = vertsolve1(
                p_start[n],
                p_ref[n],
                d0[n],
                stp_args1,
                tol,
            )
    return s, t, p


def process_arrays(s, t, p, axis=-1):
    # need a better name for this...
    if axis != -1 and axis != s.ndim - 1:
        s = np.moveaxis(s, axis, -1)
        t = np.moveaxis(t, axis, -1)
        if p.ndim == s.ndim:
            p = np.moveaxis(p, axis, -1)
    if p.ndim < s.ndim:
        p = np.broadcast_to(p, s.shape)
    s = np.require(s, dtype=np.float64, requirements="C")
    t = np.require(t, dtype=np.float64, requirements="C")
    p = np.require(p, dtype=np.float64, requirements="C")
    s_coef = linear_coefficients(p, s)
    t_coef = linear_coefficients(p, t)
    # Assume s and t have the same nan locations for missing
    # profiles or depths below the bottom.
    ngood = (~np.isnan(s)).sum(axis=-1)
    return ngood, (p, s, s_coef, t, t_coef)
