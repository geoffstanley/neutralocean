import ctypes
import functools

import numpy as np
import numba

import gsw

# from neutral_surfaces._densjmd95 import rho_bsq
from neutral_surfaces._densjmd95 import rho as rho_jmd95
from neutral_surfaces import interp_ppc
from neutral_surfaces._zero import guess_to_bounds, brent


# The following shows how we can access the C library scalar functions that
# are used by GSW-Python, since it is much faster for our jit functions to
# go straight to them rather than going via the Python ufunc wrappers.
dllname = gsw._gsw_ufuncs.__file__
gswlib = ctypes.cdll.LoadLibrary(dllname)
rho_gsw_ctypes = gswlib.gsw_rho  # In-situ density.
rho_gsw_ctypes.restype = ctypes.c_double
rho_gsw_ctypes.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)


# Wrapping the ctypes function with a jit reduces call overhead and makes it
# hashable, which is required for the caching we do via the lru_cache
# decorator.
@numba.njit
def rho_gsw(s, t, p):
    return rho_gsw_ctypes(s, t, p)


# You can add more; and a user can add to this dictionary in an interactive
# session or script, most simply by exposing it at the package level.
eosdict = dict(jmd95=rho_jmd95, gsw=rho_gsw)


def process_arrays(S, T, P, axis=-1):
    # need a better name for this...
    if axis not in (-1, S.ndim - 1):
        S = np.moveaxis(S, axis, -1)
        T = np.moveaxis(T, axis, -1)
        if P.ndim == S.ndim:
            P = np.moveaxis(P, axis, -1)
    if P.ndim < S.ndim:
        P = np.broadcast_to(P, S.shape)
    S = np.require(S, dtype=np.float64, requirements="C")
    T = np.require(T, dtype=np.float64, requirements="C")
    P = np.require(P, dtype=np.float64, requirements="C")
    # Assume S and T have the same nan locations for missing
    # profiles or depths below the bottom.

    n_good = find_first_nan(S)

    return S, T, P, n_good


def approx_neutral_surf(
    S,
    T,
    P,
    ans_type,
    target,
    ref=None,
    eos="jmd95",
    axis=-1,
    tol=1e-4,
    grav=None,
    rho_c=None,
):
    """
    ...format and fill in the rest later, but the new things are

    ans_type : string, either "sigma" or "delta"
    ref : p_ref, or (s_ref, t_ref)
    eos : string (key into dictionary of eos functions) or a callable

    """
    S, T, P, n_good = process_arrays(S, T, P, axis=axis)
    Sppc = interp_ppc.linear_coefficients(P, S)
    Tppc = interp_ppc.linear_coefficients(P, T)

    # Select equation of state and make its Boussinesq version if needed
    if isinstance(eos, str) and eos in eosdict.keys():
        eos = eosdict[eos]

    if not callable(eos):
        raise TypeError(
            'eos must be a function or a string that is either "sigma" or "delta"'
        )

    if grav != None and rho_c != None:
        z_to_p = 1e-4 * grav * rho_c
        eos = make_bsq(eos, z_to_p)

    func, vertsolve = make_workers(eos, ans_type)

    if isinstance(target, tuple):
        p0 = target[-1]  # target pressure
        n0 = target[:-1]  # index to water column which intersects the surface
        #                   at the target pressure

        if ans_type == "sigma":
            ref = target[-1]
        elif ans_type == "delta":
            # evaluate salinity and temperature at the chosen location
            s0, t0 = interp_ppc.val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], p0)
            ref = (s0, t0)

        # Choose iso-value that will intersect cast n0 at p0.
        d0 = func(p0, P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], ref, 0.0)
    else:
        d0 = target  # target sigma or delta isosurface value
        if ans_type == "sigma":
            if ref == None:
                raise TypeError(
                    'Must specify reference pressure by providing "ref" a scalar'
                )
        elif ans_type == "delta":
            if ref == None:
                raise TypeError(
                    'Must specify reference salinity and temperature by providing "ref" a 2-tuple of scalars'
                )

    # Solve non-linear root finding problem in each cast
    s = np.empty(n_good.shape, dtype=np.float64)
    t = np.empty(n_good.shape, dtype=np.float64)
    p = mid_pressure(P, n_good)
    vertsolve(s, t, p, P, S, Sppc, T, Tppc, n_good, ref, d0, tol)
    return s, t, p


@functools.lru_cache(maxsize=10)
def make_workers(eos, ans_type):
    if ans_type == "sigma":

        @numba.njit
        def func(p, P, S, Sppc, T, Tppc, ref, d0):
            s, t = interp_ppc.val2_0d(P, S, Sppc, T, Tppc, p)
            return eos(s, t, ref) - d0

    elif ans_type == "delta":

        @numba.njit
        def func(p, P, S, Sppc, T, Tppc, ref, d0):
            s, t = interp_ppc.val2_0d(P, S, Sppc, T, Tppc, p)
            return eos(s, t, p) - eos(ref[0], ref[1], p) - d0

    else:
        raise NameError(f'Unknown ans_type "{ans_type}"')

    @numba.njit
    def guess_to_bounds_eos(args, x, lb, ub):
        return guess_to_bounds(func, args, x, lb, ub)

    @numba.njit
    def brent_eos(args, a, b, t):
        return brent(func, args, a, b, t)

    @numba.njit
    def vertsolve(s, t, p, P, S, Sppc, T, Tppc, n_good, ref, d0, tol):
        # Note!  mutates s, t, p

        for n in np.ndindex(n_good.shape):
            k = n_good[n]
            if k > 1:

                # Select this water column
                tup = (*n, slice(k))
                Pn = P[tup]
                Sn = S[tup]
                Tn = T[tup]
                Sppcn = Sppc[tup]
                Tppcn = Tppc[tup]

                # Search for a sign-change, expanding outward from an initial guess
                lb, ub = guess_to_bounds_eos(
                    (Pn, Sn, Sppcn, Tn, Tppcn, ref, d0), p[n], Pn[0], Pn[-1]
                )

                if not np.isnan(lb):
                    # A sign change was discovered, so a root exists in the interval.
                    # Solve the nonlinear root-finding problem using Brent's method
                    p[n] = brent_eos((Pn, Sn, Sppcn, Tn, Tppcn, ref, d0), lb, ub, tol)

                    # Interpolate S and T onto the updated surface
                    s[n], t[n] = interp_ppc.val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])
                else:
                    s[n], t[n], p[n] = np.nan, np.nan, np.nan
            else:
                # only one grid cell so cannot interpolate.
                # This will ensure s,t,p all have the same nan structure
                s[n], t[n], p[n] = np.nan, np.nan, np.nan

        return None

    return func, vertsolve


@functools.lru_cache(maxsize=10)
def make_bsq(eos, z_to_p):
    @numba.njit
    def eos_bsq(s, t, z):
        return eos(s, t, z * z_to_p)

    return eos_bsq


@numba.njit
def find_first_nan(a):
    """
    find_first_nan(a)

    Find the index to the first nan in a along the last axis.
    If no nan's are present, the length of the last dimension is returned.
    """
    nk = a.shape[-1]
    k = np.full(a.shape[:-1], nk, dtype=np.int64)
    for n in np.ndindex(a.shape[0:-1]):
        for i in range(nk):
            if np.isnan(a[n][i]):
                k[n] = i
                break
    return k


@numba.njit
def mid_pressure(P, n_good):
    p = np.empty(P.shape[0:-1], dtype=np.float64)
    for n in np.ndindex(p.shape):
        k = n_good[n]
        if k > 0:
            p[n] = (P[n][0] + P[n][k - 1]) * 0.5
        else:
            p[n] = np.nan

    return p
