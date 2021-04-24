import ctypes
import functools

import numpy as np
import numba

import gsw

from neutral_surfaces._densjmd95 import rho_bsq
from neutral_surfaces._densjmd95 import rho as rho_jmd95
from neutral_surfaces import interp_ppc
from neutral_surfaces._zero import guess_to_bounds, brent

from neutral_surfaces._zero_freeze import make_brent_funcs


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


def pot_dens_surf(S, T, P, ref, target, eos="jmd95", axis=-1, tol=1e-4):
    """
    ...format and fill in the rest later, but the new things are:

    ref : p_ref, or (s_ref, t_ref)
    eos : string, key into dictionary of eos functions

    Hence, depending on ref, the same function works for potential density
    or specific volume anomaly surfaces.
    """
    S, T, P, n_good = process_arrays(S, T, P, axis=axis)
    Sppc = interp_ppc.linear_coefficients(P, S)
    Tppc = interp_ppc.linear_coefficients(P, T)

    func_sigma, sigma_vertsolve = make_sigma_workers(eosdict[eos], ref)

    if isinstance(target, tuple):
        p0 = target[-1]  # target pressure
        n0 = target[:-1]  # index to water column which intersects the surface
        #                   at the target pressure

        # Choose iso-value that will intersect cast n0 at p0.
        d0 = func_sigma(p0, P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], ref, 0.0)
    else:
        d0 = target

    # Solve non-linear root finding problem in each cast
    s, t, p = sigma_vertsolve(P, S, Sppc, T, Tppc, n_good, ref, d0, tol)
    return s, t, p


@functools.lru_cache(maxsize=10)
def make_sigma_workers(rho, ref):
    if np.iterable(ref):

        @numba.njit
        def func_sigma(p, P, S, Sppc, T, Tppc, ref, d0):
            # s, t = linear_eval2(p, *stp_args)
            s, t = interp_ppc.val2_0d(P, S, Sppc, T, Tppc, p)
            return rho(s, t, p) - rho(ref[0], ref[1], p) - d0

    else:

        @numba.njit
        def func_sigma(p, P, S, Sppc, T, Tppc, ref, d0):
            # s, t = linear_eval2(p, *stp_args)
            s, t = interp_ppc.val2_0d(P, S, Sppc, T, Tppc, p)
            return rho(s, t, ref) - d0

    guess_to_bounds, brent = make_brent_funcs(func_sigma)

    @numba.njit
    def sigma_vertsolve(P, S, Sppc, T, Tppc, n_good, ref, d0, tol):

        s = np.full(n_good.shape, np.nan, dtype=np.float64)
        t = s.copy()
        p = s.copy()

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

                # Use mid-depth as initial guess
                pn = (Pn[0] + Pn[-1]) * 0.5

                # Search for a sign-change, expanding outward from an initial guess
                lb, ub = guess_to_bounds(
                    (Pn, Sn, Sppcn, Tn, Tppcn, ref, d0), pn, Pn[0], Pn[-1]
                )

                if not np.isnan(lb):
                    # A sign change was discovered, so a root exists in the interval.
                    # Solve the nonlinear root-finding problem using Brent's method
                    p[n] = brent((Pn, Sn, Sppcn, Tn, Tppcn, ref, d0), lb, ub, tol)

                    # Interpolate S and T onto the updated surface
                    s[n], t[n] = interp_ppc.val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

            # else:
            # only one grid cell so cannot interpolate.
            # This will ensure s,t,p all have the same nan structure

        return s, t, p

    return func_sigma, sigma_vertsolve


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
