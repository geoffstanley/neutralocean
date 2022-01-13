"""Functions to handle updating the pressure / depth of a surface, by 
solving a nonlinear equation in the vertical dimension of each water column"""
import numpy as np
import numba
import functools

from neutralocean.fzero import guess_to_bounds, brent
from neutralocean.interp_ppc import val2_0d


@functools.lru_cache(maxsize=10)
def _make_vertsolve(eos, ans_type):

    if ans_type == "omega":

        def f(*args):
            _vertsolve_omega(*args, eos)
            return None

    elif ans_type == "sigma":

        def f(*args):
            return _vertsolve(*args, eos, _zero_sigma)

    elif ans_type == "delta":

        def f(*args):
            return _vertsolve(*args, eos, _zero_delta)

    else:
        raise NameError(f'Unknown ans_type "{ans_type}"')

    return f


@numba.njit
def _vertsolve(S, T, P, Sppc, Tppc, n_good, ref, d0, tol_p, eos, zero_func):

    s = np.full(n_good.shape, np.nan)
    t = np.full(n_good.shape, np.nan)
    p = np.full(n_good.shape, np.nan)

    for n in np.ndindex(s.shape):
        k = n_good[n]
        if k > 1:

            # Select this water column
            tup = (*n, slice(k))
            Sn = S[tup]
            Tn = T[tup]
            Pn = P[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]

            args = (Sn, Tn, Pn, Sppcn, Tppcn, ref, d0, eos)

            # Use mid-pressure as initial guess
            pn = (Pn[0] + Pn[-1]) * 0.5

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_func, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_func, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

    return s, t, p


@numba.njit
def _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p, eos):
    # Note!  mutates s, t, p

    for n in np.ndindex(s.shape):
        ϕn = ϕ[n]
        k = n_good[n]
        if k > 1 and np.isfinite(ϕn):

            # Select this water column
            tup = (*n, slice(k))
            Sn = S[tup]
            Tn = T[tup]
            Pn = P[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]
            pn = p[n]

            # Evaluate difference between (a) eos at location on the cast where the
            # pressure or depth is p, and (b) eos at location on the cast where the
            # pressure or depth is pin_p (where the surface currently is) plus the density
            # perturbation d.  Part (b) is precomputed as r0.  Here, eos always
            # evaluated at the pressure or depth of the original position, pin_p; this is
            # to calculate locally referenced potential density with reference pressure
            # pin_p.
            args = (Sn, Tn, Pn, Sppcn, Tppcn, pn, eos(s[n], t[n], pn) + ϕn, eos)

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(_zero_sigma, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(_zero_sigma, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

            else:
                # Ensure s,t,p all have the same nan structure
                s[n], t[n], p[n] = np.nan, np.nan, np.nan

        else:
            # ϕ is nan, or only one grid cell so cannot interpolate.
            # Ensure s,t,p all have the same nan structure
            s[n], t[n], p[n] = np.nan, np.nan, np.nan

    return None


@numba.njit
def _zero_sigma(p, S, T, P, Sppc, Tppc, ref_p, isoval, eos):
    # Evaluate the potential density in a given cast, minus a given isovalue
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    return eos(s, t, ref_p) - isoval


@numba.njit
def _zero_delta(p, S, T, P, Sppc, Tppc, ref, isoval, eos):
    # Evaluate the specific volume (or in-situ density) anomaly in a given cast,
    # minus a given isovalue
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    return eos(s, t, p) - eos(ref[0], ref[1], p) - isoval
