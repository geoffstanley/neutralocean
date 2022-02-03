"""Functions to handle updating the pressure / depth of a surface, by 
solving a nonlinear equation in the vertical dimension of each water column"""
import numpy as np
import numba as nb
import functools

from neutralocean.fzero import guess_to_bounds, brent
from neutralocean.ppinterp import ppval1_two


@functools.lru_cache(maxsize=10)
def _make_vertsolve(eos, ppc_fn, ans_type):

    if ans_type == "omega":

        @nb.njit
        def f(*args):
            _vertsolve_omega(*args, eos, ppc_fn)
            return None

    elif ans_type == "potential":

        @nb.njit
        def f(*args):
            return _vertsolve(*args, eos, ppc_fn, _zero_potential)

    elif ans_type == "anomaly":

        @nb.njit
        def f(*args):
            return _vertsolve(*args, eos, ppc_fn, _zero_anomaly)

    else:
        raise NameError(f'Unknown ans_type "{ans_type}"')

    return f


@nb.njit
def _vertsolve(S, T, P, n_good, ref, d0, tol_p, eos, ppc_fn, zero_func):

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

            Sppcn = ppc_fn(Pn, Sn)
            Tppcn = ppc_fn(Pn, Tn)

            # args = (Sn, Tn, Pn, ref, d0, eos, interp_fn)
            # args = (Sn, Sppcn, Tn, Tppcn, Pn, ref, d0, eos) # Actually slower than combining Sn into Sppcn
            args = (Sppcn, Tppcn, Pn, ref, d0, eos)

            # Use mid-pressure as initial guess
            pn = (Pn[0] + Pn[-1]) * 0.5

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_func, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_func, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                # s[n], t[n] = interp_fn(p[n], Pn, Sn, Tn)
                # s[n], t[n] = interp2_1d(p[n], Pn, Sn, Sppcn, Tn, Tppcn)
                s[n], t[n] = ppval1_two(p[n], Pn, Sppcn, Tppcn, 0)

    return s, t, p


@nb.njit
def _vertsolve_omega(s, t, p, S, T, P, n_good, ϕ, tol_p, eos, ppc_fn):
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
            pn = p[n]

            Sppcn = ppc_fn(Pn, Sn)
            Tppcn = ppc_fn(Pn, Tn)

            # Evaluate difference between (a) eos at location on the cast where the
            # pressure or depth is p, and (b) eos at location on the cast where the
            # pressure or depth is pin_p (where the surface currently is) plus the density
            # perturbation d.  Part (b) is precomputed as r0.  Here, eos always
            # evaluated at the pressure or depth of the original position, pin_p; this is
            # to calculate locally referenced potential density with reference pressure
            # pin_p.
            args = (Sppcn, Tppcn, Pn, pn, eos(s[n], t[n], pn) + ϕn, eos)

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(_zero_potential, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(_zero_potential, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = ppval1_two(p[n], Pn, Sppcn, Tppcn)

            else:
                # Ensure s,t,p all have the same nan structure
                s[n], t[n], p[n] = np.nan, np.nan, np.nan

        else:
            # ϕ is nan, or only one grid cell so cannot interpolate.
            # Ensure s,t,p all have the same nan structure
            s[n], t[n], p[n] = np.nan, np.nan, np.nan

    return None


# @nb.njit
# def _zero_potential(p, S, T, P, ref_p, isoval, eos, interp_fn):
#     # Evaluate the potential density in a given cast, minus a given isovalue
#     s, t = interp_fn(p, P, S, T)
#     return eos(s, t, ref_p) - isoval


# @nb.njit
# def _zero_potential(p, S, Sppc, T, Tppc, P, ref_p, isoval, eos):
#     # Evaluate the potential density in a given cast, minus a given isovalue
#     s, t = interp2_1d(p, P, S, Sppc, T, Tppc)
#     return eos(s, t, ref_p) - isoval


@nb.njit
def _zero_potential(p, Sppc, Tppc, P, ref_p, isoval, eos):
    # Evaluate the potential density in a given cast, minus a given isovalue
    s, t = ppval1_two(p, P, Sppc, Tppc, 0)
    return eos(s, t, ref_p) - isoval


@nb.njit
def _zero_anomaly(p, Sppc, Tppc, P, ref, isoval, eos):
    # Evaluate the specific volume (or in-situ density) anomaly in a given cast,
    # minus a given isovalue
    s, t = ppval1_two(p, P, Sppc, Tppc)
    return eos(s, t, p) - eos(ref[0], ref[1], p) - isoval
