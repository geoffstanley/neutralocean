"""Functions to handle updating the pressure / depth of a surface, by 
solving a nonlinear equation in the vertical dimension of each water column"""

import numpy as np
import numba as nb
import functools

from .fzero import guess_to_bounds, brent
from .ppinterp import ppval_1_nonan_two, valid_range_1_two


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
            return _vertsolve(*args, eos, ppc_fn, _potdens_1)

    elif ans_type == "anomaly":

        @nb.njit
        def f(*args):
            return _vertsolve(*args, eos, ppc_fn, _specvolanom_1)

    else:
        raise NameError(f'Unknown ans_type "{ans_type}"')

    return f


@nb.njit
def _vertsolve(S, T, P, ref, d0, tol_p, eos, ppc_fn, zero_func):

    shape2D = S.shape[0:-1]
    s = np.full(shape2D, np.nan)
    t = np.full(shape2D, np.nan)
    p = np.full(shape2D, np.nan)

    for n in np.ndindex(shape2D):

        # Select this water column
        Sn = S[n]
        Tn = T[n]
        Pn = P[n]

        k, K = valid_range_1_two(Sn, Pn)  # S and T have same nan-structure

        if K - k > 1:

            Pn = Pn[k:K]
            Sppc = ppc_fn(Pn, Sn[k:K])
            Tppc = ppc_fn(Pn, Tn[k:K])

            args = (Pn, Sppc, Tppc, ref, eos, d0)

            # Use mid-pressure as initial guess
            pn = (Pn[0] + Pn[-1]) * 0.5

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_func, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_func, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = ppval_1_nonan_two(p[n], Pn, Sppc, Tppc, 0)

    return s, t, p


@nb.njit
def _vertsolve_omega(s, t, p, S, T, P, ϕ, tol_p, eos, ppc_fn):
    # Note!  mutates s, t, p

    for n in np.ndindex(s.shape):
        ϕn = ϕ[n]
        if ϕn == 0.0:
            continue  # leave s,t,p unchanged (avoid creating errors of size tol_p)

        # Select this water column
        Sn = S[n]
        Tn = T[n]
        Pn = P[n]

        k, K = valid_range_1_two(Sn, Pn)  # S and T have same nan-structure
        if K - k > 1 and np.isfinite(ϕn):

            pn = p[n]
            Pn = Pn[k:K]

            # Build interpolant's coefficients for this water column
            Sppc = ppc_fn(Pn, Sn[k:K])
            Tppc = ppc_fn(Pn, Tn[k:K])

            # Evaluate difference between
            # (a) eos at location on the cast where the pressure or depth is p, and
            # (b) eos at location on the cast where the pressure or depth is pn
            # (where the surface currently is) plus the density perturbation d.
            # Part (b) is precomputed.  Here, eos always evaluated at the
            # pressure or depth of the original position, pn, i.e. we calculate
            # locally referenced potential density.
            args = (Pn, Sppc, Tppc, pn, eos, eos(s[n], t[n], pn) + ϕn)

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(_potdens_1, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(_potdens_1, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = ppval_1_nonan_two(p[n], Pn, Sppc, Tppc)

            else:
                # Ensure s,t,p all have the same nan structure
                s[n], t[n], p[n] = np.nan, np.nan, np.nan

        else:
            # ϕ is nan, or only one grid cell so cannot interpolate.
            # Ensure s,t,p all have the same nan structure
            s[n], t[n], p[n] = np.nan, np.nan, np.nan

    return None


@nb.njit
def _potdens_1(p, P, Sppc, Tppc, ref_p, eos, isoval=0.0):
    # Potential density referenced to `ref_p` evaluated at `p`, minus a given `isoval`.
    # Note: `P`, `Sppc`, `Tppc` must have no NaNs.
    s, t = ppval_1_nonan_two(p, P, Sppc, Tppc, 0)
    return eos(s, t, ref_p) - isoval


@nb.njit
def _specvolanom_1(p, P, Sppc, Tppc, ref, eos, isoval=0.0):
    # Specific volume (or in-situ density) anomaly with reference S and T given by `ref`
    # evaluated at `p`, minus a given `isoval`.
    # Note: `P`, `Sppc`, `Tppc` must have no NaNs.
    s, t = ppval_1_nonan_two(p, P, Sppc, Tppc)
    return eos(s, t, p) - eos(ref[0], ref[1], p) - isoval
