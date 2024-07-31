"""Neutral Trajectory and related functions"""

import numpy as np
import numba as nb

from neutralocean.ppinterp import make_pp, ppval_1_nonan_two
from neutralocean.eos.tools import load_eos
from neutralocean.fzero import guess_to_bounds, brent
from neutralocean.ppinterp import valid_range_1_two
from neutralocean.lib import _process_casts


@nb.njit
def _pot_dens_diff(p, sB, tB, pB, P, Sppc, Tppc, eos):
    # Evaluate difference between (a) eos at location on the cast (S, T, P)
    # where the pressure or depth is p, and (b) eos of the bottle (sB, tB, pB).
    # Here, eos is always evaluated at the average pressure or depth,
    # (p + pB)/2.
    s, t = ppval_1_nonan_two(p, P, Sppc, Tppc)
    p_avg = (pB + p) * 0.5
    return eos(sB, tB, p_avg) - eos(s, t, p_avg)


def ntp_bottle_to_cast(
    sB,
    tB,
    pB,
    S,
    T,
    P,
    tol_p=1e-4,
    interp="linear",
    eos="gsw",
    grav=None,
    rho_c=None,
):
    """Find the neutral tangent plane from a bottle to a cast

    Finds a point on the cast salinity, temperature, and pressure `(S, T, P)`
    where the salinity, temperature, and pressure `(s, t, p)` is neutrally
    related to a bottle of salinity, temperature and pressure `(sB, tB, pB)`.
    That is, the density of `(s, t, p_avg)` very nearly equals the density
    of `(sB, tB, p_avg)`, where `p_avg = (p + pB) / 2`.  Within `tol_p` of this
    point on the cast, there is a point where these two densities are exactly
    equal.

    Parameters
    ----------
    sB, tB, pB : float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth of the bottle

    S, T, P : 1D array of float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth of data points on the cast.  `P` must increase
        monotonically along its last dimension.

    Returns
    -------
    s, t, p : float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth at a point on the cast that is very nearly
        neutrally related to the bottle.

    Other Parameters
    ----------------
    tol_p : float, Default 1e-4

        Error tolerance in terms of pressure or depth when searching for a root
        of the nonlinear equation.  Units are the same as `P`.

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use `'linear'` for linear
        interpolation, and `'pchip'` for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `ppinterp`.

    eos : str or function, Default 'gsw'

        The equation of state for the density or specific volume as a function
        of `S`, `T`, and pressure (if non-Boussinesq) or depth(if Boussinesq).

        If a str, can be any of the strings accepted by `neutralocean.eos.tools.load_eos`.

        If a function, this should be `@numba.njit` decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    grav : float, Default None
        Gravitational acceleration [m s-2].  When non-Boussinesq, pass `None`.

    rho_c : float, Default None
        Boussinesq reference density [kg m-3].  When non-Boussinesq, pass `None`.

    """

    eos = load_eos(eos, "", grav, rho_c)
    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=False)

    k, K = valid_range_1_two(S, P)  # S and T have same nan-structure

    return _ntp_bottle_to_cast(sB, tB, pB, S, T, P, k, K, tol_p, eos, ppc_fn)


@nb.njit
def _ntp_bottle_to_cast(sB, tB, pB, S, T, P, k, K, tol_p, eos, ppc_fn):
    """Fast version of `ntp_bottle_to_cast`.

    Parameters
    ----------
    sB, tB, pB, S, T, P, tol_p : float
        See `ntp_bottle_to_cast`. Note that `S`, `T`, `P` should have no NaNs.

    k, K : int
        `k` is the index to the first finite value in `S + P`.
        `K` is the index to the first NaN value in `S + P` after `k`.
        If `S + P` is all NaN, then `k = len(S)`.
        If `S + P` is all NaN or if `(S + P)[k:]` is all finite, then `K = len(S)`.
        See `neutralocean.ppinterp.lib.valid_range_1`.

    eos : function
        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure or depth inputs.

        This function should be `@numba.njit` decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    ppc_fn : function
        Function that calculates piecewise polynomial coefficients, such as
        returned by `neutralocean.ppinterp.make_pp`

    Returns
    -------
    s, t, p : float
        See `ntp_bottle_to_cast`
    """

    if K - k > 1:
        # Trim data to valid range and build interpolants
        P = P[k:K]
        Sppc = ppc_fn(P, S[k:K])
        Tppc = ppc_fn(P, T[k:K])

        # return _ntp_bottle_to_cast_ppc(tol_p, sB, tB, pB, P, Sppc, Tppc, eos)
        p = _ntp_bottle_to_cast_ppc(tol_p, sB, tB, pB, P, Sppc, Tppc, eos)
        if np.isfinite(p):
            s, t = ppval_1_nonan_two(p, P, Sppc, Tppc)
        else:
            s, t = np.nan, np.nan

    else:  # K - k <= 1, so at most one valid data site. Can't interpolate that.
        s, t, p = np.nan, np.nan, np.nan

    return s, t, p


@nb.njit
def _ntp_bottle_to_cast_ppc(tol_p, sB, tB, pB, P, Sppc, Tppc, eos):
    """Fast version of `ntp_bottle_to_cast`, with pre-built interpolants.

    Parameters
    ----------
    sB, tB, pB, P, tol_p, eos :
        See `_ntp_bottle_to_cast`
    Sppc, Tppc : ndarray
        Piecewise Polynomial Coefficients for `S` and `T` of `_ntp_bottle_to_cast`.

    Returns
    -------
    p : float
        See `ntp_bottle_to_cast`
    """

    args = (sB, tB, pB, P, Sppc, Tppc, eos)

    # Search for a sign-change, expanding outward from an initial guess
    lb, ub = guess_to_bounds(_pot_dens_diff, pB, P[0], P[-1], args)

    if np.isfinite(lb):
        # A sign change was discovered, so a root exists in the interval.
        # Solve the nonlinear root-finding problem using Brent's method
        return brent(_pot_dens_diff, lb, ub, tol_p, args)

        # Interpolate S and T onto the updated surface
        # s, t = ppval_1_nonan_two(p, P, Sppc, Tppc)

    else:
        # s, t, p = np.nan, np.nan, np.nan
        return np.nan

    # return s, t, p


def neutral_trajectory(
    S,
    T,
    P,
    p0,
    vert_dim=-1,
    tol_p=1e-4,
    interp="linear",
    eos="gsw",
    grav=None,
    rho_c=None,
):
    """Calculate a neutral trajectory through a sequence of casts.

    Given a sequence of casts with hydrographic properties `(S, T, P)`, calculate
    a neutral trajectory starting from the first cast at pressure `p0`, or
    starting from a bottle prior to the first cast with hydrographic properties
    `(s0, t0, p0)`.

    Parameters
    ----------
    S, T, P : 2D ndarray or xarray

        1D data specifying the practical / Absolute salinity, and potential /
        Conservative temperature, and pressure / depth down a 1D sequence of casts.
        The first dimension specifies the cast number, while the second provides
        data on that cast; e.g. `S[i, :]` is the salinity down cast `i`.

    p0 : float

        The pressure / depth at which to begin the neutral trajectory on the first cast

    Returns
    -------
    s, t, p : 1D ndarray

        practical / Absolute Salinity, potential / Conservative Temperature,
        and pressure / depth along the neutral trajectory.

    Other Parameters
    ----------------
    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes` in `potential_surf`.

    tol_p, interp, eos, grav, rho_c :

        See `ntp_bottle_to_cast`

    """

    eos = load_eos(eos, "", grav, rho_c)
    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=False)
    S, T, P = _process_casts(S, T, P, vert_dim)

    nc, nk = S.shape
    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)) || all(size(P) == [nk, 1]), 'P must be [nk,nc] or [nk,1]')

    s = np.full(nc, np.nan)
    t = np.full(nc, np.nan)
    p = np.full(nc, np.nan)

    # Loop over casts
    for c in range(0, nc):
        Sc = S[c, :]
        Tc = T[c, :]
        Pc = P[c, :]
        k, K = valid_range_1_two(Sc, Pc)

        if c == 0:
            # Evaluate S and T on first cast at p0
            Sppc = ppc_fn(Pc[k:K], Sc[k:K])
            Tppc = ppc_fn(Pc[k:K], Tc[k:K])
            s[0], t[0] = ppval_1_nonan_two(p0, Pc[k:K], Sppc, Tppc)
            p[0] = p0
        else:
            # Make a neutral connection from previous bottle to the cast (S[c,:], T[c,:], P[c,:])
            s[c], t[c], p[c] = _ntp_bottle_to_cast(
                s[c - 1],
                t[c - 1],
                p[c - 1],
                Sc,
                Tc,
                Pc,
                k,
                K,
                tol_p,
                eos,
                ppc_fn,
            )

        if np.isnan(p[c]):
            # The neutral trajectory incropped or outcropped
            break

    return s, t, p
