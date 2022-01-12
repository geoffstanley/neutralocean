"""Neutral Trajectory and related functions"""

import numpy as np
import numba

from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d
from neutral_surfaces.eos.eostools import make_eos
from neutral_surfaces.fzero import guess_to_bounds, brent
from neutral_surfaces.lib import find_first_nan


@numba.njit
def _func(p, sB, tB, pB, S, T, P, Sppc, Tppc, eos):
    # Evaluate difference between (a) eos at location on the cast (S, T, P)
    # where the pressure or depth is p, and (b) eos of the bottle (sB, tB, pB)
    # here, eos is always evaluated at the average pressure or depth, (p +
    # pB)/2.
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
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
    interp_fn=linear_coeffs,
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

    S, T, P : 1D ndarray of float

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

    interp_fn : function, Default ``linear_coeffs``

        Function that calculates coefficients of piecewise polynomial
        interpolants of `S` and `T` as functions of `P`.  Options include
        ``linear_coeffs`` and ``pchip_coeffs`` from ``interp_ppc.py``.

    eos : str or function, Default 'gsw'

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure inputs.  For Boussinesq models, provide `grav`
        and `rho_c`, so this function with third input pressure will be
        converted to a function with third input depth.

        If a function, this should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

        If a str, can be either 'gsw' to use TEOS-10
        or 'jmd' to use Jackett and McDougall (1995) [1]_.

    grav : float, Default None

        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None

        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    Notes
    -----
    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388

    """

    eos = make_eos(eos, grav, rho_c)
    Sppc = interp_fn(P, S)
    Tppc = interp_fn(P, T)
    n_good = find_first_nan(S)

    return _ntp_bottle_to_cast(sB, tB, pB, S, T, P, Sppc, Tppc, n_good, eos, tol_p)


@numba.njit
def _ntp_bottle_to_cast(sB, tB, pB, S, T, P, Sppc, Tppc, n_good, eos, tol_p):
    """Find the neutral tangent plane from a bottle to a cast

    Fast version of `ntp_bottle_to_cast`, with all inputs supplied.  See
    documentation for `ntp_bottle_to_cast`.

    Parameters
    ----------
    sB, tB, pB : float
        See ntp_bottle_to_cast

    S, T, P : ndarray
        See ntp_bottle_to_cast

    Sppc, Tppc : ndarray

        Piecewise Polynomial Coefficients for `S` and `T` as functions of `P`.
        Computed these as ``Sppc = interp_fn(P, S)`` and ``Tppc
        = interp_fn(P, T)`` where `interp_fn` is ``linear_coeffs`` or
        ``pchip_coeffs`` from ``interp_ppc.py``.

    n_good : int

        Number of valid (non-NaN) data points on the cast.  That is,
        ``S[0:n_good-1]``, ``T[0:n_good-1]``, and ``P[0:n_good-1]`` should all
        be non-NaN.  Compute this as ``n_good = find_first_nan(S)``

    eos : function
        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure or depth inputs.

        This function should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    tol_p : float, Default 1e-4
        See ntp_bottle_to_cast

    Returns
    -------
    s, t, p : float
        See ntp_bottle_to_cast
    """

    if n_good > 1:

        args = (sB, tB, pB, S, T, P, Sppc, Tppc, eos)

        # Search for a sign-change, expanding outward from an initial guess
        lb, ub = guess_to_bounds(_func, pB, P[0], P[n_good - 1], args)

        if np.isfinite(lb):
            # A sign change was discovered, so a root exists in the interval.
            # Solve the nonlinear root-finding problem using Brent's method
            p = brent(_func, lb, ub, tol_p, args)

            # Interpolate S and T onto the updated surface
            s, t = val2_0d(P, S, Sppc, T, Tppc, p)

        else:
            s, t, p = np.nan, np.nan, np.nan

    else:
        s, t, p = np.nan, np.nan, np.nan

    return s, t, p


# To do: add vert_dim argument
def neutral_trajectory(
    S,
    T,
    P,
    p0,
    s0=None,
    t0=None,
    tol_p=1e-4,
    interp_fn=linear_coeffs,
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
    S, T, P : 2D ndarray

        1D data specifying the practical / Absolute salinity, and potential /
        Conservative temperature, and pressure / depth down a 1D sequence of casts

    p0 : float

        The pressure / depth at which to begin the neutral trajectory on the first cast

    s0, t0 : float, optional

        If provided, the first step of the neutral trajectory is a neutral
        connection a bottle with salinity s0, temperature t0, and pressure /
        depth p0 to the first cast.

    Returns
    -------
    s, t, p : 1D ndarray

        practical / Absolute Salinity, potential / Conservative Temperature,
        and pressure / depth along the neutral trajectory.

    Other Parameters
    ----------------
    tol_p : float, Default 1e-4

        Error tolerance when root-finding to update the pressure / depth of
        the surface in each water column. Units are the same as `P`.

    interp_fn : function, Default ``linear_coeffs``

        Function that calculates coefficients of piecewise polynomial
        interpolants of `S` and `T` as functions of `P`.  Options include
        ``linear_coeffs`` and ``pchip_coeffs`` from ``interp_ppc.py``.

    eos : str or function, Default 'gsw'

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure (not depth) inputs.

        If a function, this should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

        If a str, can be either 'gsw' to use TEOS-10
        or 'jmd' to use Jackett and McDougall (1995) [1]_.

    grav : float, Default None

        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None

        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    Notes
    -----
    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388
    """

    eos = make_eos(eos, grav, rho_c)

    nk, nc = S.shape
    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)) || all(size(P) == [nk, 1]), 'P must be [nk,nc] or [nk,1]')

    s = np.full(nc, np.nan)
    t = np.full(nc, np.nan)
    p = np.full(nc, np.nan)

    # Evaluate S and T on first cast at p0
    Sc = S[:, 0]
    Tc = T[:, 0]
    Pc = P[:, 0]
    Sppc = interp_fn(Pc, Sc)
    Tppc = interp_fn(Pc, Tc)
    s[0], t[0] = val2_0d(Pc, Sppc, Tppc, p0)
    p[0] = p0

    # Loop over remaining casts
    for c in range(1, nc):

        Sc = S[:, c]
        Tc = T[:, c]
        Pc = P[:, c]

        # Interpolate Sc and Tc as piecewise polynomials of P
        Sppc = interp_fn(Pc, Sc)
        Tppc = interp_fn(Pc, Tc)

        # Make a neutral connection from previous bottle (s0,t0,p0) to the cast (S[:,c], T[:,c], P[:,c])
        K = np.sum(np.isfinite(Sc))
        s[c], t[c], p[c] = _ntp_bottle_to_cast(
            s[c - 1], t[c - 1], p[c - 1], Sc, Tc, Pc, Sppc, Tppc, K, eos, tol_p
        )

        if np.isnan(p[c]):
            # The neutral trajectory incropped or outcropped
            break

    return s, t, p
