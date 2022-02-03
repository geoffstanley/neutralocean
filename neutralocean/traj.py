"""Neutral Trajectory and related functions"""

import numpy as np
import numba as nb

from neutralocean.ppinterp import select_ppc, ppval1_two
from neutralocean.eos.tools import make_eos
from neutralocean.fzero import guess_to_bounds, brent
from neutralocean.lib import find_first_nan


@nb.njit
def _func(p, sB, tB, pB, Sppc, Tppc, P, eos):
    # Evaluate difference between (a) eos at location on the cast (S, T, P)
    # where the pressure or depth is p, and (b) eos of the bottle (sB, tB, pB)
    # here, eos is always evaluated at the average pressure or depth, (p +
    # pB)/2.
    s, t = ppval1_two(p, P, Sppc, Tppc)
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

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use 'linear' for linear
        interpolation, and 'pchip' for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `interp1d`.

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
    ppc_fn = select_ppc(interp, "1")
    n_good = find_first_nan(S)

    Sppc = ppc_fn(P, S)
    Tppc = ppc_fn(P, T)

    return _ntp_bottle_to_cast(sB, tB, pB, Sppc, Tppc, P, n_good, tol_p, eos)


@nb.njit
def _ntp_bottle_to_cast(sB, tB, pB, Sppc, Tppc, P, n_good, tol_p, eos):
    """Find the neutral tangent plane from a bottle to a cast

    Fast version of `ntp_bottle_to_cast`, with all inputs supplied.

    Parameters
    ----------
    sB, tB, pB : float
        See ntp_bottle_to_cast

    Sppc, Tppc : ndarray
        Piecewise Polynomial Coefficients for interpolants of Salinity and
        Temperature in terms of `P`.

    P : ndarray
        See ntp_bottle_to_cast

    n_good : int

        Number of valid (non-NaN) data points on the cast.  That is,
        ``S[0:n_good-1]``, ``T[0:n_good-1]``, and ``P[0:n_good-1]`` should all
        be non-NaN.  Compute this as ``n_good = find_first_nan(S)``, where `S`
        and `T` are the salinity and temperature on the cast, from which `Sppc`
        and `Tppc` were constucted.

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

        args = (sB, tB, pB, Sppc, Tppc, P, eos)

        # Search for a sign-change, expanding outward from an initial guess
        lb, ub = guess_to_bounds(_func, pB, P[0], P[n_good - 1], args)

        if np.isfinite(lb):
            # A sign change was discovered, so a root exists in the interval.
            # Solve the nonlinear root-finding problem using Brent's method
            p = brent(_func, lb, ub, tol_p, args)

            # Interpolate S and T onto the updated surface
            s, t = ppval1_two(p, P, Sppc, Tppc)

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
    S, T, P : 2D ndarray

        1D data specifying the practical / Absolute salinity, and potential /
        Conservative temperature, and pressure / depth down a 1D sequence of casts

    p0 : float

        The pressure / depth at which to begin the neutral trajectory on the first cast

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

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use 'linear' for linear
        interpolation, and 'pchip' for Piecewise Cubic Hermite Interpolating
        Polynomials.

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
    ppc_fn = select_ppc(interp, "1")

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
    Sppc = ppc_fn(Sc, Pc)
    Tppc = ppc_fn(Tc, Pc)
    s[0], t[0] = interp(p0, Pc, Sppc, Tppc)
    p[0] = p0

    # Loop over remaining casts
    for c in range(1, nc):

        Sc = S[:, c]
        Tc = T[:, c]
        Pc = P[:, c]
        Sppc = ppc_fn(Sc, Pc)
        Tppc = ppc_fn(Tc, Pc)

        # Make a neutral connection from previous bottle to the cast (S[:,c], T[:,c], P[:,c])
        K = np.sum(np.isfinite(Sc))
        s[c], t[c], p[c] = _ntp_bottle_to_cast(
            s[c - 1], t[c - 1], p[c - 1], Sppc, Tppc, Pc, Sc, Tc, K, tol_p, eos
        )

        if np.isnan(p[c]):
            # The neutral trajectory incropped or outcropped
            break

    return s, t, p
