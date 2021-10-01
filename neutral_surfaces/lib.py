import numpy as np
import numba

from neutral_surfaces.fzero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val_0d_i, dval_0d_i
from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t


@numba.njit
def find_first_nan(a):
    """The index to the first NaN along the last axis

    Parameters
    ----------
    a : ndarray
        Input array possibly containing some NaN elements

    Returns
    -------
    k : ndarray of int
        The index to the first NaN along each 1D array making up `a`, as in the
        following example with `a` being 3D.
        If all `a[i,j,:]` are not NaN, then `k[i,j] = a.shape[-1]`.
        Otherwise, `a[i,j,k[i,j]-1]` is not NaN, but `a[i,j,k[i,j]]` is NaN.
    """
    nk = a.shape[-1]
    k = np.full(a.shape[:-1], nk, dtype=np.int64)
    for n in np.ndindex(a.shape[0:-1]):
        for i in range(nk):
            if np.isnan(a[n][i]):
                k[n] = i
                break
    return k


def _process_wrap(wrap, s=None):
    """Convert to a tuple of `int`s specifying which horizontal dimensions are periodic"""
    if isinstance(wrap, str):
        wrap = (wrap,)  # Convert single string to tuple
    if not isinstance(wrap, (tuple, list)):
        raise TypeError("wrap must be a tuple or list or str")
    if all(isinstance(x, str) for x in wrap):
        try:
            # Convert dim names to tuple of bool
            wrap = tuple(x in wrap for x in s.dims)
        except:
            raise TypeError(
                "With wrap provided as strings, S must have a .dims attribute"
            )

    # type checking on final value
    if not isinstance(wrap, (tuple, list)) or len(wrap) != 2:
        raise TypeError(
            "wrap must be a two element (logical) array "
            "or a string (or array of strings) referring to dimensions in xarray S"
        )
    return wrap


def _process_vert_dim(vert_dim, S):
    """Extract int of dimension in S named by vert_dim"""
    if isinstance(vert_dim, str):
        try:
            vert_dim = S.dims.index(vert_dim)
        except:
            raise ValueError(f"vert_dim = {vert_dim} not found in S.dims")
    return vert_dim


def xr_to_np(S):
    """Convert xarray into numpy array"""
    if hasattr(S, "values"):
        S = S.values
    return S


def _process_casts(S, vert_dim):
    """Make individual casts contiguous in memory and extract numpy array from xarray

    Parameters
    ----------
    S : ndarray or xarray.DataArray
        ocean data such as salinity, temperature, or pressure

    vert_dim : int or str, Default None
        Specifies which dimension of `S` is vertical.
        If `S` is an `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` (e.g. -1 indexes the last dimension).
        If `S` is an `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S`.

    Returns
    -------
    S : ndarray
        input data, possibly re-arranged to have `vert_dim` the last dimension
    """
    S = xr_to_np(S)

    if S.ndim > 1 and vert_dim not in (-1, S.ndim - 1):
        S = np.moveaxis(S, vert_dim, -1)

    S = np.require(S, dtype=np.float64, requirements="C")

    return S


def _ntp_ϵ_error1(s, t, p, eos_s_t, wrap, shift):
    # Calculate neutrality error on a surface in one direction.

    sa, ta, pa = (avg(x, shift, wrap) for x in (s, t, p))

    ds, dt = (dif(x, shift, wrap) for x in (s, t))

    rsa, rta = eos_s_t(sa, ta, pa)
    return rsa * ds + rta * dt


def ntp_ϵ_errors(s, t, p, eos_s_t, wrap):
    # Calculate neutrality error on a surface.
    # Use backward differences; results are on the U, V grids.

    wrap = _process_wrap(wrap, s)
    s, t, p = (xr_to_np(x) for x in (s, t, p))

    ϵx = _ntp_ϵ_error1(s, t, p, eos_s_t, wrap, im1)
    ϵy = _ntp_ϵ_error1(s, t, p, eos_s_t, wrap, jm1)

    return ϵx, ϵy


def ntp_ϵ_errors_norms(
    s, t, p, eos_s_t, wrap, dist1_iJ=1., dist1_Ij=1., dist2_Ij=1., dist2_iJ=1.
):

    ϵ_iJ, ϵ_Ij = ntp_ϵ_errors(s, t, p, eos_s_t, wrap)

    ni, nj = s.shape

    # fmt: off
    # Expand grid distances.  Soft notation: i = I-1/2; j = J-1/2
    dist1_iJ = np.broadcast_to(dist1_iJ, (ni, nj))  # Distance [m] in 1st dim centred at (I-1/2, J)
    dist1_Ij = np.broadcast_to(dist1_Ij, (ni, nj))  # Distance [m] in 1st dim centred at (I, J-1/2)
    dist2_Ij = np.broadcast_to(dist2_Ij, (ni, nj))  # Distance [m] in 2nd dim centred at (I, J-1/2)
    dist2_iJ = np.broadcast_to(dist2_iJ, (ni, nj))  # Distance [m] in 2nd dim centred at (I-1/2, J)
    AREA_iJ = dist1_iJ * dist2_iJ  # Area [m^2] centred at (I-1/2, J)
    AREA_Ij = dist1_Ij * dist2_Ij  # Area [m^2] centred at (I, J-1/2)

    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances dist1_iJ and dist2_Ij.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     AREA_iJ ./ dist1_iJ.^2 = dist2_iJ ./ dist1_iJ ,
    # and AREA_Ij ./ dist2_Ij.^2 = dist1_Ij ./ dist2_Ij .
    ϵ_RMS = np.sqrt(
        (np.nansum(dist2_iJ / dist1_iJ * ϵ_iJ ** 2) + np.nansum(dist1_Ij / dist2_Ij * ϵ_Ij ** 2)) /
        (   np.sum(AREA_iJ * np.isfinite(ϵ_iJ))     +    np.sum(AREA_Ij * np.isfinite(ϵ_Ij)))
    )


    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances dist1_iJ and dist2_Ij.
    # Thus, the numerator of L1 norm needs to multiply epsilon by
    #     AREA_iJ ./ dist1_iJ = dist2_iJ ,
    # and AREA_Ij ./ dist2_Ij = dist1_Ij .
    ϵ_MAV = (
        (np.nansum(dist2_iJ        * abs(ϵ_iJ)) + np.nansum(dist1_Ij         * abs(ϵ_Ij)))
        / ( np.sum(AREA_iJ * np.isfinite(ϵ_iJ)) +     np.sum(AREA_Ij * np.isfinite(ϵ_Ij)))
       )
    # fmt: on

    return ϵ_RMS, ϵ_MAV


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


def im1(F, wrap, fill=np.nan):  # G[i,j] == F[i-1,j]
    G = np.roll(F, 1, axis=0)
    if not wrap[0]:
        G[0, :] = fill
    return G


def ip1(F, wrap, fill=np.nan):  # G[i,j] == F[i+1,j]
    G = np.roll(F, -1, axis=0)
    if not wrap[0]:
        G[-1, :] = fill
    return G


def jm1(F, wrap, fill=np.nan):  # G[i,j] == F[i,j-1]
    G = np.roll(F, 1, axis=1)
    if not wrap[1]:
        G[:, 0] = fill
    return G


def jp1(F, wrap, fill=np.nan):  # G[i,j] == F[i,j+1]
    G = np.roll(F, -1, axis=1)
    if not wrap[1]:
        G[:, -1] = fill
    return G


def avg(F, shift, wrap):
    return (F + shift(F, wrap)) * 0.5


def dif(F, shift, wrap):
    return F - shift(F, wrap)


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
        Conservative temperature, and pressure down a 1D sequence of casts

    p0 : float

        The pressure at which to begin the neutral trajectory on the first cast

    s0, t0 : float, optional

        If provided, the first step of the neutral trajectory is a neutral
        connection a bottle with salinity s0, temperature t0, and pressure p0
        to the first cast.

    Returns
    -------
    s, t, p : 1D ndarray

        practical / Absolute Salinity, potential / Conservative Temperature,
        and pressure / depth along the neutral trajectory.

    Other Parameters
    ----------------
    tol_p : float, Default 1e-4

        Error tolerance when root-finding to update the pressure or depth of
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


# CHECK VALUE from MATLAB:
# >> veronis_density(0, S(:,i0,j0), T(:,i0,j0), Z, 10, 1500, 1, @ppc_linterp)
# 1027.770044499011
# def veronis_density(S, T, P, p1, **kwargs):
def veronis_density(
    S,
    T,
    P,
    p1,
    p_ref=0.0,
    p0=None,
    dp=1.0,
    interp_fn=linear_coeffs,
    eos="gsw",
    eos_s_t=None,
    grav=None,
    rho_c=None,
):
    """The surface density plus the integrated vertical gradient of Locally
    Referenced Potential Density

    Determines the Veronis density [1]_ [2]_ at vertical position `p1` on a
    cast with hydrographic properties `(S, T, P)`.  The Veronis density is
    the potential density (referenced to `p_ref`) evaluated at `p0` on the
    cast, plus the integral (dP) of the vertical (d/dP) derivative of Locally
    Referenced Potential Density (LRPD) from `P = p0` to `P = p1`.  The
    vertical (d/dP) derivative of LRPD is `rho_S dS/dP + rho_T dT/dP` where
    `rho_S` and `rho_T` are the partial derivatives of density with respect
    to `S` and `T`, and `dS/dP` and `dT/dP` are the derivatives of `S` and
    `T` with respect to `P` in the water column.  If `p0` or `p1` are outside
    the range of `P`, NaN is returned.

    Parameters
    ----------
    S, T, P : 1D ndarray of float

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure or depth of data points on the cast.  `P` must increase
        monotonically along its last dimension.

    p1 : float

        Pressure or depth at which the Veronis density is evaluated

    Returns
    -------
    d : float

        Veronis density

    Other Parameters
    ----------------
    p_ref : float, Default 0.

        reference pressure or depth for potential density

    p0 : float, Default `P[0]`

        Pressure or depth at which the potential density is evaluated

    dp : float, Default 1.

        Maximum interval of pressure or depth in trapezoidal numerical
        integration

    interp_fn : function, Default ``linear_coeffs``

        Function that calculates coefficients of piecewise polynomial
        interpolants of `S` and `T` as functions of `P`.  Options include
        ``linear_coeffs`` and ``pchip_coeffs`` from ``interp_ppc.py``.

    eos : str or function, Default 'gsw'

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure (not depth) inputs.

        If a str, can be either 'gsw' to use TEOS-10 or 'jmd' to use Jackett
        and McDougall (1995) [1]_.

    eos_s_t : str or function, Default None

        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T` as a function of `S`, `T`, and
        pressure (not depth) inputs.

        If a function, this need not be @numba.njit decorated but should be
        vectorized, as it will be called a few times with ndarray inputs.

        If a str, the same options apply as for `eos`. If None and `eos` is a
        str, then this defaults to the same str as `eos`.

    grav : float, Default None

        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None

        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    Notes
    -----
    The result of this function can serve as a density label for an
    approximately neutral surface. However, this is NOT the same as a value
    of the Jackett and McDougall (1997) Neutral Density variable. This is
    true even if you were to provide this function with the same cast that
    Jackett and McDougall (1997) used to initially label their Neutral
    Density variable, namely the cast at 188 deg E, 4 deg S, from the Levitus
    (1982) ocean atlas. Some difference would remain, because of differences
    in numerics, and because of a subsequent smoothing step in the Jackett
    and McDougall (1997) algorithm. This function merely allows one to label
    an approximately neutral surface with a density value that is INTERNALLY
    consistent within the dataset where one's surface lives. This function is
    NOT to compare density values against those from any other dataset, such
    as 1997 Neutral Density.

    .. [1] Veronis, G. (1972). On properties of seawater defined by temperature,
    salinity, and pressure. Journal of Marine Research, 30(2), 227.

    .. [2] Stanley, McDougall, Barker 2021, Algorithmic improvements to finding
     approximately neutral surfaces, Journal of Advances in Earth System
     Modelling, 13(5).
    """

    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)), 'P must be same size as S')
    # assert(isvector(S), 'S, T, P must be 1D. (Veronis density is only useful for one water column at a time!)')
    # assert(isscalar(p0), 'p0 must be a scalar')
    # assert(isscalar(p1), 'p1 must be a scalar')

    if p0 is None:
        p0 = P[0]

    if eos_s_t is None and isinstance(eos, str):
        eos_s_t = eos
    elif isinstance(eos, str) and isinstance(eos_s_t, str) and eos != eos_s_t:
        raise ValueError("eos and eos_s_t, if strings, must be the same string")
    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)

    # Interpolate S and T as piecewise polynomials of P
    Sppc = interp_fn(P, S)
    Tppc = interp_fn(P, T)

    # i = np.searchsorted(X,x) is such that:
    #   i = 0                   if x <= X[0]
    #   i = len(X)              if X[-1] < x or np.isnan(x)
    #   X[i-1] < x <= X[i]      otherwise
    # Having guaranteed X[0] < x <= X[-1] and x is not nan, then
    #   X[i-1] < x <= X[i]  and  1 <= i <= len(X)-1  in all cases,
    if (
        np.isnan(p0)
        or p0 < P[0]
        or P[-1] < p0
        or np.isnan(p1)
        or p1 < P[0]
        or P[-1] < p1
    ):
        return np.nan

    if p0 == P[0]:
        k0 = 0  # p0 == P[0]
    else:
        k0 = np.searchsorted(P, p0)  # P[k0-1] < p0 <= P[k0]

    if p1 == P[0]:
        k1 = 0
    else:
        k1 = np.searchsorted(P, p1)

    # Integrate from p0 to P[k0]
    d1 = _int_x_k(p0, k0, dp, P, S, T, Sppc, Tppc, eos_s_t)

    # Integrate from P[k0] to P[k1]
    for k in range(k0, k1):
        # Integrate from P[k] to P[k+1]
        d1 += _int_x_k(P[k], k + 1, dp, P, S, T, Sppc, Tppc, eos_s_t)

    # Integrate from p1 to P[k1], and subtract this
    d1 -= _int_x_k(p1, k1, dp, P, S, T, Sppc, Tppc, eos_s_t)

    # Calculate potential density, referenced to p_ref, at p0
    s0, t0 = val2_0d(P, S, Sppc, T, Tppc, p0)
    d0 = eos(s0, t0, p_ref)

    return d0 + d1


def _int_x_k(p, k, dp, P, S, T, Sppc, Tppc, eos_s_t):
    # Integrate from p to P[k] using trapezoidal integration with spacing dp or smaller

    n = np.int(np.ceil((P[k] - p) / dp)) + 1  # points between p and P[k], inclusive
    p_ = np.linspace(p, P[k], n)  # intervals are not larger than dp

    # Use piecewise polynomial coefficients as provided. Be sure to pass the
    # index of this part to avoid any issues evaluating a discontinuous piecewise
    # polynomial (probably the derivative, if either) at the knot.
    s_ = np.zeros(n)
    t_ = np.zeros(n)
    dsdp_ = np.zeros(n)
    dtdp_ = np.zeros(n)
    for i in range(n):
        s_[i] = val_0d_i(P, S, Sppc, p_[i], k - 1)
        t_[i] = val_0d_i(P, T, Tppc, p_[i], k - 1)
        dsdp_[i] = dval_0d_i(P, S, Sppc, p_[i], 1, k - 1)
        dtdp_[i] = dval_0d_i(P, T, Tppc, p_[i], 1, k - 1)

    # To use linear interpolation internally, replace the above lines with the following 7 lines
    # dp = P[k] - P[k-1]
    # dsdp_ = (S[k] - S[k-1]) / dp
    # dtdp_ = (T[k] - T[k-1]) / dp
    # s0 = ( S[k-1] * (P[k] - p) + S[k] * (p - P[k-1]) ) / dp
    # t0 = ( T[k-1] * (P[k] - p) + T[k] * (p - P[k-1]) ) / dp
    # s_ = np.linspace(s0, S[k], n)
    # t_ = np.linspace(t0, T[k], n)

    rs_, rt_ = eos_s_t(s_, t_, p_)
    y_ = rs_ * dsdp_ + rt_ * dtdp_
    return np.trapz(y_, x=p_)


# def veronis_label(p_ref, t_ref, S, T, P, p, pin, eos, eos_s_t, dp=1, interpfn=linear_coeffs):
# 1. Do a temporal neutral_trajectory from (t,i0,j0,p[i0,j0]) to (t_ref,i0,j0,p0)
# 2. Evaluate veronis density at (t_ref, i0, j0, p0)
