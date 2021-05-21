import numpy as np
import numba

from neutral_surfaces._zero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val_0d_i, dval_0d_i


def _ntp_error1(s, t, p, eos_s_t, wrap, shift):
    # Calculate neutrality error on a surface in one direction.
    sa = avg(s, shift, wrap)
    ta = avg(t, shift, wrap)
    pa = avg(p, shift, wrap)

    ds = dif(s, shift, wrap)
    dt = dif(t, shift, wrap)

    rsa, rta = eos_s_t(sa, ta, pa)
    return rsa * ds + rta * dt


def ntp_errors(s, t, p, eos_s_t, wrap):
    # Calculate neutrality error on a surface.
    # Use backward differences; results are on the U, V grids.

    ϵx = _ntp_error1(s, t, p, eos_s_t, wrap, im1)
    ϵy = _ntp_error1(s, t, p, eos_s_t, wrap, jm1)

    return ϵx, ϵy


def ϵ_norms(
    s, t, p, eos_s_t, wrap, DIST1_iJ=1, DIST2_Ij=1, DIST2_iJ=1, DIST1_Ij=1, *args
):

    ϵ_iJ, ϵ_Ij = ntp_errors(s, t, p, eos_s_t, wrap)

    if len(args) == 2:
        AREA_iJ = args[0]  # Area [m^2] centred at (I-1/2, J)
        AREA_Ij = args[1]  # Area [m^2] centred at (I, J-1/2)
    else:
        AREA_iJ = DIST1_iJ * DIST2_iJ  # Area [m^2] centred at (I-1/2, J)
        AREA_Ij = DIST1_Ij * DIST2_Ij  # Area [m^2] centred at (I, J-1/2)

    # fmt: off
    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances DIST1_iJ and DIST2_Ij.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     AREA_iJ ./ DIST1_iJ.^2 = DIST2_iJ ./ DIST1_iJ ,
    # and AREA_Ij ./ DIST2_Ij.^2 = DIST1_Ij ./ DIST2_Ij .
    ϵ_L2 = np.sqrt(
        (np.nansum(DIST2_iJ / DIST1_iJ * ϵ_iJ ** 2) + np.nansum(DIST1_Ij / DIST2_Ij * ϵ_Ij ** 2)) /
        (   np.sum(AREA_iJ * np.isfinite(ϵ_iJ))     +    np.sum(AREA_Ij * np.isfinite(ϵ_Ij)))
    )


    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances DIST1_iJ and DIST2_Ij.
    # Thus, the numerator of L1 norm needs to multiply epsilon by
    #     AREA_iJ ./ DIST1_iJ = DIST2_iJ ,
    # and AREA_Ij ./ DIST2_Ij = DIST1_Ij .
    ϵ_L1 = (
        (np.nansum(DIST2_iJ        * abs(ϵ_iJ)) + np.nansum(DIST1_Ij         * abs(ϵ_Ij)))
        / ( np.sum(AREA_iJ * np.isfinite(ϵ_iJ)) +     np.sum(AREA_Ij * np.isfinite(ϵ_Ij)))
       )
    # fmt: on

    return ϵ_L2, ϵ_L1


@numba.njit
def _func(p, sB, tB, pB, S, T, P, Sppc, Tppc, eos):
    # Evaluate difference between (a) eos at location on the cast (S, T, P)
    # where the pressure or depth is p, and (b) eos of the bottle (sB, tB, pB)
    # here, eos is always evaluated at the average pressure or depth, (p +
    # pB)/2.
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    p_avg = (pB + p) * 0.5
    return eos(sB, tB, p_avg) - eos(s, t, p_avg)


@numba.njit
def ntp_bottle_to_cast(sB, tB, pB, S, T, P, Sppc, Tppc, k, tol_p, eos):
    # Py dev:  removed "success" output

    # NTP_BOTTLE_TO_CAST  Find a bottle's level of neutral buoyancy in a water
    #                    column, using the Neutral Tangent Plane relationship.
    #
    # [p, s, t] = ntp_bottle_to_cast(Sppc, Tppc, P, k, sB, tB, pB, tol_p)
    # finds (s, t, p), with precision in p of tolp, that is at the level of
    # neutral buoyancy for a fluid bottle of (sB, tB, pB) in a water column of
    # with piecewise polynomial interpolants for S and T given by Sppc and Tppc
    # with knots at P(1:k).  Specifically, s and t are given by
    #   [s,t] = ppc_val(P, Sppc, Tppc, p)
    # and p satisfies
    #      eos(s, t, p') = eos(sB, tB, p')
    # where eos is the equation of state given by eos.m in MATLAB's path,
    # and   p' is in the range [p_avg - tol_p/2, p_avg + tol_p/2],
    # and   p_avg = (pB + p) / 2 is the average of the fluid bottle's original
    #                          and final pressure or depth.
    #
    # [p, s, t, success] = ntp_bottle_to_cast(...)
    # returns a flag value success that is true if a valid solution was found,
    # false otherwise.
    #
    # For a non-Boussinesq ocean, P, pB, and p are pressure.
    # For a Boussinesq ocean, P, pB, and p are depth.
    #
    #
    # --- Input:
    # Sppc [O, K-1]: coefficients for piecewise polynomial for practical
    #                   / Absolute Salinity in terms of P
    # Tppc [O, K-1]: coefficients for piecewise polynomial for potential
    #                   / Conservative Temperature in terms of P
    # P [K, 1]: pressure or depth in water column
    # k [1, 1]: number of valid (non-NaN) data points in the water column.
    #          Specifically, Sppc(end,1:k) and Tppc(end,1:k) must all be valid.
    # sB [1 , 1]: practical / Absolute salinity of current bottle
    # tB [1 , 1]: potential / Conservative temperature of current bottle
    # pB [1 , 1]: pressure or depth of current bottle
    # tol_p [1, 1]: tolerance for solving the level of neutral buoyancy (same
    #             units as P and pB)
    #
    # Note: physical units for Sppc, Tppc, P, sB, tB, pB, p, s, t  are
    # determined by eos.m.
    #
    # Note: P must increase monotonically along its first dimension.
    #
    #
    # --- Output:
    # p [1, 1]: pressure or depth in water column at level of neutral buoyancy
    # s [1, 1]: practical / Absolute salinity in water column at level of neutral buoyancy
    # t [1, 1]: potential / Conservative temperature in water column at level of neutral buoyancy
    # success [1,1]: true if a valid solution was found, false otherwise.

    # Author(s) : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

    if k > 1:

        args = (sB, tB, pB, S, T, P, Sppc, Tppc, eos)

        # Search for a sign-change, expanding outward from an initial guess
        lb, ub = guess_to_bounds(_func, args, pB, P[0], P[k - 1])

        if np.isfinite(lb):
            # A sign change was discovered, so a root exists in the interval.
            # Solve the nonlinear root-finding problem using Brent's method
            p = brent(_func, args, lb, ub, tol_p)

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


def neutral_trajectory(eos, S, T, P, p0, s0, t0, interpfn=linear_coeffs, tol_p=1e-6):
    # NEUTRAL_TRAJECTORY  Calculate a neutral trajectory through a sequence of casts.
    #
    #
    # [p,s,t] = neutral_trajectory(S, T, P, p0)
    # calculates a discrete neutral trajectory through the consecutive casts
    # (S(:,c), T(:,c), P(:,c)) for increasing c, beginning at depth or pressure
    # p0 on cast c=1.  The output are 1D arrays p, s, and t, whose c'th
    # elements provide the depth / pressure, salinity, and temperature values
    # on the c'th cast along the neutral trajectory. The equation of state for
    # density is given by eos.m in the PATH.
    #
    # [p,s,t] = neutral_trajectory(S, T, P, p0, s0, t0)
    # as above, but the first step is a discrete neutral trajectory from the
    # bottle (s0, t0, p0) to the cast (S(:,1), T(:,1), P(:,1)).
    #
    # ... = neutral_trajectory(..., interpfn)
    # uses interpfn (a function handle) to interpolate S and T as piecewise
    # polynomials of P. By default, interpfn = @ppc_linterp to use linear
    # interpolation. Other functions from the PPC toolbox can be used, e.g.
    # @ppc_pchip and @ppc_makima.
    #
    # ... = neutral_trajectory(..., tolp)
    # evaluates the discrete neutral trajectory with an accuracy of tolp [m or dbar].
    # By default, tolp = 1e-6 [m or dbar].
    #
    # Provide [] for any optional arguments that are required only to provide
    # a value for an argument later in the list.
    #
    #
    # --- Input:
    #  S [nk, nc]: practical / Absolute Salinity values on a cast
    #  T [nk, nc]: potential / Conservative Temperature values on a cast
    #  P [nk, nc]: pressure / depth values on a cast
    #  p0 [1, 1]: pressure / depth of starting bottle
    #  s0 [1, 1]: practical / Absolute Salinity of starting bottle
    #  t0 [1, 1]: potential / Conservative Temperature of starting bottle
    #  interpfn [function handle]: function to calcualte piecewise polynomial coefficients
    #  tolp [1, 1]: error tolerance in vertical for neutral trajectory calculations
    #
    #
    # --- Output:
    #  p [1, nc]: pressure / depth along the neutral trajectory
    #  s [1, nc]: practical / Absolute Salinity along the neutral trajectory
    #  t [1, nc]: potential / Conservative Temperature along the neutral trajectory

    # Author(s) : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

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
    Sppc = interpfn(Pc, Sc)
    Tppc = interpfn(Pc, Tc)
    s[0], t[0] = val2_0d(Pc, Sppc, Tppc, p0)
    p[0] = p0

    # Loop over remaining casts
    for c in range(1, nc):

        Sc = S[:, c]
        Tc = T[:, c]
        Pc = P[:, c]

        # Interpolate Sc and Tc as piecewise polynomials of P
        Sppc = interpfn(Pc, Sc)
        Tppc = interpfn(Pc, Tc)

        # Make a neutral connection from previous bottle (s0,t0,p0) to the cast (S[:,c], T[:,c], P[:,c])
        K = np.sum(np.isfinite(Sc))
        s[c], t[c], p[c] = ntp_bottle_to_cast(
            s[c - 1], t[c - 1], p[c - 1], Sc, Tc, Pc, Sppc, Tppc, K, tol_p, eos
        )

        if np.isnan(p[c]):
            # The neutral trajectory incropped or outcropped
            break

    return s, t, p


# CHECK VALUE from MATLAB:
# >> veronis_density(0, S(:,i0,j0), T(:,i0,j0), Z, 10, 1500, 1, @ppc_linterp)
# 1027.770044499011


def veronis_density(p_ref, S, T, P, p0, p1, eos, eos_s_t, dp=1, interpfn=linear_coeffs):
    # VERONIS_DENSITY  The surface density plus the integrated vertical
    #                  gradient of Locally Referenced Potential Density.
    #
    #
    # d1 = veronis_density(p_ref, S, T, P, p0, p1)
    # determines the Veronis density d1 at vertical position p1 on a cast with
    # practical / Absolute salinity S and potential / Conservative temperature
    # T values at depth or pressure values P.  The Veronis density is given by
    # the potential density (with reference pressure / depth p_ref) evaluated
    # at p0 on the cast, plus the integral of the vertical (d/dX) derivative of
    # Locally Referenced Potential Density (LRPD) from P = p0 to P = p1. The
    # vertical (d/dP) derivative of LRPD is rho_S dS/dP + rho_T dT/dP where
    # rho_S and rho_T are the partial derivatives of density with respect to S
    # and T, and dS/dP and dT/dP are the derivatives of S and T with respect to
    # P.  The equation of state for density is given by eos.m in the PATH, and
    # its partial derivatives with respect to S and T are given by eos_s_t.m in
    # the PATH.  If p0 or p1 are outside the range of P, d1 is returned as NaN.
    #
    # d1 = veronis_density(..., dp)
    # specifies the maximum interval size used in the trapezoidal numerical
    # integration.  If omitted, the default size is 1 unit of P (1m or 1 dbar).

    # d1 = veronis_density(..., interpfn)
    # uses interpfn (a function handle) to interpolate S and T as piecewise polynomials of P.
    # If interpfn = @ppc_linterp, the result is the same as if interpfn were omitted
    # and linear interpolation were performed native to this code.  Other functions
    # from the PPC toolbox can be used, e.g. ppc_pchip and ppc_makima.
    #
    # Provide [] for any optional arguments that are required only to provide
    # a value for an argument later in the list.
    #
    #
    # --- Input:
    # p_ref [1,1]: reference pressure / depth to evaluate potential density at p0
    # S [nk, nt] : practical / Absolute Salinity values on a cast
    # T [nk, nt] : potential / Conservative Temperature values on a cast
    # P [nk, nt] : pressure / depth values on a cast
    # p0 [1, 1]  : pressure / depth that starts the integral
    # p1 [1, 1]  : pressure / depth that ends the integral
    # dp [1, 1]  : maximum interval of pressure / depth in numerical integration
    # interpfn [function handle]: function to calcualte piecewise polynomial coefficients
    #
    #
    # --- Output:
    #  d1 [1, 1]: Veronis density
    #
    #
    # --- Discussion:
    # The result of this function can serve as a density label for an
    # approximately neutral surface. However, this is NOT the same as a value
    # of the Jackett and McDougall (1997) Neutral Density variable. This is
    # true even if you were to provide this function with the same cast that
    # Jackett and McDougall (1997) used to initially label their Neutral
    # Density variable, namely the cast at 188 deg E, 4 deg S, from the Levitus
    # (1982) ocean atlas. Some difference would remain, because of differences
    # in numerics, and because of a subsequent smoothing step in the Jackett
    # and McDougall (1997) algorithm. This function merely allows one to label
    # an approximately neutral surface with a density value that is INTERNALLY
    # consistent within the dataset where one's surface lives. This function is
    # NOT to compare density values against those from any other dataset, such
    # as 1997 Neutral Density.
    #
    #
    # --- References:
    # Veronis, G. (1972). On properties of seawater defined by temperature,
    # salinity, and pressure. Journal of Marine Research, 30(2), 227.
    #
    # Stanley, G. J., McDougall, T. J., & Barker, P. M. (2021). Algorithmic
    # improvements to finding approximately neutral surfaces. Journal of
    # Advances in Modeling Earth Systems, submitted.

    # assert(all(size(T) == size(S)), 'T must be same size as S')
    # assert(all(size(P) == size(S)), 'P must be same size as S')
    # assert(isvector(S), 'S, T, P must be 1D. (Veronis density is only useful for one water column at a time!)')
    # assert(isscalar(p0), 'p0 must be a scalar')
    # assert(isscalar(p1), 'p1 must be a scalar')

    # Interpolate S and T as piecewise polynomials of P
    Sppc = interpfn(P, S)
    Tppc = interpfn(P, T)

    # Calculate potential density, referenced to p_ref, at p0
    s0, t0 = val2_0d(P, S, Sppc, T, Tppc, p0)
    d0 = eos(s0, t0, p_ref)

    # i = searchsorted(X,x) is such that:
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
    d1 = d0 + _int_x_k(p0, k0, dp, P, S, T, Sppc, Tppc, eos_s_t)

    # Integrate from P[k0] to P[k1]
    for k in range(k0, k1):
        # Integrate from P[k] to P[k+1]
        d1 = d1 + _int_x_k(P[k], k + 1, dp, P, S, T, Sppc, Tppc, eos_s_t)

    # Integrate from p1 to P[k1], and subtract this
    d1 = d1 - _int_x_k(p1, k1, dp, P, S, T, Sppc, Tppc, eos_s_t)

    return d1


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
