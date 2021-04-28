import numpy as np
import numba

from neutral_surfaces._densjmd95 import rho_bsq, rho_s_t_bsq

from neutral_surfaces._zero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import val2_0d


def ntp_errors(s, t, p, wrap):
    # Get errors by backward differences, and leave on the U, V grid.

    def helper(s, t, p, shift):
        def avg(F):
            return (F + shift(F, wrap)) * 0.5

        def dif(F):
            return F - shift(F, wrap)

        sa = avg(s)
        ta = avg(t)
        pa = avg(p)

        rsa, rta = rho_s_t_bsq(sa, ta, pa)
        return rsa * dif(s) + rta * dif(t)

    ϵx = helper(s, t, p, im1)
    ϵy = helper(s, t, p, jm1)

    return ϵx, ϵy


def ϵ_norms(s, t, p, wrap, DIST1_iJ, DIST2_Ij, DIST2_iJ, DIST1_Ij, *args):

    ϵ_iJ, ϵ_Ij = ntp_errors(s, t, p, wrap)

    if len(args) == 2:
        AREA_iJ = args[0]  # Area [m^2] centred at (I-1/2, J)
        AREA_Ij = args[1]  # Area [m^2] centred at (I, J-1/2)
    else:
        AREA_iJ = DIST1_iJ * DIST2_iJ  # Area [m^2] centred at (I-1/2, J)
        AREA_Ij = DIST1_Ij * DIST2_Ij  # Area [m^2] centred at (I, J-1/2)

    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances DIST1_iJ and DIST2_Ij.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     AREA_iJ ./ DIST1_iJ.^2 = DIST2_iJ ./ DIST1_iJ ,
    # and AREA_Ij ./ DIST2_Ij.^2 = DIST1_Ij ./ DIST2_Ij .
    ϵ_L2 = np.sqrt(
        (
            np.nansum(DIST2_iJ / DIST1_iJ * ϵ_iJ ** 2)
            + np.nansum(DIST1_Ij / DIST2_Ij * ϵ_Ij ** 2)
        )
        / (np.sum(AREA_iJ * np.isfinite(ϵ_iJ)) + np.sum(AREA_Ij * np.isfinite(ϵ_Ij)))
    )

    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are AREA_iJ and AREA_Ij.
    # But also need to divide epsilon by grid distances DIST1_iJ and DIST2_Ij.
    # Thus, the numerator of L1 norm needs to multiply epsilon by
    #     AREA_iJ ./ DIST1_iJ = DIST2_iJ ,
    # and AREA_Ij ./ DIST2_Ij = DIST1_Ij .
    ϵ_L1 = (np.nansum(DIST2_iJ * abs(ϵ_iJ)) + np.nansum(DIST1_Ij * abs(ϵ_Ij))) / (
        np.sum(AREA_iJ * np.isfinite(ϵ_iJ)) + np.sum(AREA_Ij * np.isfinite(ϵ_Ij))
    )

    return ϵ_L2, ϵ_L1


@numba.njit
def ntp_bottle_to_cast(sB, tB, pB, P, S, Sppc, T, Tppc, k, tolp):
    # Py dev:  removed "success" output

    # NTP_BOTTLE_TO_CAST  Find a bottle's level of neutral buoyancy in a water
    #                    column, using the Neutral Tangent Plane relationship.
    #
    # [p, s, t] = ntp_bottle_to_cast(Sppc, Tppc, P, k, sB, tB, pB, tolp)
    # finds (s, t, p), with precision in p of tolp, that is at the level of
    # neutral buoyancy for a fluid bottle of (sB, tB, pB) in a water column of
    # with piecewise polynomial interpolants for S and T given by Sppc and Tppc
    # with knots at P(1:k).  Specifically, s and t are given by
    #   [s,t] = ppc_val(P, Sppc, Tppc, p)
    # and p satisfies
    #      eos(s, t, p') = eos(sB, tB, p')
    # where eos is the equation of state given by eos.m in MATLAB's path,
    # and   p' is in the range [p_avg - tolp/2, p_avg + tolp/2],
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
    # tolp [1, 1]: tolerance for solving the level of neutral buoyancy (same
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

        args = (sB, tB, pB, P, S, Sppc, T, Tppc)

        # Search for a sign-change, expanding outward from an initial guess
        lb, ub = guess_to_bounds(myfcn, args, pB, P[0], P[k - 1])

        if np.isfinite(lb):
            # A sign change was discovered, so a root exists in the interval.
            # Solve the nonlinear root-finding problem using Brent's method
            p = brent(myfcn, args, lb, ub, tolp)

            # Interpolate S and T onto the updated surface
            s, t = val2_0d(P, S, Sppc, T, Tppc, p)

        else:
            s = np.nan
            t = np.nan
            p = np.nan

    else:
        s = np.nan
        t = np.nan
        p = np.nan

    return s, t, p


@numba.njit
def myfcn(p, sB, tB, pB, P, S, Sppc, T, Tppc):
    # Evaluate difference between (a) eos at location on the cast (S, T, P)
    # where the pressure or depth is p, and (b) eos of the bottle (sB, tB, pB);
    # here, eos is always evaluated at the average pressure or depth, (p +
    # pB)/2.
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    p_avg = (pB + p) * 0.5
    return rho_bsq(sB, tB, p_avg) - rho_bsq(s, t, p_avg)


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