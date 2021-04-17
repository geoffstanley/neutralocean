import numpy as np
import numba

from neutral_surfaces._densjmd95 import rho_s_t_bsq


def ntp_errors(s, t, p, wrap):
    # Get errors by backward differences, and leave on the U, V grid.

    def helper(s, t, p, shift):
        def avg(F):
            return (F + shift(F)) * 0.5

        def dif(F):
            return F - shift(F)

        sa = avg(s)
        ta = avg(t)
        pa = avg(p)

        rsa, rta = rho_s_t_bsq(sa, ta, pa)
        return rsa * dif(s) + rta * dif(t)

    # Begin calculations for errors in x direction
    def im1(F):  # G[i,j] == F[i-1,j]
        G = np.roll(F, 1, axis=0)
        if wrap[0]:
            G[0, :] = np.nan
        return G

    ϵx = helper(s, t, p, im1)

    # Begin calculations for errors in y direction.
    def jm1(F):  # G[i,j] == F[i,j-1]
        G = np.roll(F, 1, axis=1)
        if wrap[1]:
            G[:, 0] = np.nan
        return G

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
    ϵL2 = np.sqrt(
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
    ϵL1 = (np.nansum(DIST2_iJ * abs(ϵ_iJ)) + np.nansum(DIST1_Ij * abs(ϵ_Ij))) / (
        np.sum(AREA_iJ * np.isfinite(ϵ_iJ)) + np.sum(AREA_Ij * np.isfinite(ϵ_Ij))
    )

    return ϵL2, ϵL1
