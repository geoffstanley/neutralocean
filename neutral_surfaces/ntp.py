"""Neutral Tangent Plane and related functions"""

import numpy as np

from neutral_surfaces.lib import _process_wrap, xr_to_np


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
    s, t, p, eos_s_t, wrap, dist1_iJ=1.0, dist1_Ij=1.0, dist2_Ij=1.0, dist2_iJ=1.0
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
