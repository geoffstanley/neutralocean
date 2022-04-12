"""Neutral Tangent Plane and related functions"""

import numpy as np
import numba as nb

from neutralocean.lib import xr_to_np
from neutralocean.grid.graph import edges_binary_fcn


def ntp_ϵ_errors(s, t, p, eos_s_t, edges, dist=1.0):
    """
    Calculate ϵ neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p : ndarray
        practical / Absolute Salinity, potential / Conservative temperature,
        and pressure / depth on the surface

    eos_s_t : function
        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T` as a function of `S`, `T`, and `P`
        inputs.

    # TODO: update docs
    wrap : tuple of bool, or tuple of str

        Specifies which dimensions are periodic.

        As a tuple of bool, this must be length two.  The first and second
        dimensions of `s` and `t` are periodic iff ``wrap[0]`` and
        ``wrap[1]`` is True, respectively.

        As a tuple of str, simply name the periodic dimensions of `s` and
        `t`.

    dist1_iJ, dist2_Ij, : float or ndarray, Default 1.0

        Grid distances [m] in either the 1st or 2nd lateral dimension, and
        centred at the location specified.  The naming uses a soft notation:
        the central grid point is(I,J), and i = I-1/2 and j = J-1/2.  Thus,
        `dist1_iJ[5,3]` is the distance between cells (5,3) and (4,3), while
        `dist2_Ij[5,3]` is the distance between cells (5,3) and (5,2).

    Returns
    -------
    ϵx, ϵy : ndarray
        The ϵ neutrality errors (in the first and second lateral directions)
        on the surface.  Results live on the half grids, midway between where
        `s`, `t`, and `p` live.
    """

    s, t, p = (xr_to_np(x) for x in (s, t, p))

    sa, ta, pa = (edges_binary_fcn(x, edges, avg1) for x in (s, t, p))
    ds, dt = (edges_binary_fcn(x, edges, dif1) for x in (s, t))

    rsa, rta = eos_s_t(sa, ta, pa)
    ϵ = rsa * ds + rta * dt
    if dist is not float or dist != 1.0:
        ϵ = ϵ / dist
    return ϵ


def ntp_ϵ_errors_norms(s, t, p, eos_s_t, edges, dist=1.0, distperp=1.0):
    """
    Calculate norms of the ϵ neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p, eos_s_t, wrap :
        See ntp_ϵ_errors

    dist : 1d array of float
        Distance [m] between nodes connected by edges

    distperp : 1d array of float
        Distance [m] of the face between nodes connected by edges

    dist1_iJ, dist1_Ij, dist2_Ij, dist2_iJ : float or ndarray, Default 1.0

        Grid distances [m] in either the 1st or 2nd lateral dimension, and
        centred at the location specified.  The naming uses a soft notation:
        the central grid point is(I,J), and i = I-1/2 and j = J-1/2.  Thus,
        `dist1_iJ[5,3]` is the distance between cells (5,3) and (4,3), while
        `dist2_iJ[5,3]` is the distance of the face between cells (5,3) and
        (4,3). Similarly, `dist2_Ij[5,3]` is the distance between cells
        (5,3) and (5,2), while `dist1_Ij[5,3]` is the distance of the face
        between cells (5,3) and (5,2).

    Returns
    -------
    ϵ_RMS : float
        Area-weighted root-mean-square of the ϵ neutrality error on the surface

    ϵ_MAV : TYPE
        Area-weighted mean-absolute-value of the ϵ neutrality error on the surface

    """

    # Calculate ϵ neutrality errors.  Here, treat all distances = 1.
    # The actual distances will be handled in computing the norms
    ϵ = ntp_ϵ_errors(s, t, p, eos_s_t, edges)

    area = dist * distperp  # Area [m^2] centred on edges

    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are `area`.
    # But also need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     area / dist^2 = distperp / dist,
    ϵ_RMS = np.sqrt(np.nansum(distperp / dist * ϵ**2) / np.sum(area * np.isfinite(ϵ)))

    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are `area`.
    # But also need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L1 norm needs to multiply epsilon by
    #     area ./ dist = distperp ,
    ϵ_MAV = np.nansum(distperp * abs(ϵ)) / np.sum(area * np.isfinite(ϵ))

    return ϵ_RMS, ϵ_MAV


# def im1(F, wrap, fill=np.nan):  # G[i,j] == F[i-1,j]
#     G = np.roll(F, 1, axis=0)
#     if not wrap[0]:
#         G[0, :] = fill
#     return G


# def ip1(F, wrap, fill=np.nan):  # G[i,j] == F[i+1,j]
#     G = np.roll(F, -1, axis=0)
#     if not wrap[0]:
#         G[-1, :] = fill
#     return G


# def jm1(F, wrap, fill=np.nan):  # G[i,j] == F[i,j-1]
#     G = np.roll(F, 1, axis=1)
#     if not wrap[1]:
#         G[:, 0] = fill
#     return G


# def jp1(F, wrap, fill=np.nan):  # G[i,j] == F[i,j+1]
#     G = np.roll(F, -1, axis=1)
#     if not wrap[1]:
#         G[:, -1] = fill
#     return G


# def avg(F, shift, wrap):
#     return (F + shift(F, wrap)) * 0.5


# def dif(F, shift, wrap):
#     return F - shift(F, wrap)


@nb.njit
def avg1(a, b):
    return (a + b) * 0.5


@nb.njit
def dif1(a, b):
    return a - b


# def avg(a, adj):
#     return (a + shift(a, adj)) * 0.5

# def dif(a, adj):
#     return a - shift(a, adj)


# # 180 µs
# @nb.njit
# def shpy(a, A4, d):
#     sh = a.shape
#     b = np.empty(a.size + 1, dtype=a.dtype)
#     b[0:-1] = a.reshape(-1)
#     b[-1] = np.nan
#     b = b[A4[:, d]]
#     return b.reshape(sh)


# # 217 µs
# @nb.njit
# def shpy2(a, A4, d):
#     sh = a.shape
#     b = a.reshape(-1)[A4[:, d]]
#     b[A4[:, d] == -1] = np.nan
#     return b.reshape(sh)


# # 148 µs
# @nb.njit
# def sh(a, A4, d):
#     b = np.empty(a.size, dtype=a.dtype)
#     sh = a.shape
#     a = a.reshape(-1)
#     for i in range(A4.shape[0]):
#         n = A4[i, d]
#         if n == -1:
#             b[i] = np.nan
#         else:
#             b[i] = a[n]
#     return b.reshape(sh)


# # 112 µs
# @nb.njit
# def sh0pre(a, A4, d, b):
#     sh = a.shape
#     a = a.reshape(-1)
#     for i in range(A4.shape[0]):
#         if A4[i, d] >= 0:
#             b[i] = a[A4[i, d]]
#         else:
#             b[i] = np.nan
#     b = b.reshape(sh)


# # 116 µs
# @nb.njit
# def sh01pre(a, A1, b):
#     sh = a.shape
#     a = a.reshape(-1)
#     for i in range(len(A1)):
#         if A1[i] >= 0:
#             b[i] = a[A1[i]]
#         else:
#             b[i] = np.nan
#     b = b.reshape(sh)


# # 149 µs
# @nb.njit
# def sh1(a, A4, d):
#     b = np.empty(a.size, dtype=a.dtype)
#     sh = a.shape
#     a = a.reshape(-1)
#     A1 = A4[:, d]
#     for i in range(A4.shape[0]):
#         n = A1[i]
#         if n == -1:
#             b[i] = np.nan
#         else:
#             b[i] = a[n]
#     return b.reshape(sh)
