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

    edges : ndarray of int

        A 2D array with `edges.shape[1] == 2` that specifies pairs of water
        columns that are adjacent.  Each water column (including land) is
        indexed by an integer: 0, 1, 2, ... N.  So, the water columns indexed
        by `edges[i,0]` and `edges[i,1]` are adjacent, for any valid `i`.

        For a rectilinear grid (e.g. latitude-longitude), use
            `neutralocean.grid.rectilinear.build_edges`

        For a tiled rectilinear grid, such as works with XGCM, use
            `neutralocean.grid.xgcm.build_edges_and_geometry`

        For a general grid given as a graph, use
            `neutralocean.grid.graph.graph_to_edges`

    dist : array or float

        Distance [m] between nodes connected by edges. `dist[i]` is the
        distance between water columns `edges[i,0]` and `edges[i,1]`.

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


def ntp_ϵ_errors_norms(s, t, p, eos_s_t, edges, geometry=(1.0, 1.0)):
    """
    Calculate norms of the ϵ neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p, eos_s_t, edges :
        See ntp_ϵ_errors

    geometry : tuple
        The geometry of the horizontal grid, a tuple of length 2 with the following:

        dist : 1d array of float
            Distance [m] between nodes connected by edges.
            E.g. `dist[i]` is the distance between water columns `edges[i,0]`
            and `edges[i,1]`.

        distperp : 1d array of float
            Distance [m] of the face between nodes connected by edges.
            E.g. `distperp[i]` is the distance of the face between water
            columns `edges[i,0]` and `edges[i,1]`.

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

    dist, distperp = geometry

    area = dist * distperp  # Area [m^2] centred on edges

    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are `area`.
    # But also need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     area / dist^2 = distperp / dist,
    ϵ_RMS = np.sqrt(
        np.nansum(distperp / dist * ϵ**2) / np.sum(area * np.isfinite(ϵ))
    )

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
