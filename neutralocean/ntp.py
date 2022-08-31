"""Neutral Tangent Plane and related functions"""

import numpy as np
import numba as nb

from neutralocean.lib import xr_to_np
from neutralocean.grid.graph import graph_binary_fcn
from neutralocean.eos.tools import make_eos_s_t


def ntp_ϵ_errors(s, t, p, grid, eos_s_t="gsw", grav=None, rho_c=None):
    """
    Calculate ϵ neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p, eos_s_t, grav, rho_c :
        See `ntp_ϵ_errors_norms`.

    grid : dict
        See `ntp_ϵ_errors_norms`.
        Need not have the `distperp` element, as this is not used.
        If the `dist` element is missing, a value of 1.0 will be used.
        Can alternatively pass a 2 element tuple that is just `grid['edges']`,
        in which case `dist` will be taken as 1.0.


    Returns
    -------
    ϵ : array
        The ϵ neutrality errors on the surface.
        `ϵ[i]` is the neutrality error between nodes `a[i]` and `b[i]`, where
        `a, b = grid['edges']`.
    """

    if isinstance(grid, tuple):
        edges = grid
        dist = 1.0
    elif isinstance(grid, dict):
        edges = grid["edges"]
        dist = grid.get("dist", 1.0)

    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)

    s, t, p = (xr_to_np(x) for x in (s, t, p))

    sa, ta, pa = (graph_binary_fcn(edges, x, avg1) for x in (s, t, p))
    ds, dt = (graph_binary_fcn(edges, x, dif1) for x in (s, t))

    rsa, rta = eos_s_t(sa, ta, pa)
    ϵ = rsa * ds + rta * dt
    if dist is not float or dist != 1.0:
        ϵ = ϵ / dist
    return ϵ


def ntp_ϵ_errors_norms(s, t, p, grid, eos_s_t="gsw", grav=None, rho_c=None):
    """
    Calculate norms of the ϵ neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p : ndarray
        practical / Absolute Salinity, potential / Conservative temperature,
        and pressure / depth on the surface

    grid : dict
        Containing the following:

        edges : tuple of length 2, Required
            Each element is an array of int of length E, where E is the number of
            edges in the grid's graph, i.e. the number of pairs of adjacent water
            columns (including land) in the grid.
            If `edges = (a, b)`, the nodes (water columns) whose linear indices are
            `a[i]` and `b[i]` are adjacent.
            If one of `a[i]` or `b[i]` is less than 0, that edge does not exist. # TODO:  Is this necessary?

        dist : 1d array, Default 1.0
            Horizontal distance between adjacent water columns (nodes).
            `dist[i]` is the distance between nodes whose linear indices are
            `edges[0][i]` and `edges[1][i]`.

        distperp : 1d array, Default 1.0
            Horizontal distance of the face between adjacent water columns (nodes).
            `distperp[i]` is the distance of the interface between nodes whose
            linear indices are `edges[0][i]` and `edges[1][i]`.

    eos_s_t : function
        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T` as a function of `S`, `T`, and `P`
        inputs.

    grav : float, Default None
        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None
        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    Returns
    -------
    ϵ_RMS : float
        Area-weighted root-mean-square of the ϵ neutrality error on the surface

    ϵ_MAV : TYPE
        Area-weighted mean-absolute-value of the ϵ neutrality error on the surface

    """

    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)

    # Calculate ϵ neutrality errors.  Here, treat all distances = 1.
    # The actual distances will be handled in computing the norms
    ϵ = ntp_ϵ_errors(s, t, p, {"edges": grid["edges"]}, eos_s_t)

    area = grid["dist"] * grid["distperp"]  # Area [m^2] centred on edges

    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # Here, weights are `area`.
    # But also need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     area / dist^2 = distperp / dist,
    ϵ_RMS = np.sqrt(
        np.nansum(grid["distperp"] / grid["dist"] * ϵ**2)
        / np.sum(area * np.isfinite(ϵ))
    )

    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are `area`.
    # But also need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L1 norm needs to multiply epsilon by
    #     area ./ dist = distperp ,
    ϵ_MAV = np.nansum(grid["distperp"] * abs(ϵ)) / np.sum(
        area * np.isfinite(ϵ)
    )

    return ϵ_RMS, ϵ_MAV


@nb.njit
def avg1(a, b):
    return (a + b) * 0.5


@nb.njit
def dif1(a, b):
    return a - b
