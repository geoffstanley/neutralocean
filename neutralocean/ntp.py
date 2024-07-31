"""Neutral Tangent Plane and related functions"""

import numpy as np
import numba as nb

from neutralocean.lib import xr_to_np
from neutralocean.eos.tools import load_eos


def ntp_epsilon_errors(s, t, p, grid, eos="gsw", grav=None, rho_c=None):
    """
    Calculate epsilon neutrality errors on an approximately neutral surface

    Parameters
    ----------
    s, t, p, eos_s_t, grav, rho_c :
        See `ntp_epsilon_errors_norms`.

    grid : dict
        See `ntp_epsilon_errors_norms`.
        Need not have the `distperp` element, as this is not used.
        If the `dist` element is missing, a value of 1.0 will be used.
        Can alternatively pass a 2 element tuple that is just `grid['edges']`,
        in which case `dist` will be taken as 1.0.

    eos, grav, rho_c : 
        See `ntp_epsilon_errors_norms`.

    Returns
    -------
    e : array
        The epsilon neutrality errors on the surface.
        `e[i]` is the neutrality error from node `a[i]` to node `b[i]`, where
        `a, b = grid['edges']`.
    """

    if isinstance(grid, tuple):
        edges = grid
        dist = 1.0
    elif isinstance(grid, dict):
        edges = grid["edges"]
        dist = grid.get("dist", 1.0)

    eos_s_t = load_eos(eos, "_s_t", grav, rho_c)

    s, t, p = (np.reshape(xr_to_np(x), -1) for x in (s, t, p))
    e = _ntp_epsilon_error1(s, t, p, edges[0], edges[1], eos_s_t)

    if dist is not float or dist != 1.0:
        e = e / dist
    return e


def ntp_epsilon_errors_norms(
    s, t, p, grid, eos_s_t="gsw", grav=None, rho_c=None
):
    """
    Calculate norms of the epsilon neutrality errors on an approximately neutral surface

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

        dist : 1d array, Default 1.0
            Horizontal distance between adjacent water columns (nodes).
            `dist[i]` is the distance between nodes whose linear indices are
            `edges[0][i]` and `edges[1][i]`.

        distperp : 1d array, Default 1.0
            Horizontal distance of the face between adjacent water columns (nodes).
            `distperp[i]` is the distance of the interface between nodes whose
            linear indices are `edges[0][i]` and `edges[1][i]`.

    eos_s_t : str or function, Default 'gsw'

        Specification for the equation of state.

        If a str, can be any of the strings accepted by `neutralocean.eos.tools.load_eos`.

        If a function, must take three inputs corresponding to `S`, `T`, and
        `P`, and output the density (or specific volume). This can be made as, e.g.,
        `eos = neutralocean.eos.load_eos('gsw')`
        for a non-Boussinesq ocean, or as
        `eos = neutralocean.eos.load_eos('gsw', "", grav, rho_c)`
        for a Boussinesq ocean with `grav` and `rho_c` (see inputs below).

        The function should be `@numba.njit` decorated and need not be vectorized
        -- it will be called many times with scalar inputs.

        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T` as a function of `S`, `T`, and `P`
        inputs.

    grav : float, Default None
        Gravitational acceleration [m s-2].  When non-Boussinesq, pass `None`.

    rho_c : float, Default None
        Boussinesq reference density [kg m-3].  When non-Boussinesq, pass `None`.

    Returns
    -------
    e_RMS : float
        Area-weighted root-mean-square of the epsilon neutrality error on the surface

    e_MAV : TYPE
        Area-weighted mean-absolute-value of the epsilon neutrality error on the surface

    """

    eos_s_t = load_eos(eos_s_t, "_s_t", grav, rho_c)

    # Calculate epsilon neutrality errors.  Here, treat all distances = 1.
    # The actual distances will be handled in computing the norms
    e = ntp_epsilon_errors(s, t, p, {"edges": grid["edges"]}, eos_s_t)

    area = grid["dist"] * grid["distperp"]  # Area [m^2] centred on edges


    # L2 norm of vector [a_i], weighted by vector [w_i], is sqrt( sum( w_i * a_i^2 ) / sum( w_i ) )
    # L1 norm of vector [a_i], weighted by vector [w_i], is sum( w_i * |a_i| ) / sum( w_i )
    # Here, weights are `area`.
    denom = np.sum(area * np.isfinite(e))
    if denom == 0.0:
        # This happens when a surface is one, isolated grid cell.
        return 0.0, 0.0

    # Still need to divide epsilon by grid distances `dist`.
    # Thus, the numerator of L2 norm needs to multiply epsilon^2 by
    #     area / dist^2 = distperp / dist,
    # and the numerator of L1 norm needs to multiply epsilon by
    #     area ./ dist = distperp.
    e_RMS = np.sqrt(np.nansum(grid["distperp"] / grid["dist"] * e**2) / denom)
    e_MAV = np.nansum(grid["distperp"] * abs(e)) / denom

    return e_RMS, e_MAV


@nb.njit
def _ntp_epsilon_error1(s, t, p, a, b, eos_s_t):
    e = np.empty(len(a), dtype=s.dtype)
    for i in range(len(a)):
        a_ = a[i]
        b_ = b[i]
        rs, rt = eos_s_t(
            0.5 * (s[a_] + s[b_]),
            0.5 * (t[a_] + t[b_]),
            0.5 * (p[a_] + p[b_]),
        )
        e[i] = rs * (s[b_] - s[a_]) + rt * (t[b_] - t[a_])
    return e
