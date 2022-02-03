"""Omega surfaces"""

import numpy as np
from time import time
from scipy.sparse import csc_matrix

# from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky

from neutralocean.surface.trad import _traditional_surf
from neutralocean.surface._vertsolve import _make_vertsolve
from neutralocean.interp1d import make_interpolator
from neutralocean.ppinterp import select_ppc
from neutralocean.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutralocean.ntp import ntp_ϵ_errors_norms
from neutralocean.lib import (
    xr_to_np,
    _xr_in,
    _xr_out,
    _process_pin_cast,
    _process_wrap,
    _process_casts,
    _process_n_good,
    _process_eos,
)
from neutralocean.mixed_layer import mixed_layer


def omega_surf(S, T, P, **kwargs):
    """Calculate an omega surface from structured ocean data.

    Given 3D salinity, temperature, and pressure or depth data arranged on a
    rectilinear grid, calculate a 2D omega surface [1]_ [2]_, which is a
    highly accurate approximately neutral surface.

    Parameters
    ----------
    S, T, P : ndarray or xarray.DataArray
        See `potential_surf`

    p_init : ndarray, Default None

        Pressure or depth on the initial approximately neutral surface.

        See Examples section.

    ref : float, or tuple of float of length 2

        If `p_init` is None, the reference value(s) for the initial potential
        density surface or in-situ density (specific volume) anomaly surface
        that initializes the omega surface algorithm. If `ref` is a scalar, a
        potential density urface is used, and if `ref` is None, the reference
        `P` is `pin_p` (i.e. taken local to the pinning location). If `ref`
        is a tuple of length two, a in-situ density anomaly surface is used,
        and if `ref` is (None, None), then the reference `S` and `T` values
        are taken local to the pinning location (pressure or depth `pin_p` on
        the pinning cast `pin_cast`).

        See Examples section.

    isoval : float

        Isovalue for the initial potential density or in-situ density anomaly
        surface when `p_init` is not given.  Units are same as returned by
        the function specified by `eos`.

        See Examples section.

    pin_p, pin_cast :
        See `potential_surf`

    Returns
    -------
    s, t, p : ndarray or xarray.DataArray

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure / depth on surface

    d : dict

        Diagnostics.  The first four give information going into the `i`'th
        iteration (e.g. the 0'th element is about the initial surface).  The
        others give information about what the `i`'th iteration did (and hence
        their 0'th elements are irrelevant).

        ``"ϵ_MAV"`` : array of float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.

        ``"ϵ_RMS"`` : array of float

            As ``"ϵ_MAV"`` but for the Root Mean Square.

        ``"n_wet"``: array of float

            Number of wet casts (surface points).

        ``"timer"`` : array of float

            Time spent on each iteration, excluding set-up (approximately) and diagnostics.

        ``"ϕ_MAV"`` : array of float

            Mean Absolute Value of the Locally Referenced Potential Density
            perturbation, per iteration

        ``"Δp_MAV"`` : array of float

            Mean Absolute Value of the pressure or depth change from one
            iteration to the next

        ``"Δp_RMS"`` : array of float

            Root Mean Square of the pressure or depth change from one
            iteration to the next

        ``"Δp_Linf"`` : array of float

            Maximum absolute value (infinity norm) of the pressure or depth
            change from one iteration to the next

        ``"n_newly_wet"`` : array of int

            Number of casts that are newly wet, per iteration

        ``"timer_bfs"`` : array of float

            Time spent in Breadth-First Search including wetting, per iteration.

        ``"timer_mat"`` : array of float

            Time spent building and solving the matrix problem, per iteration.

        ``"timer_update"`` : array of float

            Time spent vertically updating the surface.

    Other Parameters
    ----------------
    wrap, vert_dim, dist1_iJ, dist1_Ij, dist2_Ij, dist2_iJ, grav, rho_c,
    interp, n_good, diags, output, TOL_P_SOLVER :

        See `potential_surf`

    eos : str or tuple of functions, Default 'gsw'

        As in `potential_surf`, excluding the option to pass a single function.
        The omega surface algorithm relies on knowing the partial derivatives
        of the equation of state with respect to salinity and temperature, so
        the `eos_s_t` function is also required.

    ITER_MIN : int, Default 1

        Minimum number of iterations.

    ITER_MAX : int, Default 10

        Maximum number of iterations.

    ITER_START_WETTING : int, Default 1

        Iteration on which wetting begins.  Set to `np.inf` (`ITER_MAX` + 1
        would also do) to deactivate.

    ITER_STOP_WETTING : int, Default 5

        The last iteration on which to perform wetting.  This can be useful to
        avoid pesky water columns that repeatedly wet then dry.

    TOL_LRPD_MAV : float, Default 1e-7

        Exit iterations when the mean absolute value of the Locally Referenced
        Potential Density perturbation that updates the surface from one
        iteration to the next is less than this value.  Units are [kg m-3],
        even if `eos` returns a specific volume.  Set to 0 to deactivate.

    TOL_P_CHANGE_RMS : float, Default 0.0

        Exit iterations when the root mean square of the pressure or depth
        change on the surface from one iteration to the next is less than this
        value. Set to 0 to deactivate. Units are the same as `P` [dbar or m].

    p_ml : ndarray or dict, Default None

        If a dict, the pressure or depth at the base of the mixed layer is
        computed using `mixed_layer` with p_ml passed as keyword arguments,
        enabling control over the parameters in that function.
        See `mixed_layer` for details.

        If an ndarray (of the same shape as the lateral dimensions of `S`),
        the pressure or depth at the base of the mixed layer in each water
        column.

        When the surface's pressure is shallower than `p_ml` in any water
        column, it is set to NaN (a "dry" water column). This is not applied
        to the initial surface, but only to the surface after the first
        iteration, as the initial surface could be very far from neutral.

        If None, the mixed layer is not removed.

    Examples
    --------
    omega surfaces require a pinning cast and initial surface.  The surface is
    iteratively updated while remaining fixed at the pinning cast.  The
    initial surface can be provided directly, as the surface with pressure or
    depth given by `p_init`, in the following method:

    >>> omega_surf(S, T, P, pin_cast, p_init, ...)

    Alternatively, a
    potential density surface
    or a
    in-situ density (specific volume) anomaly surface
    can be used as the initial
    surface.  To do this, use one of the following two methods

    >>> omega_surf(S, T, P, ref, isoval, pin_cast, ...)

    >>> omega_surf(S, T, P, ref, pin_p, pin_cast, ...)

    For more info on these methods, see the Examples section of `potential_surf`.
    Note that `pin_cast` is always a required input.  Note that `ref` is
    needed to distinguish which of the two types of traditional surfaces will
    be used as the initial surface.

    Notes
    -----
    See `potential_surf` Notes.

    .. [1] Stanley, McDougall, Barker 2021, Algorithmic improvements to finding
     approximately neutral surfaces, Journal of Advances in Earth System
     Modelling, 13(5).

    .. [2] Klocker, McDougall, Jackett 2009, A new method of forming approximately
     neutral surfaces, Ocean Science, 5, 155-172.
    """

    ref = kwargs.get("ref")
    pin_p = kwargs.get("pin_p")
    pin_cast = kwargs.get("pin_cast")
    p_init = kwargs.get("p_init")
    vert_dim = kwargs.get("vert_dim", -1)
    p_ml = kwargs.get("p_ml")
    wrap = kwargs.get("wrap")
    diags = kwargs.get("diags", True)
    output = kwargs.get("output", True)
    eos = kwargs.get("eos", "gsw")
    rho_c = kwargs.get("rho_c")
    grav = kwargs.get("grav")
    ITER_MIN = kwargs.get("ITER_MIN", 1)
    ITER_MAX = kwargs.get("ITER_MAX", 10)
    ITER_START_WETTING = kwargs.get("ITER_START_WETTING", 1)
    ITER_STOP_WETTING = kwargs.get("ITER_STOP_WETTING", 5)
    TOL_P_SOLVER = kwargs.get("TOL_P_SOLVER", 1e-4)
    TOL_LRPD_MAV = kwargs.get("TOL_LRPD_MAV", 1e-7)
    TOL_P_CHANGE_RMS = kwargs.get("TOL_P_CHANGE_RMS", 0.0)
    # fmt: off
    # grid distances.  (soft notation: i = I-1/2; j = J-1/2)
    # dist1_iJ = kwargs.get('dist1_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)
    # dist1_Ij = kwargs.get('dist1_Ij', 1.) # Distance [m] in 1st dim centred at (I, J-1/2)
    # dist2_Ij = kwargs.get('dist2_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_iJ = kwargs.get('dist2_iJ', 1.) # Distance [m] in 2nd dim centred at (I-1/2, J)
    # fmt: on
    # dist2on1_iJ = dist2_iJ / dist1_iJ
    # dist1on2_Ij = dist1_Ij / dist2_Ij
    geom = [
        kwargs.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ")
    ]

    n_good = kwargs.get("n_good")
    interp = kwargs.get("interp", "linear")

    ppc_fn = select_ppc(interp, "1")
    interp_u_two = make_interpolator(interp, 0, "u", True)

    sxr, txr, pxr = (_xr_in(X, vert_dim) for X in (S, T, P))  # before _process_casts
    pin_cast = _process_pin_cast(pin_cast, S)  # call before _process_casts
    wrap = _process_wrap(wrap, sxr, True)  # call before _process_casts
    S, T, P = _process_casts(S, T, P, vert_dim)
    n_good = _process_n_good(S, n_good)  # call after _process_casts
    eos, eos_s_t = _process_eos(eos, grav, rho_c, need_s_t=True)
    ni, nj = n_good.shape

    # Prepare grid ratios for matrix problem.
    if not np.all(geom == 1.0):
        geom = [np.broadcast_to(x, (ni, nj)) for x in geom]
    dist2on1_iJ = geom[3] / geom[0]  # dist2_iJ / dist1_iJ
    dist1on2_Ij = geom[1] / geom[2]  # dist1_Ij / dist2_Ij

    if not isinstance(pin_cast, (tuple, list)):
        raise TypeError("`pin_cast` must be a tuple or list")

    pin_cast_1 = np.ravel_multi_index(pin_cast, (ni, nj))  # linear index

    # Pre-calculate grid adjacency needed for Breadth First Search:
    A4 = grid_adjacency((ni, nj), 4, wrap)  # using 4-connectivity

    if eos(34.5, 3.0, 1000.0) < 1.0:
        # Convert from a density tolerance [kg m^-3] to a specific volume tolerance [m^3 kg^-1]
        TOL_LRPD_MAV = TOL_LRPD_MAV * 1000.0 ** 2

    # Pre-allocate arrays for diagnostics
    if diags:
        d = {
            "ϵ_MAV": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "ϵ_RMS": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "ϕ_MAV": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_MAV": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_RMS": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_Linf": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "n_wet": np.zeros(ITER_MAX + 1, dtype=int),
            "n_newly_wet": np.zeros(ITER_MAX + 1, dtype=int),
            "timer_bfs": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_mat": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_update": np.zeros(ITER_MAX + 1, dtype=np.float64),
        }
    else:
        d = dict()

    timer = time()
    if p_init is None:
        # Calculate an initial "potential" or "anomaly" surface
        if isinstance(ref, (tuple, list)) and len(ref) == 2:
            ans_type = "anomaly"
        else:
            ans_type = "potential"

        # Update arguments with pre-processed values
        kwargs["n_good"] = n_good
        kwargs["wrap"] = wrap
        kwargs["vert_dim"] = -1  # Since S, T, P already reordered
        kwargs["diags"] = False  # Will make our own diags next
        kwargs["eos"] = eos
        s, t, p, _ = _traditional_surf(ans_type, S, T, P, **kwargs)

    else:
        # Handling and error checking on p_init
        p_init = xr_to_np(p_init)
        if not isinstance(p_init, np.ndarray):
            raise TypeError(
                'If provided, "p_init" or "p_init.values" must be an ndarray'
            )
        if p_init.shape != (ni, nj):
            raise ValueError(
                f'"p_init" should contain a 2D array of size ({ni}, {nj});'
                f" found size {p_init.shape}"
            )

        if pin_p is not None and pin_p != p_init[pin_cast]:
            raise ValueError("pin_p does not match p_init at pin_cast")

        p = p_init.copy()

        # Interpolate S and T onto the surface
        s, t = interp_u_two(p, P, S, T)

    pin_p = p[pin_cast]

    if np.isnan(p[pin_cast]):
        raise RuntimeError("The initial surface is NaN at the reference cast.")

    # Calculate bottom of mixed layer from given options
    if ITER_MAX > 1 and isinstance(p_ml, dict):
        # Compute the mixed layer from parameter inputs
        p_ml = mixed_layer(S, T, P, eos, **p_ml)

    if p_ml is None:
        # Prepare array as needed for bfs_conncomp1_wet
        p_ml = np.full((ni, nj), -np.inf)
        # p_ml = np.broadcast_to(-np.inf, (ni, nj))  # DEV: Doesn't work with @numba.njit

    # ensure same nan structure between s, t, and p. Just in case user gives
    # np.full((ni,nj), 1000) for a 1000dbar isobaric surface, for example
    p[np.isnan(s)] = np.nan

    if diags:
        d["timer"][0] = time() - timer

        ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, p, eos_s_t, wrap, *geom)
        d["ϵ_RMS"][0], d["ϵ_MAV"][0] = ϵ_RMS, ϵ_MAV

        n_wet = np.sum(np.isfinite(p))
        d["n_wet"][0] = n_wet

        if output:
            print(
                "iter |"
                "    MAV(ϕ)     |"
                "    RMS(Δp)      |"
                " # wet casts (# new) |"
                "     RMS(ϵ)     |"
                " time (s)"
            )
            print(
                f"{0:4d} |"
                f"                                 |"
                f" {d['n_wet'][0]:11d}         |"
                f" {ϵ_RMS:.8e} |"
                f" {d['timer'][0]:.3f}"
            )

    vertsolve = _make_vertsolve(eos, ppc_fn, "omega")

    # --- Begin iterations
    # Note: the surface exists wherever p is non-nan.  The nan structure of s
    # and t is made to match that of p when the vertical solve step is done.
    Δp_RMS = 0.0  # ensure this is defined; needed if TOL_P_CHANGE_RMS == 0
    for iter_ in range(1, ITER_MAX + 1):
        timer = time()

        # --- Remove the Mixed Layer
        if iter_ > 1 and p_ml[0, 0] != -np.inf:
            p[p < p_ml] = np.nan

        # --- Determine the connected component containing the reference cast, via Breadth First Search
        timer_loc = time()
        if iter_ >= ITER_START_WETTING and iter_ <= ITER_STOP_WETTING:
            qu, qt, n_newly_wet = bfs_conncomp1_wet(
                s,
                t,
                p,
                S,
                T,
                P,
                n_good,
                A4,
                pin_cast_1,
                TOL_P_SOLVER,
                eos,
                ppc_fn,
                p_ml=p_ml,
            )
        else:
            qu, qt = bfs_conncomp1(np.isfinite(p.flatten()), A4, pin_cast_1)
            n_newly_wet = 0
        timer_bfs = time() - timer_loc

        # --- Solve global matrix problem for the exactly determined Poisson equation
        timer_loc = time()
        ϕ = _omega_matsolve_poisson(
            s, t, p, dist2on1_iJ, dist1on2_Ij, wrap, A4, qu, qt, pin_cast, eos_s_t
        )
        timer_mat = time() - timer_loc

        # --- Update the surface (mutating s, t, p by vertsolve)
        timer_loc = time()
        p_old = p.copy()  # Record old surface for pinning and diagnostics
        vertsolve(s, t, p, S, T, P, n_good, ϕ, TOL_P_SOLVER)

        # DEV:  time seems indistinguishable from using factory function as above
        # _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, TOL_P_SOLVER, eos)

        # Force p to stay constant at the reference column, identically.
        # This avoids any intolerance from the vertical solver.
        p[pin_cast] = pin_p

        timer_update = time() - timer_loc

        # --- Closing Remarks
        ϕ_MAV = np.nanmean(abs(ϕ))
        if diags or TOL_P_CHANGE_RMS > 0:
            Δp = p - p_old
            Δp_RMS = np.sqrt(np.nanmean(Δp ** 2))

        if diags:

            d["timer"][iter_] = time() - timer

            Δp_MAV = np.nanmean(abs(Δp))
            Δp_Linf = np.nanmax(abs(Δp))

            # Diagnostics about what THIS iteration did
            d["ϕ_MAV"][iter_] = ϕ_MAV
            d["Δp_MAV"][iter_] = Δp_MAV
            d["Δp_RMS"][iter_] = Δp_RMS
            d["Δp_Linf"][iter_] = Δp_Linf
            d["n_newly_wet"][iter_] = n_newly_wet

            d["timer_mat"][iter_] = timer_mat
            d["timer_update"][iter_] = timer_update
            d["timer_bfs"][iter_] = timer_bfs

            # Diagnostics about the state AFTER this iteration
            ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, p, eos_s_t, wrap, *geom)
            d["ϵ_RMS"][iter_], d["ϵ_MAV"][iter_] = ϵ_RMS, ϵ_MAV

            n_wet = np.sum(np.isfinite(p))
            d["n_wet"][iter_] = n_wet

            if output:
                print(
                    f"{iter_:4d} |"
                    f" {ϕ_MAV:.8e} |"
                    f" {Δp_RMS:.8e} |"
                    f" {n_wet:11d} ({n_newly_wet:5}) |"
                    f" {ϵ_RMS:.8e} |"
                    f" {d['timer'][iter_]:.3f}"
                )

        # --- Check for convergence
        if (ϕ_MAV < TOL_LRPD_MAV or Δp_RMS < TOL_P_CHANGE_RMS) and iter_ >= ITER_MIN:
            break

    if diags:
        # Trim diagnostics
        for k, v in d.items():
            d[k] = v[0 : iter_ + (k in ("ϵ_MAV", "ϵ_RMS"))]

    s, t, p = (_xr_out(x, xxr) for (x, xxr) in ((s, sxr), (t, txr), (p, pxr)))

    return s, t, p, d


def _omega_matsolve_poisson(
    s, t, p, dist2on1_iJ, dist1on2_Ij, wrap, A4, qu, qt, mr, eos_s_t
):
    """Solve the Poisson formulation of the omega-surface global matrix problem

    Parameters
    ----------
    s, t, p : ndarray

        Salinity, temperature, pressure on the surface

    dist2on1_iJ : ndarray or float

        The grid distance in the second dimension divided by the grid distance
        in the first dimension, both centred at (I-1/2,J). Equivalently, the
        square root of the area of a grid cell centred at(I-1/2,J), divided
        by the distance from (I-1,J) to (I,J).

    dist1on2_Ij : ndarray or float

        The grid distance in the first dimension divided by the grid distance
        in the second dimension, both centred at (I-1/2,J). Equivalently, the
        square root of the area of a grid cell centred at(I,J-1/2), divided
        by the distance from (I,J-1) to (I,J).

    wrap : tuple of bool of length 2

        ``wrap(i)`` is true iff the domain is periodic in the i'th lateral
        dimension.

    A4 : ndarray

        four-connectivity adjacency matrix, computed as
        ``A4 = grid_adjacency(s.shape, 4, wrap)``.
        See `grid_adjacency` in `bfs.py`

    qu : ndarray

        The nodes visited by the BFS in order from 0 to `qt`(see bfs_conncomp1
        in bfs.py).

    qt : int

        The tail index of `qu` (see bfs_conncomp1 in bfs.py).

    mr : int

        Linear index to the reference cast, at which ϕ will be zero

    eos_s_t : function

        Function returning the partial derivatives of the equation of state
        with respect to S and T.

    Returns
    -------
    ϕ : ndarray

        Locally referenced potential density (LRPD) perturbation.  Vertically heaving the surface
        so that its LRPD changes by ϕ will yield a more neutral surface.
    """

    ni, nj = p.shape

    # The value nij appears in A4 to index neighbours that would go across a
    # non-periodic boundary
    nij = ni * nj

    # --- Build & solve sparse matrix problem
    ϕ = np.full(nij, np.nan, dtype=np.float64)

    # If there is only one water column, there are no equations to solve,
    # and the solution is simply phi = 0 at that water column, and nan elsewhere.
    # Note, qt > 0 (N >= 1) should be guaranteed by omega_surf(), so N <= 1 should
    # imply N == 1.  If qt > 0 weren't guaranteed, this could throw an error.
    N = qt + 1  # Number of water columns
    if N <= 1:  # There are definitely no equations to solve
        ϕ[qu[0]] = 0.0  # Leave this isolated pixel at current pressure
        return ϕ.reshape(ni, nj)

    # Collect & sort linear indices to all pixels in this region
    # sorting here makes matrix better structured; overall speedup.
    m = np.sort(qu[0 : qt + 1])

    # If both gridding variables are 1, then grid is uniform
    UNIFORM_GRID = (
        isinstance(dist2on1_iJ, float)
        and dist2on1_iJ == 1
        and isinstance(dist1on2_Ij, float)
        and dist1on2_Ij == 1
    )

    # Begin building D = divergence of ϵ,
    # and L = Laplacian operator (compact representation)

    # L refers to neighbours in this order (so does A4, except without the 5'th entry):
    # . 1 .
    # 0 4 3
    # . 2 .
    IM = 0  # (I  ,J-1)
    MJ = 1  # (I-1,J  )
    PJ = 2  # (I+1,J  )
    IP = 3  # (I  ,J+1)
    IJ = 4  # (I  ,J  )
    L = np.zeros((ni, nj, 5))  # pre-alloc space

    # Create views into L
    L_IM = L[:, :, IM]
    L_MJ = L[:, :, MJ]
    L_PJ = L[:, :, PJ]
    L_IP = L[:, :, IP]
    L_IJ = L[:, :, IJ]

    # Aliases
    sm = s
    tm = t
    pm = p

    # --- m = (i, j) & n = (i-1, j),  then also n = (i+1, j) by symmetry
    sn = im1(sm)
    tn = im1(tm)
    pn = im1(pm)
    if not wrap[0]:
        sn[0, :] = np.nan

    # A stripped down version of ntp_ϵ_errors
    vs, vt = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # (vs, vt) = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 1500)  # DEV: testing omega software to find potential density surface()
    ϵ = vs * (sm - sn) + vt * (tm - tn)

    bad = np.isnan(ϵ)
    ϵ[bad] = 0.0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = dist2on1_iJ.copy()
        fac[bad] = 0.0
        ϵ *= fac  # scale ϵ

    D = -ϵ + ip1(ϵ)

    L_IJ[:] = fac + ip1(fac)

    L_MJ[:] = -fac

    L_PJ[:] = -ip1(fac)

    # --- m = (i, j) & n = (i, j-1),  then also n = (i, j+1) by symmetry
    sn = jm1(sm)
    tn = jm1(tm)
    pn = jm1(pm)
    if not wrap[1]:
        sn[:, 0] = np.nan

    # A stripped down version of ntp_ϵ_errors
    (vs, vt) = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # (vs, vt) = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 1500)  # DEV: testing omega software to find potential density surface()

    ϵ = vs * (sm - sn) + vt * (tm - tn)
    bad = np.isnan(ϵ)
    ϵ[bad] = 0.0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = dist1on2_Ij.copy()
        fac[bad] = 0.0
        ϵ *= fac  # scale ϵ

    D += -ϵ + jp1(ϵ)

    L_IJ[:] += fac + jp1(fac)

    L_IM[:] = -fac

    L_IP[:] = -jp1(fac)

    # --- Build matrix
    # `remap` changes from linear indices for the entire 2D space (0, 1, ..., ni*nj-1) into linear
    # indices for the current connected component (0, 1, ..., N-1)
    # If the domain were doubly periodic, we would want `remap` to be a 2D array
    # of size (ni,nj). However, with a potentially non-periodic domain, we need
    # one more value for `A4` to index into.  Hence we use `remap` as a vector
    # with ni*nj+1 elements, the last one corresponding to non-periodic boundaries.
    # Water columns that are not in this connected component, and dry water columns (i.e. land),
    # and the fake water column for non-periodic boundaries are all left
    # to have a remap value of -1.
    remap = np.full(nij + 1, -1, dtype=int)
    remap[m] = np.arange(N)

    # Pin surface at mr by changing the mr'th equation to be 1 * ϕ[mr] = 0.
    D[mr] = 0.0
    L[mr] = 0.0
    L[mr][IJ] = 1.0

    L = L.reshape((nij, 5))
    D = D.reshape(nij)

    # The above change renders the mr'th column on all rows irrelevant
    # since ϕ[mr] will be zero.  So, we may also set this column to 0
    # which we do here by setting the appropriate links in L to 0. This
    # maintains symmetry of the matrix, enabling the use of a Cholesky solver.
    mrI = np.ravel_multi_index(mr, (ni, nj))  # get linear index for mr
    if A4[mrI, IP] != nij:
        L[A4[mrI, IP], IM] = 0
    if A4[mrI, PJ] != nij:
        L[A4[mrI, PJ], MJ] = 0
    if A4[mrI, MJ] != nij:
        L[A4[mrI, MJ], PJ] = 0
    if A4[mrI, IM] != nij:
        L[A4[mrI, IM], IP] = 0

    # Build the RHS of the matrix problem
    rhs = D[m]

    # Build indices for the rows of the sparse matrix, namely
    # [[0,0,0,0,0], ..., [N-1,N-1,N-1,N-1,N-1]]
    r = np.repeat(range(N), 5).reshape(N, 5)

    # Build indices for the columns of the sparse matrix
    # `remap` changes global indices to local indices for this region, numbered 0, 1, ... N-1
    # Below is equiv to ``c = remap[A5[m]]`` for A5 built with 5 connectivity
    c = np.column_stack((remap[A4[m]], np.arange(N)))

    # Build the values of the sparse matrix
    v = L[m]

    # Prune the entries to
    # (a) ignore connections to adjacent pixels that are dry (including those
    #     that are "adjacent" across a non-periodic boundary), and
    # (b) ignore the upper triangle of the matrix, since cholesky only
    #     accessses the lower triangular part of the matrix
    good = (c >= 0) & (r >= c)

    # DEV: Could try exiting here, and do csc_matrix, spsolve inside main
    # function, so that this can be njit'ed.  But numba doesn't support
    # np.roll as we need it...  (nor ravel_multi_index, but we could just do
    # that one ourselves)
    # return r[good], c[good], v[good], N, rhs, m

    # Build the sparse matrix; with N rows & N columns
    mat = csc_matrix((v[good], (r[good], c[good])), shape=(N, N))

    # --- Solve the matrix problem
    factor = cholesky(mat)
    ϕ[m] = factor(rhs)

    # spsolve (requires ``good = (c >= 0)`` above) is slower than using cholesky
    # ϕ[m] = spsolve(mat, rhs)

    return ϕ.reshape(ni, nj)


def im1(F):  # G[i,j] == F[i-1,j]
    return np.roll(F, 1, axis=0)


def ip1(F):  # G[i,j] == F[i+1,j]
    return np.roll(F, -1, axis=0)


def jm1(F):  # G[i,j] == F[i,j-1]
    return np.roll(F, 1, axis=1)


def jp1(F):  # G[i,j] == F[i,j+1]
    return np.roll(F, -1, axis=1)
