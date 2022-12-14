"""Omega surfaces"""
import numpy as np
from time import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve

from neutralocean.surface.isopycnal import _isopycnal
from neutralocean.surface._vertsolve import _make_vertsolve
from neutralocean.ppinterp import make_pp
from neutralocean.bfs import bfs_conncomp1, bfs_conncomp1_wet
from neutralocean.grid.graph import edges_to_graph
from neutralocean.ntp import ntp_epsilon_errors, ntp_epsilon_errors_norms
from neutralocean.lib import (
    xr_to_np,
    _xrs_in,
    _xr_out,
    _process_pin_cast,
    _process_casts,
    _process_eos,
    aggsum,
)
from neutralocean.mixed_layer import mixed_layer


def omega_surf(S, T, P, grid, **kwargs):
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

        Diagnostics.
        The first four listed below give information going into the `i`'th
        iteration (e.g. the 0'th element is about the initial surface).
        The rest give information about what the `i`'th iteration did (and
        hence their 0'th elements are irrelevant).

        `"e_MAV"` : array of float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.

        `"e_RMS"` : array of float

            As `"e_MAV"` but for the Root Mean Square.

        `"n_wet"`: array of float

            Number of wet casts (surface points).

        `"timer"` : array of float

            Time spent on each iteration, excluding set-up (approximately) and diagnostics.

        `"ϕ_MAV"` : array of float

            Mean Absolute Value of the Locally Referenced Potential Density
            perturbation, per iteration

        `"Δp_MAV"` : array of float

            Mean Absolute Value of the pressure or depth change from one
            iteration to the next

        `"Δp_RMS"` : array of float

            Root Mean Square of the pressure or depth change from one
            iteration to the next

        `"Δp_Linf"` : array of float

            Maximum absolute value (infinity norm) of the pressure or depth
            change from one iteration to the next

        `"n_newly_wet"` : array of int

            Number of casts that are newly wet, per iteration

        `"timer_bfs"` : array of float

            Time spent in Breadth-First Search including wetting, per iteration.

        `"timer_mat"` : array of float

            Time spent building and solving the matrix problem, per iteration.

        `"timer_update"` : array of float

            Time spent vertically updating the surface.

    Other Parameters
    ----------------
    grid, vert_dim, grav, rho_c, interp, diags, output, TOL_P_SOLVER :

        See `potential_surf`

    eos : str or tuple of functions, Default 'gsw'

        As in `potential_surf`, excluding the option to pass a single function.
        The omega surface algorithm relies on knowing the partial derivatives
        of the equation of state with respect to salinity and temperature, so
        the `eos_s_t` function is also required if given as a tuple of functions.

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

    OMEGA_FORMULATION : str, Default 'poisson'

        Specify how the matrix problem is set up and solved.  Options are
            - `'poisson'`, to solve the Poisson problem as in [1]_ with Cholesky, or
            - `'gradient'`, to solve the overdetermined gradient equations as in [2]_
              using LSQR.

    Examples
    --------
    omega surfaces require a pinning cast and initial surface.  The surface is
    iteratively updated while remaining fixed at the pinning cast.  The
    initial surface can be provided directly, as the surface with pressure or
    depth given by `p_init`, in the following method:

    >>> omega_surf(S, T, P, pin_cast, p_init, ...)

    Alternatively, a potential density surface or an in-situ density (specific
    volume) anomaly surface can be used as the initial surface.  To do this,
    use one of the following two methods:

    >>> omega_surf(S, T, P, ref, isoval, pin_cast, ...)

    >>> omega_surf(S, T, P, ref, pin_p, pin_cast, ...)

    For more info on these methods, see the Examples section of `potential_surf`.
    Note that `pin_cast` is always a required input.  Note that `ref` is
    needed to distinguish which of the two types of isopycnals will be used as
    the initial surface.

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
    OMEGA_FORMULATION = kwargs.get("OMEGA_FORMULATION", "poisson")
    interp = kwargs.get("interp", "linear")

    # Build function that calculates coefficients of a piecewise polynomial
    # interpolant, doing 1 problem at a time, and knowing there will be no nans
    # in the input data.
    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=False)

    # Build function that evaluates two piecewise polynomial interpolants (with
    # the same set of independent data), doing n problems at a time.
    interp_two = make_pp(interp, kind="u", out="interp", num_dep_vars=2)

    sxr, txr, pxr = _xrs_in(S, T, P, vert_dim)  # before _process_casts
    pin_cast = _process_pin_cast(pin_cast, S)  # call before _process_casts
    S, T, P = _process_casts(S, T, P, vert_dim)
    eos, eos_s_t = _process_eos(eos, grav, rho_c, need_s_t=True)

    # Save shape of horizontal dimensions, then flatten horiz dims to 1D.
    surf_shape = S.shape[0:-1]
    N = np.prod(surf_shape)  # number of nodes (water columns)
    S, T, P = (np.reshape(X, (N, -1)) for X in (S, T, P))

    # Update pinning cast to a linear index.
    pin_cast = np.ravel_multi_index(pin_cast, surf_shape)

    # Prepare grid ratios for matrix problem.
    distratio = grid["distperp"] / grid["dist"]

    # Pre-calculate grid adjacency needed for Breadth First Search
    edges = grid["edges"]
    graph = edges_to_graph(edges, N)

    if OMEGA_FORMULATION.lower() == "poisson":
        global_solver = _omega_matsolve_poisson
    elif OMEGA_FORMULATION.lower() == "gradient":
        global_solver = _omega_matsolve_gradient
        distratio = np.sqrt(distratio)
    else:
        raise ValueError(
            f"Unknown OMEGA_FORMULATION. Given {OMEGA_FORMULATION}"
        )

    if eos(34.5, 3.0, 1000.0) < 1.0:
        # Convert from a density tolerance [kg m^-3] to a specific volume tolerance [m^3 kg^-1]
        TOL_LRPD_MAV = TOL_LRPD_MAV / 1000.0**2

    # Pre-allocate arrays for diagnostics
    if diags:
        d = {
            "e_MAV": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "e_RMS": np.zeros(ITER_MAX + 1, dtype=np.float64),
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
        kwargs["vert_dim"] = -1  # Since S, T, P already reordered
        kwargs["diags"] = False  # Will make our own diags next
        kwargs["eos"] = eos
        kwargs["pin_cast"] = pin_cast  # update with the 1D value
        s, t, p, _ = _isopycnal(ans_type, S, T, P, **kwargs)

    else:
        # Handling and error checking on p_init
        p_init = xr_to_np(p_init)
        if not isinstance(p_init, np.ndarray):
            raise TypeError(
                'If provided, "p_init" or "p_init.values" must be an ndarray'
            )
        if p_init.shape != surf_shape:
            raise ValueError(
                f'"p_init" should contain a 2D array of size {surf_shape};'
                f" found size {p_init.shape}"
            )
        p_init = np.reshape(p_init, -1)  # now reshape to 1D

        if pin_p is not None and pin_p != p_init[pin_cast]:
            raise ValueError("pin_p does not match p_init at pin_cast")

        p = p_init.copy()

        # Interpolate S and T onto the surface
        # TODO: Update this to handle ice shelf cavity friendly interpolation
        s, t = interp_two(p, P, S, T)

    pin_p = p[pin_cast]

    if np.isnan(p[pin_cast]):
        raise RuntimeError("The initial surface is NaN at the reference cast.")

    if ITER_MAX > 1 and isinstance(p_ml, dict):
        # Compute the mixed layer from parameter inputs
        p_ml = mixed_layer(S, T, P, eos, **p_ml)

    if p_ml is None:
        # Prepare array as needed for bfs_conncomp1_wet
        p_ml = np.full(p.shape, -np.inf)
        # p_ml = np.broadcast_to(-np.inf, (ni, nj))  # DEV: Doesn't work with @numba.njit
    else:
        # Ensure p_ml is 1D array, in case it was given as a 2D array.
        p_ml = np.reshape(p_ml, -1)

    # ensure same nan structure between s, t, and p. Just in case user gives
    # np.full((ni,nj), 1000) for a 1000dbar isobaric surface, for example
    p[np.isnan(s)] = np.nan

    if diags:
        d["timer"][0] = time() - timer

        e_RMS, e_MAV = ntp_epsilon_errors_norms(s, t, p, grid, eos_s_t)
        d["e_RMS"][0], d["e_MAV"][0] = e_RMS, e_MAV

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
                f" {e_RMS:.8e} |"
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
        if p_ml is not None and iter_ > 1:
            p[p < p_ml] = np.nan

        # --- Determine the connected component containing the reference cast, via Breadth First Search
        timer_loc = time()
        if iter_ >= ITER_START_WETTING and iter_ <= ITER_STOP_WETTING:
            bfsqu, n_newly_wet = bfs_conncomp1_wet(
                graph.indptr,
                graph.indices,
                pin_cast,
                s,
                t,
                p,
                S,
                T,
                P,
                TOL_P_SOLVER,
                eos,
                ppc_fn,
                p_ml,
            )
        else:
            bfsqu = bfs_conncomp1(
                graph.indptr, graph.indices, pin_cast, np.isfinite(p)
            )
            n_newly_wet = 0
        timer_bfs = time() - timer_loc

        # --- Solve global matrix problem for the exactly determined Poisson equation
        timer_loc = time()
        # Pre-sort the BFS queue: tests in both MATLAB and Python on OCCA data
        # show this gives an overall speedup for both Poisson and Gradient formulations.
        bfsqu = np.sort(bfsqu)
        ϕ = global_solver(s, t, p, edges, distratio, bfsqu, pin_cast, eos_s_t)
        timer_mat = time() - timer_loc

        # --- Update the surface (mutating s, t, p by vertsolve)
        timer_loc = time()
        p_old = p.copy()  # Record old surface for pinning and diagnostics
        vertsolve(s, t, p, S, T, P, ϕ, TOL_P_SOLVER)

        # DEV:  time seems indistinguishable from using factory function as above
        # _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, ϕ, TOL_P_SOLVER, eos)

        # Force p to stay constant at the reference column, identically.
        # This avoids any intolerance from the vertical solver.
        p[pin_cast] = pin_p

        timer_update = time() - timer_loc

        # --- Closing Remarks
        ϕ_MAV = np.nanmean(abs(ϕ))
        if diags or TOL_P_CHANGE_RMS > 0:
            Δp = p - p_old
            Δp_RMS = np.sqrt(np.nanmean(Δp**2))

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
            e_RMS, e_MAV = ntp_epsilon_errors_norms(s, t, p, grid, eos_s_t)
            d["e_RMS"][iter_], d["e_MAV"][iter_] = e_RMS, e_MAV

            n_wet = np.sum(np.isfinite(p))
            d["n_wet"][iter_] = n_wet

            if output:
                print(
                    f"{iter_:4d} |"
                    f" {ϕ_MAV:.8e} |"
                    f" {Δp_RMS:.8e} |"
                    f" {n_wet:11d} ({n_newly_wet:5}) |"
                    f" {e_RMS:.8e} |"
                    f" {d['timer'][iter_]:.3f}"
                )

        # --- Check for convergence
        if (
            ϕ_MAV < TOL_LRPD_MAV or Δp_RMS < TOL_P_CHANGE_RMS
        ) and iter_ >= ITER_MIN:
            break

    if diags:
        # Trim diagnostics
        for k, v in d.items():
            d[k] = v[0 : iter_ + (k in ("e_MAV", "e_RMS"))]

    # Reshape (from 1D arrays) and put into DataArrays if appropriate
    s, t, p = (
        _xr_out(np.reshape(x, surf_shape), xxr)
        for (x, xxr) in ((s, sxr), (t, txr), (p, pxr))
    )

    return s, t, p, d


def _omega_matsolve_gradient(s, t, p, edges, sqrtdistratio, m, mref, eos_s_t):
    """Solve the Gradient formulation of the omega-surface global matrix problem

    Parameters
    ----------
    s, t, p, edges, m, mref, eos_s_t :
        See `_omega_matsolve_poisson`

    sqrtdistratio : array
        The square-root of the distance of the interface between adjacent
        water columns divided by the square-root of the distance between
        adjacent water columns.  That is, `np.sqrt(distperp / dist)`,
        where `distperp` and `dist` are as in the `geoemtry` input to
        `omega_surf`.

    Returns
    -------
    ϕ : ndarray
        See `_omega_matsolve_poisson`

    """
    N = len(m)  # Number of water columns
    ϕ = np.full(p.size, np.nan, dtype=p.dtype)

    # If there is only one water column, there are no equations to solve,
    # then the solution is simply phi = 0 at that water column, and nan elsewhere.
    # Note, N >= 1 should be guaranteed by omega_surf(), so N <= 1 should imply
    # N == 1.  If N >= 1 weren't guaranteed (m empty), this would throw an error.
    if N <= 1:  # There are definitely no equations to solve
        ϕ[m[0]] = 0.0  # Leave this isolated pixel at current pressure
        return ϕ.reshape(p.shape)

    a, b, e, fac, ref = _omega_matsolve_helper(
        s, t, p, edges, sqrtdistratio, m, mref, eos_s_t
    )

    rhs = np.concatenate((-e, [0.0]))  # add 0 for pinning equation

    # Build columns for matrix, including extra entry for pinning equation.
    # Note m[ref] is the reference cast, so the ref'th entry in the solution
    # vector of the matrix column corresponds to ϕ at the reference cast.
    c = np.concatenate((a, b, [ref]))

    # E = number rows in matrix. Round down ignores pinning equation
    E = len(c) // 2

    # r = [0, 1, ..., E-1, 0, 1, ..., E-1, E]
    r = np.concatenate((np.tile(np.arange(E), 2), [E]))

    # When distratio = 1, v is [1, 1, ..., 1, -1, -1, ..., -1, 1e-2]
    v = np.concatenate((-fac, fac, [1e-2]))

    mat = csc_matrix((v, (r, c)), shape=(E + 1, N))

    sol = lsqr(mat, rhs)[0]

    # Heave solution to be exactly 0 at pinning cast (to fix any intolerance
    # caused by lsqr converging before reaching the exact solution)
    sol -= sol[ref]

    ϕ[m] = sol

    ϕ = ϕ.reshape(p.shape)
    return ϕ


def _omega_matsolve_poisson(s, t, p, edges, distratio, m, mref, eos_s_t):
    """Solve the Poisson formulation of the omega-surface global matrix problem

    Parameters
    ----------
    s, t, p : ndarray

        Salinity, temperature, pressure on the surface

    edges : tuple of length 2 of 1D arrays

        See grid['edges'] from `omega_surf`

    distratio : array

        The distance of the interface between adjacent water columns divided by
        the distance between adjacent water columns, in the same order as `edges`.
        That is, `grid['distperp'] / grid['dist']` where `grid` is as input
        to `omega_surf`.

    m : array

        Linear indices to the nodes in order of the BFS, ie all casts in this
        connected component.
        That is, `m = qu[0:qt+1]` where `qu` and `qt` are outputs from
        `bfs_conncomp1` in bfs.py.

    mref : int

        Linear index to the reference cast, at which ϕ will be zero

    eos_s_t : function

        Function returning the partial derivatives of the equation of state
        with respect to S and T.

    Returns
    -------
    ϕ : ndarray

        Locally referenced potential density (LRPD) perturbation.
        Vertically heaving the surface so that its LRPD in water column m
        increases by the m'th element of ϕ will yield a more neutral surface.
    """
    N = len(m)  # number of water columns
    ϕ = np.full(p.size, np.nan, dtype=p.dtype)  # prealloc space

    # If there is only one water column, there are no equations to solve,
    # then the solution is simply phi = 0 at that water column, and nan elsewhere.
    # Note, N >= 1 should be guaranteed by omega_surf(), so N <= 1 should imply
    # N == 1.  If N >= 1 weren't guaranteed (m empty), this would throw an error.
    if N <= 1:  # There are definitely no equations to solve
        ϕ[m[0]] = 0.0  # Leave this isolated pixel at current pressure
        return ϕ.reshape(p.shape)

    a, b, e, fac, ref = _omega_matsolve_helper(
        s, t, p, edges, distratio, m, mref, eos_s_t
    )

    # Prepare diagonal entries of negative Laplacian.  For uniform geometry,
    # this simply counts the number of edges incident upon each node.  For
    # rectilinear grids, this value will be 4 for a typical node, but can be
    # less near boundaries of the connected component.
    diag = aggsum(fac, a, N) + aggsum(fac, b, N)

    # Divergence of ϵ:  D = ∑_{n ∈ N(m)} ε_{mn}.
    # Note, ϵ = ϵ_{ab} = rs * (sb - sa) + rt * (tb - ta).
    # For the connected component only.
    D = aggsum(e, a, N) - aggsum(e, b, N)

    # Build the rows, columns, and values of the sparse matrix
    r = np.concatenate((a, b, np.arange(N)))
    c = np.concatenate((b, a, np.arange(N)))
    v = np.concatenate((-fac, -fac, diag))  # negative Laplacian

    # Build the (negative) Laplacian sparse matrix with N rows and N columns
    L = csc_matrix((v, (r, c)), shape=(N, N))

    # Pinning surface at reference cast by ADDING the equation
    # 1 * ϕ[ref] = 0 to the ref'th equation.  Note, m[ref] is the reference cast.
    # If the BFS queue (m) were not sorted, then ref == 0 and m[0] would be the
    # reference cast, since the BFS is initialized from the reference cast.
    L[ref, ref] += 1

    # Alternative pinning strategy: change the mref'th equation to be 1 * ϕ[mref] = 0.
    # Then, since ϕ[mref] = 0, values of L in mref'th column are irrelevant, so set
    # these to zero to maintain symmetry.
    # Resulting output is same as above to many sig figs.
    # L[:, ref] = 0
    # L[ref, :] = 0
    # L[ref, ref] = 1
    # D[ref] = 0

    # Solve the matrix problem, L ϕ = D
    ϕ[m] = spsolve(L, D)

    # Fix machine precision errors by applying pinning condition exactly
    ϕ[mref] = 0.0

    ϕ = ϕ.reshape(p.shape)
    return ϕ


def _omega_matsolve_helper(s, t, p, edges, distratio, m, mref, eos_s_t):
    """
    Compute ϵ neutrality errors and geometry on the list of edges in the graph
    that are between pairs of "wet" casts.  Also prune the list of edges to
    just these edges, and remap the nodes they are incident upon from being
    labelled (0, 1, ... ni*nj-1) for the entire 2D grid, to being labelled
    (0, 1, ..., N-1) where N is the number of wet casts.
    """

    N = len(m)

    # `remap` changes from linear indices (0, 1, ..., ni*nj-1) for the entire
    # space (including land), into linear indices (0, 1, ..., N-1) for the
    # current connected component
    remap = np.full(p.size, -1, dtype=int)
    remap[m] = np.arange(N)

    # Select a subset of edges, namely those between two "wet" casts
    a, b = edges
    ge = (remap[a] >= 0) & (remap[b] >= 0)  # good edges
    a, b = a[ge], b[ge]

    e = ntp_epsilon_errors(s, t, p, (a, b), eos_s_t)

    if np.isscalar(distratio):
        fac = np.full(e.shape, distratio)
        if distratio != 1.0:
            e *= fac  # scale e
    else:
        fac = distratio[ge]
        e *= fac  # scale e

    # Henceforth we only refer to nodes in the connected component, so remap edges now
    a, b = remap[a], remap[b]

    # Also remap the index for the reference cast from 2D space to 1D list of wet casts
    ref = remap[mref]

    return a, b, e, fac, ref
