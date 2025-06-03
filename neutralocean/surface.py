"""
Calculate approximately neutral surfaces in the ocean.
Three such surfaces are currently supported: 
    potential density (or specific volume) surfaces,
    in-situ density (or specific volume) anomaly surfaces, and
    omega surfaces.
"""

import numpy as np
from time import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr, spsolve

from .eos import load_eos
from ._vertsolve import _make_vertsolve
from .ppinterp import make_pp, ppval_1_two, valid_range_1
from .ntp import ntp_epsilon_errors, ntp_epsilon_errors_norms
from .lib import (
    xr_to_np,
    _xrs_in,
    _xr_out,
    _process_pin_cast,
    _process_casts,
    aggsum,
    local_functions,
)
from .bfs import bfs_conncomp1, bfs_conncomp1_wet_perim
from .grid.graph import edges_to_csr
from .mixed_layer import mld


def potential_surf(S, T, P, **kw):
    """Calculate a potential density (or specific volume) surface.

    Given practical / Absolute salinity `S`, potential / Conservative
    temperature `T`, and pressure (when non-Boussinesq) or depth
    (when Boussinesq) `P`, and given a reference pressure / depth `ref`,
    calculate an isosurface of `eos(S, T, ref)` where `eos` is the equation
    of state.

    Parameters
    ----------
    S, T : ndarray or xarray.DataArray

        3D practical / Absolute salinity and potential / Conservative
        temperature --- that is, a 2D array of 1D water columns or "casts"

    P : ndarray or xarray.DataArray

        In the non-Boussinesq case, `P` is the 3D pressure, sharing the same
        dimensions as `S` and `T`.

        In the Boussinesq case, `P` is the depth and can be 3D with the same
        structure as `S` and `T`, or can be 1D with as many elements as there
        are in the vertical dimension of `S` and `T`.

    ref : float

        The reference pressure or depth, in the same units as `P`.

        If `ref` is None, the reference value is taken as `pin_p`.

        See Examples section.

    isoval : float

        Isovalue of the potential density surface.
        Units are same as returned by `eos`.

        See Examples section.

    pin_p : float

        Pressure or depth at which the surface intersects the cast `pin_cast`.

        See Examples section.

    pin_cast : tuple or list of int of length 2

        Index for cast where surface is at pressure or depth `pin_p`.

        See Examples section.

    Returns
    -------
    s, t, p : ndarray or xarray.DataArray

        practical / Absolute salinity, potential / Conservative temperature,
        and pressure / depth on the surface

    d : dict

        Diagnostics.

        `"e_MAV"` : float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.

        `"e_RMS"` : float

            As `"e_MAV"` but for the Root Mean Square.

        `"n_wet"`: float

            Number of wet casts (surface points).

        `"timer"` : float

            Time spent on the whole algorithm, excluding set-up and diagnostics.

        `"ref"` : float

            Reference pressure / depth for surface (matching input `ref` if given,
            otherwise this is calculated internally).

        `"isoval"` : float

            Isovalue of potential density for surface (matching input `isoval`
            if given, otherwise this is calculated internally).

    Other Parameters
    ----------------
    grid : dict
        Containing the following:

        edges : tuple of length 2
            Each element is an array of int of length E, where E is the number of
            edges in the grid's graph, i.e. the number of pairs of adjacent water
            columns (including land) in the grid.
            If `edges = (a, b)`, the nodes (water columns) whose linear indices are
            `a[i]` and `b[i]` are adjacent.
            Required if `diags` is True

        dist : 1d array
            Horizontal distance between adjacent water columns (nodes).
            `dist[i]` is the distance between nodes whose linear indices are
            `edges[0][i]` and `edges[1][i]`.
            If absent, a value of 1.0 is assumed for all edges.

        distperp : 1d array
            Horizontal distance of the face between adjacent water columns (nodes).
            `distperp[i]` is the distance of the interface between nodes whose
            linear indices are `edges[0][i]` and `edges[1][i]`.
            If absent, a value of 1.0 is assumed for all edges.

        For a rectilinear grid (e.g. latitude-longitude), use
            `neutralocean.grid.rectilinear.build_grid`

        For a tiled rectilinear grid, such as works with XGCM, use
            `neutralocean.grid.xgcm.build_grid`

        For a general grid given as a graph, use
            `neutralocean.grid.graph.build_grid`

        Also see the examples in `neutralocean.examples`.

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes`.

    eos : function, Default `neutralocean.eos.gsw.specvol`

        Function taking three inputs corresponding to (`S, T, P)`, and
        outputting the in-situ density or specific volume.
        Should be `@numba.njit` decorated and need not be vectorized.

    eos_s_t : function, `neutralocean.eos.gsw.specvol_s_t`

        Function taking three inputs corresponding to (`S, T, P)`, and
        outputting a tuple containing the partial derivatives of the equation of
        state with respect to `S` and `T`.
        Should be `@numba.njit` decorated and need not be vectorized.
        Note: This is only used when `diags` is `True`.

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use `'linear'` for linear
        interpolation, and `'pchip'` for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `ppinterp`.

    diags : bool, Default True

        If True, calculate diagnostics (4th output).  If False, 4th output is
        an empty dict.

    output : bool, Default True

        If `True`, prints diagnostic output during computation.
        `diags` must be `True` for this to have any effect.
        To redirect this output to a file, do the following
        >>> import sys
        >>> tmp = sys.stdout
        >>> sys.stdout = file_id = open('myfile.txt', 'w')
        >>> # Now call this function ...
        >>> sys.stdout = tmp
        >>> file_id.close()

    TOL_P_SOLVER : float, Default 1e-4

        Error tolerance when root-finding to update the pressure or depth of
        the surface in each water column. Units are the same as `P`.

    Examples
    --------
    The output surface must be specified by some combination of reference
    pressure / depth `ref`, isovalue `isoval`, pinning cast `pin_cast` and
    pinning pressure `pin_p`.  The following methods are valid, and listed in
    order of precedence (e.g. `pin_cast` and `pin_p` are not used if both
    `ref` and `isoval` are given).

    >>> potential_surf(S, T, P, ref=___, isoval=___, ...)

    This finds the surface with given reference pressure / depth and the given
    isovalue.

    >>> potential_surf(S, T, P, ref=___, pin_p=___, pin_cast=___, ...)

    This finds the surface with given reference pressure / depth that intersects
    the given cast at the given pressure or depth.

    >>> potential_surf(S, T, P, pin_p=___, pin_cast=___, ...)

    This is as for the previous method, but selects the reference pressure /
    depth as `pin_p` (i.e. the local `P` value at the given cast's given
    pressure or depth).

    Notes
    -----
    This code will internally re-arrange `S`, `T`, `P` to have the vertical
    dimension last, so that the data for an individual water column is
    contiguous in memory.  If you call this function many times, consider
    using `lib._process_casts` to pre-process your `S`, `T`, `P` inputs to
    have the vertical dimension last.
    """

    return _isopycnal("potential", S, T, P, **kw)


def anomaly_surf(S, T, P, **kw):
    """Calculate a specific volume (or in-situ density) anomaly surface.

    Given practical / Absolute salinity `S`, potential / Conservative
    temperature `T`, and pressure / depth `P`, and given reference values
    `S0` and `T0`, calculate  an isosurface of `eos(S, T, P) - eos(S0, T0, P)`
    where `eos` is the equation of state.

    In a non-Boussinesq ocean, `P` is pressure.  Also, if one is computing
    geostrophic streamfunctions, it is most convenient if `eos` provides the
    specific volume.

    In a Boussinesq ocean, `P` is depth.  Also, if one is computing
    geostrophic streamfunctions, it is most convenient if `eos` provides the
    in-situ density.

    Parameters
    ----------
    S, T, P : ndarray or xarray.DataArray
        See `potential_surf`

    ref : tuple of float of length 2

        The reference S and T values.

        If `ref` is None or has a None element, the reference values are taken
        from the local `S` and `T` at the pressure or depth `pin_p` on the
        pinning cast `pin_cast`.

    isoval, pin_p, pin_cast :
        See `potential_surf`

    Returns
    -------
    s, t, p, d :
        See `potential_surf`.
        Note `d["ref"]` returns a 2 element tuple, namely `ref` as here.

    Other Parameters
    ----------------
    grid, vert_dim, eos, interp, diags, output, TOL_P_SOLVER :
        See `potential_surf`

    Examples
    --------
    The output surface must be specified by some combination of reference
    salinity and temperature `ref`, isovalue `isoval`, pinning cast `pin_cast` and
    pinning pressure `pin_p`.  The following methods are valid, and listed in
    order of precedence (e.g. `pin_cast` and `pin_p` are not used if both
    `ref` and `isoval` are given).

    >>> anomaly_surf(S, T, P, ref=___, isoval=___, ...)

    This finds the surface with given reference salinity and temperature and
    the given isovalue.

    >>> anomaly_surf(S, T, P, ref=___, pin_p=___, pin_cast=___, ...)

    This finds the surface with given reference salinity and temperature that
    intersects the given cast at the given pressure or depth.

    >>> anomaly_surf(S, T, P, pin_p=___, pin_cast=___, ...)

    This is as for the previous method, but selects the reference salinity and
    temperature from the local `S` and `T` values at the given cast's given
    pressure or depth.

    Notes
    -----
    See `potential_surf`.
    """

    return _isopycnal("anomaly", S, T, P, **kw)


def _isopycnal(ans_type, S, T, P, **kw):
    """Calculate an isosurface of potential density or specific volume anomaly.

    Inputs are as in `potential_surf` and `anomaly_surf`, but first input is a
    string specifying "potential" or "anomaly" """

    ref = kw.get("ref")
    isoval = kw.get("isoval")
    pin_cast = kw.get("pin_cast")
    pin_p = kw.get("pin_p")
    vert_dim = kw.get("vert_dim", -1)
    TOL_P_SOLVER = kw.get("TOL_P_SOLVER", 1e-4)
    eos = kw.get("eos")
    eos_s_t = kw.get("eos_s_t")
    diags = kw.get("diags", True)
    output = kw.get("output", True)
    grid = kw.get("grid")
    interp = kw.get("interp", "linear")
    
    rho_c = kw.get("rho_c")
    grav = kw.get("grav")
    if grav is not None or rho_c is not None or isinstance(eos, str):
        raise ValueError(
            "`grav` and `rho_c` and `eos` as a string are no longer supported. "
            "Pass `eos` and `eos_s_t` as functions, which can be obtained from "
            "`neutralocean.load_eos`. See the `examples` folder for examples."
        )

    # Build function that calculates coefficients of a piecewise polynomial
    # interpolant, doing 1 problem at a time, and knowing there will be no nans
    # in the input data.
    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=False)

    # Process arguments
    sxr, txr, pxr = _xrs_in(S, T, P, vert_dim)  # before _process_casts
    pin_cast = _process_pin_cast(pin_cast, S)  # call before _process_casts
    S, T, P = _process_casts(S, T, P, vert_dim)
    if eos is None and eos_s_t is None:
        eos = load_eos("gsw")
        eos_s_t = load_eos("gsw", "_s_t")
    if diags and not callable(eos_s_t):
        raise ValueError("eos_s_t must be callable when diags is True")
    if diags and not (isinstance(grid, dict) and "edges" in grid):
        raise ValueError("grid['edges'] must be provided when diags is True")

    # Error checking on (ref, isoval, pin_cast, pin_p), then convert this
    # selection to (ref, isoval) pair
    _check_ref(ans_type, ref, isoval, pin_cast, pin_p, S)
    ref, isoval = _choose_ref_isoval(
        ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, ppc_fn
    )

    # Solve non-linear root finding problem in each cast
    vertsolve = _make_vertsolve(eos, ppc_fn, ans_type)
    timer = time()
    s, t, p = vertsolve(S, T, P, ref, isoval, TOL_P_SOLVER)

    if pin_p is not None:  # pin_cast must also be valid
        # Adjust the surface at the pinning cast slightly, to match the pinning
        # pressure / depth.  This fixes small deviations of order `TOL_P_SOLVER`
        n = pin_cast
        p[n] = pin_p
        Sn, Tn, Pn = S[n], T[n], P[n]
        k, K = valid_range_1(Sn + Pn)  # Sn and Tn have same nan-structure
        Sppcn = ppc_fn(Pn[k:K], Sn[k:K])
        Tppcn = ppc_fn(Pn[k:K], Tn[k:K])
        s[n], t[n] = ppval_1_two(pin_p, Pn[k:K], Sppcn, Tppcn)

    d = dict()
    d["ref"] = ref
    d["isoval"] = isoval
    if diags:
        d["timer"] = time() - timer
        e_RMS, e_MAV = ntp_epsilon_errors_norms(s, t, p, grid, eos_s_t)
        d["e_RMS"], d["e_MAV"] = e_RMS, e_MAV

        n_wet = np.sum(np.isfinite(p))
        d["n_wet"] = n_wet
        if output:
            print(
                f"{ans_type} done" f" | {n_wet:11d} wet casts" f" | RMS(ϵ) = {e_RMS:.8e}",
                f" | {d['timer']:.3f} sec",
            )

    s, t, p = (_xr_out(x, xxr) for (x, xxr) in ((s, sxr), (t, txr), (p, pxr)))
    return s, t, p, d


def _check_ref(ans_type, ref, isoval, pin_cast, pin_p, S):
    """Error checking on ref / isoval / pin_cast / pin_p combinations for "potential"
    and "anomaly" surfaces
    """
    # First check None values to validate one of the following options:
    # >>> _isopycnal(ans_type, S, T, P, ref, isoval)
    # >>> _isopycnal(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _isopycnal(ans_type, S, T, P, pin_cast, pin_p)
    if ref is None:
        if pin_cast is None or pin_p is None:
            raise TypeError(
                'If "ref" is not provided, "pin_cast" and "pin_p" must be provided'
            )
    else:  # ref is not None
        if isoval is None and (pin_cast is None or pin_p is None):
            raise TypeError(
                'If "ref" is provided, either "isoval" must be provided or'
                ' "pin_cast" and "pin_p" must be provided'
            )

    # Error checking on ref
    if ref is not None:
        if ans_type == "potential":
            if not isinstance(ref, float):
                raise TypeError(
                    'For "potential" surfaces, if provided "ref" must be a float'
                )
        else:  # ans_type == "anomaly"
            if not (isinstance(ref, (tuple, list)) and len(ref) == 2):
                raise TypeError(
                    'For "anomaly" surfaces, if provided "ref" must be 2 element tuple/list of float'
                )

    # Error checking on pin_cast.  Let dict inputs (for xarray) pass through fine...
    if not isinstance(pin_cast, (type(None), dict)):
        try:
            S[pin_cast]
        except:
            raise ValueError(
                'If provided, "pin_cast" must be able to index all but the'
                " vertical dimension of S"
            )

    # Error checking on pin_p
    if not isinstance(pin_p, (type(None), float)):
        raise TypeError('If provided, "pin_p" must be a float')


def _choose_ref_isoval(ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, ppc_fn):
    # Handle the three valid calls in the following order of precedence:
    # >>> _isopycnal(ans_type, S, T, P, ref, isoval)
    # >>> _isopycnal(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _isopycnal(ans_type, S, T, P, pin_cast, pin_p)
    if isoval is None:  # => pin_cast and pin_p are both not None
        n = pin_cast  # evaluate S and T on the surface at the chosen location
        # s0, t0 = interp_fn(pin_p, P[n], S[n], T[n])

        Sn, Tn, Pn = S[n], T[n], P[n]
        k, K = valid_range_1(Sn + Pn)  # Sn and Tn have same nan-structure
        Sppcn = ppc_fn(Pn[k:K], Sn[k:K])
        Tppcn = ppc_fn(Pn[k:K], Tn[k:K])
        s0, t0 = ppval_1_two(pin_p, Pn[k:K], Sppcn, Tppcn)

        if ans_type == "potential":
            if ref is None:
                ref = pin_p
            isoval = eos(s0, t0, ref)
        else:  # ans_type == "anomaly"
            if ref is None or any(x is None for x in ref):
                ref = (s0, t0)
            isoval = eos(s0, t0, pin_p) - eos(
                ref[0], ref[1], pin_p
            )  #  == 0 when ref = (s0, t0)

    return ref, isoval


def omega_surf(S, T, P, grid, pin_cast, p_init, **kw):
    """Calculate an omega surface from structured ocean data.

    Given 3D salinity, temperature, and pressure or depth data arranged on a
    rectilinear grid, calculate a 2D omega surface [1]_ [2]_, which is a
    highly accurate approximately neutral surface.

    Parameters
    ----------
    S, T, P : ndarray or xarray.DataArray

        See `potential_surf`

    pin_cast : int or tuple of int

        Index for cast where surface is kept at fixed pressure or depth.

    p_init : float or ndarray or xarray.DataArray

        If array, pressure or depth of the initial approximately neutral
        surface. Must be the same shape as `S` less its vertical dimension.

        If float, pressure or depth at `pin_cast`. The initial surface is
        generated by iteratively wetting the perimeter of the region, beginning
        with the reference cast, `pin_cast`.

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
    grid, vert_dim, interp, eos, eos_s_t, diags, output, TOL_P_SOLVER :

        See `potential_surf`
        Note: `eos_s_t` is required.

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
        computed using `mld` with p_ml passed as keyword arguments,
        enabling control over the parameters in that function.
        See `mld` for details.

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

    Notes
    -----
    See `potential_surf` Notes.

    .. [1] Stanley, G. J., McDougall, T. J., & Barker, P. M. (2021). Algorithmic
       Improvements to Finding Approximately Neutral Surfaces. Journal of
       Advances in Modeling Earth Systems, 13(5), e2020MS002436.

    .. [2] Klocker, A., McDougall, T. J., & Jackett, D. R. (2009). A new method
       for forming approximately neutral surfaces. Ocean Science, 5 (2), 155-172.

    """

    pin_c = pin_cast  # alias
    vert_dim = kw.get("vert_dim", -1)
    p_ml = kw.get("p_ml")
    diags = kw.get("diags", True)
    output = kw.get("output", True)
    eos = kw.get("eos")
    eos_s_t = kw.get("eos_s_t")
    interp = kw.get("interp", "linear")
    ITER_MIN = kw.get("ITER_MIN", 1)
    ITER_MAX = kw.get("ITER_MAX", 10)
    ITER_START_WETTING = kw.get("ITER_START_WETTING", 1)
    ITER_STOP_WETTING = kw.get("ITER_STOP_WETTING", 5)
    TOL_P_SOLVER = kw.get("TOL_P_SOLVER", 1e-4)
    TOL_LRPD_MAV = kw.get("TOL_LRPD_MAV", 1e-7)
    TOL_P_CHANGE_RMS = kw.get("TOL_P_CHANGE_RMS", 0.0)
    OMEGA_FORMULATION = kw.get("OMEGA_FORMULATION", "poisson")
    ITER_WET_PERIM = kw.get("ITER_WET_PERIM", np.iinfo(int).max)

    rho_c = kw.get("rho_c")
    grav = kw.get("grav")
    if grav is not None or rho_c is not None:
        raise ValueError(
            "grav and rho_c are no longer supported. Pass `eos` and `eos_s_t`. See the `examples` folder for examples."
        )

    # Build function that calculates coefficients of a piecewise polynomial
    # interpolant, doing 1 problem at a time, and knowing there will be no nans
    # in the input data.
    ppc_fn = make_pp(interp, kind="1", out="coeffs", nans=False)

    # Build function that evaluates two piecewise polynomial interpolants (with
    # the same set of independent data), doing n problems at a time.
    interp_two = make_pp(interp, kind="u", out="interp", num_dep_vars=2)

    sxr, txr, pxr = _xrs_in(S, T, P, vert_dim)  # before _process_casts
    pin_c = _process_pin_cast(pin_c, S)  # call before _process_casts
    S, T, P = _process_casts(S, T, P, vert_dim)
    if eos is None and eos_s_t is None:
        eos = load_eos("gsw")
        eos_s_t = load_eos("gsw", "_s_t")

    # Save shape of horizontal dimensions, then flatten horiz dims to 1D.
    surf_shape = S.shape[0:-1]
    N = np.prod(surf_shape)  # number of nodes (water columns)
    S, T, P = (np.reshape(X, (N, -1)) for X in (S, T, P))

    # Update pinning cast to a linear index, unless already so
    if isinstance(pin_c, (tuple, list)):
        pin_c = np.ravel_multi_index(pin_c, surf_shape)

    # Prepare grid ratios for matrix problem.
    distratio = grid["distperp"] / grid["dist"]

    # Pre-compute mixed layer
    if isinstance(p_ml, dict):
        # Compute the mixed layer from parameter inputs
        p_ml = mld(S, T, P, eos, **p_ml)
    if p_ml is None:
        # Prepare array as needed for bfs_conncomp1_wet
        p_ml = np.full(surf_shape, -np.inf)
        # p_ml = np.broadcast_to(-np.inf, (ni, nj))  # DEV: Doesn't work with @numba.njit
    # Ensure p_ml is 1D array, in case it was given as a 2D array.
    p_ml = np.reshape(p_ml, -1)

    # Pre-calculate grid adjacency needed for Breadth First Search
    edges = grid["edges"]
    indptr, indices = edges_to_csr(edges, N)
    bfsargs = (
        indptr,
        indices,
        pin_c,
        S,
        T,
        P,
        TOL_P_SOLVER,
        eos,
        ppc_fn,
        p_ml,
        ITER_WET_PERIM,
    )

    if OMEGA_FORMULATION.lower() == "poisson":
        global_solver = _omega_matsolve_poisson
    elif OMEGA_FORMULATION.lower() == "gradient":
        global_solver = _omega_matsolve_gradient
        distratio = np.sqrt(distratio)
    else:
        raise ValueError(f"Unknown OMEGA_FORMULATION. Given {OMEGA_FORMULATION}")

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
    if np.isscalar(p_init):
        pin_p = p_init

        p, s, t = (np.full(N, np.nan) for _ in range(3))
        p[pin_c] = pin_p

        # Interpolate S and T to P = pin_p at pin_c
        s[pin_c], t[pin_c] = interp_two(pin_p, P[pin_c], S[pin_c], T[pin_c], 0)

        # Mutate s, t, p by NTP linking perimeter until convergence.
        _ = bfs_conncomp1_wet_perim(s, t, p, *bfsargs)

    else:
        # Handling and error checking on p_init
        p_init = xr_to_np(p_init)
        if not isinstance(p_init, np.ndarray):
            raise TypeError('If provided, "p_init" or "p_init.values" must be an ndarray')
        if p_init.shape != surf_shape:
            raise ValueError(
                f'"p_init" should contain a 2D array of size {surf_shape};'
                f" found size {p_init.shape}"
            )
        p_init = np.reshape(p_init, -1)  # now reshape to 1D
        pin_p = p_init[pin_c]
        p = p_init.copy()

        # Interpolate S and T onto the surface
        s, t = interp_two(p, P, S, T, 0)

    if np.isnan(s[pin_c]):
        raise RuntimeError("The initial surface is NaN at the reference cast.")

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
            n_newly_wet = bfs_conncomp1_wet_perim(s, t, p, *bfsargs)
            qu = np.nonzero(np.isfinite(p))[0]  # sorted
        else:
            qu = bfs_conncomp1(indptr, indices, pin_c, np.isfinite(p))
            # Pre-sort the BFS queue: tests in both MATLAB and Python on OCCA data
            # show this gives an overall speedup for both Poisson and Gradient formulations.
            qu = np.sort(qu)
            n_newly_wet = 0
        timer_bfs = time() - timer_loc

        # --- Solve global matrix problem for the exactly determined Poisson equation
        timer_loc = time()
        ϕ = global_solver(s, t, p, edges, distratio, qu, pin_c, eos_s_t)
        timer_mat = time() - timer_loc

        # --- Update the surface (mutating s, t, p by vertsolve)
        timer_loc = time()
        p_old = p.copy()  # Record old surface for pinning and diagnostics
        vertsolve(s, t, p, S, T, P, ϕ, TOL_P_SOLVER)

        # DEV:  time seems indistinguishable from using factory function as above
        # _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, ϕ, TOL_P_SOLVER, eos)

        # Force p to stay constant at the reference column, identically.
        # This avoids any intolerance from the vertical solver.
        p[pin_c] = pin_p

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
        if (ϕ_MAV < TOL_LRPD_MAV or Δp_RMS < TOL_P_CHANGE_RMS) and iter_ >= ITER_MIN:
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

    # Get list of edges (a,b), ϵ neutrality errors, and geometric factors
    # for the current connected component containing the reference cast, and
    # map everything onto a set of N
    a, b, e, fac, ref = _omega_matsolve_helper(
        s, t, p, edges, distratio, m, mref, eos_s_t
    )

    # Divergence of ϵ,
    # D = ∑_{n ∈ N(m)} ϵₘₙ
    #   = ∑_{j=1}^E  δ_{m, aⱼ} ϵ_{m, bⱼ}  +  ∑_{j=1}^E  δ_{m, bⱼ} ϵ_{m, aⱼ}
    #   = ∑_{j=1}^E  δ_{m, aⱼ} ϵ_{m, bⱼ}  -  ∑_{j=1}^E  δ_{m, bⱼ} ϵ_{aⱼ, m}
    # achieved by
    D = aggsum(e, a, N) - aggsum(e, b, N)

    # Prepare diagonal entries of negative Laplacian.  For uniform geometry,
    # this simply counts the number of edges incident upon each node.  For
    # rectilinear grids, this value will be 4 for a typical node, but can be
    # less near boundaries of the connected component.
    diag = aggsum(fac, a, N) + aggsum(fac, b, N)

    # Build the rows, columns, and values of the sparse matrix
    r = np.concatenate((a, b, np.arange(N)))
    c = np.concatenate((b, a, np.arange(N)))
    v = np.concatenate((-fac, -fac, diag))  # negative Laplacian

    # Build the negative Laplacian sparse matrix with N rows and N columns
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
    that are between pairs of "wet" casts.  Then, prune the list of edges to
    just these "wet" edges that are within the connected component containing
    the reference cast (whose linear index in the full space is `mref`), and
    remap all nodes (water columns) from being labelled (0, 1, ... ni*nj-1)
    for the entire 2D grid, to being labelled (0, 1, ..., N-1) where N is the
    number of wet casts in this connected component.  The connected component
    is determined by `m`, which is a list of nodes in the label set of the
    full space, (0, 1, ... ni*nj-1).
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


__all__ = local_functions(locals(), __name__)
