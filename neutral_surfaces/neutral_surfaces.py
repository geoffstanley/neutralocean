"""
Calculate approximately neutral surfaces from structured ocean data. 
"""

import numpy as np
from time import time

from neutral_surfaces._vertsolve import _make_vertsolve
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val2
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces.ntp import ntp_ϵ_errors_norms
from neutral_surfaces.lib import (
    xr_to_np,
    _xr_in,
    _xr_out,
    _process_args,
)
from neutral_surfaces._omega import _omega_matsolve_poisson


def sigma_surf(S, T, P, **kwargs):
    """Calculate a potential density surface from structured ocean data.

    Given practical / Absolute salinity `S`, potential / Conservative
    temperature `T`, and pressure (when non-Boussinesq) or depth
    (when Boussinesq) `P` arranged on a rectilinear grid, and given a
    reference pressure `ref`, calculate an isosurface of `eos(S, T, ref)`
    where `eos` is the equation of state.

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

        The reference pressure or depth, in the same units as P.

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

        ``"ϵ_MAV"`` : float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.

        ``"ϵ_RMS"`` : float

            As ``"ϵ_MAV"`` but for the Root Mean Square.

        ``"n_wet"``: float

            Number of wet casts (surface points).

        ``"timer"`` : float

            Time spent on the whole algorithm, excluding set-up and diagnostics.


    Other Parameters
    ----------------
    wrap : tuple of bool, or tuple of str

        Specifies which dimensions are periodic.

        As a tuple of bool, this must be length two.  The first or second
        non-vertical dimension of `S` and `T` is periodic iff ``wrap[0]`` or
        ``wrap[1]`` is True, respectively.

        As a tuple of str, simply name the periodic dimensions of `S` and
        `T`.

        Required if diags is True

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes`.

    dist1_iJ, dist1_Ij, dist2_Ij, dist2_iJ : float or ndarray, Default 1.0

        Grid distances [m] in either the 1st or 2nd lateral dimension, and
        centred at the location specified.  The naming uses a soft notation:
        the central grid point is(I,J), and i = I-1/2 and j = J-1/2.  Thus,
        `dist1_iJ[5,3]` is the distance between cells (5,3) and (4,3), while
        `dist2_iJ[5,3]` is the distance of the face between cells (5,3) and
        (4,3). Similarly, `dist2_Ij[5,3]` is the distance between cells
        (5,3) and (5,2), while `dist1_Ij[5,3]` is the distance of the face
        between cells (5,3) and (5,2).

    eos : str or function, Default 'gsw'

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure inputs.  For Boussinesq models, provide `grav`
        and `rho_c`, so this function with third input pressure will be
        converted to a function with third input depth.

        If a function, this should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

        If a str, can be either 'gsw' to use the TEOS-10 specific volume
        or 'jmd95' to use the Jackett and McDougall (1995) in-situ density
        [1]_.

    eos_s_t : str or function, Default None

        Equation of state for the partial derivatives of density or specific
        volume with respect to `S` and `T` as a function of `S`, `T`, and
        pressure (not depth) inputs.

        If a function, this need not be @numba.njit decorated but should be
        vectorized, as it will be called a few times with ndarray inputs.

        If a str, the same options apply as for `eos`. If None and `eos` is a
        str, then this defaults to the same str as `eos`.

    grav : float, Default None

        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None

        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    interp_fn : function, Default ``linear_coeffs``

        Function that calculates coefficients of piecewise polynomial
        interpolants of `S` and `T` as functions of `P`.  Options include
        ``linear_coeffs`` and ``pchip_coeffs`` from ``interp_ppc.py``.

    Sppc, Tppc : ndarray, Default None

        Pre-computed Piecewise Polynomial Coefficients for `S` and `T` as
        functions of `P`. If None, these are computed as ``Sppc = interp_fn
        (S, P)`` and ``Tppc = interp_fn(T,P)``.

    n_good : ndarray, Default None

        Pre-computed number of ocean data points in each water column.
        If None, this is computed as ``n_good = lib.find_first_nan(S)``.

    diags : bool, Default True

        If True, calculate diagnostics (4th output).  If False, 4th output is
        an empty dict.

    output : bool, Default True

        If True, prints diagnostic output during computation.
        `diags` must be True for this to have any effect.
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

    >>> sigma_surf(S, T, P, ref=___, isoval=___, ...)

    This finds the surface with given reference pressure / depth and the given
    isovalue.

    >>> sigma_surf(S, T, P, ref=___, pin_p=___, pin_cast=___, ...)

    This finds the surface with given reference pressure / depth that intersects
    the given cast at the given pressure or depth.

    >>> sigma_surf(S, T, P, pin_p=___, pin_cast=___, ...)

    This is as for the previous method, but selects the reference pressure /
    depth as `pin_p` (i.e. the local `P` value at the given cast's given
    pressure or depth).

    Notes
    -----
    This code will internally re-arrange `S`, `T`, `P` to have the vertical
    dimension last, so that the data for an individual water column is
    contiguous in memory.  If you call this function many times, consider
    using ``neutral_surfaces._process_casts`` to pre-process your `S`, `T`,
    `P` inputs to have the vertical dimension last.  Also consider
    pre-computing `Sppc`, `Tppc`, and `n_good` (see the documentation for
    these inputs).

    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388
    """

    return _sigma_delta_surf("sigma", S, T, P, **kwargs)


def delta_surf(S, T, P, **kwargs):
    """Calculate a specific volume (or in-situ density) anomaly surface from structured ocean data.

    Given practical / Absolute salinity `S`, potential / Conservative
    temperature `T`, and pressure / depth `P` arranged on a rectilinear grid,
    and given a reference values `S0` and `T0`, calculate an isosurface of
    `eos(S, T, P) - eos(S0, T0, P)` where `eos` is the equation of state.

    In a non-Boussinesq ocean, `P` is pressure.  Also, if one is computing
    geostrophic streamfunctions, it is most convenient if `eos` provides the
    specific volume.

    In a Boussinesq ocean, `P` is depth.  Also, if one is computing
    geostrophic streamfunctions, it is most convenient if `eos` provides the
    in-situ density.

    Parameters
    ----------
    S, T, P : ndarray or xarray.DataArray
        See `sigma_surf`

    ref : tuple of float of length 2

        The reference S and T values.

        If `ref` is None or has a None element, the reference values are taken
        from the local `S` and `T` at the pressure or depth `pin_p` on the
        pinning cast `pin_cast`.

    isoval, pin_p, pin_cast :
        See `sigma_surf`

    Returns
    -------
    s, t, p, d :
        See `sigma_surf`

    Examples
    --------
    The output surface must be specified by some combination of reference
    salinity and temperature `ref`, isovalue `isoval`, pinning cast `pin_cast` and
    pinning pressure `pin_p`.  The following methods are valid, and listed in
    order of precedence (e.g. `pin_cast` and `pin_p` are not used if both
    `ref` and `isoval` are given).

    >>> delta_surf(S, T, P, ref=___, isoval=___, ...)

    This finds the surface with given reference salinity and temperature and
    the given isovalue.

    >>> delta_surf(S, T, P, ref=___, pin_p=___, pin_cast=___, ...)

    This finds the surface with given reference salinity and temperature that
    intersects the given cast at the given pressure or depth.

    >>> delta_surf(S, T, P, pin_p=___, pin_cast=___, ...)

    This is as for the previous method, but selects the reference salinity and
    temperature from the local `S` and `T` values at the given cast's given
    pressure or depth.

    Notes
    -----
    See `sigma_surf`.
    """

    return _sigma_delta_surf("delta", S, T, P, **kwargs)


def _sigma_delta_surf(ans_type, S, T, P, **kwargs):
    """Core function to calculate "sigma" or "delta" surfaces.
    Inputs are as in `sigma_surf` and `delta_surf`, but first input is a string
    specifying "sigma" or "delta" """

    ref = kwargs.get("ref")
    isoval = kwargs.get("isoval")
    pin_cast = kwargs.get("pin_cast")
    pin_p = kwargs.get("pin_p")
    vert_dim = kwargs.get("vert_dim", -1)
    TOL_P_SOLVER = kwargs.get("TOL_P_SOLVER", 1e-4)
    eos = kwargs.get("eos", "gsw")
    eos_s_t = kwargs.get("eos_s_t")
    wrap = kwargs.get("wrap")
    diags = kwargs.get("diags", True)
    output = kwargs.get("output", True)
    geom = (
        kwargs.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ")
    )

    n_good = kwargs.get("n_good")
    Sppc = kwargs.get("Sppc")
    Tppc = kwargs.get("Tppc")
    interp_fn = kwargs.get("interp_fn", linear_coeffs)

    d = dict()
    sxr, txr, pxr = _xr_in(S, T, P, vert_dim)  # must call before _process_casts
    S, T, P, Sppc, Tppc, n_good, wrap, eos, eos_s_t = _process_args(
        S, T, P, vert_dim, wrap, diags, eos, eos_s_t, interp_fn, Sppc, Tppc, n_good
    )
    ni, nj = n_good.shape

    # Error checking on (ref, isoval, pin_cast, pin_p), then convert this
    # selection to (ref, isoval) pair
    _check_ref(ans_type, ref, isoval, pin_cast, pin_p, ni, nj)
    ref, isoval = _choose_ref_isoval(
        ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, Sppc, Tppc
    )

    # Solve non-linear root finding problem in each cast
    vertsolve = _make_vertsolve(eos, ans_type)
    timer = time()
    s, t, p = vertsolve(S, T, P, Sppc, Tppc, n_good, ref, isoval, TOL_P_SOLVER)

    if pin_p is not None:  # pin_cast must also be valid
        # Adjust the surface at the pinning cast slightly, to match the pinning
        # pressure / depth.  This fixes small deviations of order `TOL_P_SOLVER`
        n0 = pin_cast
        p[n0] = pin_p
        s[n0], t[n0] = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], pin_p)

    if diags:
        d["timer"] = time() - timer
        ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, p, eos_s_t, wrap, *geom)
        d["ϵ_RMS"], d["ϵ_MAV"] = ϵ_RMS, ϵ_MAV

        n_wet = np.sum(np.isfinite(p))
        d["n_wet"] = n_wet
        if output:
            print(
                f"{ans_type} done"
                f" | {n_wet:11d} wet casts"
                f" | RMS(ϵ) = {ϵ_RMS:.8e}",
                f" | {d['timer']:.3f} sec",
            )

    s, t, p = _xr_out(s, t, p, sxr, txr, pxr)
    return s, t, p, d


def omega_surf(S, T, P, **kwargs):
    """Calculate an omega surface from structured ocean data.

    Given 3D salinity, temperature, and pressure or depth data arranged on a
    rectilinear grid, calculate a 2D omega surface [1]_ [2]_, which is a
    highly accurate approximately neutral surface.

    Parameters
    ----------
    S, T, P : ndarray or xarray.DataArray
        See `sigma_surf`

    p_init : ndarray, Default None

        Pressure or depth on the initial approximately neutral surface.
        Omit to initialize with a "sigma" or "delta" surface.

        See Examples section.

    ref : float, or tuple of float of length 2

        If `p_init` is None, the reference value(s) for the initial "sigma"
        surface or "delta" surface that begins the iterations. If `ref` is a
        scalar, a "sigma" surface is used, and if `ref` is None, the
        reference pressure is `pin_p` (i.e. taken local to the pinning
        location). If `ref` is a tuple of length two, a "delta" surface is
        used, and if `ref` is(None, None), then the reference `S` and `T`
        values are taken local to the pinning location (pressure or depth
        `pin_p` on the pinning cast `pin_cast`).

        See Examples section.

    isoval : float

        Isovalue for the initial "sigma" or "delta" surface if `p_init` is not
        given.  Units are same as returned by `eos`.

        See Examples section.

    pin_p, pin_cast :
        See `sigma_surf`

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
    See `sigma_surf`.

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

    p_ml : ndarray, Default None

        Mixed Layer pressure or depth, which is removed from the surface on
        each iteration after the first. Pass None to not remove the mixed
        layer.

    Examples
    --------
    omega surfaces require a pinning cast and initial surface.  The surface is
    iteratively updated while remaining fixed at the pinning cast.  The
    initial surface can be provided directly, as the surface with pressure or
    depth given by `p_init`, in the following method:

    >>> omega_surf(S, T, P, pin_cast, p_init, ...)

    Alternatively, a "sigma" or "delta" surface can be used as the initial
    surface.  To do this, use one of the following two methods

    >>> omega_surf(S, T, P, ref, isoval, pin_cast, ...)

    >>> omega_surf(S, T, P, ref, pin_p, pin_cast, ...)

    For more info on these methods, see the Examples section of "sigma_surf".
    Note that `pin_cast` is always a required input.  Note that `ref` is
    needed to distinguish if the initial surface will be a "sigma" or "delta"
    surface.

    Notes
    -----
    See `sigma_surf` Notes.

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
    eos_s_t = kwargs.get("eos_s_t")
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
    # dist1_Ij = kwargs.get('dist1_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_Ij = kwargs.get('dist2_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_iJ = kwargs.get('dist2_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)
    # fmt: on
    # dist2on1_iJ = dist2_iJ / dist1_iJ
    # dist1on2_Ij = dist1_Ij / dist2_Ij
    geom = [
        kwargs.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ")
    ]

    n_good = kwargs.get("n_good")
    Sppc = kwargs.get("Sppc")
    Tppc = kwargs.get("Tppc")
    interp_fn = kwargs.get("interp_fn", linear_coeffs)

    dist2on1_iJ = geom[3] / geom[0]  # dist2_iJ / dist1_iJ
    dist1on2_Ij = geom[1] / geom[2]  # dist1_Ij / dist2_Ij

    sxr, txr, pxr = _xr_in(S, T, P, vert_dim)  # must call before _process_casts
    S, T, P, Sppc, Tppc, n_good, wrap, eos, eos_s_t = _process_args(
        S, T, P, vert_dim, wrap, diags, eos, eos_s_t, interp_fn, Sppc, Tppc, n_good
    )
    ni, nj = n_good.shape

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
        # Calculate an initial "sigma" or "delta" surface
        if isinstance(ref, (tuple, list)) and len(ref) == 2:
            ans_type = "delta"
        else:
            ans_type = "sigma"

        # Update arguments with pre-processed values
        kwargs["Sppc"] = Sppc
        kwargs["Tppc"] = Tppc
        kwargs["n_good"] = n_good
        kwargs["eos"] = eos
        kwargs["eos_s_t"] = eos_s_t
        kwargs["wrap"] = wrap
        kwargs["vert_dim"] = -1  # Since S, T, P already reordered
        kwargs["diags"] = False  # Will make our own diags next
        s, t, p, _ = _sigma_delta_surf(ans_type, S, T, P, **kwargs)

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
        s, t = val2(P, S, Sppc, T, Tppc, p)

    pin_p = p[pin_cast]

    if np.isnan(p[pin_cast]):
        raise RuntimeError("The initial surface is NaN at the reference cast.")

    # Get mixed layer: the pressure of the mixed layer
    # if ITER_MAX > 1 && if isstruct(p_ml)
    #   # Compute the mixed layer from parameter inputs
    #   p_ml = mixed_layer(S, T, P, ML)
    # end

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

    vertsolve = _make_vertsolve(eos, "omega")

    # --- Begin iterations
    # Note: the surface exists wherever p is non-nan.  The nan structure of s
    # and t is made to match that of p when the vertical solve step is done.
    Δp_RMS = 0.0  # ensure this is defined; needed if TOL_P_CHANGE_RMS == 0
    for iter_ in range(1, ITER_MAX + 1):
        timer = time()

        # --- Remove the Mixed Layer
        # But keep it for the first iteration, which may be initialized from a
        # not very neutral surface
        if iter_ > 1 and p_ml is not None:
            p[p < p_ml] = np.nan

        # --- Determine the connected component containing the reference cast, via Breadth First Search
        timer_loc = time()
        if iter_ >= ITER_START_WETTING and iter_ <= ITER_STOP_WETTING:
            qu, qt, n_newly_wet = bfs_conncomp1_wet(
                s, t, p, S, T, P, Sppc, Tppc, n_good, A4, pin_cast_1, TOL_P_SOLVER, eos
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
        vertsolve(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, TOL_P_SOLVER)

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

    s, t, p = _xr_out(s, t, p, sxr, txr, pxr)

    return s, t, p, d


def _check_ref(ans_type, ref, isoval, pin_cast, pin_p, ni, nj):
    """Error checking on ref / isoval / pin_cast / pin_p combinations for "sigma"
    and "delta" surfaces
    """
    # First check None values to validate one of the following options:
    # >>> _sigma_delta_surf(ans_type, S, T, P, ref, isoval)
    # >>> _sigma_delta_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _sigma_delta_surf(ans_type, S, T, P, pin_cast, pin_p)
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
        if ans_type == "sigma":
            if not isinstance(ref, float):
                raise TypeError(
                    'For "sigma" surfaces, if provided "ref" must be a float'
                )
        else:  # ans_type == "delta"
            if not (isinstance(ref, (tuple, list)) and len(ref) == 2):
                raise TypeError(
                    'For "delta" surfaces, if provided "ref" must be 2 element tuple/list of float'
                )

    # Error checking on pin_cast
    if pin_cast is not None:
        if (
            isinstance(pin_cast, (tuple, list))
            and len(pin_cast) == 2
            and all(isinstance(x, int) for x in pin_cast)
        ):
            if (
                pin_cast[0] < 0
                or pin_cast[1] < 0
                or pin_cast[0] >= ni
                or pin_cast[1] >= nj
            ):
                raise ValueError(
                    '"pin_cast" must index a cast within the domain; '
                    f'found "pin_cast" = {pin_cast} outside the bounds (0,{ni-1}) x (0,{nj-1})'
                )
        else:
            raise TypeError(
                'If provided, "pin_cast" must be a tuple or list of 2 integers'
            )

    # Error checking on pin_p
    if not isinstance(pin_p, (type(None), float)):
        raise TypeError('If provided, "pin_p" must be a float')


def _choose_ref_isoval(
    ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, Sppc, Tppc
):
    # Handle the three valid calls in the following order of precedence:
    # >>> _sigma_delta_surf(ans_type, S, T, P, ref, isoval)
    # >>> _sigma_delta_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _sigma_delta_surf(ans_type, S, T, P, pin_cast, pin_p)
    if isoval is None:  # => pin_cast and pin_p are both not None
        n0 = pin_cast  # evaluate S and T on the surface at the chosen location
        s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], pin_p)

        if ans_type == "sigma":
            if ref is None:
                ref = pin_p
            isoval = eos(s0, t0, ref)
        else:  # ans_type == "delta"
            if ref is None or any(x is None for x in ref):
                ref = (s0, t0)
            isoval = eos(s0, t0, pin_p) - eos(
                ref[0], ref[1], pin_p
            )  #  == 0 when ref = (s0, t0)

    return ref, isoval
