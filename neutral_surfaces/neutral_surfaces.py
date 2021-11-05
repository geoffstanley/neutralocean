"""
Calculate approximately neutral surface from 3D ocean data. 
"""

import numpy as np
import xarray as xr
from time import time

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t
from neutral_surfaces._vertsolve import _make_vertsolve
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val2
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces.lib import (
    ntp_ϵ_errors_norms,
    find_first_nan,
    xr_to_np,
    _process_wrap,
    _process_casts,
    _process_vert_dim,
)
from neutral_surfaces._omega import _omega_matsolve_poisson


def approx_neutral_surf(ans_type, S, T, P, **kwargs):
    """Calculate approximately neutral surface from structured 3D ocean data.

    Given 3D salinity, temperature, and pressure or depth data arranged on a
    rectilinear grid, calculate a 2D potential density surface, specific
    volume (or in-situ density) anomaly surface, or an omega surface.

    Parameters
    ----------
    ans_type : str

        'sigma', to compute a potential density surface,

        'delta', to compute an in-situ density anomaly or specific volume
        anomaly surface, or

        'omega', to compute an omega-surface [1]_ [2]_.

    S, T : ndarray or xarray.DataArray

        3D practical / Absolute salinity and potential / Conservative
        temperature

    P : ndarray or xarray.DataArray

        In the non-Boussinesq case, `P` is the 3D pressure, sharing the same
        dimensions as `S` and `T`.

        In the Boussinesq case, `P` is the depth and can be 3D with the same
        structure as `S` and `T`, or can be 1D with as many elements as there
        are in the vertical dimension of `S` and `T`.

    ref : float, or tuple of float of length 2

        If `ans_type` == "sigma", a float giving the reference pressure or 
        depth, in the same units as P.

        If `ans_type` == "delta", a tuple or list of two floats, giving the
        reference S and T values.

        If `ans_type` == "omega" and `p_init` is None, the reference value(s)
        for the initial "sigma" surface (if `ref` is a scalar) or "delta"
        surface (if `ref` is a tuple of length two).  To use local reference
        values (advised), pass `ref` as None or (None, None).

        Whenever `ref` is None or has a None element, the reference value(s)
        are taken from the local ocean properties at `(pin_cast, pin_p)`.

        See Examples section.

    isoval : float

        Isovalue for "sigma" or "delta" surface.
        Units are same as returned by `eos`.

        See Examples section.

    pin_p : float

        Pressure or depth at which the surface is fixed in cast `pin_cast`.

        See Examples section.

    pin_cast : tuple or list of int of length 2

        Index for cast where surface is fixed at pressure or depth `pin_p`.

        See Examples section.

    Returns
    -------
    s : ndarray or xarray.DataArray

        practical / Absolute salinity on surface

    t : ndarray or xarray.DataArray

        potential / Conservative temperature on surface

    p : ndarray or xarray.DataArray

        pressure or depth on surface

    d : dict

        Diagnostics.  The first four are given for all surface types. 
        For "omega" surfaces, all diagnostics are given.  The first four
        give information going into the `i`'th iteration, i.e. the 0'th element
        is about the initial surface.  The others give information about what
        the `i`'th iteration did, and hence their 0'th elements are meaningless.
        
        ``"ϵ_MAV"`` : float or array of float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.  When `ans_type` == "omega", ``d["ϵ_MAV"][0]``
            is this value for the initial surface, and ``d["ϵ_MAV"][i]`` is
            this value after the `i`'th iteration.

        ``"ϵ_RMS"`` : float or array of float

            As ``"ϵ_MAV"`` but for the Root Mean Square.
            
        ``"n_wet"``: float or array of float
        
            Number of wet casts (surface points).

        ``"timer"`` : float or array of float

            Time spent on the whole algorithm, excluding set-up and diagnostics.

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

        ``"timer_matbuild"`` : array of float

            Time spent building the matrix problem, per iteration.

        ``"timer_matsolve"`` : array of float

            Time spent solving the matrix problem, per iteration.

        ``"timer_update"`` : array of float

            Time spent vertically updating the surface.

    Other Parameters
    ----------------
    wrap : tuple of bool, or tuple of str

        Specifies which dimensions are periodic.

        As a tuple of bool, this must be length two.  The first or second
        non-vertical dimension of `S` and `T` is periodic iff ``wrap[0]`` or
        ``wrap[1]`` is True, respectively.

        As a tuple of str, simply name the periodic dimensions of `S` and
        `T`.

        Required if `ans_type` is "omega" or diags is True

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes`.


    dist1_iJ : float or ndarray, Default 1.

        Distance [m] in 1st lateral dimension centred at (I-1/2, J).
        The naming uses a soft notation, where i = I-1/2.

    dist2_Ij : float or ndarray, Default 1.

        Distance [m] in 2nd lateral dimension centred at (I, J-1/2).
        The naming uses a soft notation, where j = J-1/2.

    dist2_iJ : float or ndarray, Default 1.

        Distance [m] in 2nd lateral dimension centred at (I-1/2, J).

    dist1_Ij : float or nndarray, Default 1.

        Distance [m] in 1st lateral dimension centred at (I, J-1/2).

    eos : str or function, Default 'gsw'

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and pressure inputs.  For Boussinesq models, provide `grav`
        and `rho_c`, so this function with third input pressure will be
        converted to a function with third input depth. 

        If a function, this should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

        If a str, can be either 'gsw' to use TEOS-10
        or 'jmd95' to use Jackett and McDougall (1995) [3]_.

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

    TOL_P_SOLVER : float, Default 1e-4

        Error tolerance when root-finding to update the pressure or depth of
        the surface in each water column. Units are the same as `P`.

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

        If True, calculate diagnostics (4th output) for each iteration.

        If False, 4th output is an empty dict.

    output : bool, Default True

        If True, prints diagnostic output during computation.
        `diags` must be True for this to have any effect.
        To redirect this output to a file, do the following
        >>> import sys
        >>> tmp = sys.stdout
        >>> sys.stdout = file_id = open('myfile.txt', 'w')
        >>> approx_neutral_surf(...)
        >>> sys.stdout = tmp
        >>> file_id.close()
        
    **Other Parameters specific to "omega" surfaces**

    p_init : ndarray, Default None

        Pressure or depth on the initial approximately neutral surface.  Pass
        None to initialize with a "sigma" or "delta" surface.
        See Examples section.

    p_ml : ndarray, Default None

        Mixed Layer pressure or depth, which is removed from the surface on
        each iteration after the first. Pass None to not remove the mixed
        layer.

    ITER_MIN : int, Default 1

        Minimum number of "omega" iterations.

    ITER_MAX : int, Default 10

        Maximum number of "omega" iterations.

    ITER_START_WETTING : int, Default 1

        Iteration on which wetting begins.  Set to `ITER_MAX` + 1 to deactivate.

    ITER_STOP_WETTING : int, Default 5

        The last iteration on which to perform wetting.  This can be useful to
        avoid pesky water columns that repeatedly wet then dry.

    TOL_LRPD_MAV : float, Default 1e-7

        Exit iterations when the mean absolute value of the Locally Referenced
        Potential Density perturbation that updates the surface from one
        iteration to the next is less than this value.  Units are [kg m-3],
        even if `eos` returns a specific volume.  Set to 0 to deactivate.

    TOL_P_CHANGE_RMS : float, Default 0.

        Exit iterations when the root mean square of the pressure or depth
        change on the surface from one iteration to the next is less than this
        value. Set to 0 to deactivate. Units are the same as `P` [dbar or m].


    Examples
    --------
    If `ans_type` is "sigma" or "delta", the surface to be calculated must be
    specified by some combination of reference value(s) `ref`, isovalue `isoval`,
    pinning cast `pin_cast` and pinning pressure `pin_p`.  The following methods
    are valid, and listed in order of precedence, e.g. `pin_cast` and `pin_p` are
    not used if both `ref` and `isoval` are given.

    >>> approx_neutral_surf(ans_type, S, T, P, ref, isoval, ...)

    This finds the surface with given reference value(s) and the given
    isovalue.

    >>> approx_neutral_surf(ans_type, S, T, P, ref, pin_p, pin_cast, ...)

    This finds the surface with given reference value(s) that intersects
    the given cast at the given pressure or depth.

    >>> approx_neutral_surf(ans_type, S, T, P, pin_p, pin_cast, ...)

    This is as for the previous method, but selects the reference value(s)
    from the ocean data at the given cast's given pressure or depth.

    If `ans_type` == "omega", a pinning cast and initial surface must be
    given.  This can be done by using the above three methods to initialize
    the "omega" algorithm with a "sigma" or "delta" surface, but note the
    first method must also specify `pin_cast`, as the "omega" algorithm
    iteratively adjusts the surface while always intersecing the pinning cast
    at a fixed pressure.  Note if `ref` is unspecified (as in the third case
    above), the initial surface will be a "delta" surface.  Alternatively,
    the initial surface can be specified directly, as follows:

    >>> approx_neutral_surf("omega", S, T, P, pin_cast, p_init, ...)

    Notes
    -----
    This code will internally re-arrange `S`, `T`, `P` to have the vertical
    dimension last, so that the data for an individual water column is
    contiguous in memory.  If you call this function many times, consider
    using ``neutral_surfaces.unpack_STP`` to pre-processing your `S`, `T`,
    `P` inputs to have the vertical dimension last.  Also consider
    pre-computing `Sppc`, `Tppc`, and `n_good` (see the documentation for
    these inputs).

    .. [1] Stanley, McDougall, Barker 2021, Algorithmic improvements to finding
     approximately neutral surfaces, Journal of Advances in Earth System
     Modelling, 13(5).

    .. [2] Klocker, McDougall, Jackett 2009, A new method of forming approximately
     neutral surfaces, Ocean Science, 5, 155-172.

    .. [3] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388
    """

    # Get extra arguments
    # fmt: off
    ref = kwargs.get('ref')
    isoval = kwargs.get('isoval')
    pin_cast = kwargs.get('pin_cast')
    pin_p = kwargs.get('pin_p')
    p_init = kwargs.get('p_init')

    # grid distances.  (soft notation: i = I-1/2; j = J-1/2)
    # dist1_iJ = kwargs.get('dist1_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)
    # dist1_Ij = kwargs.get('dist1_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_Ij = kwargs.get('dist2_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_iJ = kwargs.get('dist2_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)

    wrap = kwargs.get('wrap')
    vert_dim = kwargs.get('vert_dim', -1)

    eos = kwargs.get('eos', 'gsw')
    eos_s_t = kwargs.get('eos_s_t')
    grav = kwargs.get('grav')
    rho_c = kwargs.get('rho_c')

    interp_fn = kwargs.get('interp_fn', linear_coeffs)
    n_good = kwargs.get('n_good')
    Sppc = kwargs.get('Sppc')
    Tppc = kwargs.get('Tppc')

    # output = kwargs.get('output', True)
    diags = kwargs.get('diags', True)
    # fmt: on

    if ans_type not in ("sigma", "delta", "omega"):
        raise ValueError('"ans_type" must be one of ("sigma", "delta", "omega")')

    # Prepare xarray container for outputs if xarrays given for inputs
    s_ = None
    S_is_xr = isinstance(S, xr.core.dataarray.DataArray)
    T_is_xr = isinstance(T, xr.core.dataarray.DataArray)
    P_is_xr = isinstance(P, xr.core.dataarray.DataArray)
    if S_is_xr:
        s_ = xr.full_like(S.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if T_is_xr:
        t_ = xr.full_like(T.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if P_is_xr:
        p_ = xr.full_like(P.isel({vert_dim: 0}).drop_vars(vert_dim), 0)

    if wrap is None:
        if diags or ans_type == "omega":
            raise ValueError(
                'wrap must be given if diags is True or ans_type is "omega"'
            )
    else:
        wrap = _process_wrap(wrap, s_)

    # Process 3D hydrography
    vert_dim = _process_vert_dim(vert_dim, S)
    if P.ndim < S.ndim:
        P = np.broadcast_to(P, S.shape)
    S, T, P = (_process_casts(x, vert_dim) for x in (S, T, P))
    ni, nj, nk = S.shape
    if n_good is None:
        n_good = find_first_nan(S)

    # Compute interpolants for S and T casts (unless already provided)
    if Sppc is None or Sppc.shape[0:-1] != (ni, nj, nk - 1):
        Sppc = interp_fn(P, S)
    if Tppc is None or Tppc.shape[0:-1] != (ni, nj, nk - 1):
        Tppc = interp_fn(P, T)
    # Error checking on ref, isoval, pin_cast, pin_p
    _check_ref(ref, isoval, pin_cast, pin_p, ans_type, ni, nj)

    # Handling and error checking on p_init
    if p_init is not None and ans_type == "omega":
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

    # Process equation of state function and make cache functions
    if eos_s_t is None and isinstance(eos, str):
        eos_s_t = eos
    elif isinstance(eos, str) and isinstance(eos_s_t, str) and eos != eos_s_t:
        raise ValueError("eos and eos_s_t, if strings, must be the same string")
    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)

    args = (S, T, P, Sppc, Tppc, n_good, eos, eos_s_t, wrap)

    if ans_type == "omega":

        # Check omega args: must have pin_cast.
        # Checks on ref, isoval, pin_p will be done later, if needed.
        if pin_cast is None:
            raise TypeError('`pin_cast` must be given when `ans_type` is "omega"')

        # fmt: off
        opts = {k: kwargs[k] for k in kwargs.keys()
            if k in (
                "diags", "output",
                "ref", "isoval", "pin_p", "pin_cast", "p_init",
                "p_ml",
            ) or k[0:4] in ("ITER", "TOL_", "dist")
        }
        # fmt: on
        s, t, p, d = omega_surf(*args, **opts)
        # s, t, p, d = omega_surf(*args, kwargs)  # DEV: alternate strat: pass all kwargs as a single argument

        # DEV: A third strat is to include (Sppc, Tppc, n_good, tol_p, eos,
        # eos_s_t, wrap) as named arguments in approx_neutral_surf argument
        # list, so they don't go into kwargs and then **kwargs can be passed
        # straight through without worrying about having two "Sppc" vars, e.g.

    else:  # ans_type in ("sigma", "delta")

        # fmt: off
        opts = {k: kwargs[k] for k in kwargs.keys()
            if k in ("diags", "output",
                "ref", "isoval", "pin_p", "pin_cast"
            )
        }
        # fmt: on
        s, t, p, d = sigma_delta_surf(ans_type, *args, **opts)

    # Return xarrays if inputs were xarrays
    if S_is_xr:
        s_.data = s
        s = s_
    if T_is_xr:
        t_.data = t
        t = t_
    if P_is_xr:
        p_.data = p
        p = p_

    return s, t, p, d


def _sigma_delta_surf(ans_type, S, T, P, Sppc, Tppc, n_good, eos, **opts):
    """Efficient function for computing "sigma" or "delta" surfaces
    Inputs are as in `approx_neutral_surface`"""
    ref = opts.get("ref")
    isoval = opts.get("isoval")
    pin_cast = opts.get("pin_cast")
    pin_p = opts.get("pin_p")

    TOL_P_SOLVER = opts.get("TOL_P_SOLVER", 1e-4)

    ref, isoval = _choose_ref_isoval(
        ref, isoval, pin_cast, pin_p, ans_type, eos, S, T, P, Sppc, Tppc
    )

    # Solve non-linear root finding problem in each cast
    vertsolve = _make_vertsolve(eos, ans_type)
    s, t, p = vertsolve(S, T, P, Sppc, Tppc, n_good, ref, isoval, TOL_P_SOLVER)
    return s, t, p


def sigma_delta_surf(ans_type, S, T, P, Sppc, Tppc, n_good, eos, eos_s_t, wrap, **opts):

    timer = time()

    diags = opts.get("diags", True)
    output = opts.get("output", True)
    geom = (opts.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ"))

    s, t, p = _sigma_delta_surf(ans_type, S, T, P, Sppc, Tppc, n_good, eos, **opts)

    d = dict()
    if diags:
        d["timer"] = time() - timer
        ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, p, eos_s_t, wrap, *geom)
        d["ϵ_RMS"], d["ϵ_MAV"] = ϵ_RMS, ϵ_MAV
        
        n_wet = np.sum(np.isfinite(p))
        d["n_wet"] = n_wet
        if output:
            print(
                f"{ans_type} done"
                f" | {d['timer']:5.2f} sec"
                f" | {d['n_wet']:4} wet casts"
                f" | log_10(rms(ϵ)) = {np.log10(ϵ_RMS) : 9.6f}",
            )

    return s, t, p, d


def omega_surf(S, T, P, Sppc, Tppc, n_good, eos, eos_s_t, wrap, **opts):
    """Efficient function for computing "omega" surfaces, given initial surface
    Inputs are as in `approx_neutral_surface`"""

    timer = time()

    ref = opts.get("ref")
    pin_p = opts.get("pin_p")
    pin_cast = opts.get("pin_cast")
    p_init = opts.get("p_init")
    p_ml = opts.get("p_ml")
    diags = opts.get("diags", True)
    output = opts.get("output", True)
    ITER_MIN = opts.get("ITER_MIN", 1)
    ITER_MAX = opts.get("ITER_MAX", 10)
    ITER_START_WETTING = opts.get("ITER_START_WETTING", 1)
    ITER_STOP_WETTING = opts.get("ITER_STOP_WETTING", 5)
    TOL_P_SOLVER = opts.get("TOL_P_SOLVER", 1e-4)
    TOL_LRPD_MAV = opts.get("TOL_LRPD_MAV", 1e-7)
    TOL_P_CHANGE_RMS = opts.get("TOL_P_CHANGE_RMS", 0.0)
    # fmt: off
    # grid distances.  (soft notation: i = I-1/2; j = J-1/2)
    # dist1_iJ = kwargs.get('dist1_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)
    # dist1_Ij = kwargs.get('dist1_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_Ij = kwargs.get('dist2_Ij', 1.) # Distance [m] in 2nd dim centred at (I, J-1/2)
    # dist2_iJ = kwargs.get('dist2_iJ', 1.) # Distance [m] in 1st dim centred at (I-1/2, J)
    # fmt: on
    # dist2on1_iJ = dist2_iJ / dist1_iJ
    # dist1on2_Ij = dist1_Ij / dist2_Ij
    geom = [opts.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ")]

    dist2on1_iJ = geom[3] / geom[0]  # dist2_iJ / dist1_iJ
    dist1on2_Ij = geom[1] / geom[2]  # dist1_Ij / dist2_Ij

    ni, nj = n_good.shape

    pin_cast_1 = np.ravel_multi_index(pin_cast, (ni, nj))  # linear index

    # Pre-calculate grid adjacency needed for Breadth First Search:
    # all grid points that are adjacent to all grid points
    #A5 = grid_adjacency((ni, nj), 5, wrap)  # using 5-connectivity
    #A4 = A5[:, 0:-1]  # using 4-connectivity
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
            "timer_matbuild": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_matsolve": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_update": np.zeros(ITER_MAX + 1, dtype=np.float64),
        }
    else:
        d = dict()

    if p_init is None:
        # Calculate an initial "sigma" or "delta" surface
        ref = opts.get("ref")
        if isinstance(ref, (tuple, list)) and len(ref) == 2:
            ans_type = "delta"
            if any(x is None for x in ref):
                ref = None  # reset (None, None) or similar trigger
        else:
            ans_type = "sigma"

        s, t, p = _sigma_delta_surf(ans_type, S, T, P, Sppc, Tppc, n_good, eos, **opts)

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

    if diags:
        ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, p, eos_s_t, wrap, *geom)
        d["ϵ_RMS"][0], d["ϵ_MAV"][0] = ϵ_RMS, ϵ_MAV
        
        n_wet = np.sum(np.isfinite(p))
        d["n_wet"][0] = n_wet

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

    vertsolve = _make_vertsolve(eos, "omega")

    if diags:
        d["timer"][0] = time() - timer
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
                f" {d['n_wet'][0]:11}         |"
                f" {ϵ_RMS:.8e} |"
                f" {d['timer'][0]:.3f}"
            )

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
        ϕ, timer_matbuild = _omega_matsolve_poisson(
            s, t, p, dist2on1_iJ, dist1on2_Ij, wrap, A4, qu, qt, pin_cast, eos_s_t
        )
        timer_solver = time() - timer_loc - timer_matbuild

        # --- Update the surface
        timer_loc = time()
        p_old = p.copy()  # Record old surface for pinning and diagnostics
        vertsolve(
            s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, TOL_P_SOLVER
        )  # mutates s, t, p

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

            d["timer_matbuild"][iter_] = timer_matbuild
            d["timer_matsolve"][iter_] = timer_solver
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
                    f" {n_wet:11} ({n_newly_wet:5}) |"
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

    return s, t, p, d


def _check_ref(ref, isoval, pin_cast, pin_p, ans_type, ni, nj):
    """Error checking on ref / isoval / pin_cast / pin_p combinations for "sigma"
    and "delta" surfaces, only
    """
    # First check None values to validate one of the following options:
    # >>> approx_neutral_surf(ans_type, S, T, P, ref, isoval)
    # >>> approx_neutral_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> approx_neutral_surf(ans_type, S, T, P, pin_cast, pin_p)
    if ref is None:
        if pin_cast is None or pin_p is None:
            raise TypeError(
                'If "ref" is not provided and ans_type is "sigma" or "delta",'
                ' "pin_cast" and "pin_p" must be provided'
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
                    'If provided, "ref" must be float when "ans_type" is "sigma"'
                )
        else:  # ans_type == "delta"
            if not (isinstance(ref, (tuple, list)) and len(ref) == 2):
                raise TypeError(
                    'If provided, "ref" must be 2 element tuple/list of float when "ans_type" is "delta"'
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
    ref, isoval, pin_cast, pin_p, ans_type, eos, S, T, P, Sppc, Tppc
):
    # Handle the three valid calls in the following order of precedence:
    # >>> approx_neutral_surf(ans_type, S, T, P, ref, isoval)
    # >>> approx_neutral_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> approx_neutral_surf(ans_type, S, T, P, pin_cast, pin_p)
    if isoval is None:  # => pin_cast and pin_p are both not None
        n0 = pin_cast  # evaluate S and T on the surface at the chosen location
        s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], pin_p)

        if ans_type == "sigma":
            if ref is None:
                ref = pin_p
            isoval = eos(s0, t0, ref)
        else:  # ans_type == "delta"
            if ref is None:
                ref = (s0, t0)
            elif not (
                isinstance(ref, (tuple, list))
                and len(ref) == 2
                and all(isinstance(x, float) for x in ref)
            ):
                raise TypeError(
                    '"ref" must be None or tuple/list of 2 floats when "ans_type" is "delta"'
                )
            # isoval == 0. when ref = (s0,t0)
            isoval = eos(s0, t0, pin_p) - eos(ref[0], ref[1], pin_p)

    return ref, isoval
