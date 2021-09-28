"""
Calculate approximately neutral surface from 3D ocean data. 
"""

import numpy as np
import xarray as xr
import numba
import sys
from time import time

import functools

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t
from neutral_surfaces.fzero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val2
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces.lib import (
    ntp_ϵ_errors_norms,
    find_first_nan,
    _process_wrap,
    _process_input,
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

        If `ans_type` == "sigma", a float giving the reference pressure or depth,
        in the same units as P.

        If `ans_type` == "delta", a tuple or list of two floats, giving the
        reference S and T values.

        If `ans_type` == "omega" and `p_init` is None, the reference value
        for the initial "sigma" surface (if `ref` is a scalar) or "delta"
        surface (if `ref` is a tuple of length two).

        Whenever `ref` is None or has a None element, the reference value(s)
        are taken from the local ocean properties at `(pin_loc, pin_p)`.

        See Examples section.

    isoval : float

        Isovalue for "sigma" or "delta" surface.
        Units are same as output by `eos`.

        See Examples section.

    pin_loc : tuple or list of int of length 2

        Location of water column where surface is fixed at pressure or depth
        `pin_p`.

        See Examples section.

    pin_p : float

        Pressure or depth at which the surface is fixed in water column
        `pin_loc`.

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

        Diagnostics for the algorithm.  The first four apply to all surface
        types, and are per- iteration when `ans_type` == "omega".

        ``"ϵ_MAV"`` : array of float

            Mean Absolute Value of the ϵ neutrality error on the surface,
            area-weighted.  Units are those of `eos` return values divided by
            those of `dist*` inputs.  When `ans_type` == "omega", ``d["ϵ_MAV"][0]``
            is this value for the initial surface, and ``d["ϵ_MAV"][i]`` is
            this value after the `i`'th iteration.

        ``"ϵ_RMS"`` : array of float

            As ``"ϵ_MAV"`` but for the Root Mean Square.

        ``"timer"`` : array of float

            Time spent on the whole algorithm, excluding diagnostics.

        ``"timer_update"`` : array of float

            Time spent vertically updating the surface.

        ``"ϕ_MAV"`` : array of float

            Mean Absolute Value of the Locally Referenced Potential Density
            perturbation, per iteration.

        ``"Δp_MAV"`` : array of float

            Mean Absolute Value of the pressure or depth change from one
            iteration to the next.

        ``"Δp_RMS"`` : array of float

            Root Mean Square of the pressure or depth change from one
            iteration to the next.

        ``"Δp_Linf"`` : array of float

            Maximum absolute value (infinity norm) of the pressure or depth
            change from one iteration to the next.

        ``"n_wet"`` : array of int

            Number of casts that are newly wet, per iteration

        ``"timer_bfs"`` : array of float

            Time spent in Breadth-First Search including wetting, per iteration.

        ``"timer_matbuild"`` : array of float

            Time spent building the matrix problem, per iteration.

        ``"timer_matsolve"`` : array of float

            Time spent solving the matrix problem, per iteration.

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
        `S`, `T`, and pressure (not depth) inputs.

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

    tol_p : float, Default 1e-4

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

    verbose : int, Default 1

        0, show no output
        1, show a moderate level of information. Sets diags = True.

    file_name : str, Default None

        Name of text file where info is output when `verbose` > 0.  Pass None
        to output to stdout.

    **Other Parameters specific to "omega" surfaces**

    p_init : ndarray, Default None

        Pressure or depth on the initial approximately neutral surface.  Pass
        None to initialize with a "sigma" or "delta" surface.
        See Examples section.

    ML : ndarray, Default None

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
    The surface to be calculated must be specified by some combination of
    reference value(s) `ref`, isovalue `isoval`, pinning water column
    `pin_loc` and pinning pressure `pin_p`.  The following methods are valid.
    They are listed in order of precedence, e.g. `pin_loc` and `pin_p` are not
    used if both `ref` and `isoval` are given.

    >>> approx_neutral_surf(ans_type, S, T, P, wrap, ref, isoval)

    This finds the surface with given reference value(s) and the given
    isovalue.

    >>> approx_neutral_surf(ans_type, S, T, P, wrap, ref, pin_loc, pin_p)

    This finds the surface with given reference value(s) that intersects
    the given water column at the given pressure or depth.

    >>> approx_neutral_surf(ans_type, S, T, P, wrap, pin_loc, pin_p)

    This is as for the previous method, but selects the reference value(s)
    from the ocean data at the given water column's given pressure or depth.

    If `ans_type` == "omega", the above 3 methods are applied to find the
    initial surface, which is iteratively improved upon.  A 4th method is

    >>> approx_neutral_surf("omega", S, T, P, wrap, pin_loc, p_init)

    This specifies the initial surface by its pressure or depth throughout the
    ocean.  The "omega" surface will be pinned to intersect the given water
    column at the same pressure or depth as the initial surface.

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

    timer = time()

    if ans_type not in ("sigma", "delta", "omega"):
        raise ValueError('"ans_type" must be one of ("sigma", "delta", "omega")')

    # Get extra arguments
    # fmt: off
    ref = kwargs.get('ref')
    pin_loc = kwargs.get('pin_loc')
    pin_p = kwargs.get('pin_p')
    isoval = kwargs.get('isoval')
    p_init = kwargs.get("p_init")

    # grid distances.  (soft notation: i = I-1/2; j = J-1/2)
    dist1_iJ = kwargs.get('dist1_iJ', 1) # Distance [m] in 1st dim centred at (I-1/2, J)
    dist2_Ij = kwargs.get('dist2_Ij', 1) # Distance [m] in 2nd dim centred at (I, J-1/2)
    dist2_iJ = kwargs.get('dist2_iJ', 1) # Distance [m] in 1st dim centred at (I-1/2, J)
    dist1_Ij = kwargs.get('dist1_Ij', 1) # Distance [m] in 2nd dim centred at (I, J-1/2)

    wrap = kwargs.get('wrap')
    vert_dim = kwargs.get('vert_dim', -1)

    eos = kwargs.get('eos', 'gsw')
    eos_s_t = kwargs.get('eos_s_t')
    grav = kwargs.get('grav')
    rho_c = kwargs.get('rho_c')

    tol_p = kwargs.get('tol_p', 1e-4)
    
    interp_fn = kwargs.get('interp_fn', linear_coeffs)
    n_good = kwargs.get('n_good')
    Sppc = kwargs.get('Sppc')
    Tppc = kwargs.get('Tppc')

    file_name = kwargs.get('file_name')
    verbose = kwargs.get('verbose', 1)
    diags = kwargs.get('diags', True)
    # fmt: on

    if diags is False:
        verbose = 0

    if verbose > 0:
        diags = True

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
    S, T, P = (_process_input(x, vert_dim) for x in (S, T, P))
    ni, nj, nk = S.shape
    if n_good is None:
        n_good = find_first_nan(S)

    # Compute interpolants for S and T casts (unless already provided)
    if Sppc is None or Sppc.shape[0:-1] != (ni, nj, nk - 1):
        Sppc = interp_fn(P, S)
    if Tppc is None or Tppc.shape[0:-1] != (ni, nj, nk - 1):
        Tppc = interp_fn(P, T)

    # Process equation of state function and make cache functions
    if eos_s_t is None and isinstance(eos, str):
        eos_s_t = eos
    elif isinstance(eos, str) and isinstance(eos_s_t, str) and eos != eos_s_t:
        raise ValueError("eos and eos_s_t, if strings, must be the same string")
    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)
    vertsolve = _make_vertsolve(eos, ans_type)

    # Error checking on ref / isoval / pin_loc / pin_p combinations.
    # The valid options are:
    # >>> approx_neutral_surf(ans_type, S, T, P, wrap, ref, isoval)
    # >>> approx_neutral_surf(ans_type, S, T, P, wrap, ref, pin_loc, pin_p)
    # >>> approx_neutral_surf(ans_type, S, T, P, wrap, pin_loc, pin_p)
    # >>> approx_neutral_surf("omega" , S, T, P, wrap, pin_loc, p_init)
    if ref is None:
        if ans_type in ("sigma", "delta"):
            if pin_loc is None or pin_p is None:
                raise TypeError(
                    'If "ref" is not provided and ans_type is "sigma" or "delta",'
                    ' "pin_loc" and "pin_p" must be provided'
                )
        else:  # ans_type == "omega"
            if pin_loc is None and (pin_p is None or p_init is None):
                raise TypeError(
                    'If "ref" is not provided and ans_type is "omega",'
                    ' "pin_loc" must be provided, and either "pin_p" or "p_init"'
                    " must be provided"
                )
    else:  # ref is not None
        if isoval is None and (pin_loc is None or pin_p is None):
            raise TypeError(
                'If "ref" is provided, either "isoval" must be provided or'
                ' "pin_loc" and "pin_p" must be provided'
            )

    # Error checking on pin_loc
    if pin_loc is not None:
        if (
            isinstance(pin_loc, (tuple, list))
            and all(isinstance(x, int) for x in pin_loc)
            and len(pin_loc) == 2
        ):
            if pin_loc[0] < 0 or pin_loc[1] < 0 or pin_loc[0] >= ni or pin_loc[1] >= nj:
                raise ValueError(
                    '"pin_loc" must index a cast within the domain;'
                    f'found "pin_loc" = {pin_loc} outside the bounds (0,{ni-1}) x (0,{nj-1})'
                )
        else:
            raise TypeError(
                'If provided, "pin_loc" must be a tuple or list of 2 integers'
            )

    # Error checking on pin_p
    if not isinstance(pin_p, (type(None), float)):
        raise TypeError('If provided, "pin_p" must be a float')

    # Error checking on p_init
    if p_init is not None:
        if not isinstance(p_init, np.ndarray):
            raise TypeError('If provided, "p_init" must be an ndarray')
        if p_init.shape != (ni, nj):
            raise ValueError(
                f'"p_init" should be a 2D array of size ({ni}, {nj});'
                f"found size {p_init.shape}"
            )

    # Get ratios of distances and expand to [ni,nj] for ntp_ϵ_errors_norm
    # DEV:  The following broadcast_to calls are probably not general enough...
    # If dist2_Ij is a vector of length nj, for instance, this crashes.
    dist1_iJ = np.broadcast_to(dist1_iJ, (ni, nj))
    dist1_Ij = np.broadcast_to(dist1_Ij, (ni, nj))
    dist2_Ij = np.broadcast_to(dist2_Ij, (ni, nj))
    dist2_iJ = np.broadcast_to(dist2_iJ, (ni, nj))
    areaiJ = dist1_iJ * dist2_iJ
    areaIj = dist1_Ij * dist2_Ij

    d = dict()

    if file_name is None:
        file_id = sys.stdout
    else:
        file_id = open(file_name, "w")

    if ans_type in ("sigma", "delta"):

        # Handle the three valid calls in the following order of precedence:
        # >>> approx_neutral_surf(ans_type, S, T, P, ref, isoval)
        # >>> approx_neutral_surf(ans_type, S, T, P, ref, pin_loc, pin_p)
        # >>> approx_neutral_surf(ans_type, S, T, P, pin_loc, pin_p)
        if isoval is None:  # => pin_loc and pin_p are both not None
            n0 = pin_loc  # evaluate S and T on the surface at the chosen location
            s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], pin_p)

            if ans_type == "sigma":
                if ref is None:
                    ref = pin_p
                elif not isinstance(ref, float):
                    raise TypeError(
                        '"ref" must be None or float when "ans_type" is "sigma"'
                    )
                isoval = eos(s0, t0, ref)
            else:
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
                isoval = eos(s0, t0, pin_p) - eos(
                    ref[0], ref[1], pin_p
                )  # == 0. when ref = (s0,t0)

        # Solve non-linear root finding problem in each cast
        timer_loc = time()
        s, t, p = vertsolve(S, T, P, Sppc, Tppc, n_good, ref, isoval, tol_p)
        timer_update = time() - timer_loc

        # Diagnostics
        if diags:
            d["timer_update"] = timer_update
            d["timer"] = time() - timer
            d["ϵ_RMS"], d["ϵ_MAV"] = ntp_ϵ_errors_norms(
                s,
                t,
                p,
                eos_s_t,
                wrap,
                dist1_iJ,
                dist2_Ij,
                dist2_iJ,
                dist1_Ij,
                areaiJ,
                areaIj,
            )
            if verbose > 0:
                print(
                    f"{ans_type} done"
                    f" | {d['timer']:5.2f} sec"
                    f" | log_10(rms(ϵ)) = {np.log10(d['ϵ_RMS']) : 9.6f}",
                    file=file_id,
                )

    else:  # ans_type == 'omega':

        # --- Get extra arguments
        ML = kwargs.get("ML")
        ITER_MIN = kwargs.get("ITER_MIN", 1)
        ITER_MAX = kwargs.get("ITER_MAX", 10)
        ITER_START_WETTING = kwargs.get("ITER_START_WETTING", 1)
        ITER_STOP_WETTING = kwargs.get("ITER_STOP_WETTING", 5)
        TOL_LRPD_MAV = kwargs.get("TOL_LRPD_MAV", 1e-7)
        TOL_P_CHANGE_RMS = kwargs.get("TOL_P_CHANGE_RMS", 0.0)

        dist2on1_iJ = dist2_iJ / dist1_iJ
        dist1on2_Ij = dist1_Ij / dist2_Ij

        pin_loc_1 = np.ravel_multi_index(pin_loc, (ni, nj))  # linear index

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
                "timer_bfs": np.zeros(ITER_MAX + 1, dtype=np.float64),
                "timer_matbuild": np.zeros(ITER_MAX + 1, dtype=np.float64),
                "timer_matsolve": np.zeros(ITER_MAX + 1, dtype=np.float64),
                "timer_update": np.zeros(ITER_MAX + 1, dtype=np.float64),
            }

        # Calculate an initial surface through `pin` if no initial surface given
        if p_init is None:
            if isinstance(ref, (tuple, list)) and len(ref) == 2:
                ans_type_init = "delta"
                if any(x is None for x in ref):
                    ref = None  # reset (None, None) or similar trigger
            else:
                ans_type_init = "sigma"

            s, t, p, d_ = approx_neutral_surf(
                ans_type_init,
                S,
                T,
                P,
                ref=ref,
                isoval=isoval,
                pin_loc=pin_loc,
                pin_p=pin_p,
                wrap=wrap,
                vert_dim=-1,
                dist1_iJ=dist1_iJ,
                dist2_Ij=dist2_Ij,
                dist2_iJ=dist2_iJ,
                dist1_Ij=dist1_Ij,
                eos=eos,
                eos_s_t=eos_s_t,
                grav=grav,
                rho_c=rho_c,
                tol_p=tol_p,
                interp_fn=interp_fn,
                Sppc=Sppc,
                Tppc=Tppc,
                n_good=n_good,
                diags=diags,
                verbose=verbose,
                file_name=file_name,
            )

            if diags:
                d["ϵ_MAV"][0] = d_["ϵ_MAV"]
                d["ϵ_RMS"][0] = d_["ϵ_RMS"]

        else:
            p = p_init.copy()

            if pin_p is not None and pin_p != p_init[pin_loc]:
                raise ValueError("pin_p does not match p_init at pin_loc")

            # Interpolate S and T onto the surface
            s, t = val2(P, S, Sppc, T, Tppc, p)

            # Diagnostics
            d["ϵ_RMS"][0], d["ϵ_MAV"][0] = ntp_ϵ_errors_norms(
                s,
                t,
                p,
                eos_s_t,
                wrap,
                dist1_iJ,
                dist2_Ij,
                dist2_iJ,
                dist1_Ij,
                areaiJ,
                areaIj,
            )

        if pin_p is None:
            pin_p = p[pin_loc]

        # Pre-calculate things for Breadth First Search:
        # all grid points that are adjacent to all grid points, using 5-connectivity
        A5 = grid_adjacency((ni, nj), 5, wrap)
        # all grid points that are adjacent to all grid points, using 4-connectivity
        A4 = A5[:, 0:-1]

        # Get ML: the pressure of the mixed layer
        # if ITER_MAX > 1 && if isstruct(ML)
        #   # Compute the mixed layer from parameter inputs
        #   ML = mixed_layer(S, T, P, ML)
        # end

        # ensure same nan structure between s, t, and p. Just in case user gives
        # np.full((ni,nj), 1000) for a 1000dbar isobaric surface, for example
        p[np.isnan(s)] = np.nan

        d["timer"][0] = time() - timer
        if verbose > 0:
            print(
                f"{ans_type} initialized "
                f" | {d['timer'][0]:5.2f} sec"
                f" | log_10(rms(ϵ)) = {np.log10(d['ϵ_RMS'][0] ):9.6f}",
                file=file_id,
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
            if iter_ > 1 and ML is not None:
                p[p < ML] = np.nan

            # --- Determine the connected component containing the reference cast, via Breadth First Search
            timer_loc = time()
            if iter_ >= ITER_START_WETTING and iter_ <= ITER_STOP_WETTING:
                qu, qt, n_wet = bfs_conncomp1_wet(
                    s, t, p, S, T, P, Sppc, Tppc, n_good, A4, pin_loc_1, tol_p, eos
                )
            else:
                qu, qt = bfs_conncomp1(np.isfinite(p.flatten()), A4, pin_loc_1)
                n_wet = 0
            timer_bfs = time() - timer_loc
            if qt < 0:
                raise RuntimeError(
                    "The surface is NaN at the reference cast. Probably the initial surface was NaN here."
                )

            # --- Solve global matrix problem for the exactly determined Poisson equation
            timer_loc = time()
            ϕ, timer_matbuild = _omega_matsolve_poisson(
                s, t, p, dist2on1_iJ, dist1on2_Ij, wrap, A5, qu, qt, pin_loc, eos_s_t
            )
            timer_solver = time() - timer_loc - timer_matbuild

            # --- Update the surface
            timer_loc = time()
            p_old = p.copy()  # Record old surface for pinning and diagnostics
            vertsolve(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p)  # mutates s, t, p

            # DEV:  time seems indistinguishable from using factory function as above
            # _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p, eos)

            # Force p to stay constant at the reference column, identically.
            # This avoids any intolerance from the vertical solver.
            p[pin_loc] = pin_p

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
                d["n_wet"][iter_] = n_wet

                d["timer_matbuild"][iter_] = timer_matbuild
                d["timer_matsolve"][iter_] = timer_solver
                d["timer_update"][iter_] = timer_update
                d["timer_bfs"][iter_] = timer_bfs

                # Diagnostics about the state AFTER this iteration
                d["ϵ_RMS"][iter_], d["ϵ_MAV"][iter_] = ntp_ϵ_errors_norms(
                    s,
                    t,
                    p,
                    eos_s_t,
                    wrap,
                    dist1_iJ,
                    dist2_Ij,
                    dist2_iJ,
                    dist1_Ij,
                    areaiJ,
                    areaIj,
                )

                if verbose > 0:
                    print(
                        f"{ans_type} iter {iter_:02d} done"
                        f" | {d['timer'][iter_]:5.2f} sec"
                        f" | log_10(rms(ϵ)) = {np.log10(d['ϵ_RMS'][iter_]):9.6f}"
                        f" | ϕ MAV = {ϕ_MAV:.6e}"
                        f" | {n_wet:4} casts newly wet"
                        f" | Δp RMS = {Δp_RMS:.6e}",
                        file=file_id,
                    )

            # --- Check for convergence
            if (
                ϕ_MAV < TOL_LRPD_MAV or Δp_RMS < TOL_P_CHANGE_RMS
            ) and iter_ >= ITER_MIN:
                break

    if file_name is not None:
        file_id.close()

    if diags and ans_type == "omega":
        # Trim diagnostic output
        for k, v in d.items():
            d[k] = v[0 : iter_ + (k in ("ϵ_MAV", "ϵ_RMS"))]

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


def unpack_STP(S, T, P, vert_dim):
    """Process ocean data to make water columns contiguous in memory.

    Also extracts the ndarray data underlying xarray inputs.

    Parameters
    ----------
    S, T : ndarray or xarray.DataArray

        3D practical / Absolute salinity and potential / Conservative
        temperature

    P : ndarray or xarray.DataArray

        In the non-Boussinesq case, `P` is the 3D pressure, sharing the same
        dimensions as `S` and `T`.

        In the Boussinesq case, `P` is the depth and can be 3D with the same
        structure as `S` and `T`, or can be 1D with as many elements as there
        are in the vertical dimension of `S` and `T`.

    vert_dim : int or str, Default -1

        Specifies which dimension of `S`, `T` (and `P` if 3D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

    Returns
    -------
    S, T, P : ndarray

    vertdim : int

    """

    # DEV:  if inputs are 2D (i.e. for a hydrographic section), expand them here
    # to be 3D?  Would want to modify s,t,p output though too...

    # Extract numpy arrays from xarrays
    if isinstance(S, xr.core.dataarray.DataArray):
        # Assume S, T are all xarrays, with same dimension ordering
        if isinstance(vert_dim, str) and vert_dim in S.dims:
            vert_dim = S.dims.index(vert_dim)
        S = S.values
        T = T.values
    if isinstance(P, xr.core.dataarray.DataArray):
        P = P.values

    if vert_dim not in (-1, S.ndim - 1):
        S = np.moveaxis(S, vert_dim, -1)
        T = np.moveaxis(T, vert_dim, -1)
        if P.ndim == S.ndim:
            P = np.moveaxis(P, vert_dim, -1)
    if P.ndim < S.ndim:
        P = np.broadcast_to(P, S.shape)
    S = np.require(S, dtype=np.float64, requirements="C")
    T = np.require(T, dtype=np.float64, requirements="C")
    P = np.require(P, dtype=np.float64, requirements="C")
    # Assume S and T have the same nan locations for missing
    # profiles or depths below the bottom.

    if not (P.shape == S.shape or P.ndim == 1 and len(P) == S.shape[-1]):
        raise TypeError(
            "P must match dimensions of S, or be 1D matching the last dimension of S;"
            f"found P.shape = {P.shape} but S.shape = {S.shape}"
        )

    vert_dim = -1  # Now vert_dim shows the last dimension of a numpy array

    return S, T, P, vert_dim


@functools.lru_cache(maxsize=10)
def _make_vertsolve(eos, ans_type):

    if ans_type == "omega":

        def f(*args):
            _vertsolve_omega(*args, eos)
            return None

    elif ans_type == "sigma":

        def f(*args):
            return _vertsolve(*args, eos, zero_sigma)

    elif ans_type == "delta":

        def f(*args):
            return _vertsolve(*args, eos, zero_delta)

    else:
        raise NameError(f'Unknown ans_type "{ans_type}"')

    return f


@numba.njit
def _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p, eos):
    # Note!  mutates s, t, p

    for n in np.ndindex(s.shape):
        ϕn = ϕ[n]
        k = n_good[n]
        if k > 1 and np.isfinite(ϕn):

            # Select this water column
            tup = (*n, slice(k))
            Sn = S[tup]
            Tn = T[tup]
            Pn = P[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]
            pn = p[n]

            # Evaluate difference between (a) eos at location on the cast where the
            # pressure or depth is p, and (b) eos at location on the cast where the
            # pressure or depth is pin_p (where the surface currently is) plus the density
            # perturbation d.  Part (b) is precomputed as r0.  Here, eos always
            # evaluated at the pressure or depth of the original position, pin_p; this is
            # to calculate locally referenced potential density with reference pressure
            # pin_p.
            args = (Sn, Tn, Pn, Sppcn, Tppcn, pn, eos(s[n], t[n], pn) + ϕn, eos)

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_sigma, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_sigma, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

            else:
                # Ensure s,t,p all have the same nan structure
                s[n], t[n], p[n] = np.nan, np.nan, np.nan

        else:
            # ϕ is nan, or only one grid cell so cannot interpolate.
            # Ensure s,t,p all have the same nan structure
            s[n], t[n], p[n] = np.nan, np.nan, np.nan

    return None


@numba.njit
def _vertsolve(S, T, P, Sppc, Tppc, n_good, ref, d0, tol_p, eos, zero_func):

    s = np.full(n_good.shape, np.nan)
    t = np.full(n_good.shape, np.nan)
    p = np.full(n_good.shape, np.nan)

    for n in np.ndindex(s.shape):
        k = n_good[n]
        if k > 1:

            # Select this water column
            tup = (*n, slice(k))
            Sn = S[tup]
            Tn = T[tup]
            Pn = P[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]

            args = (Sn, Tn, Pn, Sppcn, Tppcn, ref, d0, eos)

            # Use mid-pressure as initial guess
            pn = (Pn[0] + Pn[-1]) * 0.5

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_func, pn, Pn[0], Pn[-1], args)

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_func, lb, ub, tol_p, args)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

    return s, t, p


@numba.njit
def zero_sigma(p, S, T, P, Sppc, Tppc, ref_p, isoval, eos):
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    return eos(s, t, ref_p) - isoval


@numba.njit
def zero_delta(p, S, T, P, Sppc, Tppc, ref, isoval, eos):
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)
    return eos(s, t, p) - eos(ref[0], ref[1], p) - isoval
