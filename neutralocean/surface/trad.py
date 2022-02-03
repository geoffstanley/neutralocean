"""
Calculate traditional approximately neutral surfaces, namely potential density
surfaces and in-situ density (or specific volume) anomaly surfaces, from
structured ocean data. 
"""

import numpy as np
from time import time

from neutralocean.surface._vertsolve import _make_vertsolve
from neutralocean.ppinterp import ppval1_two, select_ppc
from neutralocean.ntp import ntp_ϵ_errors_norms
from neutralocean.lib import (
    _xr_in,
    _xr_out,
    _process_pin_cast,
    _process_wrap,
    _process_casts,
    _process_n_good,
    _process_eos,
)


def potential_surf(S, T, P, **kwargs):
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

        ``"ref"`` : float

            Reference pressure / depth for surface (matching input `ref` if given,
            otherwise this is calculated internally).

        ``"isoval"`` : float

            Isovalue of potential density for surface (matching input `ref` if
            given, otherwise this is calculated internally).

    Other Parameters
    ----------------
    wrap : tuple of bool, or tuple of str

        Specifies which dimensions are periodic.

        As a tuple of bool, this must be length two.  The first or second
        non-vertical dimension of `S` and `T` is periodic iff ``wrap[0]`` or
        ``wrap[1]`` is True, respectively.

        As a tuple of str, simply name the periodic dimensions of `S` and
        `T`.

        Required if `diags` is True

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


    eos : str or function or tuple of functions, Default 'gsw'

        Specification for the equation of state.

        If a str, can be any of the strings accepted by
        `neutralocean.eos.tools.make_eos`,
        e.g. 'jmd95', 'jmdfwg06', 'gsw'.

        If a function, must take three inputs corresponding to `S`, `T`, and
        `P`, and output the density (or specific volume).  This form is not
        allowed when `diags` is True.  This can be made as, e.g.,
        `eos = neutralocean.eos.make_eos('gsw')`
        for a non-Boussinesq ocean, or as
        `eos = neutralocean.eos.make_eos('gsw', grav, rho_c)`
        for a Boussinesq ocean with `grav` and `rho_c` (see inputs below).

        If a tuple of functions, the first element must be a function for the
        equation of state as above, and the second element must be a function
        taking the same three inputs as above and returning two outputs, namely
        the partial derivatives of the equation of state with respect to `S`
        and `T`.  The second element can be made as, e.g.,
        `eos = neutralocean.eos.make_eos_s_t('gsw', grav, rho_c)`

        The function (or the first element of the tuple of functions) should be
        @numba.njit decorated and need not be vectorized -- it will be called
        many times with scalar inputs.

    grav : float, Default None
        Gravitational acceleration [m s-2].  When non-Boussinesq, pass None.

    rho_c : float, Default None
        Boussinesq reference desnity [kg m-3].  When non-Boussinesq, pass None.

    interp : str, Default 'linear'

        Method for vertical interpolation.  Use 'linear' for linear
        interpolation, and 'pchip' for Piecewise Cubic Hermite Interpolating
        Polynomials.  Other interpolants can be added through the subpackage,
        `ppinterp`.

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
    using ``lib._process_casts`` to pre-process your `S`, `T`, `P` inputs to
    have the vertical dimension last.
    """

    return _traditional_surf("potential", S, T, P, **kwargs)


def anomaly_surf(S, T, P, **kwargs):
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
        Note d["ref"] returns a 2 element tuple, namely `ref` as here.

    Other Parameters
    ----------------
    wrap, vert_dim, dist1_iJ, dist1_Ij, dist2_Ij, dist2_iJ, eos, grav, rho_c,
    interp, n_good, diags, output, TOL_P_SOLVER :
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

    return _traditional_surf("anomaly", S, T, P, **kwargs)


def _traditional_surf(ans_type, S, T, P, **kwargs):
    """Core function to calculate "potential" or "anomaly" surfaces.
    Inputs are as in `potential` and `anomaly`, but first input is a string
    specifying "potential" or "anomaly" """

    ref = kwargs.get("ref")
    isoval = kwargs.get("isoval")
    pin_cast = kwargs.get("pin_cast")
    pin_p = kwargs.get("pin_p")
    vert_dim = kwargs.get("vert_dim", -1)
    TOL_P_SOLVER = kwargs.get("TOL_P_SOLVER", 1e-4)
    eos = kwargs.get("eos", "gsw")
    rho_c = kwargs.get("rho_c")
    grav = kwargs.get("grav")
    wrap = kwargs.get("wrap")
    diags = kwargs.get("diags", True)
    output = kwargs.get("output", True)
    geom = (
        kwargs.get(x, 1.0) for x in ("dist1_iJ", "dist1_Ij", "dist2_Ij", "dist2_iJ")
    )

    n_good = kwargs.get("n_good")
    interp = kwargs.get("interp", "linear")

    # interp_1_twice = make_interpolator(interp, 0, "1", True)
    ppc_fn = select_ppc(interp, "1")

    # Process arguments
    sxr, txr, pxr = (_xr_in(X, vert_dim) for X in (S, T, P))  # before _process_casts
    pin_cast = _process_pin_cast(pin_cast, S)  # call before _process_casts
    wrap = _process_wrap(wrap, sxr, diags)  # call before _process_casts
    S, T, P = _process_casts(S, T, P, vert_dim)
    n_good = _process_n_good(S, n_good)  # call after _process_casts
    eos, eos_s_t = _process_eos(eos, grav, rho_c, need_s_t=diags)

    ni, nj = n_good.shape

    # Error checking on (ref, isoval, pin_cast, pin_p), then convert this
    # selection to (ref, isoval) pair
    _check_ref(ans_type, ref, isoval, pin_cast, pin_p, ni, nj)
    ref, isoval = _choose_ref_isoval(
        ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, ppc_fn
    )

    # Solve non-linear root finding problem in each cast
    vertsolve = _make_vertsolve(eos, ppc_fn, ans_type)
    timer = time()
    s, t, p = vertsolve(S, T, P, n_good, ref, isoval, TOL_P_SOLVER)

    if pin_p is not None:  # pin_cast must also be valid
        # Adjust the surface at the pinning cast slightly, to match the pinning
        # pressure / depth.  This fixes small deviations of order `TOL_P_SOLVER`
        n0 = pin_cast
        p[n0] = pin_p
        Sppcn0 = ppc_fn(P[n0], S[n0])
        Tppcn0 = ppc_fn(P[n0], T[n0])
        s[n0], t[n0] = ppval1_two(pin_p, P[n0], Sppcn0, Tppcn0)

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
                f" | {n_wet:11d} wet casts"
                f" | RMS(ϵ) = {ϵ_RMS:.8e}",
                f" | {d['timer']:.3f} sec",
            )

        d["ref"] = ref
        d["isoval"] = isoval

    s, t, p = (_xr_out(x, xxr) for (x, xxr) in ((s, sxr), (t, txr), (p, pxr)))
    return s, t, p, d


def _check_ref(ans_type, ref, isoval, pin_cast, pin_p, ni, nj):
    """Error checking on ref / isoval / pin_cast / pin_p combinations for "potential"
    and "anomaly" surfaces
    """
    # First check None values to validate one of the following options:
    # >>> _traditional_surf(ans_type, S, T, P, ref, isoval)
    # >>> _traditional_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _traditional_surf(ans_type, S, T, P, pin_cast, pin_p)
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


def _choose_ref_isoval(ans_type, ref, isoval, pin_cast, pin_p, eos, S, T, P, ppc_fn):
    # Handle the three valid calls in the following order of precedence:
    # >>> _traditional_surf(ans_type, S, T, P, ref, isoval)
    # >>> _traditional_surf(ans_type, S, T, P, ref, pin_cast, pin_p)
    # >>> _traditional_surf(ans_type, S, T, P, pin_cast, pin_p)
    if isoval is None:  # => pin_cast and pin_p are both not None
        n0 = pin_cast  # evaluate S and T on the surface at the chosen location
        # s0, t0 = interp_fn(pin_p, P[n0], S[n0], T[n0])

        Sppc = ppc_fn(P[n0], S[n0])
        Tppc = ppc_fn(P[n0], T[n0])
        s0, t0 = ppval1_two(pin_p, P[n0], Sppc, Tppc)

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
