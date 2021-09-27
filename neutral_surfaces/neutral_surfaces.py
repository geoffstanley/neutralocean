import numpy as np
import xarray as xr
import numba
import sys
from time import time

import functools

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t
from neutral_surfaces._zero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val2
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces.lib import ϵ_norms
from neutral_surfaces._omega import _omega_matsolve_poisson

# signatures:
# approx_neutral_surf('sigma', *args, pin)
# approx_neutral_surf('sigma', *args, pin, ref_p)
# approx_neutral_surf('sigma', *args, isoval, ref_p)

# approx_neutral_surf('omega', *args, pin)
# approx_neutral_surf('omega', *args, pin, ref_p)


def approx_neutral_surf(ans_type, S, T, P, wrap, vert_dim, **kwargs):
    """
    Parameters
    ----------
    ans_type : str
        'sigma', 'delta', or 'omega'
    S : numpy.ndarray or xarray.DataArray
        3D practical / Absolute salinity.
    T : TYPE
        3D potential / Conservative temperature    
    P : TYPE
        1D or 3D pressure or depth
    wrap : tuple of bool (S is numpy array), or tuple of str (if S is xarray)
        specifying which dimensions are periodic
    vert_dim : int (S is numpy array) or str (S is xarray)
        Specifies which dimension of S, T, P is vertical
    ...and more!
    

    Returns
    -------
    s : numpy.ndarray or xarray.DataArray
        practical / Absolute salinity on surface
    t : numpy.ndarray or xarray.DataArray
        potential / Conservative temperature on surface
    p : numpy.ndarray or xarray.DataArray
        pressure or depth on surface
    d : dict
        diagnostics

    """

    mytime = time()

    if ans_type not in ("sigma", "delta", "omega"):
        raise ValueError("ans_type must be one of ('sigma', 'delta', 'omega')")

    # Get extra arguments
    # fmt: off
    ref = kwargs.get('ref')
    pin = kwargs.get('pin')
    
    # grid distances.  (soft notation: i = I-1/2; j = J-1/2)
    dist1_iJ = kwargs.get('dist1_iJ', 1)  # Distance [m] in 1st dim centred at (I-1/2, J)
    dist2_Ij = kwargs.get('dist2_Ij', 1)  # Distance [m] in 2nd dim centred at (I, J-1/2)
    dist2_iJ = kwargs.get('dist2_iJ', 1)  # Distance [m] in 1st dim centred at (I-1/2, J)
    dist1_Ij = kwargs.get('dist1_Ij', 1)  # Distance [m] in 2nd dim centred at (I, J-1/2)
    
    eos = kwargs.get('eos', 'gsw')
    eos_s_t = kwargs.get('eos_s_t')
    grav = kwargs.get('grav')
    rho_c = kwargs.get('rho_c')
    
    interp_fn = kwargs.get('interp_fn', linear_coeffs)
    n_good = kwargs.get('n_good')
    Sppc = kwargs.get('Sppc')
    Tppc = kwargs.get('Tppc')
    
    file_name = kwargs.get('file_name') # name for file where textual info is output; None for stdout
    verbose = kwargs.get('verbose', 1)  # show a moderate level of information. Requires diags == true
    diags = kwargs.get('diags', True) # return diagnostics for each iteration
    
    # Error tolerance for root-finding to update surface; same units as P [dbar] or [m]
    tol_p = kwargs.get('tol_p', 1e-4) 
    # fmt: on

    # Prepare xarray container for outputs if xarrays given for inputs
    S_is_xr = isinstance(S, xr.core.dataarray.DataArray)
    T_is_xr = isinstance(T, xr.core.dataarray.DataArray)
    P_is_xr = isinstance(P, xr.core.dataarray.DataArray)
    if S_is_xr:
        s_ = xr.full_like(S.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if T_is_xr:
        t_ = xr.full_like(T.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if P_is_xr:
        p_ = xr.full_like(P.isel({vert_dim: 0}).drop_vars(vert_dim), 0)

    # Process 3D hydrography
    S, T, P, vert_dim = unpack_STP(S, T, P, vert_dim)
    ni, nj, nk = S.shape  # Get size of 3D hydrography
    if n_good is None:
        n_good = find_first_nan(S)

    # Compute interpolants for S and T casts (unless already provided)
    if (
        Sppc is None
        or Sppc.shape[0:3] != (ni, nj, nk - 1)
        or Tppc is None
        or Tppc.shape[0:3] != (ni, nj, nk - 1)
    ):
        Sppc = interp_fn(P, S)
        Tppc = interp_fn(P, T)

    # Process equation of state function
    if eos_s_t is None and isinstance(eos, str):
        eos_s_t = eos
    elif isinstance(eos, str) and isinstance(eos_s_t, str) and eos != eos_s_t:
        raise ValueError("eos and eos_s_t, if strings, must be the same string")
    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)
    vertsolve = make_vertsolve(eos, ans_type)

    # Process wrap
    if isinstance(wrap, str):
        wrap = (wrap,)  # Convert single string to tuple
    if not isinstance(wrap, (tuple, list)):
        raise TypeError("wrap must be a tuple or list")
    if S_is_xr and all(
        isinstance(x, str) for x in wrap
    ):  # Convert dim names to Tuple of Bool
        wrap = tuple(x in wrap for x in s_.dims)
    if not isinstance(wrap, (tuple, list)) or len(wrap) != 2:
        raise TypeError(
            "wrap must be a two element (logical) array"
            " or an array of strings referring to dimensions in xarray S"
        )

    # Error checking on pin
    if isinstance(pin, (tuple, list)):
        if len(pin) in (2, 3):
            if not all(isinstance(x, int) for x in pin[0:2]):
                raise TypeError('First 2 elements of "pin" must be integers')
            if pin[0] < 0 or pin[1] < 0 or pin[0] >= ni or pin[1] >= nj:
                raise ValueError(
                    '"pin" must index a cast within the domain;'
                    f'found "pin" = {pin} outside the bounds (0,{ni-1}) x (0,{nj-1})'
                )
        else:
            raise TypeError('If provided, "pin" must be a 2 or 3 element vector')

    # Calculate the ratios of distances, and auto expand to [ni,nj] sizes, for eps_norms()
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

        # Get extra arguments
        isoval = kwargs.get("isoval", None)

        if isinstance(pin, (tuple, list)) and len(pin) == 3:
            p0 = pin[-1]  # pin pressure
            n0 = pin[:-1]  # index to water column which intersects the surface
            #                at the pin pressure

            # evaluate S and T on the surface at the chosen location
            s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], p0)

            # Choose reference value(s) and isovalue that will intersect cast n0 at p0
            if ans_type == "sigma":
                if ref is None:
                    ref = pin[-1]
                isoval = eos(s0, t0, ref)

            else:  # ans_type == 'delta'
                if ref is None:
                    ref = (s0, t0)
                    isoval = 0.0
                else:
                    isoval = eos(s0, t0, p0) - eos(ref[0], ref[1], p0)

        elif ref is None or isoval is None:
            if ans_type == "sigma":
                raise TypeError(
                    'Without a 3 element vector for "pin", must provide'
                    ' "ref" (scalar) and "isoval" (scalar)'
                )
            else:  # ans_type == 'delta'
                raise TypeError(
                    'Without a 3 element vector for "pin", must provide'
                    ' "ref" (2 element vector) and "isoval" (scalar)'
                )

        # Solve non-linear root finding problem in each cast
        mytime = time()
        s, t, p = vertsolve(S, T, P, Sppc, Tppc, n_good, ref, isoval, tol_p)
        d["timer"] = time() - mytime

        # Diagnostics
        d["ϵ_L2"], d["ϵ_L1"] = ϵ_norms(
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
                f" | log_10(|ϵ|_2) = {np.log10(d['ϵ_L2']) : 9.6f}",
                file=file_id,
            )

    else:  # ans_type == 'omega':

        # --- Get extra arguments
        # fmt: off
        p_init = kwargs.get("p_init", None)
        
        ML = kwargs.get("ML")  # Mixed layer pressure or depth to remove
    
        ITER_MIN = kwargs.get("ITER_MIN", 1)  # min number of iterations
        ITER_MAX = kwargs.get("ITER_MAX", 10)  # max number of iterations
        ITER_START_WETTING = kwargs.get("ITER_START_WETTING", 1)  # start wetting on this iteration (first iteration is 1)
        ITER_STOP_WETTING = kwargs.get("ITER_STOP_WETTING", 5)  # stop wetting after this many iterations (useful to avoid adding then removing some pesky casts)
    
        # Exit iterations when the L2 change of pressure (or depth) on the surface
        # is less than this value. Set to 0 to deactivate. Units are the same as P [dbar or m].
        TOL_LRPD_L1 = kwargs.get("TOL_LRPD_L1", 1e-7)
    
        # Exit iterations when the L1 change of the Locally Referenced Potential
        # Density perturbation is less than this value [kg m^-3].  Set to 0 to deactivate.
        TOL_P_CHANGE_L2 = kwargs.get("TOL_P_CHANGE_L2", 0.0)
        # fmt: on

        dist2on1_iJ = dist2_iJ / dist1_iJ
        dist1on2_Ij = dist1_Ij / dist2_Ij

        ref_cast = pin[0:2]
        I_ref = np.ravel_multi_index(ref_cast, (ni, nj))  # linear index

        if eos(34.5, 3.0, 1000.0) < 1.0:
            # Convert from a density tolerance [kg m^-3] to a specific volume tolerance [m^3 kg^-1]
            TOL_LRPD_L1 = TOL_LRPD_L1 * 1000.0 ** 2

        # Pre-allocate arrays for diagnostics
        d = {
            "ϵ_L1": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "ϵ_L2": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "ϕ_L1": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_L1": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_L2": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "Δp_Linf": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "freshly_wet": np.zeros(ITER_MAX + 1, dtype=int),
            "timer_bfs": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_matbuild": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_matsolve": np.zeros(ITER_MAX + 1, dtype=np.float64),
            "timer_update": np.zeros(ITER_MAX + 1, dtype=np.float64),
        }

        # Calculate an initial surface through `pin` if none given
        if p_init is None:
            s, t, p, d_ = approx_neutral_surf(
                "sigma",
                S,
                T,
                P,
                wrap,
                vert_dim,
                ref=ref,
                pin=pin,
                n_good=n_good,
                Sppc=Sppc,
                Tppc=Tppc,
                eos=eos,
                eos_s_t=eos_s_t,
                dist1_iJ=dist1_iJ,
                dist2_Ij=dist2_Ij,
                dist2_iJ=dist2_iJ,
                dist1_Ij=dist1_Ij,
                tol_p=tol_p,
            )

            d["ϵ_L1"][0] = d_["ϵ_L1"]
            d["ϵ_L2"][0] = d_["ϵ_L2"]

        else:
            p = p_init.copy()

            if len(pin) == 3 and pin[-1] != p_init[ref_cast]:
                raise RuntimeError("pin[-1] does not match p_init at ref_cast")

            # Interpolate S and T onto the surface
            s, t = val2(P, S, Sppc, T, Tppc, p)

            # Diagnostics
            d["ϵ_L2"][0], d["ϵ_L1"][0] = ϵ_norms(
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

        if len(pin) == 3:
            p0 = pin[-1]
        else:
            p0 = p[ref_cast]

        # Pre-calculate things for Breadth First Search:
        # all grid points that are adjacent to all grid points, using 5-connectivity
        A5 = grid_adjacency((ni, nj), 5, wrap)
        # all grid points that are adjacent to all grid points, using 4-connectivity
        A4 = A5[:, 0:-1]

        # Get ML: the pressure of the mixed layer
        # if ITER_MAX > 1 && if isstruct(OPTS.ML)
        #   # Compute the mixed layer from parameter inputs
        #   ML = mixed_layer(S, T, P, ML)
        # end

        # ensure same nan structure between s, t, and p. Just in case user gives
        # np.full((ni,nj), 1000) for a 1000dbar isobaric surface, for example
        p[np.isnan(s)] = np.nan

        d["timer"][0] = time() - mytime
        if diags and verbose > 0:
            print(
                f"{ans_type} initialized "
                f" | {d['timer'][0]:5.2f} sec"
                f" | log_10(|ϵ|_2) = {np.log10(d['ϵ_L2'][0] ):9.6f}",
                file=file_id,
            )

        # --- Begin iterations
        # Note: the surface exists wherever p is non-nan.  The nan structure of s
        # and t is made to match that of p when the vertical solve step is done.
        Δp_L2 = 0.0  # ensure this is defined; needed if OPTS.TOL_P_CHANGE_L2 == 0
        for iter_ in range(1, ITER_MAX + 1):
            iter_time = time()

            # --- Remove the Mixed Layer
            # But keep it for the first iteration; which may be initialized from a
            # not very neutral surface
            if iter_ > 1 and ML != None:
                p[p < ML] = np.nan

            # --- Determine the connected component containing the reference cast, via Breadth First Search
            mytime = time()
            if iter_ >= ITER_START_WETTING and iter_ <= ITER_STOP_WETTING:
                qu, qt, freshly_wet = bfs_conncomp1_wet(
                    s, t, p, S, T, P, Sppc, Tppc, n_good, A4, I_ref, tol_p, eos
                )
            else:
                qu, qt = bfs_conncomp1(np.isfinite(p.flatten()), A4, I_ref)
                freshly_wet = 0
            timer_bfs = time() - mytime
            if qt < 0:
                raise RuntimeError(
                    "The surface is NaN at the reference cast. Probably the initial surface was NaN here."
                )

            # --- Solve global matrix problem for the exactly determined Poisson equation
            mytime = time()
            ϕ, timer_matbuild = _omega_matsolve_poisson(
                s, t, p, dist2on1_iJ, dist1on2_Ij, wrap, A5, qu, qt, ref_cast, eos_s_t
            )
            timer_solver = time() - mytime - timer_matbuild

            # --- Update the surface
            mytime = time()
            p_old = p.copy()  # Record old surface for pinning & d
            vertsolve(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p)  # mutates s, t, p

            # DEV:  time seems indistinguishable from using factory function as above
            # _vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p, eos)

            # Force p to stay constant at the reference column, identically. This
            # avoids any intolerance from the vertical solver.
            p[ref_cast] = p0

            timer_update = time() - mytime

            # --- Closing Remarks
            ϕ_L1 = np.nanmean(abs(ϕ))  # Actually MAV, not L1 norm!
            if diags or TOL_P_CHANGE_L2 > 0:
                Δp = p - p_old
                Δp_L2 = np.sqrt(np.nanmean(Δp ** 2))  # Actually RMS, not L1 norm!

            if diags:

                d["timer"][iter_] = time() - iter_time

                Δp_L1 = np.nanmean(abs(Δp))
                Δp_Linf = np.nanmax(abs(Δp))

                # Diagnostics about what THIS iteration did
                d["ϕ_L1"][iter_] = ϕ_L1
                d["Δp_L1"][iter_] = Δp_L1
                d["Δp_L2"][iter_] = Δp_L2
                d["Δp_Linf"][iter_] = Δp_Linf
                d["freshly_wet"][iter_] = freshly_wet

                d["timer_matbuild"][iter_] = timer_matbuild
                d["timer_matsolve"][iter_] = timer_solver
                d["timer_update"][iter_] = timer_update
                d["timer_bfs"][iter_] = timer_bfs

                # Diagnostics about the state AFTER this iteration
                ϵ_L2, ϵ_L1 = ϵ_norms(
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

                # mean_p = np.nanmean(p)
                # mean_eos = np.nanmean(eos(s, t, p))
                d["ϵ_L1"][iter_] = ϵ_L1
                d["ϵ_L2"][iter_] = ϵ_L2
                # d['mean_p'][iter_+1]    = mean_p
                # d['mean_eos'][iter_+1]  = mean_eos

                if verbose > 0:
                    print(
                        f"{ans_type} iter {iter_:02d} done"
                        f" | {d['timer'][iter_]:5.2f} sec"
                        f" | log_10(|ϵ|_2) = {np.log10(ϵ_L2):9.6f}"
                        f" | |ϕ|_1 = {ϕ_L1:.6e}"
                        f" | {freshly_wet:4} casts freshly wet"
                        f" | |Δp|_2 = {Δp_L2:.6e}",
                        file=file_id,
                    )

            # --- Check for convergence
            if (ϕ_L1 < TOL_LRPD_L1 or Δp_L2 < TOL_P_CHANGE_L2) and iter_ >= ITER_MIN:
                break

    if file_name != None:
        file_id.close()

    if diags and ans_type == "omega":
        # Trim diagnostic output
        for k, v in d.items():
            d[k] = v[0 : iter_ + (k in ("ϵ_L1", "ϵ_L2"))]

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
    # DEV:  if inputs are 2D (i.e. for a hydrographic section), expand them here
    # to be 3D?  Would want to modify s,t,p output though too...

    # Extract numpy arrays from xarrays
    if isinstance(S, xr.core.dataarray.DataArray):
        # Assume S, T are all xarrays, with same dimension ordering
        if vert_dim in S.dims:
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


@numba.njit
def find_first_nan(a):
    """
    find_first_nan(a)

    Find the index to the first nan in a along the last axis.
    If no nan's are present, the length of the last dimension is returned.
    """
    nk = a.shape[-1]
    k = np.full(a.shape[:-1], nk, dtype=np.int64)
    for n in np.ndindex(a.shape[0:-1]):
        for i in range(nk):
            if np.isnan(a[n][i]):
                k[n] = i
                break
    return k


@functools.lru_cache(maxsize=10)
def make_vertsolve(eos, ans_type):

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
            # pressure or depth is p0 (where the surface currently is) plus the density
            # perturbation d.  Part (b) is precomputed as r0.  Here, eos always
            # evaluated at the pressure or depth of the original position, p0; this is
            # to calculate locally referenced potential density with reference pressure
            # p0.
            args = (Sn, Tn, Pn, Sppcn, Tppcn, pn, eos(s[n], t[n], pn) + ϕn, eos)

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(zero_sigma, args, pn, Pn[0], Pn[-1])

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_sigma, args, lb, ub, tol_p)

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
            lb, ub = guess_to_bounds(zero_func, args, pn, Pn[0], Pn[-1])

            if np.isfinite(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(zero_func, args, lb, ub, tol_p)

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
