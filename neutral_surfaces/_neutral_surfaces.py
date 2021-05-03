import numpy as np
import numba
import sys
from time import time

import functools

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t
from neutral_surfaces._zero import guess_to_bounds, brent
from neutral_surfaces.interp_ppc import linear_coeffs, val2_0d, val2
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces.lib import ϵ_norms
from neutral_surfaces.omega_surface import omega_matsolve_poisson


# signatures:
# pot_dens_surf(*args, ref_p, isoval)
# pot_dens_surf(*args, ref_p, pin)
# pot_dens_surf(*args, pin)
def pot_dens_surf(
    S,
    T,
    P,
    ref=None,
    isoval=None,
    pin=None,

    axis=-1,
    n_good=None,

    Sppc=None,
    Tppc=None,
    interp_fn=linear_coeffs,

    eos='jmd95',
    grav=None,
    rho_c=None,

    tol_p=1e-4,
):

    S, T, P, Sppc, Tppc, n_good = process_STPppc(
        S, T, P, pin, axis, n_good, Sppc, Tppc, interp_fn
    )

    eos = make_eos(eos, grav, rho_c)

    if isinstance(pin, (tuple, list)) and len(pin) == 3:
        p0 = pin[-1]  # pin pressure
        n0 = pin[:-1]  # index to water column which intersects the surface
        #                   at the pin pressure

        # evaluate S and T on the surface at the chosen location
        s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], p0)

        if ref is None:
            ref = pin[-1]

        # Choose iso-value that will intersect cast n0 at p0.
        isoval = eos(s0, t0, ref)

    elif ref is None or isoval is None:
        raise TypeError(
            'Without a 3 element vector for "pin", must provide'
            ' "ref" and "isoval" (both scalars)'
        )

    # Solve non-linear root finding problem in each cast
    f = make_vertsolve(eos, 'sigma')
    return f(S, T, P, Sppc, Tppc, n_good, ref, isoval, tol_p)


def delta_surf(
    S,
    T,
    P,
    ref=None,
    isoval=None,
    pin=None,

    axis=-1,
    n_good=None,

    Sppc=None,
    Tppc=None,
    interp_fn=linear_coeffs,

    eos='jmd95',
    grav=None,
    rho_c=None,

    tol_p=1e-4,
):

    S, T, P, Sppc, Tppc, n_good = process_STPppc(
        S, T, P, pin, axis, n_good, Sppc, Tppc, interp_fn
    )

    eos = make_eos(eos, grav, rho_c)

    if isinstance(pin, (tuple, list)) and len(pin) == 3:
        p0 = pin[-1]  # pin pressure
        n0 = pin[:-1]  # index to water column which intersects the surface
        #                   at the pin pressure

        # evaluate S and T on the surface at the chosen location
        s0, t0 = val2_0d(P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], p0)

        # Choose iso-value that will intersect cast n0 at p0.
        if ref is None:
            ref = (s0, t0)
            isoval = 0.0
        else:
            isoval = eos(s0, t0, p0) - eos(ref[0], ref[1], p0)

    elif ref is None or isoval is None:
        raise TypeError(
            'Without a 3 element vector for "pin", must provide'
            ' "ref" (2 element vector) and "isoval" (scalar)'
        )

    # Solve non-linear root finding problem in each cast
    f = make_vertsolve(eos, 'delta')
    return f(S, T, P, Sppc, Tppc, n_good, ref, isoval, tol_p)


def omega_surf(
    S,
    T,
    P,
    wrap,
    pin,
    p_init=None,

    # params about S, T, P
    axis=-1,
    n_good=None,

    # params about interpolation
    Sppc=None,
    Tppc=None,
    interp_fn=linear_coeffs,

    # params about equation of state
    eos='jmd95',
    eos_s_t='jmd95',
    grav=None,
    rho_c=None,

    ML=None,  # mixed layer

    # params about grid
    DIST1_iJ=1,
    DIST2_Ij=1,
    DIST2_iJ=1,
    DIST1_Ij=1,

    # params about iteration
    ITER_MIN=1,
    ITER_MAX=10,
    ITER_START_WETTING=1,
    ITER_STOP_WETTING=5,

    # params about tolerances / convergence
    TOL_LRPD_L1=1e-7,
    TOL_P_CHANGE_L2=0.0,
    tol_p=1e-4,

    # params about diagnostics
    DIAGS=True,
    VERBOSE=1,
    FILE_NAME=None,

    # params for pot_dens_surf in case p_init=None
    ref=None,
):

    S, T, P, Sppc, Tppc, n_good = process_STPppc(
        S, T, P, pin, axis, n_good, Sppc, Tppc, interp_fn
    )

    if isinstance(eos, str) and not isinstance(eos_s_t, str) or eos != eos_s_t:
        raise ValueError(
            'eos and eos_s_t, if strings, must be the same string')

    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)

    # # --- Get extra arguments
    # # fmt: off

    # # Below uses soft notation, similar to that in MOM6: i = I-1/2; j = J-1/2
    # DIST1_iJ = kwargs.get('DIST1_iJ', 1)  # Distance [m] in 1st dim centred at (I-1/2, J)
    # DIST2_Ij = kwargs.get('DIST2_Ij', 1)  # Distance [m] in 2nd dim centred at (I, J-1/2)
    # DIST2_iJ = kwargs.get('DIST2_iJ', 1)  # Distance [m] in 1st dim centred at (I-1/2, J)
    # DIST1_Ij = kwargs.get('DIST1_Ij', 1)  # Distance [m] in 2nd dim centred at (I, J-1/2)

    # ML = kwargs.get("ML")  # Mixed layer pressure or depth to remove

    # ITER_MIN = kwargs.get("ITER_MIN", 1)  # min number of iterations
    # ITER_MAX = kwargs.get("ITER_MAX", 10)  # max number of iterations
    # ITER_START_WETTING = kwargs.get("ITER_START_WETTING", 1)  # start wetting on this iteration (first iteration is 1)
    # ITER_STOP_WETTING = kwargs.get("ITER_STOP_WETTING", 5)  # stop wetting after this many iterations (useful to avoid adding then removing some pesky casts)

    # # Exit iterations when the L2 change of pressure (or depth) on the surface
    # # is less than this value. Set to 0 to deactivate. Units are the same as P [dbar or m].
    # TOL_LRPD_L1 = kwargs.get("TOL_LRPD_L1", 1e-7)

    # # Exit iterations when the L1 change of the Locally Referenced Potential
    # # Density perturbation is less than this value [kg m^-3].  Set to 0 to deactivate.
    # TOL_P_CHANGE_L2 = kwargs.get("TOL_P_CHANGE_L2", 0.0)

    # # Error tolerance when root-finding to update surface, in the same units as
    # # P [dbar] or [m].
    # tol_p = kwargs.get("tol_p", 1e-4)

    # VERBOSE = kwargs.get("VERBOSE", 1)  # show a moderate level of information. Requires DIAGS == true
    # DIAGS = kwargs.get("DIAGS", True)  # return diagnostics for each iteration
    # FILE_NAME = kwargs.get("FILE_NAME")  # output textual info to this file
    # # fmt: on

    # --- Process extra args
    ni, nj, nk = S.shape  # Get size of 3D hydrography

    if not isinstance(wrap, (tuple, list)) or len(wrap) != 2:
        raise TypeError("wrap must be a two element (logical) vector")

    ref_cast = pin[0:2]
    I_ref = np.ravel_multi_index(ref_cast, (ni, nj))  # linear index

    if eos(34.5, 3.0, 1000.0) < 1.0:
        # Convert from a density tolerance [kg m^-3] to a specific volume tolerance [m^3 kg^-1]
        TOL_LRPD_L1 = TOL_LRPD_L1 * 1000.0 ** 2

    # Calculate the ratios of distances, and auto expand to [ni,nj] sizes, for eps_norms()
    # DEV:  The following broadcast_to calls are probably not general enough...
    # If DIST2_Ij is a vector of length nj, for instance, this crashes.
    DIST1_iJ = np.broadcast_to(DIST1_iJ, (ni, nj))
    DIST1_Ij = np.broadcast_to(DIST1_Ij, (ni, nj))
    DIST2_Ij = np.broadcast_to(DIST2_Ij, (ni, nj))
    DIST2_iJ = np.broadcast_to(DIST2_iJ, (ni, nj))
    AREA_iJ = DIST1_iJ * DIST2_iJ
    AREA_Ij = DIST1_Ij * DIST2_Ij
    DIST2on1_iJ = DIST2_iJ / DIST1_iJ
    DIST1on2_Ij = DIST1_Ij / DIST2_Ij

    if FILE_NAME is None:
        file_id = sys.stdout
    else:
        file_id = open(FILE_NAME, "w")

    # Calculate an initial surface through `pin` if none given
    if p_init is None:
        s, t, p = pot_dens_surf(
            S, T, P, ref=ref, pin=pin, n_good=n_good, Sppc=Sppc, Tppc=Tppc, eos=eos, tol_p=tol_p
        )

    else:
        p = p_init.copy()

        if len(pin) == 3 and pin[-1] != p_init[ref_cast]:
            raise RuntimeError('pin[-1] does not match p_init at ref_cast')

        # Interpolate S and T onto the surface
        s, t = val2(P, S, Sppc, T, Tppc, p)

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

    # --- Prepare diagnostics
    if DIAGS:
        diags = {
            "ϵ_L1": np.empty(ITER_MAX + 1, dtype=np.float64),
            "ϵ_L2": np.empty(ITER_MAX + 1, dtype=np.float64),
            "ϕ_L1": np.empty(ITER_MAX, dtype=np.float64),
            "Δp_L1": np.empty(ITER_MAX, dtype=np.float64),
            "Δp_L2": np.empty(ITER_MAX, dtype=np.float64),
            "Δp_Linf": np.empty(ITER_MAX, dtype=np.float64),
            "freshly_wet": np.empty(ITER_MAX, dtype=int),
            "clocktime": np.empty(ITER_MAX, dtype=np.float64),
            "timer_solver": np.empty(ITER_MAX, dtype=np.float64),
            "timer_update": np.empty(ITER_MAX, dtype=np.float64),
            "timer_bfs": np.empty(ITER_MAX, dtype=np.float64),
        }

        # Diagnostics about state BEFORE the first iteration
        ϵ_L2, ϵ_L1 = ϵ_norms(
            s, t, p, eos_s_t, wrap, DIST1_iJ, DIST2_Ij, DIST2_iJ, DIST1_Ij, AREA_iJ, AREA_Ij
        )
        # mean_p = np.nanmean(p)
        # mean_eos = np.nanmean(eos(s, t, p))
        diags["ϵ_L1"][0] = ϵ_L1
        diags["ϵ_L2"][0] = ϵ_L2
        # diags["mean_p"][0] = mean_p
        # diags["mean_eos"][0] = mean_eos

        if VERBOSE > 0:
            print(
                f'Initial surface has log_10(|ϵ|_2) = {np.log10(ϵ_L2) : 9.6f} ..................',
                file=file_id,
            )
    else:
        diags = {}

    # --- Begin iterations
    # Note: the surface exists wherever p is non-nan.  The nan structure of s
    # and t is made to match that of p when the vertical solve step is done.
    Δp_L2 = 0.0  # ensure this is defined; needed if OPTS.TOL_P_CHANGE_L2 == 0
    vertsolve_omega = make_vertsolve(eos, 'omega')
    for iter_ in range(ITER_MAX):
        iter_time = time()

        # --- Remove the Mixed Layer
        # But keep it for the first iteration; which may be initialized from a
        # not very neutral surface()
        if iter_ + 1 > 1 and ML != None:
            p[p < ML] = np.nan

        # --- Determine the connected component containing the reference cast; via Breadth First Search
        mytime = time()
        if iter_ + 1 >= ITER_START_WETTING and iter_ + 1 <= ITER_STOP_WETTING:
            qu, qt, freshly_wet = bfs_conncomp1_wet(
                s, t, p, S, T, P, Sppc, Tppc, n_good, A4, I_ref, tol_p, eos
            )
        else:
            qu, qt = bfs_conncomp1(np.isfinite(p.flatten()), A4, I_ref)
            freshly_wet = 0
        timer_bfs = time() - mytime
        if qt < 0:
            raise RuntimeError(
                'The surface is NaN at the reference cast. Probably the initial surface was NaN here.'
            )

        # --- Solve global matrix problem for the exactly determined Poisson equation
        mytime = time()
        # r, c, v, N, rhs, m = omega_matsolve_poisson(s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A5, qu, qt, ref_cast)
        # mat = csc_matrix((v, (r, c)), shape=(N, N) )
        # sol = spsolve(mat, rhs)
        # ϕ = np.full(nij, np.nan, dtype=np.float64)
        # ϕ[m] = sol
        # ϕ = ϕ.reshape(ni, nj)
        ϕ = omega_matsolve_poisson(
            s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A5, qu, qt, ref_cast, eos_s_t
        )
        timer_solver = time() - mytime

        # --- Update the surface
        mytime = time()
        p_old = p.copy()  # Record old surface for pinning & diags
        vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p)

        # DEV:  time seems indistinguishable from using factory function as above
        #_vertsolve_omega(s, t, p, S, T, P, Sppc, Tppc, n_good, ϕ, tol_p, eos())

        # Force p to stay constant at the reference column, identically. This
        # avoids any intolerance from the vertical solver.
        p[ref_cast] = p0

        timer_update = time() - mytime

        # --- Closing Remarks
        ϕ_L1 = np.nanmean(abs(ϕ))  # Actually MAV, not L1 norm!
        if DIAGS or TOL_P_CHANGE_L2 > 0:
            Δp = p - p_old
            Δp_L2 = np.sqrt(np.nanmean(Δp ** 2))  # Actually RMS, not L1 norm!

        # fig, ax = plt.subplots()
        # cs = ax.imshow(Δp, origin='lower')
        # cbar = fig.colorbar(cs, ax=ax)

        if DIAGS:

            diags["clocktime"][iter_] = time() - iter_time

            Δp_L1 = np.nanmean(abs(Δp))
            Δp_Linf = np.nanmax(abs(Δp))

            # Diagnostics about what THIS iteration did
            diags["ϕ_L1"][iter_] = ϕ_L1
            diags["Δp_L1"][iter_] = Δp_L1
            diags["Δp_L2"][iter_] = Δp_L2
            diags["Δp_Linf"][iter_] = Δp_Linf
            diags["freshly_wet"][iter_] = freshly_wet

            diags["timer_solver"][iter_] = timer_solver
            diags["timer_update"][iter_] = timer_update
            diags["timer_bfs"][iter_] = timer_bfs

            # Diagnostics about the state AFTER this iteration
            ϵ_L2, ϵ_L1 = ϵ_norms(
                s, t, p, eos_s_t, wrap, DIST1_iJ, DIST2_Ij, DIST2_iJ, DIST1_Ij, AREA_iJ, AREA_Ij
            )

            # mean_p = np.nanmean(p)
            # mean_eos = np.nanmean(eos(s, t, p))
            diags["ϵ_L1"][iter_ + 1] = ϵ_L1
            diags["ϵ_L2"][iter_ + 1] = ϵ_L2
            # diags["mean_p"][iter_+1]    = mean_p
            # diags["mean_eos"][iter_+1]  = mean_eos

            if VERBOSE > 0:
                print(
                    f'Iter {iter_ + 1 : 2}'
                    f' [{diags["clocktime"][iter_] : 5.2f} sec]'
                    f' log_10(|ϵ|_2) = {np.log10(ϵ_L2) : 9.6f}'
                    f' by |ϕ|_1 = {ϕ_L1 : .6e};'
                    f' {freshly_wet : 4} casts freshly wet;'
                    f' |Δp|_2 = {Δp_L2 : .6e}',
                    file=file_id,
                )

        # --- Check for convergence
        if (ϕ_L1 < TOL_LRPD_L1 or Δp_L2 < TOL_P_CHANGE_L2) and iter_ + 1 >= ITER_MIN:
            break

    if FILE_NAME != None:
        file_id.close()

    if DIAGS:
        # Trim diagnostic output
        for k, v in diags.items():
            diags[k] = v[0: iter_ + 1 + (k in ("ϵ_L1", "ϵ_L2"))]

    return s, t, p, diags


def process_STP(S, T, P, axis):
    # DEV:  if inputs are 2D (i.e. for a hydrographic section), expand them here
    # to be 3D?  Would want to modify s,t,p output though too...

    if axis not in (-1, S.ndim - 1):
        S = np.moveaxis(S, axis, -1)
        T = np.moveaxis(T, axis, -1)
        if P.ndim == S.ndim:
            P = np.moveaxis(P, axis, -1)
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

    return S, T, P


def process_STPppc(
    S,
    T,
    P,
    pin=None,

    axis=-1,  # axis of the vertical dimension
    n_good=None,

    Sppc=None,
    Tppc=None,
    interp_fn=linear_coeffs,
):

    S, T, P = process_STP(S, T, P, axis=axis)

    if n_good is None:
        n_good = find_first_nan(S)

    # Get size of 3D hydrography
    ni, nj, nk = S.shape

    # Compute interpolants for S and T casts (unless already provided)
    if (
        Sppc is None
        or Sppc.shape[0:3] != (ni, nj, nk - 1)
        or Tppc is None
        or Tppc.shape[0:3] != (ni, nj, nk - 1)
    ):
        Sppc = interp_fn(P, S)
        Tppc = interp_fn(P, T)

    # check 'pin' argument of pot_dens_surf, delta_surf, omega_surf
    # DEV: <<< maybe eliminate this?  >>>
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
            raise TypeError(
                'If provided, "pin" must be a 2 or 3 element vector')

    return S, T, P, Sppc, Tppc, n_good


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

    if ans_type == 'omega':
        def f(*args):
            _vertsolve_omega(*args, eos)
            return None
    elif ans_type == 'sigma':
        def f(*args):
            return _vertsolve(*args, eos, zero_sigma)
    elif ans_type == 'delta':
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
            # pressure or depth is p, plus the density perturbation ϕ, and (b) eos at
            # location on the cast where the surface currently resides (at pressure or
            # depth p0).  The combination of d and part (b) is precomputed as ϕ_minus_rho0.
            # Here, eos always evaluated at the pressure or depth of the original position,
            # p0; this is to calculate locally referenced potential density with reference
            # pressure p0.
            args = (Sn, Tn, Pn, Sppcn, Tppcn, pn,
                    eos(s[n], t[n], pn) - ϕn, eos)

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
