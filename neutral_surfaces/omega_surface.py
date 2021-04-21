import numpy as np
import numba
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from time import time

from neutral_surfaces._neutral_surfaces import process_arrays
from neutral_surfaces._densjmd95 import rho_bsq, rho_s_t_bsq
from neutral_surfaces.interp_ppc import linear_coefficients, val2_0d, val2
from neutral_surfaces.lib import ϵ_norms
from neutral_surfaces.bfs import bfs_conncomp1, bfs_conncomp1_wet, grid_adjacency
from neutral_surfaces._zero import guess_to_bounds, brent

import matplotlib.pyplot as plt


def omega_surf(
    S,
    T,
    P,
    p,
    ref_cast,
    wrap,
    DIST1_iJ=1,  # Distance [m] in 1st dimension centred at (I-1/2, J)
    DIST2_Ij=1,  # Distance [m] in 2nd dimension centred at (I, J-1/2)
    DIST2_iJ=1,  # Distance [m] in 2nd dimension centred at (I-1/2, J)
    DIST1_Ij=1,  # Distance [m] in 1st dimension centred at (I, J-1/2)
    ML=np.zeros((0, 0)),  # Do not remove the Mixed Layer
    FIGS_SHOW=False,  # do not show figures
    # INTERPFN = ppc_linterp, # Use linear interpolation in the vertical dimension.
    Sppc=np.zeros(
        (0, 0, 0, 0)
    ),  # Pre-computed interpolation coefficients.  None given here.
    Tppc=np.zeros(
        (0, 0, 0, 0)
    ),  # Pre-computed interpolation coefficients.  None given here.
    ITER_MIN=1,  # minimum number of iterations
    ITER_MAX=10,  # maximum number of iterations
    ITER_START_WETTING=1,  # start wetting immediately
    ITER_STOP_WETTING=5,  # stop wetting after this many iterations (to avoid flip-flopping on adding then removing some nuisance casts)
    # Exit iterations when the L2 change of pressure (or depth) on the surface
    # is less than this value. Set to 0 to deactivate. Units are the same as P [dbar or m].
    TOL_LRPD_L1=1e-7,
    # Exit iterations when the L1 change of the Locally Referenced Potential
    # Density perturbation is less than this value [kg m^-3].  Set to 0 to deactivate.
    TOL_P_CHANGE_L2=0.0,
    # Error tolerance when root-finding to update surface, in the same units as
    # P [dbar] or [m].
    TOL_P_UPDATE=1e-4,
    VERBOSE=1,  # show a moderate level of information. Requires DIAGS == true
    DIAGS=True,  # return diagnostics for each iteration
    # FILE_ID = 1, # standard output to MATLAB terminal
    # eos::Function = densjmd95,
    # eos_s_t::Function = densjmd95_s_t
    # grav::Float64 = NaN,
    # ρ_c::Float64 = NaN
    axis=-1,  # axis of the vertical dimension
):

    # Above uses soft notation, similar to that in MOM6: i = I - 1/2; j = J - 1/2

    # Doco from MATLAB.  Needs updating.

    # OMEGA_SURFACE  Create an omega surface, minimizing error from the neutral tangent plane.
    #
    #
    # [p, s, t] = omega_surface(S, T, P, p, ref_cast, OPTS)
    # returns the pressure [or depth] [p], practical / Absolute salinity s, &
    # potential / Conservative temperature t on an omega surface, initialized
    # from an approximately neutral surface of [input] pressure [or depth] p
    # in an ocean whose practical / Absolute salinity & potential /
    # Conservative temperature are S & T located at datasites where the
    # pressure [or depth] is P.  The pressure of the omega surface is pinned
    # unchanging through the iterations, at the reference cast indexed by
    # ref_cast.  An omega surface attempts to minimize the L2 norm of the
    # neutrality error. The density | specific volume [either may be used] &
    # its partial derivatives with respect to S and T are given by the
    # functions eos.m & eos_s_t.m in MATLAB's path. Algorithmic parameters
    # are provided in OPTS [see "Options" below for further details].  For
    # units, see "Equation of State" below.
    #
    #
    # --- Input:
    #  S [nk, ni, nj]: practical / Absolute Salinity
    #  T [nk, ni, nj]: potential / Conservative Temperature
    #  P [nk, ni, nj] | [nk, 1]: pressure [or depth]
    #  p     [ni, nj]: pressure [or depth] on initial surface()
    #  ref_cast [1, 1] or [2, 1] : linear index | 2D index to the reference cast
    #  OPTS [struct]: options [see "Options" below]
    #
    #
    # --- Output:
    #  p [ni, nj]: pressure [or depth] on omega surface()
    #  s [ni, nj]: practical / Absolute salinity on omega surface()
    #  t [ni, nj]: potential / Conservative temperature on omega surface()
    #  diags [struct]: diagnostics such as clock time & norms of neutrality
    #                  errors.  See code for info. Programmable as needed.
    #
    # Note: physical units of S, T, P, & p are determined by eos.m.
    #
    #
    # --- Equation of State:
    # The MATLAB path* must contain two functions, eos.m & eos_s_t.m. Both
    # accept 3 inputs: S, T, & P. eos[S, T, P] returns the specific volume
    # [m^3 kg^-1] | the in-situ density [kg m^-3]. eos_s_t[S, T, P] returns
    # as its two outputs, the partial derivatives of eos with respect to S &
    # T.
    # *Note: It is not sufficient to simply have these eos functions in the
    # current working directory, because the compiled MEX functions will not be
    # able to find them there.  They must be in the MATLAB path.  If they are
    # in the current working directory, use `addpath(pwd)` to add the current
    # working directory to the top of MATLAB's path.
    #
    # For a non-Boussinesq ocean, p & P are pressure [dbar].
    #
    # For a Boussinesq ocean, p & P are actually depth [m].  It is essential
    # that these, like pressure, are positive & increasing down.
    #
    # Various equation of state functions are found in ../lib/eos/.  Simply
    # copy the desired functions to another location in the MATLAB path (such
    # as this directory) and rename them eos.m & eos_s_t.m.  Note, the
    # Boussinesq equation of state is often [but not always] just the regular
    # equation of state but using a hydrostatic pressure [10^-4 * grav * rho_c
    # * z] where grav [m s^-2] is the gravitational acceleration, rho_c [kg
    # m^-3] is the Boussinesq reference density, & z [m, positive] is the
    # depth. In such a case, simply make new eos.m & eos_p.m functions that
    # accept depth as the third input by modifying the original functions that
    # take pressure, this involves hard-coding the gravitational acceleration
    # & Boussinesq reference density into the function.  An example of a
    # Boussinesq eos.m & eos_s_t.m are given for the densjmd95 equation of
    # state, in ../lib/eos/eoscg_densjmd95_bsq.m &
    # ../lib/eos/eoscg_densjmd95_bsq_s_t.m.  Finally, note that eos.m and
    # eos_s_t.m must be compatible with MATLAB's code generation [codegen]
    # which may entail eliminating input checks and/or expansion of input()
    # variables [MATLAB's automatic expansion now handles this].
    #
    #
    # --- Options:
    # OPTS is a struct containing the following fields.
    #   FILE_ID [1, 1]: 1 to write any output to MATLAB terminal, | a file
    #       identifier as returned by fopen() to write to a file. Default: 1.
    #   FIGS_SHOW [scalar]: true to show figures of specific volume adjustment
    #       during computation. Default: false.
    #   FINAL_ROW_VALUES [scalar]: value with which to fill the final row of
    #       the sparse matrix for the purpose of selecting a unique solution
    #       for th density perturbation. This value doesn't matter in theory
    #       but in practice, excessively large | small values may degrade the
    #       numerical solution.  Values in the range of 1e-4 to 1 were tested
    #       on 1x1deg OCCA data, & all work well. Default: 1e-2.
    #   ITER_MAX [1, 1]: maximum number of iterations. Default: 10
    #   ITER_START_WETTING [scalar]: Start wetting on iterations that are
    #       >= ITER_START_WETTING. To disable wetting, set to +inf. Default: 1.
    #   ITER_STOP_WETTING [scalar]: Do wetting for iterations that are
    #       <= ITER_STOP_WETTING. To disable wetting, set to 0. Default: 5.
    #   INTERPFN [function handle]: vertical interpolation function, used to
    #       evaluate Sppc & Tppc if those are not provided.  Default:
    #       INTERPFN = @ppc_linterp.
    #   ML []: do not remove the mixed layer [default]
    #   ML [struct]: calculate the mixed layer using these parameters in mixed_layer[].
    #   ML [ni, nj]: use a pre-computed mixed layer pressure [dbar] | depth [m]
    #   Sppc [O, nk-1, ni, nj]: Coefficients for piecewise polynomials, whose
    #       knots are at P, that interpolate S as a function of P in each water
    #       column.  E.g. Sppc = ppc_linterp[P, S]
    #   Tppc [O, nk-1, ni, nj]: Coefficients for piecewise polynomials, whose
    #       knots are at P, that interpolate T as a function of P in each water
    #       column.  E.g. Tppc = ppc_linterp[P, T]
    #   TOL_LRPD_L1 [scalar]: Error tolerance in Locally Referenced Potential Density [kg m^-3].
    #       Iterations stop when the L1 norm of the LRPD change of the surface()
    #       is below this value. Even if eos gives specific volume, specify
    #       this with units of density, it will be converted. Set to 0 to
    #       ignore this stopping criterion.
    #       Default: 10^-7 kg m^-3 (chosen to give an uncertainty in pressure
    #       of roughly +/- 0.01 dbar.)
    #   TOL_P_CHANGE_L2 [scalar]: Error tolerance in change of pressure [dbar].
    #       Iterations stop when the L2 norm of the change in pressure on the
    #       surface is below this value.  Set to 0 to ignore this stopping
    #       criterion.
    #       Default: inf
    #   TOL_LSQR_REL [scalar]: Relative tolerance for LSQR. Default: 10^-6.
    #   VERBOSE [scalar]: 0 for no output, 1 for summary of each iteration
    #                     2 for detailed information on each iteration.
    #                     Default: 1.
    #   wrap [2 element array]: logical array.  wrap[i] is true iff the domain
    #       is periodic in the i'th lateral dimension.
    #
    #
    # --- References:
    # Klocker, McDougall, Jackett 2009: A new method of forming approximately
    #  neutral surfaces, Ocean Science, 5, 155-172.
    #
    # Stanley, McDougall, Barker 2021: Algorithmic improvements to finding
    #  approximately neutral surfaces, Journal of Advances in Earth System
    #  Modelling.

    # Author[s] : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

    # Acknowledgements: Adapted from "analyze_surface" by Andreas Klocker, &
    #                subsequently modified by Paul Barker and Trevor McDougall.

    # --- Notes on the code:
    # Upper case letters, e.g. S, denote 3D scalar fields [nk,ni,nj]
    # Lower case letters, e.g. s, denote 2D scalar fields    [ni,nj]
    # Developmental things are marked with a comment "DEV"

    ## Simple checks & preparations:

    S, T, P, n_good = process_arrays(S, T, P, axis=axis)

    p = p.copy()

    # Get size of 3D hydrography
    ni, nj, nk = S.shape
    nij = ni * nj

    qu = np.empty(nij, dtype=int)

    Δp_L2 = 0.0  # ensure this is defined; needed if OPTS.TOL_P_CHANGE_L2 == 0
    # Process OPTS
    assert len(wrap) == 2, "wrap must be a two element logical array"
    assert (
        ref_cast[0] >= 0 and ref_cast[1] >= 0 and ref_cast[0] < ni and ref_cast[1] < nj
    ), "ref_cast must index a cast within the domain."
    assert (P.ndim == 1 and len(P) == nk) or (
        P.ndim == 3 and P.shape == S.shape
    ), "P must match dimensions of S, or be 1D matching the first dimension of S"
    I_ref = np.ravel_multi_index(ref_cast, (ni, nj))  # Convert into linear index

    # Pre-calculate things for Breadth First Search:
    # all grid points that are adjacent to all grid points, using 5-connectivity
    A5 = grid_adjacency((ni, nj), 5, wrap)
    # all grid points that are adjacent to all grid points, using 4-connectivity
    A4 = A5[:, 0:-1]

    if rho_bsq(34.5, 3.0, 1000.0) < 1.0:
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

    ## Get ML: the pressure of the mixed layer
    # if ITER_MAX > 1 && if isstruct(OPTS.ML)
    #   # Compute the mixed layer from parameter inputs
    #   ML = mixed_layer(S, T, P, ML)
    # end

    # Compute interpolants for S and T casts (unless already provided)
    if Sppc.shape != (ni, nj, nk - 1) or Tppc.shape != (ni, nj, nk):
        Sppc = linear_coefficients(P, S)
        Tppc = linear_coefficients(P, T)

    # Interpolate S and T onto the surface
    s, t = val2(P, S, Sppc, T, Tppc, p)

    # ensure same nan structure between s, t, and p. Just in case user gives, e.g., repeat(1000,ni,nj) for a 1000dbar isobaric surface
    p[np.isnan(s)] = np.nan

    ## Prepare diagnostics
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

        # Diagnostics about state BEFORE this (first) iteration
        ϵ_L2, ϵ_L1 = ϵ_norms(
            s, t, p, wrap, DIST1_iJ, DIST2_Ij, DIST2_iJ, DIST1_Ij, AREA_iJ, AREA_Ij
        )
        # mean_p = np.nanmean(p)
        # mean_eos = np.nanmean(eos(s, t, p))
        diags["ϵ_L1"][0] = ϵ_L1
        diags["ϵ_L2"][0] = ϵ_L2
        # diags["mean_p"][0] = mean_p
        # diags["mean_eos"][0] = mean_eos

        if VERBOSE > 0:
            print(
                "Initial surface has log_10(|ϵ|_2) = %9.6f .................."
                % np.log10(ϵ_L2)
            )

    ## Begin iterations
    # Note: the surface exists wherever p is non-nan.  The nan structure of s
    # and t is made to match that of p when the vertical solve step is done.
    for iter_ in range(ITER_MAX):
        iter_time = time()

        # --- Remove the Mixed Layer
        # But keep it for the first iteration; which may be initialized from a
        # not very neutral surface()
        if iter_ > 0 and ML.size == nij:
            p[p < ML] = np.nan

        # --- Determine the connected component containing the reference cast; via Breadth First Search
        mytime = time()
        if iter_ + 1 >= ITER_START_WETTING and iter_ + 1 <= ITER_STOP_WETTING:
            qu, qt, freshly_wet = bfs_conncomp1_wet(
                s, t, p, P, S, Sppc, T, Tppc, TOL_P_UPDATE, A4, n_good, I_ref
            )
        else:
            qu, qt = bfs_conncomp1(np.isfinite(p.flatten()), A4, I_ref)
            freshly_wet = 0
        timer_bfs = time() - mytime
        assert qt >= 0, "Error: surface is NaN at the reference cast"

        # --- Solve global matrix problem for the exactly determined Poisson equation
        mytime = time()
        # r, c, v, N, rhs, m = omega_matsolve_poisson(s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A5, qu, qt, ref_cast)
        # mat = csc_matrix((v, (r, c)), shape=(N, N) )
        # sol = spsolve(mat, rhs)
        # ϕ = np.full(nij, np.nan, dtype=np.float64)
        # ϕ[m] = sol
        # ϕ = ϕ.reshape(ni, nj)
        ϕ = omega_matsolve_poisson(
            s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A5, qu, qt, ref_cast
        )
        timer_solver = time() - mytime

        # fig, ax = plt.subplots()
        # cs = ax.imshow(ϕ, origin='lower')
        # cbar = fig.colorbar(cs, ax=ax)

        # --- Update the surface()
        mytime = time()
        p_old = p.copy()  # Record old surface for pinning or diagnostic purposes.
        omega_vertsolve(s, t, p, P, S, Sppc, T, Tppc, n_good, TOL_P_UPDATE, ϕ)

        # Force p to stay constant at the reference column, identically. This
        # avoids any intolerance from the vertical solver.
        p[ref_cast] = p_old[ref_cast]

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
                s, t, p, wrap, DIST1_iJ, DIST2_Ij, DIST2_iJ, DIST1_Ij, AREA_iJ, AREA_Ij
            )

            # mean_p = np.nanmean(p)
            # mean_eos = np.nanmean(eos(s, t, p))
            diags["ϵ_L1"][iter_ + 1] = ϵ_L1
            diags["ϵ_L2"][iter_ + 1] = ϵ_L2
            # diags["mean_p"][iter_+1]    = mean_p
            # diags["mean_eos"][iter_+1]  = mean_eos

            if VERBOSE > 0:
                print(
                    "Iter %2d [%6.2f sec] log_10(|ϵ|_2) = %9.6f by |ϕ|_1 = %.6e; %4d casts freshly wet; |Δp|_2 = %.6e"
                    % (
                        iter_ + 1,
                        diags["clocktime"][iter_],
                        np.log10(ϵ_L2),
                        ϕ_L1,
                        freshly_wet,
                        Δp_L2,
                    )
                )

        # --- Check for convergence
        if (ϕ_L1 < TOL_LRPD_L1 or Δp_L2 < TOL_P_CHANGE_L2) and iter_ + 1 >= ITER_MIN:
            break

    if DIAGS:
        # Trim output
        for key, val in diags.items():
            diags[key] = val[0 : iter_ - 1 + (key in ["ϵ_L1", "ϵ_L2"])]
        return p, s, t, diags
    else:
        return p, s, t


def omega_matsolve_poisson(s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A5, qu, qt, mr):
    # Doco from MATLAB, needs updating.

    # OMEGA_MATSOLVE_POISSON  Build & solve the sparse matrix Poisson problem for omega surfaces
    #
    #
    # ϕ = omega_matsolve_poisson(s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A4, qu, qt, m_ref)
    # builds & solves the sparse matrix problem for omega surfaces in Poisson
    # form.
    #
    #
    # --- Input:
    #  s [ni, nj]: practical / Absolute Salinity on the surface()
    #  t [ni, nj]: potential / Conservative Temperature on the surface()
    #  p [ni, nj]: pressure [or depth] on the surface()
    # DIST2on1_iJ [ni, nj]: the area of a grid cell centred at [I-1/2, J]
    #   divided by the distance, squared, from [I-1,J] to [I,J].  Equivalently
    #   this is the grid distance in second dimension divided by grid distance
    #   in first dimension, both centred at [I-1/2, J].
    # DIST1on2_Ij [ni, nj]: the area of a grid cell centred at [I, J-1/2]
    #   divided by the distance, squared, from [I,J-1] to [I,J].  Equivalently
    #   this is the grid distance in first dimension divided by grid distance
    #   in second dimension, both centred at [I, J-1/2].
    # wrap [2 element array]: wrap[i] is true iff the domain is periodic in the
    #                         i'th lateral dimension.
    # A4 [4, ni*nj]: adjacency matrix  [see grid_adjacency.m]
    # qu [ni*nj,1]: the nodes visited by the BFS's in order from 1 to qt [see bfs_conncomp1.m]
    # qt [1,1]: the last valid index of qu [see bfs_conncomp1.m]
    # m_ref [1,1]  : linear index to a reference cast at which ϕ will be zero.
    #
    #
    # --- Output:
    # ϕ [ni, nj]: density perturbation attempting to satisfy the discrete
    #             version of  div grad ϕ = - div ϵ
    #             where ϵ is the neutrality error (see ntp_errors).

    # Author[s] : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

    def im1(F):  # G[i,j] == F[i-1,j]
        return np.roll(F, 1, axis=0)

    def ip1(F):  # G[i,j] == F[i+1,j]
        return np.roll(F, -1, axis=0)

    def jm1(F):  # G[i,j] == F[i,j-1]
        return np.roll(F, 1, axis=1)

    def jp1(F):  # G[i,j] == F[i,j+1]
        return np.roll(F, -1, axis=1)

    ni, nj = p.shape

    # The value nij appears in A5 to index neighbours that would go across a non-periodic boundary
    nij = ni * nj

    # If both gridding variables are 1, then grid is uniform
    UNIFORM_GRID = (
        DIST2on1_iJ.size == 1
        and DIST2on1_iJ == 1
        and DIST1on2_Ij.size == 1
        and DIST1on2_Ij == 1
    )

    ## Begin building D = divergence of ϵ, and L = Laplacian [compact representation]

    # L refers to neighbours in this order [so does A4, except without the 5'th entry]:
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

    # A stripped down version of ntp_errors[s,t,p,1,1,true,false,true]
    vs, vt = rho_s_t_bsq(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # [vs, vt] = eos_s_t[ 0.5 * (sm + sn), 0.5 * (tm + tn), 1500 ];  # DEV: testing omega software to find potential density surface()
    ϵ = vs * (sm - sn) + vt * (tm - tn)

    bad = np.isnan(ϵ)
    ϵ[bad] = 0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = DIST2on1_iJ.copy()
        fac[bad] = 0
        ϵ *= fac  # scale ϵ

    D = ϵ - ip1(ϵ)

    L_IJ[:] = fac + ip1(fac)

    L_MJ[:] = -fac

    L_PJ[:] = -ip1(fac)

    # --- m = (i, j) & n = (i, j-1),  then also n = (i, j+1) by symmetry
    sn = jm1(sm)
    tn = jm1(tm)
    pn = jm1(pm)
    if not wrap[1]:
        sn[:, 0] = np.nan

    # A stripped down version of ntp_errors[s,t,p,1,1,true,false,true]
    (vs, vt) = rho_s_t_bsq(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # [vs, vt] = eos_s_t[ 0.5 * (sm + sn), 0.5 * (tm + tn), 1500 ];  # DEV: testing omega software to find potential density surface()

    ϵ = vs * (sm - sn) + vt * (tm - tn)
    bad = np.isnan(ϵ)
    ϵ[bad] = 0.0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = DIST1on2_Ij.copy()
        fac[bad] = 0.0
        ϵ *= fac  # scale ϵ

    D += ϵ - jp1(ϵ)

    L_IJ[:] += fac + jp1(fac)

    L_IM[:] = -fac

    L_IP[:] = -jp1(fac)

    # --- Finish building L
    # For any m where all neighbours are NaN, set L[IJ,m] to 1 so that this
    # equation amounts to:  1 * ϕ[m] = 0. This keϵ ϕ[m] = 0, rather than
    # becoming NaN & infecting its neighbours.
    L_IJ[L_IJ == 0.0] = 1.0

    # --- Build & solve sparse matrix problem
    ϕ = np.full(nij, np.nan, dtype=np.float64)

    # Collect & sort linear indices to all pixels in this region
    # sorting here makes matrix better structured; overall speedup.
    m = np.sort(qu[0 : qt + 1])

    # Note: N > 0 guaranteed by qt > 0 in caller function
    N = len(m)  # Number of water columns
    if N <= 1:  # There are definitely no equations to solve
        ϕ[m[0]] = 0.0  # Leave this isolated pixel at current pressure
        return ϕ.reshape(ni, nj)

    # `remap` changes from linear indices for the entire 2D space (0, 1, ..., ni*nj-1) into linear
    # indices for the current connected component (0, 1, ..., N-1)
    # If the domain were doubly periodic, we would want `remap` to be a 2D array
    # of size (ni,nj). However, with a potentially non-periodic domain, we need
    # one more value for `A5` to index into.  Hence we use `remap` as a vector
    # with ni*nj+1 elements, the last one corresponding to non-periodic boundaries.
    # Water columns that are not in this connected component, and dry water columns (i.e. land),
    # and the fake water column for non-periodic boundaries are all left
    # to have a remap value of -1.
    remap = np.full(nij + 1, -1, dtype=int)
    remap[m] = range(N)

    # Pin surface at mr by changing the mr'th equation to be 1 * ϕ[mr] = 0.
    D[mr] = 0.0
    L[mr] = 0.0
    L[mr][IJ] = 1.0

    L = L.reshape((nij, 5))
    D = D.reshape(nij)

    # The above change renders the mr'th column on all rows irrelevant
    # since ϕ[mr] will be zero.  So, we may also set this column to 0
    # which we do here by setting the appropriate links in L to 0. This
    # maintains symmetry of the matrix, and speeds up solution by a
    # factor of about 2.
    mrI = np.ravel_multi_index(mr, (ni, nj))  # get linear index for mr
    if A5[mrI, IP] != nij:
        L[A5[mrI, IP], IM] = 0
    if A5[mrI, PJ] != nij:
        L[A5[mrI, PJ], MJ] = 0
    if A5[mrI, MJ] != nij:
        L[A5[mrI, MJ], PJ] = 0
    if A5[mrI, IM] != nij:
        L[A5[mrI, IM], IP] = 0

    # Build the RHS of the matrix problem
    rhs = D[m]

    # Build indices for the rows of the sparse matrix, namely
    # [[0,0,0,0,0], ..., [N-1,N-1,N-1,N-1,N-1]]
    r = np.repeat(range(N), 5).reshape(N, 5)

    # Build indices for the columns of the sparse matrix
    # `remap` changes global indices to local indices for this region, numbered 0, 1, ... N-1
    c = remap[A5[m]]

    # Build the values of the sparse matrix
    v = L[m]

    # Ignore connections to dry pixels (though they should have zero values anyway, this is faster)
    good = c >= 0

    # DEV: Could try exiting here, and do csc_matrix, spsolve inside main
    # function, so that this can be njit'ed.  But numba doesn't support
    # np.roll as we need it...  (nor ravel_multi_index, but we could just do
    # that one ourselves)
    # return r[good], c[good], v[good], N, rhs, m

    # Build the sparse matrix; with N rows & N columns
    mat = csc_matrix((v[good], (r[good], c[good])), shape=(N, N))

    # raise('halt')

    # Solve the matrix problem
    sol = spsolve(mat, rhs)  # DEV: check that this uses Cholesky

    # Save solution
    ϕ[m] = sol

    return ϕ.reshape(ni, nj)


@numba.njit
def omega_vertsolve(s, t, p, P, S, Sppc, T, Tppc, n_good, tol, ϕ):
    # Note!  mutates s, t, p
    # Doco from MATLAB, needs updating.

    for n in np.ndindex(n_good.shape):
        ϕn = ϕ[n]
        k = n_good[n]
        if k > 1 and np.isfinite(ϕn):

            # Select this water column
            tup = (*n, slice(k))
            Pn = P[tup]
            Sn = S[tup]
            Tn = T[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]

            args = (Pn, Sn, Sppcn, Tn, Tppcn, ϕn - rho_bsq(s[n], t[n], p[n]), p[n])

            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(func_omega, args, p[n], Pn[0], Pn[-1])

            if not np.isnan(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(func_omega, args, lb, ub, tol)

                # Interpolate S and T onto the updated surface
                s[n], t[n] = val2_0d(Pn, Sn, Sppcn, Tn, Tppcn, p[n])

            else:
                s[n] = np.nan
                t[n] = np.nan
                p[n] = np.nan

        else:
            # ϕ is nan, or only one grid cell so cannot interpolate.
            # This will ensure s,t,p all have the same nan structure
            s[n] = np.nan
            t[n] = np.nan
            p[n] = np.nan

    return None


@numba.njit
def func_omega(p, P, S, Sppc, T, Tppc, ϕ_minus_rho0, p0):
    #     Evaluate difference between (a) eos at location on the cast where the
    #     pressure or depth is p, plus the density perturbation ϕ, and (b) eos at
    #     location on the cast where the surface currently resides (at pressure or
    #     depth p0).  The combination of d and part (b) is precomputed as ϕ_minus_rho0.
    #     Here, eos always evaluated at the pressure or depth of the original position,
    #     p0; this is to calculate locally referenced potential density with reference
    #     pressure p0.

    # Interpolate S and T to the current pressure or depth
    s, t = val2_0d(P, S, Sppc, T, Tppc, p)

    # Calculate the potential density or potential specific volume difference
    return rho_bsq(s, t, p0) + ϕ_minus_rho0
