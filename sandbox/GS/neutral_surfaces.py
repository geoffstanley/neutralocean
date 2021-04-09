import numpy as np

import fzero
import densjmd95
import ppc


def pot_dens_surf(S, T, P, p_ref, var, TOL_P_UPDATE=1e-4, INTERPFN=ppc.linterp):
    """
    POT_DENS_SURF  Potential density surface by nonlinear solution in each water column.


    [p,s,t] = pot_dens_surf(S, T, P, p_ref, d0)
    finds the pressure or depth p -- and its salinity s and temperature t --
    of the isosurface d0 of potential density referenced to p_ref, in the
    ocean with practical / Absolute salinity S and potential / Conservative
    temperature T at data sites where the and pressure or depth is P.  The
    equation of state is given by eos.m in MATLAB's path, which accepts S, T,
    p_ref as its 3 inputs. For a non-Boussinesq ocean, p, P, and p_ref should
    be pressure [dbar].  For a Boussinesq ocean, p, P, and p_ref should be
    depth [m], positive and increasing down.  Algorithmic parameters are
    provided in OPTS (see "Options" below for further details).

    [p,s,t,d0] = pot_dens_surf(S, T, P, p_ref, [i0, j0, p0])
    as above but finds the surface intersecting a reference cast given by
    grid indices (i0,j0) at pressure or depth p0.

    ... = pot_dens_surf(..., OPTS)
    sets algorithmic parameters (see "Options" below for further details).


    --- Input:
    S [nk, ni, nj]: practical / Absolute salinity
    T [nk, ni, nj]: potential / Conservative temperature
    P [nk, ni, nj] or [nk, 1]: pressure [dbar] or depth [m, positive]
    p_ref [1, 1]: reference pressure [dbar] or depth [m]
    var [1, 1] or [1, 3]: isovalue of potential density, or target location
    OPTS [struct]: options (see below)

    Note: nk is the maximum number of data points per water column,
          ni is the number of data points in longitude,
          nj is the number of data points in latitude.

    Note: physical units for S, T, P, p_ref, p, d0 are determined by eos.m.

    Note: P must increase monotonically along its first dimension.


    --- Output:
    p [ni, nj]: pressure [dbar] or depth [m] of the surface
    s [ni, nj]: practical / Absolute salinity on the surface
    t [ni, nj]: potential / Conservative temperature the surface
    d0 [1, 1]: potential density [kg m^-3] or specific volume [m^3 kg^-1] of
               the surface
    diags [struct]: diagnostics of the solution and computation time


    --- Options:
    OPTS is a struct containing a subset of the following fields.
      INTERPFN [function handle]: vertical interpolation function, used to
          evaluate Sppc and Tppc if those are not provided.  E.g. INTERPFN =
          @ppc_linterp.
      Sppc [O, nk-1, ni, nj]: Coefficients for piecewise polynomials whose
          knots are P that interpolate S as a function of P in each water
          column.  E.g. Sppc = ppc_linterp(P, S);
      Tppc [O, nk-1, ni, nj]: Coefficients for piecewise polynomials whose
          knots are P that interpolate T as a function of P in each water
          column.  E.g. Tppc = ppc_linterp(P, T);
      TOL_P_UPDATE [1, 1]: error tolerance, in the same units as P [dbar] or
          [m], when root-finding to update the surface.


    --- Equation of State:
    The MATLAB path* must contain the function eos.m. This must accept 3
    inputs: S, T, and P. eos(S, T, P) returns the specific volume [m^3 kg^-1]
    or the in-situ density [kg m^-3].
    *Note: It is not sufficient to simply have these eos functions in the
    current working directory, because the compiled MEX functions will not be
    able to find them there.  They must be in the MATLAB path.  If they are
    in the current working directory, use `addpath(pwd)` to add the current
    working directory to the top of MATLAB's path.

    Author(s) : Geoff Stanley
    Email     : g.stanley@unsw.edu.au
    Email     : geoffstanley@gmail.com
    """

    def vertsolve(p, P, Sppc, Tppc, BotK, p_ref, d0, TOL):
        """
        VERTSOLVE  Helper function for pot_dens_surf, solving
                   non-linear root finding problem in each water column

        Mutates p.

        Assumes P is 1D.

        Note: to ensure the generated MEX function gives the same output as
        running this in native MATLAB, the MEX function must be generated with K
        specified as an integer class, not double: we use uint16.

        Author(s) : Geoff Stanley
        Email     : g.stanley@unsw.edu.au
        Email     : geoffstanley@gmail.com
        """

        s = np.empty(p.shape)
        t = np.empty(p.shape)

        def fn(p, P, Sppc, Tppc, p_ref, d0):
            (s, t) = ppc.val2(P, Sppc, Tppc, p)
            return densjmd95.rho(s, t, p_ref) - d0

        # Loop over each cast
        for n in np.ndindex(p.shape):
            k = BotK[n]
            if k > 1:

                # Select this water column
                Sppcn = Sppc[(*n, ...)]
                Tppcn = Tppc[(*n, ...)]

                def f(p):
                    return fn(p, P, Sppcn, Tppcn, p_ref, d0)

                # Initial guess could be nan, which would enter fzero.guess_to_bounds
                # into an infinite loop.  In this case, try initial guess at mid-depth.
                if np.isnan(p[n]):
                    p[n] = (P[0] + P[k-1]) * 0.5

                # Search for a sign-change, expanding outward from an initial guess
                (lb, ub) = fzero.guess_to_bounds(f, p[n], P[0], P[k-1])

                if ~np.isnan(lb):
                    # A sign change was discovered, so a root exists in the interval.
                    # Solve the nonlinear root-finding problem using Brent's method
                    p[n] = fzero.brent(f, lb, ub, TOL)

                    # Interpolate S and T onto the updated surface
                    (s[n], t[n]) = ppc.val2(P, Sppcn, Tppcn, p[n])
                else:
                    p[n] = np.nan
                    s[n] = np.nan
                    t[n] = np.nan
            else:
                # only one grid cell so cannot interpolate.
                # This will ensure s,t,p all have the same nan structure
                p[n] = np.nan
                s[n] = np.nan
                t[n] = np.nan

        return (s, t)

    # Interpolate S and T as piecewise polynomials of P, or use pre-computed interpolants in OPTS.
    Sppc = INTERPFN(P, S)
    Tppc = INTERPFN(P, T)

    # Count number of valid bottles per cast
    BotK = np.sum(np.isfinite(S), axis=-1)

    # Decide on the isosurface value
    if type(var) == tuple and len(var) == 1:
        d0 = var[0]
    elif type(var) == float:
        d0 = var
    elif len(var) == 3:
        i0 = var[0]
        j0 = var[1]
        p0 = var[2]

        # Select the reference cast
        Sppc0 = Sppc[i0, j0, ...]
        Tppc0 = Tppc[i0, j0, ...]

        # Choose iso-value that will intersect (i0,j0,p0).
        (s0, t0) = ppc.val2(P, Sppc0, Tppc0, p0)
        d0 = densjmd95.rho(s0, t0, p_ref)
    else:
        raise TypeError("var must be a scalar or 3 element tuple")

    # Calculate 3D field for vertical interpolation
    D = densjmd95.rho(S, T, p_ref)
    if densjmd95.rho(34.5, 3, 1000) > 1:
        # eos is in-situ density, increasing with 3rd argument
        D.sort()
    else:
        # eos is specific volume, decreasing with 3rd argument
        D[::-1].sort()

    # Get started with the discrete version (and linear interpolation)
    p = ppc.linterp(D, P, d0)

    # Start timer after all the setup has been done.
    # iter_tic = tic();

    # Solve non-linear root finding problem in each cast
    (s, t) = vertsolve(p, P, Sppc, Tppc, BotK, p_ref, d0, TOL_P_UPDATE)

    return (p, s, t)

    # if DIAGS
    #     diags = struct();
    #     diags.clocktime = toc(iter_tic);
    #     [epsL2, epsL1] = eps_norms(s, t, p, true, OPTS.WRAP, {}, OPTS.DIST1_iJ, OPTS.DIST2_Ij, OPTS.DIST2_iJ, OPTS.DIST1_Ij);
    #     diags.epsL1 = epsL1;
    #     diags.epsL2 = epsL2;
    # end
