import numpy as np
import numba

from neutral_surfaces._densjmd95 import rho_bsq
from neutral_surfaces import interp_ppc
from neutral_surfaces._zero import guess_to_bounds, brent



def process_arrays(S, T, P, axis=-1):
    # need a better name for this...
    if axis not in (-1, S.ndim -1):
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
    
    n_good = find_first_nan(S)
    
    return S, T, P, n_good

def pot_dens_surf(S, T, P, p_ref, target, axis=-1, tol=1e-4):
    S, T, P, n_good = process_arrays(S, T, P, axis=axis)
    
    Sppc = interp_ppc.linear_coefficients(P, S)
    Tppc = interp_ppc.linear_coefficients(P, T)
    
    
    if isinstance(target, tuple):
        p0 = target[-1]   # target pressure 
        n0 = target[:-1]  # index to water column which intersects the surface at the target pressure
        
        # Choose iso-value that will intersect cast n0 at p0.
        d0 = func_sigma(p0, P[n0], S[n0], Sppc[n0], T[n0], Tppc[n0], p_ref, 0.)
    else:
        d0 = target
        
    shape = S.shape[0:-1]
    p = np.empty(shape)
    p.fill(np.nan)  # Instead of linearly interpolating density below to get the initial guess, we'll just use the mid-depth. 

#     # Calculate 3D field for vertical interpolation
#     D = rho_ufunc(S, T, p_ref)
#     if rho_bsq(34.5, 3, 1000) > 1:
#         # eos is in-situ density, increasing with 3rd argument
#         D.sort()
#     else:
#         # eos is specific volume, decreasing with 3rd argument
#         D[::-1].sort()
    
#     # Get started with the discrete version (and linear interpolation)
#     Pppc = linear_coefficients(D, P)
#     for n in np.ndindex(shape):
#         p[n] = val(D[(*n,...)], P[(*n,...)], Pppc[(*n,...)], d0)  # DEV: would like a nicer way to do this than the for loop.
    
    
    # Solve non-linear root finding problem in each cast
    s, t, p = vertsolve(p, p_ref, d0, n_good, P, S, Sppc, T, Tppc, tol)
    return s, t, p
    
@numba.njit
def func_sigma(p, P, S, Sppc, T, Tppc, p_ref, d0):
    #s, t = linear_eval2(p, *stp_args)
    s, t = interp_ppc.val2(P, S, Sppc, T, Tppc, p)
    return rho_bsq(s, t, p_ref) - d0


# @numba.njit
# def func_delta(p, P, S, Sppc, T, Tppc, s_ref, t_ref, d0):
#     s, t = val2(P, S, Sppc, T, Tppc, p)
#     return rho_bsq(s, t, p) - rho_bsq(s_ref, t_ref, p) - d0

# @numba.njit
# def func_omega(p, Sppc, Tppc, P, phi_minus_rho0, p0)
# #     Evaluate difference between (a) eos at location on the cast where the
# #     pressure or depth is p, plus the density perturbation phi, and (b) eos at
# #     location on the cast where the surface currently resides (at pressure or
# #     depth p0).  The combination of d and part (b) is precomputed as phi_minus_rho0.
# #     Here, eos always evaluated at the pressure or depth of the original position, 
# #     p0; this is to calculate locally referenced potential density with reference 
# #     pressure p0.

#     # Interpolate S and T to the current pressure or depth
#     s, t = linear_eval2(p, *STPppc)

#     # Calculate the potential density or potential specific volume difference
#     return rho_bsq(s, t, p0) + phi_minus_rho0


@numba.njit
def vertsolve(p_in, p_ref, d0, n_good, P, S, Sppc, T, Tppc, tol):
    
    s = np.empty(p_in.shape, dtype=np.float64)
    s.fill(np.nan)
    t = s.copy()
    p = s.copy()
    
    for n in np.ndindex(p_in.shape):
        k = n_good[n]
        if k > 1:
            tup = (*n, slice(k))
            # Unfortunately, we can't do the following:
            # stp_args1 = tuple([arg[tup] for arg in stp_args])
            # The tuple() constructor is not supported by numba, and leaving
            # stp_args1 as a list causes problems later.
            #a = stp_args
            #stp_args1 = (a[0][tup], a[1][tup], a[2][tup], a[3][tup], a[4][tup])
            
            # Select this water column
            Pn = P[tup]
            Sn = S[tup]
            Tn = T[tup]
            Sppcn = Sppc[tup]
            Tppcn = Tppc[tup]
            
            # Initial guess could be nan, which would send guess_to_bounds
            # into an infinite loop.  In this case, try initial guess at mid-depth.
            pn = p_in[n]
            if np.isnan(pn):
                pn = (Pn[0] + Pn[-1]) * 0.5
            
            # Search for a sign-change, expanding outward from an initial guess
            lb, ub = guess_to_bounds(func_sigma, (Pn, Sn, Sppcn, Tn, Tppcn, p_ref, d0), pn, Pn[0], Pn[-1])
            
            if not np.isnan(lb):
                # A sign change was discovered, so a root exists in the interval.
                # Solve the nonlinear root-finding problem using Brent's method
                p[n] = brent(func_sigma, (Pn, Sn, Sppcn, Tn, Tppcn, p_ref, d0), lb, ub, tol)
                
                # Interpolate S and T onto the updated surface
                s[n], t[n] = interp_ppc.val2(Pn, Sn, Sppcn, Tn, Tppcn, p[n])
        
        # else:
            # only one grid cell so cannot interpolate.
            # This will ensure s,t,p all have the same nan structure
            
    return s, t, p



@numba.njit
def find_first_nan(a):
    """
    find_first_nan(a)

    Find the index to the first nan in a along the last axis. 
    If no nan's are present, the length of the last dimension is returned. 
    """
    nk = a.shape[-1]
    k = np.empty(a.shape[0:-1], dtype=np.intp)
    k.fill(nk)
    for n in np.ndindex(a.shape[0:-1]):
        for i in range(nk):
            if np.isnan(a[n][i]):
                k[n] = i
                break
    return k