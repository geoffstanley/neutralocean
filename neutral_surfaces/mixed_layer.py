""" Mixed Layer """

from neutral_surfaces.interp_ppc import linear_coeffs, val
from neutral_surfaces.eos.eostools import vectorize_eos

def mixed_layer(S, T, P, eos, pot_dens_diff=0.03, ref_p=100.0,bottle_num=2,interp_fn=linear_coeffs):
    """Calculate the pressure or depth at the bottom of the mixed layer
    
    
    ML = mixed_layer(S, T, P)
    calculates the pressure or depth of the mixed layer, ML, in an ocean
    with practical / absolute salinity S and potential / Conservative
    temperature T located at datasites where the pressure or depth is P.  The
    equation of state for either the in-situ density or the specific volume
    is given by eos.m in the path, which accepts S, T, P as its 3 inputs.
    ML is the pressure or depth at which the potential density (referenced
    to P = 100 dbar or P = 100 m) exceeds the potential density near the
    surface (the second bottle on each cast) by 0.03 kg m^-3.
    
    ... = mixed_layer(..., OPTS) overwrites the default parameters
    according to the struct OPTS (see below).
    
    
    Parameters
    ----------
    S [nk, ni, nj]: practical / Absolute salinity
    T [nk, ni, nj]: potential / Conservative temperature
    P [nk, ni, nj]: pressure [dbar] or depth [m, positive]
    OPTS [struct]: options
    OPTS.pot_dens_ref [1, 1]: the reference pressure or depth for potential
    density [dbar or m, positive]
    OPTS.pot_dens_diff [1, 1]: the potential density difference that
    determines the mixed layer [kg m^-3]
    OPTS.bottle_num [1, 1]: the bottle number on each cast where the "near
    surface" potential density is calculated [integer]
    OPTS.INTERPFN [function handle]: the vertical interpolation function
    
    Note: nk is the maximum number of data points per water column,
       ni is the number of data points in longitude,
       nj is the number of data points in latitude.
    
    Note: P must increase monotonically along the first dimension.
    
    
    Returns
    -------
    ML : ndarray
        the pressure [dbar] or depth [m, positive] of the mixed layer
    """
    
    # Ensure eos is vectorized. It's okay if eos already was.
    eos = vectorize_eos(eos) 
    
    SB = S[:,:,bottle_num : bottle_num+1] # retain singleton trailing dimension
    TB = T[:,:,bottle_num : bottle_num+1]
    
    # Calculate potential density difference between each data point and the
    # near-surface bottle
    if eos(34.5, 3, 1000) > 1:  # eos computes in-situ density
        DD =     eos(S, T, ref_p) -     eos(SB, TB, ref_p)
    else:                       # eos computes specific volume
        DD = 1 / eos(S, T, ref_p) - 1 / eos(SB, TB, ref_p)
    
    # Find the pressure or depth at which the potential density difference
    # exceeds the threshold pot_dens_diff
    Pppc = interp_fn(DD, P)
    return val(DD, P, Pppc, pot_dens_diff)
    
    
    
