""" Mixed Layer """

from neutral_surfaces.interp_ppc import linear_coeffs, val
from neutral_surfaces.eos.eostools import vectorize_eos

def mixed_layer(S, T, P, eos, pot_dens_diff=0.03, ref_p=100.0,bottle_num=1,interp_fn=linear_coeffs):
    """Calculate the pressure or depth at the bottom of the mixed layer

    The mixed layer is the pressure or depth at which the potential density 
    (referenced to `ref_p`) exceeds the potential density near the surface 
    (specifically, the n'th bottle on each cast where n = `bottle_num`) by an 
    amount of `pot_dens_diff`.
    
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
        
        Must increase monotonically along the first dimension (i.e. downwards).

    eos : function

        Equation of state for the density or specific volume as a function of
        `S`, `T`, and `P` inputs.  
        
        This function should be @numba.njit decorated and need not be
        vectorized, as it will be called many times with scalar inputs.

    pot_dens_diff : float, Default 0.03
    
        Difference in potential density [kg m-3] between the near-surface and 
        the base of the mixed layer.
        
    ref_p : float, Default 100.0
    
        Reference pressure for potential density [dbar or m].
        
    bottle_num : int, Default 2
    
        The bottle number on each cast where the "near surface" potential 
        density is calculated.
        
    interp_fn : function, Default `linear_coeffs`
    
        Function to perform vertical interpolation

        
    Returns
    -------
    ML : ndarray
        
        A 2D array giving the pressure [dbar] or depth [m, positive] of the mixed layer
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
    
    
    
