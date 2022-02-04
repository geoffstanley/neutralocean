"""
Density of Sea Water using the Jackett and McDougall 1995 [1]_ function

Functions:

rho :: computes in-situ density from salinity, potential temperature and
    pressure

rho_s_t :: compute the partial derivatives of in-situ density with
    respect to salinity and potential temperature

rho_p :: compute the partial derivative of in-situ density with
    respect to pressure

Notes:
To make Boussinesq versions of these functions, see 
`neutralocean.eos.tools.make_eos_bsq`.

To make vectorized versions of these functions, see
`neutralocean.eos.tools.vectorize_eos`.

.. [1] Jackett and McDougall, 1995, JAOT 12[4], pp. 381-388
"""

import numpy as np
import numba as nb

# Specify scalar inputs and output.  If ndarray inputs are needed, then apply
# @nb.vectorize, as in `.tools.vectorize_eos`.  A vectorized function specified
# for scalars is about twice as fast as a signatureless njit'ed function
# applied to ndarrays.
@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho(s, t, p):
    """Fast JMD95 [1]_ in-situ density.

    Parameters
    ----------
    s : float
        Practical salinity [PSS-78]
    t : float
        Potential temperature [IPTS-68]
    p : float
        Pressure [dbar]

    Returns
    -------
    rho : float
        JMD95 in-situ density [kg m-3]

    Notes
    -----
    This function is derived from densjmd95.m, documented below. Input
    checks and expansion of variables have been removed. That code used
    arrays to store coefficients, and also began by multiplying the
    input pressure by 10 (dbar -> bar). The coefficients, modified to
    not need this multiplication by 10, have been taken out of arrays
    and simply hardcoded into the later expressions.  The polynomial
    calculations have also been optimized to favour faster, nested
    multiplications.

    As such, the output of this function differs from the output of the
    original densjmd95.m function, though the difference is at the
    level of machine precision.


    % DENSJMD95    Density of sea water
    %=========================================================================
    %
    % USAGE:  dens = densjmd95(S,Theta,P)
    %
    % DESCRIPTION:
    %    Density of Sea Water using Jackett and McDougall 1995 (JAOT 12)
    %    polynomial (modified UNESCO polynomial).
    %
    % INPUT:  (all must have same dimensions)
    %   S     = salinity    [psu      (PSS-78)]
    %   Theta = potential temperature [degree C (IPTS-68)]
    %   P     = pressure    [dbar]
    %       (P may have dims 1x1, mx1, 1xn or mxn for S(mxn) )
    %
    % OUTPUT:
    %   dens = density  [kg/m^3]
    %
    % AUTHOR:  Martin Losch 2002-08-09  (mlosch@mit.edu)


    % Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388

    % created by mlosch on 2002-08-09
    % $Header: /u/gcmpack/MITgcm/utils/matlab/densjmd95.m,v 1.2 2007/02/17 23:49:43 jmc Exp $
    % $Name:  $

    .. [1] Jackett and McDougall, 1995, JAOT 12[4], pp. 381-388
    """

    s1o2 = np.sqrt(s)

    # fmt: off
    # The secant bulk modulus
    K = (           1.965933e+05 + t*( 1.444304e+03 + t*(-1.706103e+01 + t*(9.648704e-02 + t*-4.190253e-04)))
        +   s *    (5.284855e+02 + t*(-3.101089e+00 + t*( 6.283263e-02 + t*-5.084188e-04))
        +   s1o2 * (3.886640e+00 + t*( 9.085835e-02 + t* -4.619924e-03) ))
        + p * (     3.186519e+00 + t*( 2.212276e-02 + t*(-2.984642e-04 + t* 1.956415e-06))
        +   s *    (6.704388e-03 + t*(-1.847318e-04 + t*  2.059331e-07) + s1o2*1.480266e-04)
        + p * (     2.102898e-05 + t*(-1.202016e-06 + t*  1.394680e-08)
        +   s *   (-2.040237e-07 + t*( 6.128773e-09 + t*  6.207323e-11)) )) )

    # The in-situ density
    rho = (
                   999.842594   + t*( 6.793952e-02 + t*(-9.095290e-03 + t*( 1.001685e-04 + t*(-1.120083e-06 + t*6.536332e-09))))
        + s * (    8.244930e-01 + t*(-4.089900e-03 + t*( 7.643800e-05 + t*(-8.246700e-07 + t*  5.387500e-09)))
        + s1o2 * (-5.724660e-03 + t*( 1.022700e-04 + t* -1.654600e-06))
        + s    *   4.831400e-04
        )) / (1.0 - p / K )
    # fmt: on
    return rho


# If we specified a signature for scalar inputs and outputs, such as
#   @nb.njit(nb.typeof((1.0, 1.0))(nb.f8, nb.f8, nb.f8))
# then we would not be able to make a function that uses @numba.vectorize to
# wrap this function.  The tuple output messes that up.  However, we could
# make a function that uses @numba.guvectorize to wrap this function, e.g.
#   @nb.guvectorize([(nb.f8, nb.f8, nb.f8, nb.f8[:], nb.f8[:])], "(),(),()->(),()")
#   def eos_vec(s, t, p, eos_s, eos_t):
#       eos_s[0], eos_t[0] = rho_s_t(s, t, p)
# However, this appears to be about two times slower than just using @numba.njit
# without a signature specification, and calling the rho_s_t function with ndarray
# objects.  So, we'll just use @nb.njit with no signature.
@nb.njit
def rho_s_t(s, t, p):
    """
    Fast salinity and potential temperature partial derivatives of JMD95 in-situ density

    Parameters
    ----------
    s, t, p : float
        See `rho`

    Returns
    -------
        rho_s : float
            Partial derivative of JMD95 in-situ density with respect to
            practical salinity `s` [kg m-3 psu-1]
        rho_t : float
            Partial derivative of JMD95 in-situ density with respect to
            potential temperature `t`   [kg m-3 degC-1]

    Notes
    -----
    This function is derived from `rho`.
    """

    # INPUT CHECKS REMOVED

    # fmt: off
    # Coefficients nonlinear equation of state in pressure coordinates for
    # 1. density of fresh water at p = 0
    eosJMDCFw = [
       999.842594  ,
       6.793952e-02,
    -  9.095290e-03,
       1.001685e-04,
    -  1.120083e-06,
       6.536332e-09]
    # 2. density of sea water at p = 0
    eosJMDCSw = [
       8.244930e-01,
    -  4.089900e-03,
       7.643800e-05,
    -  8.246700e-07,
       5.387500e-09,
    -  5.724660e-03,
       1.022700e-04,
    -  1.654600e-06,
       4.831400e-04]
    # coefficients in pressure coordinates for
    # 3. secant bulk modulus K of fresh water at p = 0
    eosJMDCKFw = [
      1.965933e+05, # .== original * 10
      1.444304e+03, # .== original * 10
    - 1.706103e+01, # .== original * 10
      9.648704e-02, # .== original * 10
    - 4.190253e-04] # .== original * 10
    # 4. secant bulk modulus K of sea water at p = 0
    eosJMDCKSw = [
      5.284855e+02, # .== original * 10
    - 3.101089e+00, # .== original * 10
      6.283263e-02, # .== original * 10
    - 5.084188e-04, # .== original * 10
      3.886640e+00, # .== original * 10
      9.085835e-02, # .== original * 10
    - 4.619924e-03] # .== original * 10
    # 5. secant bulk modulus K of sea water at p
    eosJMDCKP = [
      3.186519e+00,
      2.212276e-02,
    - 2.984642e-04,
      1.956415e-06,
      6.704388e-03,
    - 1.847318e-04,
      2.059331e-07,
      1.480266e-04,
      2.102898e-05, # .== original / 10
    - 1.202016e-06, # .== original / 10
      1.394680e-08, # .== original / 10
    - 2.040237e-07, # .== original / 10
      6.128773e-09, # .== original / 10
      6.207323e-11] # .== original / 10
    # The above coeffs need to be defined in the function for 
    # compatability with @numba.njit
    
    s1o2 = np.sqrt(s)

    # The secant bulk modulus
    K =    (        eosJMDCKFw[0] + t*(eosJMDCKFw[1] + t*(eosJMDCKFw[2] + t*(eosJMDCKFw[3] + t*eosJMDCKFw[4])))
        +   s *    (eosJMDCKSw[0] + t*(eosJMDCKSw[1] + t*(eosJMDCKSw[2] + t* eosJMDCKSw[3]))
        +   s1o2 * (eosJMDCKSw[4] + t*(eosJMDCKSw[5] + t* eosJMDCKSw[6]) ))
        + p * (     eosJMDCKP[0] + t*(eosJMDCKP[1] + t*(eosJMDCKP[2] + t*eosJMDCKP[3]))
        +   s *    (eosJMDCKP[4] + t*(eosJMDCKP[5] + t* eosJMDCKP[6]) + s1o2*eosJMDCKP[7])
        + p * (     eosJMDCKP[8] + t*(eosJMDCKP[9] + t* eosJMDCKP[10])
        +   s *    (eosJMDCKP[11] + t*(eosJMDCKP[12] + t* eosJMDCKP[13])) )) )


    # The partial derivative of K with respect to s
    # K_s =      ( eosJMDCKSw[0] + t*(eosJMDCKSw[1] + t*(eosJMDCKSw[2] + t* eosJMDCKSw[3]))
    #     + s1o2 * (1.5*eosJMDCKSw[4] + t*(1.5*eosJMDCKSw[5] + t* (1.5*eosJMDCKSw[6])) )
    #     + p * ( eosJMDCKP[4] + t*(eosJMDCKP[5] + t* eosJMDCKP[6]) + s1o2*(1.5*eosJMDCKP[7])
    #     + p * ( eosJMDCKP[11] + t*(eosJMDCKP[12] + t* (eosJMDCKP[13])) )) )

    # The partial derivative of K with respect to t
    # K_t =    (     (eosJMDCKFw[1] + t*(2.0*eosJMDCKFw[2] + t*(3.0*eosJMDCKFw[3] + t*(4.0*eosJMDCKFw[4]))))
    #     +   s *    (eosJMDCKSw[1] + t*(2.0*eosJMDCKSw[2] + t*(3.0*eosJMDCKSw[3]))
    #     +   s1o2 * (eosJMDCKSw[5] + t*(2.0*eosJMDCKSw[6]) ))
    #     + p * (     eosJMDCKP[1] + t*(2.0*eosJMDCKP[2] + t*(3.0*eosJMDCKP[3]))
    #     +   s *    (eosJMDCKP[5] + t*(2.0*eosJMDCKP[6]))
    #     + p * (     eosJMDCKP[9] + t*(2.0*eosJMDCKP[10])
    #     +   s *    (eosJMDCKP[12] + t*(2.0*eosJMDCKP[13])) )) )

    # work =    (    eosJMDCFw[0] + t*(eosJMDCFw[1] + t*(eosJMDCFw[2] + t*(eosJMDCFw[3] + t*(eosJMDCFw[4] + t*eosJMDCFw[5]))))
    #     + s *    ( eosJMDCSw[0] + t*(eosJMDCSw[1] + t*(eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
    #     + s1o2 * ( eosJMDCSw[5] + t*(eosJMDCSw[6] + t*eosJMDCSw[7]))
    #     + s    *   eosJMDCSw[8]   # from here up is the density of sea water at p = 0
    #     )) * p / (K - p) # this prepares for the final rho_s and rho_t computations.

    # The partial derivative of sea water at p = 0, with respect to s
    # rho_0_S =    (     eosJMDCSw[0] + t*(    eosJMDCSw[1] + t*(    eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
    #     + s1o2 * ( 1.5*eosJMDCSw[5] + t*(1.5*eosJMDCSw[6] + t*(1.5*eosJMDCSw[7])))
    #     + s    * ( 2.0*eosJMDCSw[8]))

    # The partial derivative of sea water at p = 0, with respect to t
    # rho_0_T =    ( eosJMDCFw[1] + t*(2.0*eosJMDCFw[2] + t*(3.0*eosJMDCFw[3] + t*(4.0*eosJMDCFw[4] + t*(5.0*eosJMDCFw[5]))))
    #     + s *    ( eosJMDCSw[1] + t*(2.0*eosJMDCSw[2] + t*(3.0*eosJMDCSw[3] + t*(4.0*eosJMDCSw[4])))
    #     + s1o2 * ( eosJMDCSw[6] + t*(2.0*eosJMDCSw[7]))))

    # The in-situ density is defined as
    #  rho = rho_0 / (1 - p / K)
    # Taking the partial derivative w.r.t S gives
    #           /            rho_0 * p * K_S   \       1
    #  rho_s = | rho_0_S -  _________________   |  _________
    #           \            (1 - p/K) * K^2   /   1 - p / K
    # This is re-written as
    #           /                rho_0 * p * K_S    \       1
    #  rho_s = | rho_0_S * K -  __________________   |  _________
    #           \                     (K - p)       /     K - p
    # Similarly for rho_t.

    # rho_s = (rho_0_S * K - work * K_s) / (K - p)
    # rho_t = (rho_0_T * K - work * K_t) / (K - p)

    # The following expressions are faster and require less memory, though appear more gruesome
    rho_s = (
                 (     eosJMDCSw[0] + t*(    eosJMDCSw[1] + t*(    eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
        + s1o2 * ( 1.5*eosJMDCSw[5] + t*(1.5*eosJMDCSw[6] + t*(1.5*eosJMDCSw[7])))
        + s    * ( 2.0*eosJMDCSw[8])) # rho_0_S
        * K
        - 
              (    eosJMDCFw[0] + t*(eosJMDCFw[1] + t*(eosJMDCFw[2] + t*(eosJMDCFw[3] + t*(eosJMDCFw[4] + t*eosJMDCFw[5]))))
        + s *    ( eosJMDCSw[0] + t*(eosJMDCSw[1] + t*(eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
        + s1o2 * ( eosJMDCSw[5] + t*(eosJMDCSw[6] + t*eosJMDCSw[7]))
        + s    *   eosJMDCSw[8]   # from here up is the density of sea water at p = 0
        )) # rho_0
        * p / (K - p) 
        *
              ( eosJMDCKSw[0] + t*(eosJMDCKSw[1] + t*(eosJMDCKSw[2] + t* eosJMDCKSw[3]))
        + s1o2 * (1.5*eosJMDCKSw[4] + t*(1.5*eosJMDCKSw[5] + t* (1.5*eosJMDCKSw[6])) )
        + p * ( eosJMDCKP[4] + t*(eosJMDCKP[5] + t* eosJMDCKP[6]) + s1o2*(1.5*eosJMDCKP[7])
        + p * ( eosJMDCKP[11] + t*(eosJMDCKP[12] + t* (eosJMDCKP[13])) )) )  # K_s
        ) / (K - p)
    
    rho_t = (
                 ( eosJMDCFw[1] + t*(2.0*eosJMDCFw[2] + t*(3.0*eosJMDCFw[3] + t*(4.0*eosJMDCFw[4] + t*(5.0*eosJMDCFw[5]))))
        + s *    ( eosJMDCSw[1] + t*(2.0*eosJMDCSw[2] + t*(3.0*eosJMDCSw[3] + t*(4.0*eosJMDCSw[4])))
        + s1o2 * ( eosJMDCSw[6] + t*(2.0*eosJMDCSw[7])))) # rho_0_T
        * K
        - 
              (    eosJMDCFw[0] + t*(eosJMDCFw[1] + t*(eosJMDCFw[2] + t*(eosJMDCFw[3] + t*(eosJMDCFw[4] + t*eosJMDCFw[5]))))
        + s *    ( eosJMDCSw[0] + t*(eosJMDCSw[1] + t*(eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
        + s1o2 * ( eosJMDCSw[5] + t*(eosJMDCSw[6] + t*eosJMDCSw[7]))
        + s    *   eosJMDCSw[8]   # from here up is the density of sea water at p = 0
        )) # rho_0
        * p / (K - p) 
        * 
             (     (eosJMDCKFw[1] + t*(2.0*eosJMDCKFw[2] + t*(3.0*eosJMDCKFw[3] + t*(4.0*eosJMDCKFw[4]))))
        +   s *    (eosJMDCKSw[1] + t*(2.0*eosJMDCKSw[2] + t*(3.0*eosJMDCKSw[3]))
        +   s1o2 * (eosJMDCKSw[5] + t*(2.0*eosJMDCKSw[6]) ))
        + p * (     eosJMDCKP[1] + t*(2.0*eosJMDCKP[2] + t*(3.0*eosJMDCKP[3]))
        +   s *    (eosJMDCKP[5] + t*(2.0*eosJMDCKP[6]))
        + p * (     eosJMDCKP[9] + t*(2.0*eosJMDCKP[10])
        +   s *    (eosJMDCKP[12] + t*(2.0*eosJMDCKP[13])) )) ) # K_t
        ) / (K - p)

    # fmt: on

    return rho_s, rho_t


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho_p(s, t, p):
    """
    Fast pressure derivative of JMD95 in-situ density.

    Parameters
    ----------
    s, t, p : float
        See `rho`

    Returns
    -------
    rho_p : float
        Partial derivative of JMD95 in-situ density with respect to
        pressure `p` [kg m-3 dbar-1]

    Notes
    -----
    This function is derived from `rho`.
    """

    s1o2 = np.sqrt(s)

    # fmt: off
    # Coefficients nonlinear equation of state in pressure coordinates for
    # 1. density of fresh water at p = 0
    eosJMDCFw = [
       999.842594  ,
       6.793952e-02,
    -  9.095290e-03,
       1.001685e-04,
    -  1.120083e-06,
       6.536332e-09]
    # 2. density of sea water at p = 0
    eosJMDCSw = [
       8.244930e-01,
    -  4.089900e-03,
       7.643800e-05,
    -  8.246700e-07,
       5.387500e-09,
    -  5.724660e-03,
       1.022700e-04,
    -  1.654600e-06,
       4.831400e-04]
    # coefficients in pressure coordinates for
    # 3. secant bulk modulus K of fresh water at p = 0
    eosJMDCKFw = [
      1.965933e+05, # .== original * 10
      1.444304e+03, # .== original * 10
    - 1.706103e+01, # .== original * 10
      9.648704e-02, # .== original * 10
    - 4.190253e-04] # .== original * 10
    # 4. secant bulk modulus K of sea water at p = 0
    eosJMDCKSw = [
      5.284855e+02, # .== original * 10
    - 3.101089e+00, # .== original * 10
      6.283263e-02, # .== original * 10
    - 5.084188e-04, # .== original * 10
      3.886640e+00, # .== original * 10
      9.085835e-02, # .== original * 10
    - 4.619924e-03] # .== original * 10
    # 5. secant bulk modulus K of sea water at p
    eosJMDCKP = [
      3.186519e+00,
      2.212276e-02,
    - 2.984642e-04,
      1.956415e-06,
      6.704388e-03,
    - 1.847318e-04,
      2.059331e-07,
      1.480266e-04,
      2.102898e-05, # .== original / 10
    - 1.202016e-06, # .== original / 10
      1.394680e-08, # .== original / 10
    - 2.040237e-07, # .== original / 10
      6.128773e-09, # .== original / 10
      6.207323e-11] # .== original / 10
    # The above coeffs need to be defined in the function for 
    # compatability with @numba.njit

    K2 = (
        p * ( eosJMDCKP[8] + t*(eosJMDCKP[9] + t*eosJMDCKP[10])  
        + s *  (eosJMDCKP[11] + t*(eosJMDCKP[12] + t*eosJMDCKP[13])) ) 
        )

    K1plusK2 = (
        K2 +
            eosJMDCKP[0] + t*(eosJMDCKP[1] + t*(eosJMDCKP[2] + t*eosJMDCKP[3]))            # \__ these 2 lines are K1
            + s * ( eosJMDCKP[4] + t*(eosJMDCKP[5] + t*eosJMDCKP[6]) + eosJMDCKP[7]*s1o2 ) # /  
        )
    # K == bulkmod
    K = (   eosJMDCKFw[0] + t*(eosJMDCKFw[1] + t*(eosJMDCKFw[2] + t*(eosJMDCKFw[3] + t*eosJMDCKFw[4])))  # secant bulk modulus of fresh water at the surface
        + s * (   eosJMDCKSw[0] + t*(eosJMDCKSw[1] + t*(eosJMDCKSw[2] + t*eosJMDCKSw[3])) 
            + s1o2 * (eosJMDCKSw[4] + t*(eosJMDCKSw[5] + t*eosJMDCKSw[6]) ) 
        )  # secant bulk modulus of sea water at the surface
        + p * K1plusK2 # secant bulk modulus of sea water at pressure z
        )

    rho = (
        eosJMDCFw[0] + t*(eosJMDCFw[1] + t*(eosJMDCFw[2] + t*(eosJMDCFw[3] + t*(eosJMDCFw[4] + t*eosJMDCFw[5])))) # density of freshwater at the surface
        + s * ( eosJMDCSw[0] + t*(eosJMDCSw[1] + t*(eosJMDCSw[2] + t*(eosJMDCSw[3] + t*eosJMDCSw[4])))
            + s1o2 * ( eosJMDCSw[5] + t*(eosJMDCSw[6] + t*eosJMDCSw[7]) )
            + s * eosJMDCSw[8]
        ) # density of sea water at the surface
        )
    
    # (K1 + 2 * K2) is K_p, the partial derivative of K w.r.t p
    rho_p = rho * (K - p * (K1plusK2 + K2)) / (K - p)**2
    # fmt: on

    return rho_p
