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


# fmt: off
# Coefficients nonlinear equation of state in pressure coordinates for
# 1. density of fresh water at p = 0
eosJMDCFw0 =    999.842594  
eosJMDCFw1 =    6.793952e-02
eosJMDCFw2 = -  9.095290e-03
eosJMDCFw3 =    1.001685e-04
eosJMDCFw4 = -  1.120083e-06
eosJMDCFw5 =    6.536332e-09
# 2. density of sea water at p = 0
eosJMDCSw0 =    8.244930e-01
eosJMDCSw1 = -  4.089900e-03
eosJMDCSw2 =    7.643800e-05
eosJMDCSw3 = -  8.246700e-07
eosJMDCSw4 =    5.387500e-09
eosJMDCSw5 = -  5.724660e-03
eosJMDCSw6 =    1.022700e-04
eosJMDCSw7 = -  1.654600e-06
eosJMDCSw8 =    4.831400e-04
# coefficients in pressure coordinates for
# 3. secant bulk modulus K of fresh water at p = 0
eosJMDCKFw0 =   1.965933e+05 # .== original * 10
eosJMDCKFw1 =   1.444304e+03 # .== original * 10
eosJMDCKFw2 = - 1.706103e+01 # .== original * 10
eosJMDCKFw3 =   9.648704e-02 # .== original * 10
eosJMDCKFw4 = - 4.190253e-04 # .== original * 10
# 4. secant bulk modulus K of sea water at p = 0
eosJMDCKSw0 =   5.284855e+02 # .== original * 10
eosJMDCKSw1 = - 3.101089e+00 # .== original * 10
eosJMDCKSw2 =   6.283263e-02 # .== original * 10
eosJMDCKSw3 = - 5.084188e-04 # .== original * 10
eosJMDCKSw4 =   3.886640e+00 # .== original * 10
eosJMDCKSw5 =   9.085835e-02 # .== original * 10
eosJMDCKSw6 = - 4.619924e-03 # .== original * 10
# 5. secant bulk modulus K of sea water at p
eosJMDCKP0 =   3.186519e+00
eosJMDCKP1 =   2.212276e-02
eosJMDCKP2 = - 2.984642e-04
eosJMDCKP3 =   1.956415e-06
eosJMDCKP4 =   6.704388e-03
eosJMDCKP5 = - 1.847318e-04
eosJMDCKP6 =   2.059331e-07
eosJMDCKP7 =   1.480266e-04
eosJMDCKP8 =   2.102898e-05 # .== original / 10
eosJMDCKP9 = - 1.202016e-06 # .== original / 10
eosJMDCKP10 =   1.394680e-08 # .== original / 10
eosJMDCKP11 = - 2.040237e-07 # .== original / 10
eosJMDCKP12 =   6.128773e-09 # .== original / 10
eosJMDCKP13 =   6.207323e-11 # .== original / 10
# The above coeffs need to be defined in the function for
# compatability with @numba.njit
# fmt: on

# If ndarray inputs are needed, it is best to use @nb.vectorize.  That is,
# apply `.tools.vectorize_eos`.  A vectorized function specified
# for scalars is about twice as fast as a signatureless njit'ed function
# applied to ndarrays.
@nb.njit
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
    This function is derived from `densjmd95.m`, documented below. Input
    checks and expansion of variables have been removed. That code used
    arrays to store coefficients, and also began by multiplying the
    input pressure by 10 (dbar -> bar). The coefficients, modified to
    not need this multiplication by 10, have been taken out of arrays
    and simply hardcoded into the later expressions.  The polynomial
    calculations have also been optimized to favour faster, nested
    multiplications.

    As such, the output of this function differs from the output of the
    original `densjmd95.m` function, though the difference is at the
    level of machine precision.

    .. highlight:: matlab
    .. code-block:: matlab

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

    s1o2 = np.sqrt(s)

    # fmt: off
    # The secant bulk modulus
    K =    (        eosJMDCKFw0 + t*(eosJMDCKFw1 + t*(eosJMDCKFw2 + t*(eosJMDCKFw3 + t*eosJMDCKFw4)))
        +   s *    (eosJMDCKSw0 + t*(eosJMDCKSw1 + t*(eosJMDCKSw2 + t* eosJMDCKSw3))
        +   s1o2 * (eosJMDCKSw4 + t*(eosJMDCKSw5 + t* eosJMDCKSw6) ))
        + p * (     eosJMDCKP0 + t*(eosJMDCKP1 + t*(eosJMDCKP2 + t*eosJMDCKP3))
        +   s *    (eosJMDCKP4 + t*(eosJMDCKP5 + t* eosJMDCKP6) + s1o2*eosJMDCKP7)
        + p * (     eosJMDCKP8 + t*(eosJMDCKP9 + t* eosJMDCKP10)
        +   s *    (eosJMDCKP11 + t*(eosJMDCKP12 + t* eosJMDCKP13)) )) )


    # The partial derivative of K with respect to s
    # K_s =      ( eosJMDCKSw0 + t*(eosJMDCKSw1 + t*(eosJMDCKSw2 + t* eosJMDCKSw3))
    #     + s1o2 * (1.5*eosJMDCKSw4 + t*(1.5*eosJMDCKSw5 + t* (1.5*eosJMDCKSw6)) )
    #     + p * ( eosJMDCKP4 + t*(eosJMDCKP5 + t* eosJMDCKP6) + s1o2*(1.5*eosJMDCKP7)
    #     + p * ( eosJMDCKP11 + t*(eosJMDCKP12 + t* (eosJMDCKP13)) )) )

    # The partial derivative of K with respect to t
    # K_t =    (     (eosJMDCKFw1 + t*(2.0*eosJMDCKFw2 + t*(3.0*eosJMDCKFw3 + t*(4.0*eosJMDCKFw4))))
    #     +   s *    (eosJMDCKSw1 + t*(2.0*eosJMDCKSw2 + t*(3.0*eosJMDCKSw3))
    #     +   s1o2 * (eosJMDCKSw5 + t*(2.0*eosJMDCKSw6) ))
    #     + p * (     eosJMDCKP1 + t*(2.0*eosJMDCKP2 + t*(3.0*eosJMDCKP3))
    #     +   s *    (eosJMDCKP5 + t*(2.0*eosJMDCKP6))
    #     + p * (     eosJMDCKP9 + t*(2.0*eosJMDCKP10)
    #     +   s *    (eosJMDCKP12 + t*(2.0*eosJMDCKP13)) )) )

    # work =    (    eosJMDCFw0 + t*(eosJMDCFw1 + t*(eosJMDCFw2 + t*(eosJMDCFw3 + t*(eosJMDCFw4 + t*eosJMDCFw5))))
    #     + s *    ( eosJMDCSw0 + t*(eosJMDCSw1 + t*(eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
    #     + s1o2 * ( eosJMDCSw5 + t*(eosJMDCSw6 + t*eosJMDCSw7))
    #     + s    *   eosJMDCSw8   # from here up is the density of sea water at p = 0
    #     )) * p / (K - p) # this prepares for the final rho_s and rho_t computations.

    # The partial derivative of sea water at p = 0, with respect to s
    # rho_0_S =    (     eosJMDCSw0 + t*(    eosJMDCSw1 + t*(    eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
    #     + s1o2 * ( 1.5*eosJMDCSw5 + t*(1.5*eosJMDCSw6 + t*(1.5*eosJMDCSw7)))
    #     + s    * ( 2.0*eosJMDCSw8))

    # The partial derivative of sea water at p = 0, with respect to t
    # rho_0_T =    ( eosJMDCFw1 + t*(2.0*eosJMDCFw2 + t*(3.0*eosJMDCFw3 + t*(4.0*eosJMDCFw4 + t*(5.0*eosJMDCFw5))))
    #     + s *    ( eosJMDCSw1 + t*(2.0*eosJMDCSw2 + t*(3.0*eosJMDCSw3 + t*(4.0*eosJMDCSw4)))
    #     + s1o2 * ( eosJMDCSw6 + t*(2.0*eosJMDCSw7))))

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
                 (     eosJMDCSw0 + t*(    eosJMDCSw1 + t*(    eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
        + s1o2 * ( 1.5*eosJMDCSw5 + t*(1.5*eosJMDCSw6 + t*(1.5*eosJMDCSw7)))
        + s    * ( 2.0*eosJMDCSw8)) # rho_0_S
        * K
        - 
              (    eosJMDCFw0 + t*(eosJMDCFw1 + t*(eosJMDCFw2 + t*(eosJMDCFw3 + t*(eosJMDCFw4 + t*eosJMDCFw5))))
        + s *    ( eosJMDCSw0 + t*(eosJMDCSw1 + t*(eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
        + s1o2 * ( eosJMDCSw5 + t*(eosJMDCSw6 + t*eosJMDCSw7))
        + s    *   eosJMDCSw8   # from here up is the density of sea water at p = 0
        )) # rho_0
        * p / (K - p) 
        *
              ( eosJMDCKSw0 + t*(eosJMDCKSw1 + t*(eosJMDCKSw2 + t* eosJMDCKSw3))
        + s1o2 * (1.5*eosJMDCKSw4 + t*(1.5*eosJMDCKSw5 + t* (1.5*eosJMDCKSw6)) )
        + p * ( eosJMDCKP4 + t*(eosJMDCKP5 + t* eosJMDCKP6) + s1o2*(1.5*eosJMDCKP7)
        + p * ( eosJMDCKP11 + t*(eosJMDCKP12 + t* (eosJMDCKP13)) )) )  # K_s
        ) / (K - p)
    
    rho_t = (
                 ( eosJMDCFw1 + t*(2.0*eosJMDCFw2 + t*(3.0*eosJMDCFw3 + t*(4.0*eosJMDCFw4 + t*(5.0*eosJMDCFw5))))
        + s *    ( eosJMDCSw1 + t*(2.0*eosJMDCSw2 + t*(3.0*eosJMDCSw3 + t*(4.0*eosJMDCSw4)))
        + s1o2 * ( eosJMDCSw6 + t*(2.0*eosJMDCSw7)))) # rho_0_T
        * K
        - 
              (    eosJMDCFw0 + t*(eosJMDCFw1 + t*(eosJMDCFw2 + t*(eosJMDCFw3 + t*(eosJMDCFw4 + t*eosJMDCFw5))))
        + s *    ( eosJMDCSw0 + t*(eosJMDCSw1 + t*(eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
        + s1o2 * ( eosJMDCSw5 + t*(eosJMDCSw6 + t*eosJMDCSw7))
        + s    *   eosJMDCSw8   # from here up is the density of sea water at p = 0
        )) # rho_0
        * p / (K - p) 
        * 
             (     (eosJMDCKFw1 + t*(2.0*eosJMDCKFw2 + t*(3.0*eosJMDCKFw3 + t*(4.0*eosJMDCKFw4))))
        +   s *    (eosJMDCKSw1 + t*(2.0*eosJMDCKSw2 + t*(3.0*eosJMDCKSw3))
        +   s1o2 * (eosJMDCKSw5 + t*(2.0*eosJMDCKSw6) ))
        + p * (     eosJMDCKP1 + t*(2.0*eosJMDCKP2 + t*(3.0*eosJMDCKP3))
        +   s *    (eosJMDCKP5 + t*(2.0*eosJMDCKP6))
        + p * (     eosJMDCKP9 + t*(2.0*eosJMDCKP10)
        +   s *    (eosJMDCKP12 + t*(2.0*eosJMDCKP13)) )) ) # K_t
        ) / (K - p)

    # fmt: on

    return rho_s, rho_t


@nb.njit
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
    K2 = (
        p * ( eosJMDCKP8 + t*(eosJMDCKP9 + t*eosJMDCKP10)  
        + s *  (eosJMDCKP11 + t*(eosJMDCKP12 + t*eosJMDCKP13)) ) 
        )

    K1plusK2 = (
        K2 +
            eosJMDCKP0 + t*(eosJMDCKP1 + t*(eosJMDCKP2 + t*eosJMDCKP3))            # \__ these 2 lines are K1
            + s * ( eosJMDCKP4 + t*(eosJMDCKP5 + t*eosJMDCKP6) + eosJMDCKP7*s1o2 ) # /  
        )
    # K == bulkmod
    K = (   eosJMDCKFw0 + t*(eosJMDCKFw1 + t*(eosJMDCKFw2 + t*(eosJMDCKFw3 + t*eosJMDCKFw4)))  # secant bulk modulus of fresh water at the surface
        + s * (   eosJMDCKSw0 + t*(eosJMDCKSw1 + t*(eosJMDCKSw2 + t*eosJMDCKSw3)) 
            + s1o2 * (eosJMDCKSw4 + t*(eosJMDCKSw5 + t*eosJMDCKSw6) ) 
        )  # secant bulk modulus of sea water at the surface
        + p * K1plusK2 # secant bulk modulus of sea water at pressure z
        )

    rho = (
        eosJMDCFw0 + t*(eosJMDCFw1 + t*(eosJMDCFw2 + t*(eosJMDCFw3 + t*(eosJMDCFw4 + t*eosJMDCFw5)))) # density of freshwater at the surface
        + s * ( eosJMDCSw0 + t*(eosJMDCSw1 + t*(eosJMDCSw2 + t*(eosJMDCSw3 + t*eosJMDCSw4)))
            + s1o2 * ( eosJMDCSw5 + t*(eosJMDCSw6 + t*eosJMDCSw7) )
            + s * eosJMDCSw8
        ) # density of sea water at the surface
        )
    
    # (K1 + 2 * K2) is K_p, the partial derivative of K w.r.t p
    rho_p = rho * (K - p * (K1plusK2 + K2)) / (K - p)**2
    # fmt: on

    return rho_p
