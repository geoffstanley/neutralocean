"""
Density of Sea Water using the Jackett et al. (2006) [1]_ function

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

.. [1] Jackett, D. R., McDougall, T. J., Feistel, R., Wright, D. G., &
   Griffies, S. M. (2006). Algorithms for Density, Potential Temperature,
   Conservative Temperature, and the Freezing Temperature of Seawater.
   Journal of Atmospheric and Oceanic Technology, 23(12), 1709–1728.
   https://doi.org/10.1175/JTECH1946.1

History
-------
Code adapted from MOM5.1 (Griffies et al) originally written in Fortran
2022 January 14 - David Hutchinson - Translated into python
2022 January 14 - Geoff Stanley    - code optimization, partial derivatives, check vals
"""

import numpy as np
import numba as nb

# List of coefficients for the rational function
# fmt: off
a0  =  9.9984085444849347e+02
a1  =  7.3471625860981584e+00
a2  = -5.3211231792841769e-02
a3  =  3.6492439109814549e-04
a4  =  2.5880571023991390e+00
a5  = -6.7168282786692355e-03
a6  =  1.9203202055760151e-03
a7  =  1.1798263740430364e-02
a8  =  9.8920219266399117e-08
a9  =  4.6996642771754730e-06
a10 = -2.5862187075154352e-08
a11 = -3.2921414007960662e-12

b0  =  1.0000000000000000e+00 
b1  =  7.2815210113327091e-03
b2  = -4.4787265461983921e-05 
b3  =  3.3851002965802430e-07
b4  =  1.3651202389758572e-10
b5  =  1.7632126669040377e-03
b6  = -8.8066583251206474e-06
b7  = -1.8832689434804897e-10
b8  =  5.7463776745432097e-06
b9  =  1.4716275472242334e-09
b10 =  6.7103246285651894e-06
b11 = -2.4461698007024582e-17
b12 = -9.1534417604289062e-18

epsln = 1.0e-40
# fmt: on


@nb.njit
def rho(s, t, p):
    """
    Parameters
    ----------
    s : float
        Practical salinity [PSS-78]
    t : float
        Potential temperature [ITS-90]
    p : float
        Pressure [dbar]

    Returns
    -------
    rho : float
        In-situ density [kg m-3]
    """

    # Precompute some commonly used terms
    t2 = t * t

    # Rational function for density
    num = (
        a0
        + t * (a1 + t * (a2 + a3 * t))
        + s * (a4 + a5 * t + a6 * s)
        + p * (a7 + a8 * t2 + a9 * s + p * (a10 + a11 * t2))
    )

    inv_den = 1.0 / (
        b0
        + t * (b1 + t * (b2 + t * (b3 + t * b4)))
        + s * (b5 + t * (b6 + b7 * t2) + np.sqrt(s) * (b8 + b9 * t2))
        + p * (b10 + p * t * (b11 * t2 + b12 * p))
        + epsln
    )

    return num * inv_den


@nb.njit
def rho_s_t(s, t, p):
    """
    Parameters
    ----------
    s : float
        Practical salinity [PSS-78]
    t : float
        Potential temperature [ITS-90]
    p : float
        Pressure [dbar]

    Returns
    -------
    rho_s : float
        Partial derivative of in-situ density with respect to salinity [kg m-3 psu-1]

    rho_t : float
        Partial derivative of in-situ density with respect to temperature [kg m-3 degc-1]
    """

    # Precompute some commonly used terms
    t2 = t * t
    sp5 = np.sqrt(s)
    pt = p * t

    # Rational function for density
    num = (
        a0
        + t * (a1 + t * (a2 + a3 * t))
        + s * (a4 + a5 * t + a6 * s)
        + p * (a7 + a8 * t2 + a9 * s + p * (a10 + a11 * t2))
    )

    inv_den = 1.0 / (
        b0
        + t * (b1 + t * (b2 + t * (b3 + t * b4)))
        + s * (b5 + t * (b6 + b7 * t2) + sp5 * (b8 + b9 * t2))
        + p * (b10 + pt * (b11 * t2 + b12 * p))
        + epsln
    )

    # The density is
    #   rho = num / den
    # Taking the partial derivative w.r.t. s gives
    #   rho_s = (num_s - num * den_s / den ) / den
    # and similarly for rho_t

    num_s = a4 + a5 * t + 2.0 * a6 * s + p * a9
    num_t = (
        a1
        + t * (2.0 * a2 + 3.0 * a3 * t)
        + a5 * s
        + 2.0 * a8 * pt
        + 2.0 * a11 * p * pt
    )

    den_s = b5 + t * (b6 + b7 * t2) + sp5 * (1.5 * b8 + 1.5 * b9 * t2)
    den_t = (
        b1
        + t * (2.0 * b2 + t * (3.0 * b3 + 4.0 * b4 * t))
        + s * (b6 + 3.0 * b7 * t2 + 2.0 * b9 * sp5 * t)
        + 3.0 * b11 * pt * pt
        + b12 * p**3
    )

    rho_s = (num_s - num * den_s * inv_den) * inv_den
    rho_t = (num_t - num * den_t * inv_den) * inv_den

    return rho_s, rho_t


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho_p(s, t, p):
    """
    Parameters
    ----------
    s : float
        Practical salinity [PSS-78]
    t : float
        Potential temperature [ITS-90]
    p : float
        Pressure [dbar]

    Returns
    -------
    rho_p : float
        Partial derivative of in-situ density with respect to pressure [kg m-3 dbar-1]
    """

    # Precompute some commonly used terms
    t2 = t * t

    # Rational function for density
    num = (
        a0
        + t * (a1 + t * (a2 + a3 * t))
        + s * (a4 + a5 * t + a6 * s)
        + p * (a7 + a8 * t2 + a9 * s + p * (a10 + a11 * t2))
    )

    inv_den = 1.0 / (
        b0
        + t * (b1 + t * (b2 + t * (b3 + t * b4)))
        + s * (b5 + t * (b6 + b7 * t2) + np.sqrt(s) * (b8 + b9 * t2))
        + p * (b10 + p * t * (b11 * t2 + b12 * p))
        + epsln
    )

    # The density is
    #   rho = num / den
    # Taking the partial derivative w.r.t. p gives
    #   rho_p = (num_p - num * den_p / den ) / den

    num_p = a7 + a8 * t2 + a9 * s + p * (2.0 * a10 + 2.0 * a11 * t2)

    den_p = b10 + p * t * (2.0 * b11 * t2 + 3.0 * b12 * p)

    return (num_p - num * den_p * inv_den) * inv_den
