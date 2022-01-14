"""
USAGE: r = rho(s, t, p)
USAGE: rs, rt = rho_s_t(s, t, p)

Purpose
-------
Calculates the density of seawater, using the equation of state of:

Jackett, McDougall, Feistel, Wright, and Griffies (2006):
Algorithms for density, potential temperature, conservative  temperature, and freezing
temperature of seawater. Journal of Atmospheric and Oceanic Technology, 23, 1709-1728.

Inputs
------
s       Salinity (psu)
t       Potential temperature (degC)
p       Pressure (db)

Outputs
-------
rho     Density (kg m-3)

History
-------
Code adapted from MOM5.1 (Griffies et al) originally written in Fortran
2022 January 14 - David Hutchinson - Translated into python
2022 January 14 - Geoff Stanley    - code optimization, partial derivatives, check vals
"""

import numpy as np
from numba import njit, float64


@njit(float64(float64, float64, float64))
def rho(s, t, p):
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
    # fmt: on
    epsln = 1.0e-40

    # Precompute some commonly used terms
    t2 = t * t

    # Rational function for density
    num = (
        a0
        + t * (a1 + t * (a2 + a3 * t))
        + s * (a4 + a5 * t + a6 * s)
        + p * (a7 + a8 * t2 + a9 * s + p * (a10 + a11 * t2))
    )

    den = 1.0 / (
        b0
        + t * (b1 + t * (b2 + t * (b3 + t * b4)))
        + s * (b5 + t * (b6 + b7 * t2) + np.sqrt(s) * (b8 + b9 * t2))
        + p * (b10 + p * t * (b11 * t2 + b12 * p))
        + epsln
    )

    return num * den


def rho_s_t(s, t, p):
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
    # fmt: on

    epsln = 1.0e-40

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
    # Taking the partial derivative w.r.t. S gives
    #   rho_s = (num_S - num * den_S / den ) / den
    # and similarly for rho_t

    num_s = a4 + a5 * t + 2.0 * a6 * s + p * a9
    num_t = (
        a1 + t * (2.0 * a2 + 3.0 * a3 * t) + a5 * s + 2.0 * a8 * pt + 2.0 * a11 * p * pt
    )

    den_s = b5 + t * (b6 + b7 * t2) + sp5 * (1.5 * b8 + 1.5 * b9 * t2)
    den_t = (
        b1
        + t * (2.0 * b2 + t * (3.0 * b3 + 4.0 * b4 * t))
        + s * (b6 + 3.0 * b7 * t2 + 2.0 * b9 * sp5 * t)
        + 3.0 * b11 * pt * pt
        + b12 * p ** 3
    )

    rho_s = (num_s - num * den_s * inv_den) * inv_den
    rho_t = (num_t - num * den_t * inv_den) * inv_den

    return rho_s, rho_t


def _check_rho_vals():
    # Checkval from Jackett et al (2006) Appendix B:
    assert np.isclose(rho(35, 25, 2000), 1031.65056056576, rtol=0, atol=1e-11)
    assert np.isclose(rho(20, 20, 1000), 1017.72886801964, rtol=0, atol=1e-11)
    assert np.isclose(rho(40, 12, 8000), 1062.95279820631, rtol=0, atol=1e-11)


def _check_rho_derivs():
    # Check rho_s_t by centred differences
    s, t, p = (35.0, 25.0, 2000.0)
    ds, dt, dp = (1e-4, 1e-4, 1e-1)

    rho_s_centred = (rho(s + ds, t, p) - rho(s - ds, t, p)) / (2.0 * ds)
    rho_t_centred = (rho(s, t + dt, p) - rho(s, t - dt, p)) / (2.0 * dt)
    rho_p_centred = (rho(s, t, p + dp) - rho(s, t, p - dp)) / (2.0 * dp)

    rho_s, rho_t = rho_s_t(s, t, p)
    # rho_s - rho_s_centred  # 3.96301880201122e-11
    # rho_t - rho_t_centred  # -1.3336413084985566e-09
    assert np.isclose(rho_s, rho_s_centred, rtol=0, atol=1e-8)
    assert np.isclose(rho_t, rho_t_centred, rtol=0, atol=1e-8)
