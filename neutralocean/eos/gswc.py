"""
Density of Sea Water using the TEOS-10 Gibbs Seawater [1]_ function

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

.. [1] McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and 
the Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127, 
ISBN 978-0-646-55621-5. 
"""

import ctypes
import numba as nb
import gsw


# The following shows how we can access the C library scalar functions that
# are used by GSW-Python, since it is much faster for our jit functions to
# go straight to them rather than going via the Python ufunc wrappers.
dllname = gsw._gsw_ufuncs.__file__
gswlib = ctypes.cdll.LoadLibrary(dllname)
rho_gsw_ctypes = gswlib.gsw_rho  # In-situ density.
rho_gsw_ctypes.restype = ctypes.c_double
rho_gsw_ctypes.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)


# Wrapping the ctypes function with njit reduces call overhead and makes it
# hashable, which is required for the caching we do via the lru_cache
# decorator.
@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho(s, t, p):
    """
    The TEOS-10 GSW equation of state for in-situ density.

    Parameters
    ----------
    s : float
        Absolute Salinity [g kg-1]
    t : float
        Conservative Temperature [degC]
    p : float
        Pressure [dbar]

    Returns
    -------
    rho : float
        In-situ density [kg m-3]

    """
    return rho_gsw_ctypes(s, t, p)


# Can't use nb.njit on functions wrapping gsw.rho_first_derivatives:
# "Unknown attribute 'rho_first_derivatives' of type Module(<module 'gsw' from ..."
# For the partial derivatives, we only call these on the entire surface,
# so the Python ufunc wrappers are okay.
def rho_s_t(s, t, p):
    """
    The partial derivatives of the TEOS-10 GSW equation of state for in-situ
    density with respect to Absolute Salinity and Conservative Temperature

    Parameters
    ----------
    s, t, p : float or ndarray
        See `rho`

    Returns
    -------
    rho_s : float or ndarray
        Partial derivative of in-situ density with respect to Absolute
        Salinity [kg m-3 (g/kg)-1]

    rho_t : float or ndarray
        Partial derivative of in-situ density with respect to Conservative
        Temperature [kg m-3 degC-1]

    """
    # Would be better if we didn't have to compute the pressure derivative.
    rs, rt, _ = gsw.rho_first_derivatives(s, t, p)
    return rs, rt


def rho_p(s, t, p):
    """
    The partial derivatives of the TEOS-10 GSW equation of state for in-situ
    density with respect to pressure

    Parameters
    ----------
    s, t, p : float or ndarray
        See `rho`

    Returns
    -------
    rho_p : float or ndarray
        Partial derivative of in-situ density with respect to pressure [kg m-3 dbar-1]

    """
    # Would be better if we didn't have to compute the S and T derivative.
    _, _, rp = gsw.rho_first_derivatives(s, t, p)
    return rp * 1e4  # convert from [kg m-3 Pa-1] to [kg m-3 dbar-1]
