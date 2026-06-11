"""
Density of seawater using the official TEOS-10 Gibbs SeaWater toolbox.

This module wraps the external `gsw` package and exposes functions compatible
with `neutralocean.eos.tools.load_eos`.

Functions
---------
rho
    In-situ density from Absolute Salinity, Conservative Temperature, pressure.
rho_s_t
    Partial derivatives of `rho` with respect to salinity and temperature.
rho_p
    Partial derivative of `rho` with respect to pressure.
"""

import ctypes
import numba as nb

try:
    import gsw
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The optional dependency 'gsw' is required for neutralocean.eos.gsw_official. "
        "Install it with `pip install gsw`."
    ) from exc


# Access the scalar C entry point used by gsw-python for fast njit-compatible calls.
dllname = gsw._gsw_ufuncs.__file__
gswlib = ctypes.cdll.LoadLibrary(dllname)
rho_gsw_ctypes = gswlib.gsw_rho
rho_gsw_ctypes.restype = ctypes.c_double
rho_gsw_ctypes.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho(s, t, p):
    """The TEOS-10 GSW equation of state for in-situ density."""
    return rho_gsw_ctypes(s, t, p)


@nb.njit(nb.types.UniTuple(nb.f8, 2)(nb.f8, nb.f8, nb.f8))
def rho_s_t(s, t, p):
    """Partial derivatives of in-situ density wrt salinity and temperature."""
    ds = 1e-4
    dt = 1e-4
    rs = (rho(s + ds, t, p) - rho(s - ds, t, p)) / (2.0 * ds)
    rt = (rho(s, t + dt, p) - rho(s, t - dt, p)) / (2.0 * dt)
    return rs, rt


@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def rho_p(s, t, p):
    """Partial derivative of in-situ density wrt pressure [kg m-3 dbar-1]."""
    dp = 1e-1
    return (rho(s, t, p + dp) - rho(s, t, p - dp)) / (2.0 * dp)
