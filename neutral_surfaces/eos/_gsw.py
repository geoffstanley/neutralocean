import ctypes
import numba
import gsw


# The following shows how we can access the C library scalar functions that
# are used by GSW-Python, since it is much faster for our jit functions to
# go straight to them rather than going via the Python ufunc wrappers.
dllname = gsw._gsw_ufuncs.__file__
gswlib = ctypes.cdll.LoadLibrary(dllname)
rho_gsw_ctypes = gswlib.gsw_rho  # In-situ density.
rho_gsw_ctypes.restype = ctypes.c_double
rho_gsw_ctypes.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)


# Wrapping the ctypes function with a jit reduces call overhead and makes it
# hashable, which is required for the caching we do via the lru_cache
# decorator.
@numba.njit
def rho_gsw(s, t, p):
    return rho_gsw_ctypes(s, t, p)


# For the S and T derivatives, we only call these on the entire surface,
# so the Python ufunc wrappers are okay.
# Would be nicer if we didn't have to compute the pressure derivative.
def rho_s_t_gsw(s, t, p):
    rs, rt, _ = gsw.rho_first_derivatives(s, t, p)
    return rs, rt
