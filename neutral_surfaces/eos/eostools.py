import functools
import numba

from .densjmd95 import rho as rho_jmd95
from .densjmd95 import rho_s_t as rho_s_t_jmd95

from .gsw import rho as rho_gsw
from .gsw import rho_s_t as rho_s_t_gsw


def _make_eos(eos, eos_dict, grav=None, rho_c=None):
    # Convert eos as a string into a function, and make Boussinesq if needed

    if isinstance(eos, str) and eos in eos_dict.keys():
        eos = eos_dict[eos]

    if not callable(eos):
        raise TypeError(
            f'`eos` must be a function, "gsw", or "jmd95"; found eos = {eos}'
        )

    if grav != None and rho_c != None:
        eos = make_eos_bsq(eos, grav, rho_c)

    return eos


def make_eos(eos, grav=None, rho_c=None):
    eos_dict = {"jmd95": rho_jmd95, "gsw": rho_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c)


def make_eos_s_t(eos, grav=None, rho_c=None):
    eos_dict = {"jmd95": rho_s_t_jmd95, "gsw": rho_s_t_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c)


@functools.lru_cache(maxsize=10)
def make_eos_bsq(eos, grav, rho_c):
    z_to_p = 1e-4 * grav * rho_c

    @numba.njit
    def eos_bsq(s, t, z):
        return eos(s, t, z * z_to_p)

    return eos_bsq
