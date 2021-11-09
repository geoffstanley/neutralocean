"""Tools for handling the Equation of State"""
import functools
import numba

from .densjmd95 import rho as rho_jmd95
from .densjmd95 import rho_s_t as rho_s_t_jmd95

from .gsw import rho as rho_gsw
from .gsw import rho_s_t as rho_s_t_gsw


def _make_eos(eos, eos_dict, grav=None, rho_c=None):
    if isinstance(eos, str) and eos in eos_dict.keys():
        eos = eos_dict[eos]

    if not callable(eos):
        raise TypeError(
            f'First input must be a function, "gsw", or "jmd95"; found {eos}'
        )

    if grav != None and rho_c != None:
        eos = make_eos_bsq(eos, grav, rho_c)

    return eos


def make_eos(eos, grav=None, rho_c=None):
    """Make an equation of state function, possibly modified for Boussinesq

    Parameters
    ----------
    eos : str or function

        If a str, can be 'gsw' to generate the TEOS-10 specific volume
        or 'jmd95' to generate the Jackett and McDougall (1995) in-situ
        density [1]_.

        If a function, should be an equation of state as a function of
        practical / Absolute salinity, potential / Conservative temperature,
        and pressure.

    grav, rho_c : float

        Gravitational acceleration [m s-2] and Boussinesq reference density
        [kg m-3]. If both are provided, the equation of state is modified as
        appropriate for the Boussinesq approximation, in which the third
        argument is depth, not pressure. Specifically, a depth `z` is
        converted to ``1e-4 * grav * rho_c * z``, which is the hydrostatic
        pressure [dbar] at depth `z` caused by a water column of density
        `rho_c` under gravity `grav`.

    Returns
    -------
    eos : function

        The desired equation of state.

    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388
    """
    eos_dict = {"jmd95": rho_jmd95, "gsw": rho_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c)


def make_eos_s_t(eos, grav=None, rho_c=None):
    """Make a function for the partial S and T derivatives of an equation of state

    Parameters
    ----------
    eos : str or function
        See `make_eos`

    grav, rho_c : float
        See `make_eos`

    Returns
    -------
    eos_s_t : function

        Function returning two outputs, namely the partial derivatives with
        respect to its first two arguments (practical / Absolute salinity and
        potential / Conservative temperature) of the desired equation of
        state.

    .. [1] Jackett and McDougall, 1995, JAOT 12(4), pp. 381-388
    """
    eos_dict = {"jmd95": rho_s_t_jmd95, "gsw": rho_s_t_gsw}
    return _make_eos(eos, eos_dict, grav, rho_c)


@functools.lru_cache(maxsize=10)
def make_eos_bsq(eos, grav, rho_c):
    """Make a Boussinesq version of a given equation of state

    Parameters
    ----------
    eos : function
        Equation of state with inputs of salinity, temperature, pressure
    grav : float
        Gravitational acceleration [m s-2]
    rho_c : float
        Boussinesq reference density [kg m-3]

    Returns
    -------
    eos_bsq : function
        Boussinesq version of `eos`, with inputs of salinity, temperature, depth
    """
    z_to_p = 1e-4 * grav * rho_c
    @numba.njit
    def eos_bsq(s, t, z):
        return eos(s, t, z * z_to_p)
    return eos_bsq


def vectorize_eos(eos):
    """Convert an eos function that takes scalar inputs into one taking arrays
    
    Parameters
    ----------
    eos : function
        Any function taking three scalar inputs and returning one scalar output,
        such as the equation of state.  Note this does not work for functions
        returning multiple outputs (i.e. a tuple), such as a function returning
        the partial derivatives of the equation of state.

    Returns
    -------
    eos_vec : function
        A @numba.vectorize'd version of `eos`, which can take array inputs and
        returns one array output.  The array inputs' shape need not match 
        exactly, but must be broadcastable to each other.
    """
    @numba.vectorize
    def eos_vec(s, t, p):
        return eos(s, t, p)
    return eos_vec