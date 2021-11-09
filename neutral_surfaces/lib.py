"""Library of simple functions for neutral_surfaces"""

import numpy as np
import numba
import xarray as xr

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t


@numba.njit
def find_first_nan(a):
    """The index to the first NaN along the last axis

    Parameters
    ----------
    a : ndarray
        Input array possibly containing some NaN elements

    Returns
    -------
    k : ndarray of int
        The index to the first NaN along each 1D array making up `a`, as in the
        following example with `a` being 3D.
        If all ``a[i,j,:]`` are NaN, then ``k[i,j] = 0``.
        If all ``a[i,j,:]`` are not NaN, then ``k[i,j] = a.shape[-1]``.
        Otherwise, ``K = k[i,j]`` is the smallest int such that ``a[i,j,K-1]``
        is not NaN, but ``a[i,j,K]`` is NaN.
    """
    nk = a.shape[-1]
    k = np.full(a.shape[:-1], nk, dtype=np.int64)
    for n in np.ndindex(a.shape[0:-1]):
        for i in range(nk):
            if np.isnan(a[n][i]):
                k[n] = i
                break
    return k


def xr_to_np(S):
    """Convert xarray into numpy array"""
    if hasattr(S, "values"):
        S = S.values
    return S


def _xr_in(S, T, P, vert_dim):
    # Prepare xarray containers for output
    sxr, txr, pxr = None, None, None
    if isinstance(S, xr.core.dataarray.DataArray):
        sxr = xr.full_like(S.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if isinstance(T, xr.core.dataarray.DataArray):
        txr = xr.full_like(T.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    if isinstance(P, xr.core.dataarray.DataArray):
        pxr = xr.full_like(P.isel({vert_dim: 0}).drop_vars(vert_dim), 0)
    return (sxr, txr, pxr)


def _xr_out(s, t, p, sxr, txr, pxr):
    # Return xarrays if inputs were xarrays
    if isinstance(sxr, xr.core.dataarray.DataArray):
        sxr.data = s
        s = sxr
    if isinstance(txr, xr.core.dataarray.DataArray):
        txr.data = t
        t = txr
    if isinstance(pxr, xr.core.dataarray.DataArray):
        pxr.data = p
        p = pxr
    return (s, t, p)


def _process_vert_dim(vert_dim, S):
    """Convert `vert_dim` as a str naming a dimension in `S` into an int.
    If vert_dim is not a str, or if S isn't an xarray, this just returns vert_dim."""
    if isinstance(vert_dim, str) and hasattr(S, "dims"):
        try:
            vert_dim = S.dims.index(vert_dim)
        except:
            raise ValueError(f"vert_dim = {vert_dim} not found in S.dims")
    return vert_dim


def _contiguous_casts(S, vert_dim=-1):
    """Make individual casts contiguous in memory

    Parameters
    ----------
    S : ndarray
        ocean data such as salinity, temperature, or pressure

    vert_dim : int, Default -1
        Specifies which dimension of `S` is vertical.

    Returns
    -------
    S : ndarray
        input data, possibly re-arranged to have `vert_dim` the last dimension
    """

    if S.ndim > 1 and vert_dim not in (-1, S.ndim - 1):
        S = np.moveaxis(S, vert_dim, -1)

    return np.require(S, dtype=np.float64, requirements="C")


def _process_casts(S, T, P, vert_dim):
    """Make individual casts contiguous in memory and extract numpy array from xarray"""

    vert_dim = _process_vert_dim(vert_dim, S)  # Converts from str to int if needed
    if P.ndim < S.ndim:
        P = np.broadcast_to(P, S.shape)

    S, T, P = (xr_to_np(x) for x in (S, T, P))
    S, T, P = (_contiguous_casts(x, vert_dim) for x in (S, T, P))
    return S, T, P


def _process_n_good(S, n_good=None):
    if n_good is None:
        n_good = find_first_nan(S)
    return n_good


def _interp_casts(S, T, P, interp_fn, Sppc=None, Tppc=None):
    # Compute interpolants for S and T casts (unless already provided)
    ni, nj, nk = S.shape
    if Sppc is None or Sppc.shape[0:-1] != (ni, nj, nk - 1):
        Sppc = interp_fn(P, S)
    if Tppc is None or Tppc.shape[0:-1] != (ni, nj, nk - 1):
        Tppc = interp_fn(P, T)
    return Sppc, Tppc


def _process_wrap(wrap, diags=True, s=None):
    """Convert to a tuple of `int`s specifying which horizontal dimensions are periodic"""

    if wrap is None:
        if diags:
            raise ValueError(
                "wrap must be given for omega surfaces, or when `diags` is True"
            )
        else:
            return wrap

    if isinstance(wrap, str):
        wrap = (wrap,)  # Convert single string to tuple
    if not isinstance(wrap, (tuple, list)):
        raise TypeError("If given, wrap must be a tuple or list or str")
    if all(isinstance(x, str) for x in wrap):
        try:
            # Convert dim names to tuple of bool
            wrap = tuple(x in wrap for x in s.dims)
        except:
            raise TypeError(
                "With wrap provided as strings, s must have a .dims attribute"
            )

    # type checking on final value
    if not (isinstance(wrap, (tuple, list)) and len(wrap) == 2):
        raise TypeError(
            "wrap must be a two element (logical) array "
            "or a string (or array of strings) referring to dimensions in xarray S"
        )
    return wrap


def _process_eos(eos, eos_s_t, grav=None, rho_c=None):
    # Process equation of state function and make cache functions
    if eos_s_t is None and isinstance(eos, str):
        eos_s_t = eos
    elif isinstance(eos, str) and isinstance(eos_s_t, str) and eos != eos_s_t:
        raise ValueError("eos and eos_s_t, if strings, must be the same string")
    eos = make_eos(eos, grav, rho_c)
    eos_s_t = make_eos_s_t(eos_s_t, grav, rho_c)
    return eos, eos_s_t


def _process_pin_cast(pin_cast, S):
    # Convert pinning cast from a coordinate representation, suitable for ``S.sel(pin_cast)``
    # when S is an xarray, into an index representation, suitable for ``S[pin_cast]``
    # when S is an ndarray.

    # DEV: There must be a better way of doing this...
    # One issue is this always rounds one way, whereas a "nearest" neighbour
    # type behaviour would be preferred, as in xr.DataArray.sel
    if isinstance(pin_cast, dict):
        return tuple(int(S.get_index(k).searchsorted(v)) for (k, v) in pin_cast.items())
    else:
        return pin_cast


def _process_args(
    S,
    T,
    P,
    vert_dim,
    pin_cast,
    wrap,
    diags,
    eos,
    eos_s_t,
    grav,
    rho_c,
    interp_fn,
    Sppc,
    Tppc,
    n_good,
):
    sxr, txr, pxr = _xr_in(S, T, P, vert_dim)  # must call before _process_casts
    pin_cast = _process_pin_cast(pin_cast, S)  # must call before _process_casts
    wrap = _process_wrap(wrap, diags, sxr)  # must call before _process_casts
    eos, eos_s_t = _process_eos(eos, eos_s_t, grav, rho_c)
    S, T, P = _process_casts(S, T, P, vert_dim)
    Sppc, Tppc = _interp_casts(S, T, P, interp_fn, Sppc, Tppc)  # after _process_casts
    n_good = _process_n_good(S, n_good)  # must call after _process_casts
    return S, T, P, Sppc, Tppc, n_good, pin_cast, wrap, eos, eos_s_t
