"""Library of simple functions for neutral_surfaces"""

import numpy as np
import numba
import xarray as xr

from .eos import make_eos, make_eos_s_t


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


def val_bot(T, n_good):
    """Evaluate tracer at the bottom ocean grid cell

    Parameters
    ----------
    T : ndarray
        Input tracer data like salinity, temperature, pressure, or depth.

    n_good : ndarray, Default None
        Number of good (non-NaN) elements in each 1D array making up T along its
        last dimension, e.g. each water column if T is arranged with depth as
        its last dimension.  This can be calculated as `n_good = find_first_nan(x)`
        where `x` is the salinity or temperature --- namely a variable that has
        data for each water column.

    Returns
    -------
    T_bot : ndarray
        The input `T` evaluated at the bottommost ocean grid cell.

    Examples
    --------
    # Evaluate temperature, having data in each water column, at the bottom grid cell
    >>> T = np.empty((3,2,10))  # (longitude, latitude, depth), let us say
    >>> T[..., :] = np.arange(10, 0, -1)  # decreasing along depth dim from 10 to 1
    >>> T[0,0,:] = np.nan  # make cast (0,0) be land
    >>> T[0,1,3:] = np.nan  # make cast (0,1) be only 3 ocean cells deep
    >>> n_good = find_first_nan(T)
    >>> val_bot(T, n_good)
    array([[nan,  8.], [ 1.,  1.], [ 1.,  1.]])

    # Evaluate the depth at the bottom grid cell
    >>> Z = np.linspace(0, 4500, 10)  # grid cell centre's are at depths 0, 500, 1000, ..., 4500.
    >>> val_bot(Z, n_good)  # using n_good calculated from T as above
    array([[  nan, 1000.], [4500., 4500.], [4500., 4500.]])
    """

    if T.ndim == n_good.ndim + 1:
        # if n_good[i,j] == 0, this will index T[i,j,-1] which will be nan, so T_bot[i,j] == nan.
        T_bot = np.take_along_axis(T, n_good[..., None] - 1, -1).squeeze()
    elif T.ndim == 1:
        # select the bottom element and then correct land casts where n_good is 0
        T_bot = T[n_good - 1]
        T_bot[n_good == 0] = np.nan
    else:
        raise ValueError("T must be 1 dimensional or have 1 more dimension than n_good")
    return T_bot


def xr_to_np(S):
    """Convert xarray into numpy array"""
    if hasattr(S, "values"):
        S = S.values
    return S


def _xr_in(S, drop_dim):
    # Prepare xarray container for output: like input S but without dimension
    # labelled `drop_dim`
    if isinstance(S, xr.core.dataarray.DataArray):
        if drop_dim is None:
            return xr.full_like(S, 0)
        else:
            return xr.full_like(S.isel({drop_dim: 0}).drop_vars(drop_dim), 0)
    else:
        return None


def _xr_out(s, sxr):
    # Return xarrays if inputs were xarrays
    if isinstance(sxr, xr.core.dataarray.DataArray):
        sxr.data = s
        s = sxr
    return s


def _process_vert_dim(vert_dim, S):
    """Convert `vert_dim` as a str naming a dimension in `S` or a (possibly
    negative) int into an int between 0 and S.ndim-1."""
    if isinstance(vert_dim, str) and hasattr(S, "dims"):
        try:
            vert_dim = S.dims.index(vert_dim)
        except:
            raise ValueError(f"vert_dim = {vert_dim} not found in S.dims")

    return np.mod(vert_dim, S.ndim)


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

    vert_dim = _process_vert_dim(vert_dim, S)

    # Broadcast a 1D vector for P into a 3D array like S
    if P.ndim < S.ndim:
        # First make P a 3D array with its non-singleton dimension be `vert_dim`
        P = np.reshape(P, tuple(-1 if x == vert_dim else 1 for x in range(S.ndim)))
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


def _process_wrap(wrap, s=None, diags=False):
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


def _process_eos(eos, grav=None, rho_c=None, need_s_t=False):
    # Process equation of state argument and make cache functions

    eos_s_t = None
    if isinstance(eos, str):
        if need_s_t:
            eos_s_t = make_eos_s_t(eos, grav, rho_c)
        eos = make_eos(eos, grav, rho_c)
    else:
        if need_s_t:
            if isinstance(eos, (tuple, list)) and len(eos) == 2:
                eos_s_t = eos[1]
                eos = eos[0]
            if not callable(eos) or not callable(eos_s_t):
                raise ValueError(
                    "If `eos` is not a str, expected a tuple of length two"
                    " containing an eos function and an eos_s_t function."
                )
        else:
            if isinstance(eos, (tuple, list)) and len(eos) >= 1:
                eos = eos[0]
            if not callable(eos):
                raise ValueError("If `eos` is not a str, expected a function.")
    return eos, eos_s_t
