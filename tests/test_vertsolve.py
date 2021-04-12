import itertools

import pytest

import numpy as np
from neutral_surfaces._vertsolve import process_arrays, func_zero, vertsolve1, vertsolve
from neutral_surfaces._densjmd95 import rho, rho_ufunc


def make_simple_stp(shape, p_axis=-1, p_is_1d=True):
    s = np.empty(shape, dtype=float)
    t = np.empty(shape, dtype=float)
    nlevels = shape[p_axis]
    p1 = (np.linspace(0, 1, nlevels) ** 1.5) * 5000
    # Both salinity and temperature cause increase in potential density with
    # depth, for convenience in making a sanity check, below.
    s1 = np.linspace(34, 35, nlevels)
    t1 = np.linspace(25, 2, nlevels)
    ind = [np.newaxis] * s.ndim
    ind[p_axis] = slice(None)
    ind = tuple(ind)
    if p_is_1d:
        p = p1
    else:
        p = np.empty(shape, dtype=float)
        p[:] = p1[ind]
    s[:] = s1[ind]
    t[:] = t1[ind]
    return s, t, p


arg_combos = itertools.product([0, 1, 2], [True, False])


@pytest.mark.parametrize("p_axis,p_1d", arg_combos)
def test_process_arrays(p_axis, p_1d):
    shape = (3, 4, 5)
    s, t, p = make_simple_stp(shape, p_axis, p_is_1d=p_1d)
    ngood, stp_args = process_arrays(s, t, p, axis=p_axis)
    nlevels = stp_args[0].shape[-1]
    assert np.all(ngood == nlevels)
    newshape = stp_args[0].shape
    assert ngood.shape == newshape[:-1]
    for arr in stp_args[1::2]:  # s, t
        assert arr.shape == newshape
    for arr in stp_args[2::2]:  # interp coefficients
        assert arr.shape[-1] == newshape[-1] - 1
    for arr in stp_args:
        assert arr.flags.c_contiguous


def test_func_zero():
    shape = (50,)
    s, t, p = make_simple_stp(shape)
    ngood, stp_args = process_arrays(s, t, p)
    # Setting p_ref and d0 to 0 makes the function return the potential density.
    rho_upper = func_zero(p[25], 0.0, 0.0, stp_args)
    rho_lower = func_zero(p[26], 0.0, 0.0, stp_args)
    rho_mid = func_zero((0.5 * (p[25] + p[26])), 0.0, 0.0, stp_args)
    assert rho_lower > rho_mid > rho_upper
    # Interpolating the density is almost the same as interpolating S and T.
    rho_interp = 0.5 * (rho_upper + rho_lower)
    assert abs(rho_mid - rho_interp) < 1e6 * (rho_lower - rho_upper)


def test_vertsolve1():
    shape = (50,)
    s, t, p = make_simple_stp(shape)
    ngood, stp_args = process_arrays(s, t, p)
    d0 = 1026.0
    p_ref = 0.0
    p_start = 1500.0
    ss, tt, pp = vertsolve1(p_start, p_ref, d0, stp_args, 1e-8)
    rho_found = rho(ss, tt, p_ref)
    assert abs(rho_found - d0) < 1e-8


def test_vertsolve():
    shape = (3, 4, 50)
    s, t, p = make_simple_stp(shape)
    ngood, stp_args = process_arrays(s, t, p)
    d0 = 1026.0 * np.ones((3, 4), dtype=float)
    p_ref = np.zeros((3, 4), dtype=float)
    p_start = 1500.0 * np.ones((3, 4), dtype=float)
    ss, tt, pp = vertsolve(p_start, p_ref, d0, ngood, stp_args, 1e-8)
    rho_found = rho_ufunc(ss, tt, p_ref)
    assert np.all(np.abs(rho_found - d0) < 1e-8)


def test_vertsolve_with_nans():
    shape = (4, 3, 50)
    s, t, p = make_simple_stp(shape)
    # All nans for one profile.
    s[0, 0] = t[0, 0] = np.nan
    # Raise the bottom of two more.
    s[1, 0, 45:] = t[1, 0, 45:] = np.nan
    s[2, 0, 5:] = t[2, 0, 5:] = np.nan
    # One with only one level left.
    s[3, 0, 1:] = t[3, 0, 1:] = np.nan
    ngood, stp_args = process_arrays(s, t, p)
    shape = ngood.shape
    p_start = np.broadcast_to(1500.0, shape)
    p_ref = np.broadcast_to(0.0, shape)
    d0 = np.broadcast_to(1026.0, shape)
    ss, tt, pp = vertsolve(p_start, p_ref, d0, ngood, stp_args, 1e-8)
    for ind in ((0, 0), (3, 0)):
        assert np.isnan([ss[ind], tt[ind], pp[ind]]).all()
    rho_found = np.ma.masked_invalid(rho_ufunc(ss, tt, p_ref))
    assert np.ma.allclose(rho_found, d0)
    assert rho_found.size - rho_found.count() == 3
