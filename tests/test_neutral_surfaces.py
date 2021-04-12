import pytest
import numpy as np
from neutral_surfaces._neutral_surfaces import pot_dens_surf
from neutral_surfaces._densjmd95 import rho_ufunc


# Copied from test_vertsolve; consolidate later.
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


# Modified from test_vertsolve.
@pytest.mark.parametrize("target", [(3, 2, 1500.0), 1026])
def test_pot_dens_surf(target):
    shape = (4, 3, 50)
    s, t, p = make_simple_stp(shape)
    # All nans for one profile.
    s[0, 0] = t[0, 0] = np.nan
    # Raise the bottom of two more.
    s[1, 0, 45:] = t[1, 0, 45:] = np.nan
    s[2, 0, 5:] = t[2, 0, 5:] = np.nan
    # One with only one level left.
    s[3, 0, 1:] = t[3, 0, 1:] = np.nan
    p_ref = 0.0
    ss, tt, pp = pot_dens_surf(s, t, p, p_ref, target, tol=1e-8)
    for ind in ((0, 0), (3, 0)):
        assert np.isnan([ss[ind], tt[ind], pp[ind]]).all()
    rho_found = np.ma.masked_invalid(rho_ufunc(ss, tt, p_ref))
    if not isinstance(target, tuple):
        assert np.ma.allclose(rho_found, target)
    assert rho_found.size - rho_found.count() == 3
