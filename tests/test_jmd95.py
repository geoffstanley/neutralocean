import numpy as np

from neutral_surfaces._densjmd95 import rho, rho_ufunc

# Check values from JMD95 p. 388:
checkvals = (35.5, 3.0, 3000.0, 1041.83267)


def test_jmd95_scalar():
    assert np.round(rho(*checkvals[:-1]), decimals=5) == checkvals[-1]


def test_rho_ufunc_scalar():
    res = rho_ufunc(*checkvals[:-1])
    assert np.round(res, decimals=5) == checkvals[-1]


def test_rho_ufunc_array():
    # Smoketest: broadcasting
    s = np.ones((4, 5), dtype=float) * checkvals[0]
    t = np.ones((5,), dtype=float) * checkvals[1]
    p = checkvals[2]
    res = rho_ufunc(s, t, p)
    assert res.shape == s.shape
    assert np.allclose(res, checkvals[-1], rtol=0, atol=1e-5)
