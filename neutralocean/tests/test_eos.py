import numpy as np
import pytest

from neutralocean.eos.tools import vectorize_eos

from neutralocean.eos.jmd95 import rho as rho_jmd95
from neutralocean.eos.jmd95 import rho_s_t as rho_s_t_jmd95
from neutralocean.eos.jmd95 import rho_p as rho_p_jmd95

from neutralocean.eos.jmdfwg06 import rho as rho_jmdfwg06
from neutralocean.eos.jmdfwg06 import rho_s_t as rho_s_t_jmdfwg06
from neutralocean.eos.jmdfwg06 import rho_p as rho_p_jmdfwg06

from neutralocean.eos.gsw import rho as rho_gsw
from neutralocean.eos.gsw import rho_s_t as rho_s_t_gsw
from neutralocean.eos.gsw import rho_p as rho_p_gsw

checkval_jmd95 = (35.5, 3.0, 3000.0, 1041.83267)
rho_jmd95_ufunc = vectorize_eos(rho_jmd95)

# Check Values from Jackett and McDougall (1995) Appendix, p. 388
# and Jackett et al (2006) Appendix A, p. 1723
@pytest.mark.parametrize(
    "rho,checkval,decimals",
    [
        (rho_jmd95, checkval_jmd95, 5),
        (rho_jmdfwg06, (35.0, 25.0, 2000.0, 1031.65056056576), 11),
        (rho_jmdfwg06, (20.0, 20.0, 1000.0, 1017.72886801964), 11),
        (rho_jmdfwg06, (40.0, 12.0, 8000.0, 1062.95279820631), 11),
    ],
)
def test_checkval(rho, checkval, decimals):
    assert np.round(rho(*checkval[:-1]), decimals=decimals) == checkval[-1]


def test_jmd95_ufunc_scalar():
    res = rho_jmd95_ufunc(*checkval_jmd95[:-1])
    assert np.round(res, decimals=5) == checkval_jmd95[-1]


def test_jmd95_ufunc_array():
    # Smoketest: broadcasting
    s = np.ones((4, 5), dtype=float) * checkval_jmd95[0]
    t = np.ones((5,), dtype=float) * checkval_jmd95[1]
    p = checkval_jmd95[2]
    res = rho_jmd95_ufunc(s, t, p)
    assert res.shape == s.shape
    assert np.all(np.round(res, decimals=5) == checkval_jmd95[-1])


@pytest.mark.parametrize(
    "rho,rho_s_t,rho_p",
    [
        (rho_jmd95, rho_s_t_jmd95, rho_p_jmd95),
        (rho_jmdfwg06, rho_s_t_jmdfwg06, rho_p_jmdfwg06),
        (rho_gsw, rho_s_t_gsw, rho_p_gsw),
    ],
)
def test_rho_derivs(rho, rho_s_t, rho_p):
    """Check rho_s_t and rho_p functions by centred differences

    Parameters
    ----------
    rho : function
        Equation of State, in terms of (S, T, P).  e.g. `.gsw.rho`

    rho_s_t : function
        Function of (S, T, P) returning a tuple of length two, giving the
        partial derivatives of `rho` with respect to `S` and `T`.  e.g. `.gsw.rho_s_t`

    rho_p : function
        Function of (S, T, P) returning the partial derivatives of `rho` with
        respect to `P`.  e.g. `.gsw.rho_p`

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
        If the results of `rho_s_t` and `rho_p` at a (hardcoded) checkvalue
        disagree considerably with an approximation of partial derivatives
        calculated by evaluating `rho` using centred finite differences.

    """

    s, t, p = (35.0, 25.0, 2000.0)
    ds, dt, dp = (1e-4, 1e-4, 1e-1)

    rs_centred = (rho(s + ds, t, p) - rho(s - ds, t, p)) / (2.0 * ds)
    rt_centred = (rho(s, t + dt, p) - rho(s, t - dt, p)) / (2.0 * dt)
    rp_centred = (rho(s, t, p + dp) - rho(s, t, p - dp)) / (2.0 * dp)

    rs, rt = rho_s_t(s, t, p)
    rp = rho_p(s, t, p)

    assert np.isclose(rs, rs_centred, rtol=0, atol=1e-8)
    assert np.isclose(rt, rt_centred, rtol=0, atol=1e-8)
    assert np.isclose(rp, rp_centred, rtol=0, atol=1e-11)
