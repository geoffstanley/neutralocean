import numpy as np
import pytest

from neutralocean.eos.tools import vectorize_eos
from neutralocean.eos import jmd95, jmdfwg06, gsw, polyTEOS10bsq

checkval_jmd95 = (35.5, 3.0, 3000.0, 1041.83267)
rho_jmd95_ufunc = vectorize_eos(jmd95.rho)

# Check Values from Jackett and McDougall (1995) Appendix, p. 388
# and Jackett et al (2006) Appendix A, p. 1723
@pytest.mark.parametrize(
    "eos,checkval,decimals",
    [
        (jmd95.rho, checkval_jmd95, 5),
        (jmdfwg06.rho, (35.0, 25.0, 2000.0, 1031.65056056576), 11),
        (jmdfwg06.rho, (20.0, 20.0, 1000.0, 1017.72886801964), 11),
        (jmdfwg06.rho, (40.0, 12.0, 8000.0, 1062.95279820631), 11),
        (gsw.specvol, (35.0, 25.0, 2000.0, 9.694293111803510e-04), 18),
        (polyTEOS10bsq.rho_vert, (1000.0, 4.59763035), 8),
        (polyTEOS10bsq.rho_horiz, (30.0, 10.0, 1000.0, 1022.85377), 5),
    ],
)
def test_checkval(eos, checkval, decimals):
    assert np.round(eos(*checkval[:-1]), decimals=decimals) == checkval[-1]


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
    "eos,eos_s_t,eos_p",
    [
        (jmd95.rho, jmd95.rho_s_t, jmd95.rho_p),
        (jmdfwg06.rho, jmdfwg06.rho_s_t, jmdfwg06.rho_p),
        (gsw.specvol, gsw.specvol_s_t, gsw.specvol_p),
        (polyTEOS10bsq.rho, polyTEOS10bsq.rho_s_t, polyTEOS10bsq.rho_z)
    ],
)
def test_eos_derivs(eos, eos_s_t, eos_p):
    """Check first partial derivatives by centred differences

    Parameters
    ----------
    eos : function
        Function of (S, T, P) for the Equation of State.
        Can give density or specific volume.

    eos_s_t : function
        Function of (S, T, P) returning a tuple of length two, giving the
        partial derivatives of `eos` with respect to `S` and `T`.

    eos_p : function
        Function of (S, T, P) returning the partial derivatives of `eos` with
        respect to `P`.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
        If the results of `eos_s_t` and `eos_p` disagree considerably with
        an approximation of partial derivatives calculated by evaluating 
        `eos` using centred finite differences.

    """

    s, t, p = (35.0, 25.0, 2000.0)
    ds, dt, dp = (1e-4, 1e-4, 1e-1)

    rs_centred = (eos(s + ds, t, p) - eos(s - ds, t, p)) / (2.0 * ds)
    rt_centred = (eos(s, t + dt, p) - eos(s, t - dt, p)) / (2.0 * dt)
    rp_centred = (eos(s, t, p + dp) - eos(s, t, p - dp)) / (2.0 * dp)

    rs, rt = eos_s_t(s, t, p)
    rp = eos_p(s, t, p)

    assert np.isclose(rs_centred, rs, atol=0, rtol=1e-8)
    assert np.isclose(rt_centred, rt, atol=0, rtol=1e-8)
    assert np.isclose(rp_centred, rp, atol=0, rtol=1e-8)
