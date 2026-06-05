import numpy as np

import neutralocean as no
from neutralocean import mixed_layer, ntp, stability, traj


def test_load_eos_alias_matches_gsw_official():
    eos_official = no.load_eos("gsw_official")
    eos_alias = no.load_eos("gswc")

    s, t, p = (35.0, 25.0, 2000.0)
    assert np.isclose(eos_official(s, t, p), eos_alias(s, t, p), rtol=0.0, atol=0.0)


def test_module_default_eos_backends_are_density_form():
    s, t, p = (35.0, 25.0, 2000.0)
    eos_official = no.load_eos("gsw_official")
    eos_official_s_t = no.load_eos("gsw_official", "_s_t")

    assert np.isclose(mixed_layer.eos_(s, t, p), eos_official(s, t, p))
    assert np.isclose(stability.eos_(s, t, p), eos_official(s, t, p))
    assert np.isclose(traj.eos_(s, t, p), eos_official(s, t, p))

    rs0, rt0 = ntp.eos_s_t_(s, t, p)
    rs1, rt1 = eos_official_s_t(s, t, p)
    assert np.isclose(rs0, rs1)
    assert np.isclose(rt0, rt1)
