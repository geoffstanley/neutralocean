import numpy as np

import neutralocean as no
from neutralocean import mixed_layer, ntp, stability, traj


def test_module_default_eos_backends_are_bundled_gsw():
    s, t, p = (35.0, 25.0, 2000.0)
    eos_gsw = no.load_eos("gsw")
    eos_gsw_s_t = no.load_eos("gsw", "_s_t")

    assert np.isclose(mixed_layer.eos_(s, t, p), eos_gsw(s, t, p))
    assert np.isclose(stability.eos_(s, t, p), eos_gsw(s, t, p))
    assert np.isclose(traj.eos_(s, t, p), eos_gsw(s, t, p))

    rs0, rt0 = ntp.eos_s_t_(s, t, p)
    rs1, rt1 = eos_gsw_s_t(s, t, p)
    assert np.isclose(rs0, rs1)
    assert np.isclose(rt0, rt1)
