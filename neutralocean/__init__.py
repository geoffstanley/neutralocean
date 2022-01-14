"""
Calculate approximately neutral surfaces in the ocean.
"""
from neutralocean.eos.tools import make_eos, make_eos_s_t, make_eos_p
from neutralocean.surface.trad import potential_surf, anomaly_surf
from neutralocean.surface.omega import omega_surf
