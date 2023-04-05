# In[Imports]

import numpy as np

from neutralocean.surface import potential_surf, anomaly_surf, omega_surf
from neutralocean.eos import make_eos_s_t
from neutralocean.ntp import ntp_epsilon_errors

# In[Create an ocean of 4 water columns]

# Make up simple Salt, Temperature, and Pressure data for 4 casts
nc = 4  # Number of casts (water columns)
nk = 20  # Number of grid points per cast
pbot = 4000.0  # Pressure at bottom grid point

P = np.linspace(0, 1, nk) ** 3 * pbot  # pressure increasing cubicly going down

S = np.linspace(34, 36, nk).reshape((1, -1))  # saltier going down
S = S + np.linspace(0, 0.9, nc).reshape((-1, 1))  # some lateral structure

T = np.linspace(14, -2, nk).reshape((1, -1))  # warmer going down
T = T + np.linspace(0, 6, nc).reshape((-1, 1))  # some lateral structure

# Arrange the 4 casts (labelled 0, 1, 2, 3) with connections as follows:
# 0
# | \
# 1--2--3
# That is, cast 0 is connected to casts 1 and 2; cast 1 is connected to casts 0
# and 2; cast 2 is conncted to all casts; cast 3 is connected only to cast 2.
a = np.array([0, 0, 1, 2])  # a[i] and b[i] are a pair of adjacent casts
b = np.array([1, 2, 2, 3])
edges = (a, b)

# Invent distances `dist` between pairs of casts, roughly 100km
dist = np.array([1, 1.4, 1, 1]) * 1e5  # Units: [m]

# Invent distances `distperp` of the interfaces between pairs of casts.
# The product of `dist` and `distperp` gives an area associated to the region
# between casts, which is where the ϵ neutrality errors live. We seek to minimize
# these ϵ neutrality errors, weighted by these areas.
distperp = np.array([1, 1, 1, 1]) * 1e5  # Units: [m]

# Package the grid information into a dict, for neutralocean.
grid = {"edges": edges, "dist": dist, "distperp": distperp}

"""
# Note, more complex examples may have the water column adjacency specified by
# a graph structure, represented as a (sparse) matrix.  For example (with a
# dense matrix here given the smallness of this example), suppose we have two
# graphs, each with the same sparsity structure, one storing the distances
# between adjacent water columns and the other storing the length of the
# interfaces between adjacent water columns.
graph_dist = np.zeros((nc, nc))
graph_dist[0, :] = [0.0, 1.0, 1.4, 0.0]
graph_dist[1, :] = [1.0, 0.0, 1.0, 0.0]
graph_dist[2, :] = [1.4, 1.0, 0.0, 2.0]
graph_dist[3, :] = [0.0, 0.0, 2.0, 0.0]
graph_dist *= 1e5
graph_distperp = np.sign(graph_dist) * 1e5

# To convert these graphs into the format for the `grid` argument to
# neutralocean functions, simply call:
from neutralocean.grid.graph import build_grid
grid = build_grid({"dist": graph_dist, "distperp": graph_distperp})
"""


# In[Approx Neutral Surfaces]

# Here, ν = 1/ρ is the TEOS-10 specific volume,
#       S is Absolute Salinity,
#       Θ is Conservative Temperature,
#       S is Absolute Salinity,
#       p is pressure

# Potential specific volume surface, with given reference pressure and given isovalue.
# This finds the surface satisfying
#   ν(S, Θ, 0 dbar) = (1/1027.5) m3 / kg
s, t, p, d = potential_surf(
    S,
    T,
    P,
    grid=grid,
    eos="gsw",
    ref=0.0,
    isoval=1 / 1027.5,
)
print(
    f" ** The potential specific volume surface (referenced to {d['ref']}dbar)"
    f" with isovalue = {d['isoval']} m3 kg-1"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} m2 kg-1."
)

# In-situ specific volume anomaly, with given reference S and Θ values and given isovalue.
# This finds the surface satisfying
#   ν(S, Θ, p) - ν(34.5 g/kg, 4.0°C, p) = 0 m3 / kg
s0, t0 = 34.5, 4.0
s, t, p, d = anomaly_surf(
    S,
    T,
    P,
    grid=grid,
    eos="gsw",
    ref=(s0, t0),
    isoval=0.0,
)
print(
    f" ** The in-situ specific volume anomaly surface (referenced to {d['ref']})"
    f" with isovalue = {d['isoval']} m3 kg-1"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} m2 kg-1."
)


# omega-surface, initialized by a potential density surface, pinning the surface
# to be 1500dbar on cast 0.
s, t, p, d = omega_surf(S, T, P, grid, pin_cast=0, pin_p=1500.0, eos="gsw")
print(
    f" ** The omega-surface"
    f" initialized from a potential density surface (referenced to 1500 dbar)"
    f" intersecting the cast labelled '0' at pressure 1500 bar"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} m2 kg-1."
)

# Calculate ϵ neutrality errors on the latest surface, between all pairs of adjacent water columns
eos_s_t = make_eos_s_t("gsw")
e = ntp_epsilon_errors(s, t, p, grid, eos_s_t)
print("The ϵ neutrality errors on the ω-surface are as follows:")
for i in range(len(a)):
    print(f"  From cast {a[i]} to cast {b[i]}, ϵ = {e[i]} m2 kg-1")
print(
    "Note that the connection between casts 2 and 3 has virtually 0 neutrality "
    "error.  This is because cast 3 is ONLY connected to cast 2, so this link "
    "can be along the (discrete) neutral tangent plane joining cast 2 and 3. "
    "The ω-surface finds this."
)
