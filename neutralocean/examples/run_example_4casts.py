# In[Imports]

import numpy as np

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

# In[Create an ocean of 4 water columns]

nc = 4  # Number of water columns
nk = 20  # Number of grid points per water column
pbot = 4000.0  # Pressure at bottom grid point

P = np.linspace(0, 1, nk) ** 3 * pbot  # pressure increasing cubicly going down

S = np.linspace(34, 36, nk).reshape((1, -1))  # saltier going down
S = S + np.arange(0, 0.1 * nc, 0.1).reshape((-1, 1))  # some lateral structure

T = np.linspace(20, -2, nk).reshape((1, -1))  # warmer going down
T = T + np.arange(0, 0.1 * nc, 0.1).reshape((-1, 1))  # some lateral structure

# Arrange the 4 casts (labelled 0, 1, 2, 3) with connections as follows:
#   0
#   |\
#   1-2--3
edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3]])  # pairs of adjacent casts
dist = np.array([1, 1.4, 1, 2]) * 1e5  # 100km to 200km between casts
distperp = np.array([1, 1, 1, 1]) * 1e5  # ~100km interface between casts

# Note, more complex examples may have the water column adjacency specified by
# a graph structure, such as
# graph_dist = np.zeros((nc, nc))
# graph_dist[0, :] = [0.0, 1.0, 1.4, 0.0] * 1e5
# graph_dist[1, :] = [1.0, 0.0, 1.0, 0.0] * 1e5
# graph_dist[2, :] = [1.4, 1.0, 0.0, 2.0] * 1e5
# graph_dist[3, :] = [0.0, 0.0, 2.0, 0.0] * 1e5
# graph_distperp = graph_dist.sign() * 1e5
# To construct the edges, dist, and distperp arrays from a pair of such graphs,
# do the following:
# from neutralocean.grid.graph import graph_to_edges, graph_to_edge_data
# edges = graph_to_edges(graph_dist)
# dist = graph_to_edge_data(graph_dist)
# distperp = graph_to_edge_data(graph_distperp)


# Some methods below will pin the surface to reference pressure the reference cast
c0 = 0  # reference cast
p0 = 1500.0  # reference pressure

# In[Potential Density surfaces]

# Provide reference pressure (actually depth, in Boussinesq) and isovalue
s, t, p, d = potential_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    ref=0.0,
    isoval=1027.5,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}dbar)"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, p, d = potential_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    ref=0.0,
    pin_cast=c0,
    pin_p=p0,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}dbar)"
    f" intersecting the cast indexed by {c0} at pressure {p0}dbar"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide just the location to intersect `(pin_cast, pin_p)`.
# This takes the reference depth `ref` to match `pin_p`.
# Also use PCHIPs as the vertical interpolants.
s, t, p, d = potential_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    pin_cast=c0,
    pin_p=p0,
    interp="pchip",
)
print(
    f" ** The potential density surface (referenced to {d['ref']}dbar)"
    f" intersecting the cast at indexed by {c0} at pressure {p0}dbar"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# In[Delta surfaces]

# Provide reference salinity and potential temperature values
s0, t0 = 34.5, 4.0
s, t, p, d = anomaly_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    ref=(s0, t0),
    isoval=0.0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, p, d = anomaly_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    ref=(s0, t0),
    pin_cast=c0,
    pin_p=p0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {c0} at pressure {p0}dbar"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide just the location to intersect: depth `pin_p` on cast `pin_cast`
# This takes the reference S and T values from that location.
s, t, p, d = anomaly_surf(
    S,
    T,
    P,
    edges=edges,
    geometry=(dist, distperp),
    eos="gsw",
    pin_cast=c0,
    pin_p=p0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {c0} at pressure {p0}dbar"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# In[Omega surfaces]

# Initialize omega surface with a (locally referenced) potential density surface.
s, t, p, d = omega_surf(
    S, T, P, edges, geometry=(dist, distperp), pin_cast=c0, pin_p=p0, eos="gsw"
)
print(
    f" ** The omega-surface"
    f" initialized from a potential density surface (referenced to {p0}dbar)"
    f" intersecting the cast indexed by {c0} at pressure {p0}dbar"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS'][-1]} kg m-4"
)

# Initialize omega surface with a (locally referenced) in-situ density anomaly surface.
# Use PCHIP interpolation rather than the default, linear interpolation.
# Remove the mixed layer, calculated internally according to the given parameters --
#   see `mixed_layer` for details on these parameters.
s, t, p, d = omega_surf(
    S,
    T,
    P,
    edges,
    geometry=(dist, distperp),
    ref=(None, None),
    pin_cast=c0,
    pin_p=p0,
    eos="gsw",
    interp="pchip",
    p_ml={"bottle_index": 1, "ref_p": 0.0},
)
print(
    f" ** The omega-surface"
    f" initialized from an in-situ density anomaly surface (referenced locally to cast {c0} at {p0}dbar)"
    f" intersecting the cast indexed by {c0} at pressure {p0}dbar"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS'][-1]} kg m-4"
)


# In[Show more...]
# TODO: Show that ϵ is zero for the link from 2 to 3.
# Show neutral trajectory?
