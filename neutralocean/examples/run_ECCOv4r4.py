# %% Imports

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# from neutralocean.grids import neighbour4_tiled_rectilinear, xgcm_faceconns_convert

import numpy as np
import xarray as xr

# import ecco_v4_py as ecco

folder_ecco4 = "/home/stanley/work/data/ECCOv4r4/"
file_grid = folder_ecco4 + "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"

file_salt = folder_ecco4 + "nctiles_daily/SALT/SALT_2002_12_23.nc"
file_theta = folder_ecco4 + "nctiles_daily/THETA/THETA_2002_12_23.nc"

# tuple(x for x in foo.dims if x != "k")

ds = xr.open_dataset(file_salt)
S = ds.SALT.squeeze().load()
ds.close()

ds = xr.open_dataset(file_theta)
T = ds.THETA.squeeze().load()
ds.close()

# Reorder dimensions to ensure individual water columns are float64 and contiguous in memory
dims = (*tuple(x for x in S.dims if x != "k"), "k")
S, T = (x.transpose(*dims).astype(np.float64, order="C") for x in (S, T))

# This DataSet appears to be lacking info about missing values. Use S == 0.
bad = S.values == 0.0
S.values[bad] = np.nan
T.values[bad] = np.nan
del bad

n = ds.nx

g = xr.open_dataset(file_grid)
Z = -np.float64(g.Z.load().values)  # Depth vector (note positive and increasing down)


# define the connectivity between faces for the ECCOv4 LLC grid:
# fmt: off
face_connections = {'tile': {
    0: {'X':  ((12, 'Y', False), (3, 'X', False)),
        'Y':  (None,             (1, 'Y', False))},
    1: {'X':  ((11, 'Y', False), (4, 'X', False)),
        'Y':  ((0, 'Y', False),  (2, 'Y', False))},
    2: {'X':  ((10, 'Y', False), (5, 'X', False)),
        'Y':  ((1, 'Y', False),  (6, 'X', False))},
    3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
        'Y':  (None,             (4, 'Y', False))},
    4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
        'Y':  ((3, 'Y', False),  (5, 'Y', False))},
    5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
        'Y':  ((4, 'Y', False),  (6, 'Y', False))},
    6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
        'Y':  ((5, 'Y', False),  (10, 'X', False))},
    7: {'X':  ((6, 'X', False),  (8, 'X', False)),
        'Y':  ((5, 'X', False),  (10, 'Y', False))},
    8: {'X':  ((7, 'X', False),  (9, 'X', False)),
        'Y':  ((4, 'X', False),  (11, 'Y', False))},
    9: {'X':  ((8, 'X', False),  None),
        'Y':  ((3, 'X', False),  (12, 'Y', False))},
    10:{'X': ((6, 'Y', False),  (11, 'X', False)),
        'Y': ((7, 'Y', False),  (2, 'X', False))},
    11:{'X': ((10, 'X', False), (12, 'X', False)),
        'Y': ((8, 'Y', False),  (1, 'X', False))},
    12:{'X': ((11, 'X', False), None),
        'Y': ((9, 'Y', False),  (0, 'X', False))}}}
# fmt: on
nf = len(next(iter(face_connections.values())))  # len(faces_connections only value)
N = n * n * nf  # number of nodes

# F = xgcm_faceconns_convert(face_connections)
# A4 = neighbour4_tiled_rectilinear(F, n)

from neutralocean.grids import edgescompact_from_faceconns, adj_to_edges
from neutralocean.graph import (
    max_deg_from_edges,
    edges_to_adjnodes,
    edgescompact_to_adjnodes,
)

adj = edgescompact_from_faceconns(face_connections, n)
edges = adj_to_edges(adj)
max_deg = 4
# max_deg = max_deg_from_edges(edges, N)
neigh = edges_to_adjnodes(edges, N, max_deg)

A4 = edgescompact_to_adjnodes(adj)

# %%
# Select pinning cast in the middle of the domain
pin_cast = (11, 0, 14)  # Near equatorial Pacific
z0 = 1500.0

# make Boussinesq version of the Jackett and McDougall (1995) equation of state
# --- which is what OCCA used --- and its partial derivatives
grav, rho_c = 9.81, 1027.5
eos = make_eos("jmd95", grav, rho_c)
eos_s_t = make_eos_s_t("jmd95", grav, rho_c)

# %%
# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf

s, t, z, _ = potential_surf(
    S.values, T.values, Z, eos="jmd95", vert_dim=-1, ref=0.0, isoval=1027.5, diags=False
)

# %%
from neutralocean.ntp import ntp_ϵ_errors

dist = np.ones(adj.size)
eps = ntp_ϵ_errors(s, t, z, eos_s_t, edges, dist)
