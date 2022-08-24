# %% Imports

import numpy as np
import xarray as xr

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, omega_surf

from neutralocean.grid.xgcm import build_edges_and_geometry

# %% Load data
folder_ecco4 = "/home/stanley/work/data/ECCOv4r4/"  # edit as needed...
file_grid = folder_ecco4 + "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"

file_salt = folder_ecco4 + "nctiles_daily/SALT/SALT_2002_12_23.nc"
file_theta = folder_ecco4 + "nctiles_daily/THETA/THETA_2002_12_23.nc"

# Load hydrographic data
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

# size of each horizontal dimension in each square tile
n = ds.nx

g = xr.open_dataset(file_grid)

# Create depth xarray.
Z = -g.Z  # Make Z > 0 and increasing down
Z.attrs.update({"positive": "down"})  # Update attrs to match.

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

# Build list of adjacent water columns and distances between those water column pairs
dims = g.Depth.dims  # ('tile', 'j', 'i')
edges, dist, distperp = build_edges_and_geometry(
    n, face_connections, dims, g.dxC, g.dyC, g.dxG, g.dyG, "left", "left"
)


# Make Boussinesq version of the Jackett and McDougall (1995) equation of state
#  and its partial derivatives.
# TODO: double check that is what ECCOv4r4 used
grav, rho_c = 9.81, 1027.5
eos = make_eos("jmd95", grav, rho_c)
eos_s_t = make_eos_s_t("jmd95", grav, rho_c)

# %% Select pinning cast and pinning depth
# pin_cast = (11, 0, 14)  # hardcoded cast at (-127.5, 0.2)

x0, y0 = (-172, -4)
pin_cast = np.unravel_index(
    ((g.XC.values - x0) ** 2 + (g.YC.values - y0) ** 2).argmin(), g.XC.shape
)
z0 = 1500.0

# %% Build approximately neutral surfaces!

# Build potential density surface, with given reference pressure (actually depth,
# for Boussinesq) and given isovalue.  No diagnostics requested, so info about
# the grid is not needed (no `edges` and `geoemtry` provided).
s, t, z, _ = potential_surf(
    S, T, Z, eos=eos, vert_dim="k", ref=0.0, isoval=1027.5, diags=False
)

# As above, but with diagnostics.
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    edges=edges,
    geometry=(dist, distperp),
    eos=(eos, eos_s_t),
    vert_dim="k",
    ref=0.0,
    isoval=1027.5,
)

# Initialize omega surface with a (locally referenced) potential density surface.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    edges,
    geometry=(dist, distperp),
    vert_dim="k",
    pin_cast=pin_cast,
    pin_p=z0,
    eos=(eos, eos_s_t),
    interp="pchip",
    ITER_MAX=10,
    ITER_START_WETTING=1,
)
