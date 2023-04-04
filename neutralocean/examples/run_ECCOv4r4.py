# In[Imports]

import numpy as np
import xarray as xr
from os.path import expanduser

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

from neutralocean.grid.xgcm import build_grid, edgedata_to_maps
from neutralocean.ntp import ntp_epsilon_errors

# In[Load data]

print(
    "To get started, download the ECCOv4r4 grid information at\n"
    "  https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
    "\nand one day of Salinity and Temperature 3D data at\n"
    "  https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/ECCO_L4_TEMP_SALINITY_LLC0090GRID_DAILY_V4R4/OCEAN_TEMPERATURE_SALINITY_day_mean_2002-12-23_ECCO_V4r4_native_llc0090.nc"
    "\nYou will have to create an Earthdata Login first.  "
    "Then edit the below variable, `folder_ecc4`, to point to the directory"
    " where you saved these two files."
)

folder_ecco4 = expanduser("~/work/data/ECCOv4r4/")  # << EDIT AS NEEDED >>
file_grid = folder_ecco4 + "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
date = "2002-12-23"
file_ST = (
    folder_ecco4
    + "OCEAN_TEMPERATURE_SALINITY_day_mean_"
    + date
    + "_ECCO_V4r4_native_llc0090.nc"
)

# Load horizontal grid information
ds = xr.open_dataset(file_grid)
XC, YC, dxC, dyC, dyG, dxG = (
    ds[x].load() for x in ("XC", "YC", "dxC", "dyC", "dyG", "dxG")
)
ds.close()

# Load hydrographic data
ds = xr.open_dataset(file_ST)
S, T = (ds[x].squeeze().load() for x in ("SALT", "THETA"))

# Create depth xarray.
Z = -ds.Z.load()  # Make Z > 0 and increasing down
Z.attrs.update({"positive": "down"})  # Update attrs to match.

n = len(ds.i)  # size of each horizontal dimension in each square tile
ds.close()

# Get order of non-vertical dimensions
dims = tuple(x for x in S.dims if x != "k")

# Reorder dimensions to ensure individual water columns are float64 and contiguous in memory
S, T = (x.transpose(*dims, "k").astype(np.float64, order="C") for x in (S, T))

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

# Number of Faces -- len(faces_connections only value)
nf = len(next(iter(face_connections.values())))

# Build list of adjacent water columns and distances between those water column pairs
xsh = ysh = "left"
grid = build_grid(n, face_connections, dims, xsh, ysh, dxC, dyC, dyG, dxG)


# Make Boussinesq version of the Jackett and McDougall (1995) equation of state
#  and its partial derivatives.
# TODO: is this what ECCOv4r4 used?
grav, rho_c = 9.81, 1027.5
eos = make_eos("jmd95", grav, rho_c)
eos_s_t = make_eos_s_t("jmd95", grav, rho_c)

# Select pinning cast, picking the cast closest to (x0,y0)
x0, y0 = (-172, -4)  # longitude, latitude -- Pacific equatorial ocean
pin_cast = np.unravel_index(
    ((XC.values - x0) ** 2 + (YC.values - y0) ** 2).argmin(), XC.shape
)
z0 = 1500.0  # pinning depth

# In[Build approximately neutral surfaces]

# Build potential density surface, with given reference pressure (actually depth,
# for Boussinesq) and given isovalue.  No diagnostics requested, so info about
# the grid is not needed (no `edges` and `geoemtry` provided), and also eos_s_t
# is not needed.
s, t, z, _ = potential_surf(
    S, T, Z, eos=eos, vert_dim="k", ref=0.0, isoval=1027.5, diags=False
)
z_sigma = z

# Build in-situ density anomaly surface with given reference salinity and
# potential temperature values and an isovalue of 0, which means the surface
# will intersect any point where the local (S,T) equals the reference values.
# Also return diagnostics.
s0, t0 = 34.5, 4.0
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim="k",
    ref=(s0, t0),
    isoval=0.0,
)

# Build an omega surface that is initialized from the above potential density
# surface and is pinned at the cast `pin_cast` (i.e. the omega surface will have
# the same depth as the initializing potential density surface at this cast).
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid=grid,
    vert_dim="k",
    p_init=z_sigma,
    pin_cast=pin_cast,
    eos=(eos, eos_s_t),
    interp="pchip",
    ITER_MAX=10,
    ITER_START_WETTING=1,
)
z_omega = z

# In[Calculate neutrality error]

# Calculate ϵ neutrality errors on the latest surface, between all pairs of adjacent water columns
e = ntp_epsilon_errors(s, t, z, grid, eos_s_t)

# Convert the 1D array of ϵ values into two maps of ϵ neutrality errors, one
# for the errors in each of the two lateral ('i' and 'j') dimensions.  These
# neutrality errors can then be mapped or further analyzed.
ei, ej = edgedata_to_maps(e, n, face_connections, dims, xsh, ysh)

# In[Map depth difference between omega and potential]

# I've installed ecco_v4_py by cloning their git repository to ~/work/python/ECCOv4-py
# Follow https://ecco-v4-python-tutorial.readthedocs.io/Installing_Python_and_Python_Packages.html#option-1-clone-into-the-repository-using-git-recommended
# and edit path below as needed.
import sys
import matplotlib.pyplot as plt

sys.path.append("/home/stanley/work/python/ECCOv4-py/")
import ecco_v4_py as ecco

ecco.plot_tiles(
    z_omega - z_sigma,
    cmin=-200,
    cmax=200,
    fig_size=9,
    layout="latlon",
    rotate_to_latlon=True,
    Arctic_cap_tile_location=10,
    show_tile_labels=False,
    show_colorbar=True,
    show_cbar_label=True,
    cbar_label="Depth difference [m]",
)
plt.suptitle("z_omega - z_sigma")

plt.savefig("/home/stanley/Fig.png", bbox_inches="tight")

# In[Double check neutralocean's grid differencing]
from xmitgcm import open_mdsdataset
import xgcm

# To check that the grid works, we'll calculate the backwards differences of the
# sea-surface temperature in both horizontal directions. Begin by extracting SST.
k = 0  # index in vertical dimension
Tnp = T.values  # extract raw numpy data
SST = Tnp[..., k]  # slice the shallowest data from each water column

# Calculate difference between all pairs of adjacent casts
a, b = grid["edges"]
ΔT = SST.reshape(-1)[a] - SST.reshape(-1)[b]

# Decompose ΔT (a 1D array) into two 2D arrays, one for differences in each horizontal dimension
SSTx, SSTy = edgedata_to_maps(ΔT, n, face_connections, dims, xsh, ysh)


# Next, we'll repeat the above differencing using ECCO's methods.
# See https://ecco-v4-python-tutorial.readthedocs.io/VectorCalculus_ECCO_barotropicVorticity.html
print(
    "Download the grid's binary data files from "
    "< https://ndownloader.figshare.com/files/6494721 > "
    f" then exctract it so that {folder_ecco4} contains a folder 'global_oce_llc90'"
)
ds_llc = open_mdsdataset(
    folder_ecco4 + "global_oce_llc90/",
    iters=0,
    geometry="llc",
)

# rename 'face' to 'tile' to match T from the netCDF file
ds_llc = ds_llc.rename({"face": "tile"})

# Make the xgcm grid object to handle the differencing.
grid_llc = xgcm.Grid(
    ds_llc,
    periodic=False,
    face_connections=face_connections,
    coords={
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
    },
)

# Backward differences in both horizontal directions
SSTx_ = grid_llc.diff(T[..., k], "X").values
SSTy_ = grid_llc.diff(T[..., k], "Y").values


# Begin checks.

# First, check the two differencing methods are equal, everywhere
assert np.array_equal(SSTx, SSTx_, equal_nan=True)
assert np.array_equal(SSTy, SSTy_, equal_nan=True)
print(
    "Test for equivalence between neutralocean grid differencing and xgcm grid differencing: passed."
)

# Now, check that differencing makes sense at an interior point
(t, j, i) = (9, 40, 40)
assert SSTx[t, j, i] == (Tnp[t, j, i, k] - Tnp[t, j, i - 1, k])
assert SSTy[t, j, i] == (Tnp[t, j, i, k] - Tnp[t, j - 1, i, k])
print(
    f"SSTx[{t}, {j}, {i}] == (T[{t}, {j}, {i}, {k}] - T[{t}, {j}, {i - 1}, {k}])"
)
print(
    f"SSTy[{t}, {j}, {i}] == (T[{t}, {j}, {i}, {k}] - T[{t}, {j - 1}, {i}, {k}])"
)

# Check that differencing makes sense across a boundary.  Find the second point
# that is involved in the difference across the boundary.  Verify by hand that
# this second point is where it ought to be.
# In this case, we're taking a meridional difference from the southern boundary
# of tile 4, and the second point is along the northern boundary of tile 3,
# which makes sense.
t, j, i = 4, 0, 40
idx = np.nonzero(SSTy[t, j, i] == (Tnp[t, j, i, k] - Tnp[..., k]))
t_, j_, i_ = idx[0][0], idx[1][0], idx[2][0]
assert SSTy[t, j, i] == Tnp[t, j, i, k] - Tnp[t_, j_, i_, k]
print(f"SSTy[{t},{j},{i}] == T[{t},{j},{i},{k}] - T[{t_},{j_},{i_},{k}]")
# # Output:
# SSTy[4,0,40] == T[4,0,40,0] - T[3,89,40,0]
