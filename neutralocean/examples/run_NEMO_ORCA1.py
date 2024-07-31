# Show basic use of neutralocean on a tripolar grid using CMIP6 archived
# CanESM5 data, which runs NEMO 3.4 on the ORCA1 grid.
#
# For more advanced usage of neutralocean, see the run_OCCA.py example.

# In[Imports]

import xarray as xr
import pooch

from neutralocean.grid.tripolar import build_grid, edgedata_to_maps
from neutralocean.surface import potential_surf, omega_surf

# In[Load data]

# Get salt and potential temperature from CMIP6 archives (734 MB and 963 MB)
url_salt = "http://crd-esgf-drc.ec.gc.ca/thredds/fileServer/esgC_dataroot/AR6/CMIP6/CMIP/CCCma/CanESM5/esm-piControl/r1i1p1f1/Omon/so/gn/v20190429/so_Omon_CanESM5_esm-piControl_r1i1p1f1_gn_629101-630012.nc"
url_theta = "http://crd-esgf-drc.ec.gc.ca/thredds/fileServer/esgC_dataroot/AR6/CMIP6/CMIP/CCCma/CanESM5/esm-piControl/r1i1p1f1/Omon/thetao/gn/v20190429/thetao_Omon_CanESM5_esm-piControl_r1i1p1f1_gn_629101-630012.nc"

# Get ORCA1 horizontal grid distances from a Zenodo archive that is
# unassociated with the CMIP6 CanESM5 run, I suspect is the same as that which
# would have been used by CanESM5. This example is for illustrative purposes
# only! (405 MB)
url_mesh = "https://zenodo.org/record/4432892/files/ORCA1_mesh_mask.nc"

# Download data, using friendly pooch to fetch it if not already downloaded.
hash_salt = "93ba818ac7661c48686ae3af497e832667d0e72eb1968f375d9191c24f11db94"
hash_theta = "5d4085ab355c3a5526fa62d5f545a2322050e08dc24383189d458b413db99a79"
hash_mesh = "3e4560590c338dfd18de5db580f2354b11acb8ae95e13101c66461683385e777"
file_salt = pooch.retrieve(url=url_salt, known_hash=hash_salt)
file_theta = pooch.retrieve(url=url_theta, known_hash=hash_theta)
file_mesh = pooch.retrieve(url=url_mesh, known_hash=hash_mesh)


# Extract data
timestep = 0
ds = xr.open_dataset(file_salt)
nj = ds.dims["j"]  # number of grid points in the meridional y direction
ni = ds.dims["i"]  # number of grid points in the zonal x direction
S = ds["so"].isel({"time": timestep})  # 3D salinity
Z = ds["lev"]  # 1D depth at center of tracer cells

ds = xr.open_dataset(file_theta)
T = ds["thetao"].isel({"time": timestep})  # potential temperature

mesh = xr.open_dataset(file_mesh)  # contains horizontal grid information

# Squeeze out singleton dimension in horizontal grid distances and convert to numpy array
e1u, e2v, e2u, e1v = (mesh[x].data.squeeze() for x in ("e1u", "e2v", "e2u", "e1v"))

# Note: The ORCA1 tripolar horizontal grid is of size (nj, ni) == (291, 360).
# The second dimension is periodic, handling longitude's periodic nature.
# The first dimension has two different boundary conditions. The south
# (the first row) is non-periodic since Antarctica covers the South Pole and
# the ocean grid only goes to -78.3935°S. The north (the last row) is periodic
# with a flipped version of itself, due to ORCA's tripolar grid. Specifically,
# the cell at [nj - 1, i] is adjacent to the cell at [nj - 1, ni - i - 1].
# The grid metrics e1u, e2v, e2u, e1v all have size (nj+1, ni+2) == (292, 362):
# they employ padding to ease applying boundary conditions.
# See documentation in `neutralocean.grid.tripolar` for more information.

# Build the edges of the grid graph, and the metrics associated with each edge
grid = build_grid((nj, ni), e1u, e2v, e2u, e1v)

# In[Approximately Neutral Surfaces]

grav = 9.80665  # gravitational acceleration [m s-2]
rau0 = 1035.0  # Boussinesq reference density [kg m-3]

# Provide reference pressure (actually depth, in Boussinesq) and isovalue
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    grid=grid,
    eos="jmd95",
    grav=grav,
    rho_c=rau0,
    vert_dim="lev",
    ref=0.0,
    isoval=1027.5,
)
s_sigma, t_sigma, z_sigma = s, t, z  # save for later
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Pin omega surface to the cast (j0, i0) in the equatorial Pacific at depth 1500m.
j0, i0 = 100, 150
z0 = 1500.0

# Initialize omega surface with a (locally referenced) potential density surface.
# Provide grid distances. By default, does as many iterations as needed to get
# the root-mean-square change in the locally referenced potential density from
# one iteration to the next to drop below 10^-7 kg m-3, or a maximum of 10
# iterations.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid,
    pin_cast=(j0, i0),
    p_init=z0,
    vert_dim="lev",
    eos="jmd95",
    grav=grav,
    rho_c=rau0,
)
s_omega, t_omega, z_omega = s, t, z  # save for later
print(
    f" ** The omega-surface"
    f" initialized from a potential density surface (referenced to {z0}m)"
    f" intersecting the cast indexed by {(j0,i0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)


# In[Neutrality errors on a surface]
from neutralocean.ntp import ntp_epsilon_errors
from neutralocean.eos import load_eos

# Prepare function for S and T derivatives of the equation of state
eos_s_t = load_eos("jmd95", "_s_t", grav, rau0)

# Calculate ϵ neutrality errors on all pairs of adjacent water columns
e = ntp_epsilon_errors(s, t, z, grid, eos_s_t)

# Convert ϵ above into two 2D maps, one each of the "x" and "y" dimensions
ex, ey = edgedata_to_maps(e, (nj, ni))

# These can then be mapped, e.g.:
# import matplotlib.pyplot as plt
# plt.imshow(ex, origin="lower", vmin=-1e-9, vmax=1e-9)
# plt.colorbar()
