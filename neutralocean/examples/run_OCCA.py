# In[Imports]

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

# Functions to load OCCA data
from neutralocean.examples.load_OCCA import load_OCCA

from neutralocean.grid.rectilinear import build_grid, edgedata_to_maps

# In[Load OCCA data]

# Load OCCA data from the local folder containing this script.
# If it is not there, it will be downloaded to that folder.
# If that folder does not have write permissions, it will be downloaded to temp
# files, loaded into memory, then the temp files deleted.
g, S, T = load_OCCA()  # S, T arranged as (Longitude, Latitude, Depth)
ni, nj, nk = S.shape
Z = -g["RC"]  # Depth vector (note positive and increasing down)

# Select pinning cast in the middle of the domain
i0 = int(ni / 2)
j0 = int(nj / 2)
z0 = 1500.0

# make Boussinesq version of the Jackett and McDougall (1995) equation of state
# --- which is what OCCA used --- and its partial derivatives
eos = make_eos("jmd95", g["grav"], g["rho_c"])
eos_s_t = make_eos_s_t("jmd95", g["grav"], g["rho_c"])

# Build grid adjacency and distance information for neutralocean functions
grid = build_grid(
    (ni, nj), g["wrap"], g["DXCvec"], g["DYCsc"], g["DYGsc"], g["DXGvec"]
)

# In[Potential Density surfaces]

# Provide reference pressure (actually depth, in Boussinesq) and isovalue
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    grid=grid,
    eos="jmd95",
    grav=g["grav"],
    rho_c=g["rho_c"],
    vert_dim="Depth_c",
    ref=0.0,
    isoval=1027.5,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    grid=grid,
    eos="jmd95",
    grav=g["grav"],
    rho_c=g["rho_c"],
    vert_dim="Depth_c",
    ref=0.0,
    pin_cast=(i0, j0),
    pin_p=z0,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide just the location to intersect `(pin_cast, pin_p)`.
# This takes the reference depth `ref` to match `pin_p`.
# Also illustrate using xarray coordinates for pin_cast.
# Also use PCHIPs as the vertical interpolants.
# Also provide the (pre-made) equation of state and its partial derivatives w.r.t S and T.
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim="Depth_c",
    pin_cast={"Longitude_t": 180.5, "Latitude_t": 0.5},
    pin_p=z0,
    interp="pchip",
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" intersecting the cast at (180.5 E, 0.5 N) at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# In[Delta surfaces]

# Provide reference salinity and potential temperature values
s0, t0 = 34.5, 4.0
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim="Depth_c",
    ref=(s0, t0),
    isoval=0.0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim="Depth_c",
    ref=(s0, t0),
    pin_cast=(i0, j0),
    pin_p=z0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide just the location to intersect: depth `pin_p` on cast `pin_cast`
# This takes the reference S and T values from that location.
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# In[Omega surfaces]

# Initialize omega surface with a (locally referenced) potential density surface.
# Provide grid distances.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid,
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    ITER_MAX=10,
    ITER_START_WETTING=1,
)
print(
    f" ** The omega-surface"
    f" initialized from a potential density surface (referenced to {z0}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)

# Initialize omega surface with a (locally referenced) in-situ density anomaly surface.
# Use PCHIP interpolation rather than the default, linear interpolation.
# Remove the mixed layer, calculated internally according to the given parameters --
#   see `mixed_layer` for details on these parameters.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid,
    ref=(None, None),
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    interp="pchip",
    p_ml={"bottle_index": 1, "ref_p": 0.0},
    ITER_MAX=10,
    ITER_START_WETTING=1,
    TOL_P_SOLVER=1e-5,
)
print(
    f" ** The omega-surface"
    f" initialized from an in-situ density anomaly surface (referenced locally to cast {(i0,j0)} at {z0}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)


# In[Begin showing more advanced features]

import numpy as np
import numba as nb
from neutralocean.mixed_layer import mixed_layer
from neutralocean.ntp import ntp_epsilon_errors, ntp_epsilon_errors_norms
from neutralocean.label import veronis_density
from neutralocean.lib import _process_casts
from neutralocean.ppinterp import make_pp
from neutralocean.eos import make_eos_p, vectorize_eos
from neutralocean.traj import ntp_bottle_to_cast

# In[Show vertical interpolation]

# Build interpolation function using same interpolation kernel ("pchip") as
# last ANS surface constructed above.
# Allow this to operate across all water columns with one call (kind="u").
# Evaluate the interpolant rather than build its coefficients (out="interp").
# Allow that there could be NaNs in the data (nans=True).
# There will be two numpy arrays of dependent data (S, T), sharing the same
# independent data Z (num_dep_vars=2).
interp_two = make_pp(
    "pchip", kind="u", out="interp", nans=True, num_dep_vars=2
)

# Apply interpolation function to interpolate salinity and temperature onto the
# depth of the surface.  This requires working with numpy arrays, not xarrays.
# Evaluate the interpolant, not any of its derivatives (d=0).
s_, t_ = interp_two(z.values, Z, S.values, T.values, d=0)


# Check that the results of the above interpolation match (to machine precision)
# the results returned from omega_surf above.
s_check = np.allclose(s.values, s_, atol=1e-15, equal_nan=True)
t_check = np.allclose(t.values, t_, atol=1e-15, equal_nan=True)
if not (s_check and t_check):
    print(
        "Something's wrong; should be able to reconstruct salinity and "
        "temperature on surface by interpolating to surface's depth"
    )


# In[Veronis Density, used to label an approx neutral surface]
S_ref_cast = S.values[i0, j0]
T_ref_cast = T.values[i0, j0]
rho_v = veronis_density(S_ref_cast, T_ref_cast, Z, z0, eos="jmd95")
print(
    f"A surface through the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has Veronis density {rho_v} kg m-3"
)

# In[Remove mixed layer from an omega surface]

# Pre-compute depth of the mixed layer
z_ml = mixed_layer(S, T, Z, eos)

# The correct way to remove the mixed layer from the omega surface algorithm
# is to pass the p_ml parameter, e.g. pass `p_ml=z_ml` in the call below.
# However, here we'll use a more drastic approach: setting our 3D hydrographic
# data to NaN above the mixed layer.
# This serves as a test of the new interpolation methods that allow for NaN data
# in the top of the water column followed by valid (non-NaN) data in the middle
# where the interpolation is done, followed by more NaN data at the bottom.
# This is to allow for models or data with ice shelves.
# Note, this drastic NaN'ing of the input (S,T) data is not expected to give
# exactly the same results as by passing the p_ml parameter, since
# (a) with p_ml, the mixed layer removal is not applied until after the first
#     iteration, and
# (b) with PCHIPs, the interpolant just below the mixed layer is affected by
#     a few of the lower bottles in the mixed layer.
# (c) with p_ml, the surface is set to NaN only if it goes above p_ml which
#     is any real number, whereas with the data-NaN'ing approach, discrete
#     bottles are set to NaN so the surface will only be valid if it is below
#     the first valid bottle below what was NaN'ed out.


@nb.njit
def mixedlayer2nan(S, T, Z, z_ml):
    for n in np.ndindex(z_ml.shape):
        z = z_ml[n]
        for k in range(nk):
            if Z[k] < z:
                S[(*n, k)] = np.nan
                T[(*n, k)] = np.nan
            else:
                break


mixedlayer2nan(S.values, T.values, Z, z_ml)


s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid,
    ref=(None, None),
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    interp="pchip",
    ITER_MAX=10,
    ITER_START_WETTING=1,
    TOL_P_SOLVER=1e-5,
)

# potential and anomaly surfaces don't take `p_ml` as an argument, since they
# aren't iterative algorithms, so we can just remove it manually, e.g.
# z[z < z_ml] = np.nan

# In[Neutrality errors on a surface]
e_RMS, e_MAV = ntp_epsilon_errors_norms(s, t, z, grid, eos_s_t)
print(f"RMS of ϵ is {e_RMS : 4e} [kg m-4])")

# Calculate ϵ neutrality errors on all pairs of adjacent water columns
e = ntp_epsilon_errors(s, t, z, grid, eos_s_t)

# Convert ϵ above into two 2D maps, one for zonal ϵ errors and one for meridional ϵ errors
ex, ey = edgedata_to_maps(e, (ni, nj), g["wrap"])
# These can then be mapped...

# In[Neutral Tangent Plane bottle to cast]

sB, tB, zB = 35.0, 16.0, 500.0  # Thermodynamic properties of a given Bottle
S1 = S.values[180, 80, :]
T1 = T.values[180, 80, :]
s1, t1, z1 = ntp_bottle_to_cast(sB, tB, zB, S1, T1, Z)

# In[Work with Numpy arrays instead of xarrays]

# Convert S and T from xarray to numpy ndarrays, and make vertical dimension
# contiguous in memory.  If not done here in advance, this will be done each
# time potential_surf, anomaly_surf, or omega_surf is called.  Hence, this is
# the recommended method if many approx neutral surfaces will be calculated.
Snp, Tnp, Znp = _process_casts(S, T, Z, "Depth_c")

s, t, z, d = anomaly_surf(
    Snp,
    Tnp,
    Znp,
    grid=grid,
    eos=(eos, eos_s_t),
    vert_dim=-1,
    ref=(s0, t0),
    isoval=0.0,
)


# In[Calculate a large-scale potential vorticity on our surface]

# Create function for partial deriv of equation of state with respect to depth z
eos_z = make_eos_p("jmd95", g["grav"], g["rho_c"])  # for scalar inputs
eos_z = vectorize_eos(eos_z)  # for nd inputs

# Build linear interpolation function, operating over a whole dataset (kind="u"),
# evaluating the interpolant on the fly (out="interp"), for two dependent
# variables (num_dep_vars=2).
interp_two = make_pp("linear", kind="u", out="interp", num_dep_vars=2)

# Earth sidereal day period [s]
Earth_day = 86164

# Coriolis param [s-1] on tracer grid
f = 2 * (2 * np.pi / Earth_day) * np.sin(g["YCvec"] * (np.pi / 180))

# evaluate first derivative (d=1) of S(z) and T(z) at depth z.  That is, eval
# ∂S/∂Z and ∂T/∂Z, on the surface.
sz, tz = interp_two(z, Z, S.values, T.values, 1)
rs, rt = eos_s_t(s, t, z)  # ∂ρ/∂S and ∂ρ/∂T, on the surface

# ∂δ/∂z on the surface, where δ is the in-situ density anomaly
δz = rs * sz + rt * tz + (eos_z(s, t, z) - eos_z(s0, t0, z))

# large-scale potential vorticity [m-1 s-1] defined via in-situ density anomaly, on the surface
q = f * δz
