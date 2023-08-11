# In[Imports]

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

# Functions to load OCCA data
from neutralocean.examples.load_OCCA import load_OCCA

from neutralocean.grid.rectilinear import build_grid, edgedata_to_maps

# In[Load OCCA data]

g, S, T = load_OCCA()  # S, T arranged as (Longitude, Latitude, Depth)
ni, nj, nk = S.shape
Z = -g["RC"]  # Depth vector (note positive and increasing down)

# Select pinning cast in the equatorial Pacific. 
# The following surfaces will intersect cast (i0,j0) at a depth of z0.
i0, j0 = 220, 80
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
    pin_cast={"Longitude_t": 220.5, "Latitude_t": 0.5},
    pin_p=z0,
    interp="pchip",
)
assert z.values[i0, j0] == z0  # check pin_cast was indeed (i0,j0)
z_sigma = z  # save for later
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
z_delta = z  # save for later
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# In[Omega surfaces]

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
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
)
s_omega, t_omega, z_omega = s, t, z  # save for later
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
# Specify a higher max number of iterations, and quit iterations when the
# depth change from one iteration to the next has a root-mean-square value less
# than 1e-6 m.
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
    ITER_MAX=20,
    TOL_P_CHANGE_RMS=1e-6,
    TOL_LRPD_MAV=0.0,  # deactivate this exit tolerance
    TOL_P_SOLVER=1e-7,
)
z_omega_converged = z  # save for next
print(
    f" ** The omega-surface"
    f" initialized from an in-situ density anomaly surface (referenced locally to cast {(i0,j0)} at {z0}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)

# Initialize omega surface with a pre-computed surface.
# In this case, let's continue from the above omega surface, but change the
# pinning location. Since the above omega surface is very nearly converged,
# this omega surface should basically match the above one. Indeed, it will
# do one iteration and find it is below the RMS depth change tolerance, and exit.
i1, j1 = 315, 110  # North Atlantic
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    grid,
    vert_dim="Depth_c",
    pin_cast=(i1, j1),
    p_init=z_omega_converged,
    eos=(eos, eos_s_t),
    interp="pchip",
    ITER_MAX=10,
    ITER_START_WETTING=99,  # greater than ITER_MAX, so no wetting
    TOL_P_CHANGE_RMS=1e-6,
    TOL_LRPD_MAV=0.0,  # deactivate this exit tolerance
    TOL_P_SOLVER=1e-7,
)
print(
    f" ** The omega-surface"
    f" initialized from an omega surface and pinned to the cast {(i1,j1)} at {z0}m)"
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
from neutralocean.traj import ntp_bottle_to_cast, neutral_trajectory

# In[Show vertical interpolation]

# Build interpolation function using same interpolation kernel ("linear") as
# omega surface we saved above.
# Allow this to operate across all water columns with one call (kind="u").
# Evaluate the interpolant rather than build its coefficients (out="interp").
# Allow that there could be NaNs in the data (nans=True).
# There will be two numpy arrays of dependent data (S, T), sharing the same
# independent data Z (num_dep_vars=2).
interp_two = make_pp(
    "linear", kind="u", out="interp", nans=True, num_dep_vars=2
)

# Apply interpolation function to interpolate salinity and temperature onto the
# depth of the surface.  This requires working with numpy arrays, not xarrays.
# Evaluate the interpolant, not any of its derivatives (d=0).
s_, t_ = interp_two(z_omega.values, Z, S.values, T.values, 0)


# Check that the results of the above interpolation match (to machine precision)
# the results returned from omega_surf above.
s_check = np.allclose(s_omega.values, s_, atol=1e-15, equal_nan=True)
t_check = np.allclose(t_omega.values, t_, atol=1e-15, equal_nan=True)
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

# In[Neutral Trajectory]

# neutral trajectory depth along meridional section
znt = np.full(nj, np.nan)

# Build a neutral trajectory going northwards from cast at (i0, j0), starting
# at depth of z0. Put result into `znt`
Snt = S[i0, j0:nj, :]
Tnt = T[i0, j0:nj, :]
s_, t_, z_ = neutral_trajectory(Snt, Tnt, Z, z0, eos=eos)
znt[j0:nj] = z_

# As above, but go southwards.
Snt = S[i0, j0:0:-1, :]
Tnt = T[i0, j0:0:-1, :]
s_, t_, z_ = neutral_trajectory(Snt, Tnt, Z, z0, eos=eos)
znt[j0:0:-1] = z_

# Consider three other trajectories along this meridional section, following ANSs
zom = z_omega[i0, :].values  # omega
zpd = z_sigma[i0, :].values  # potential density
zda = z_delta[i0, :].values  # density anomaly

# Find index to furthest south cast that has all four trajectories
j_south = np.argmax(
    np.all(
        np.isfinite(np.array((znt, zom, zpd, zda))),
        axis=0,
    )
)
print(
    f"From (i,j)={(i0,j0)} at {z0}m moving south to j={j_south}, ...\n"
    f"Neutral Trajectory              ends at {znt[j_south]:8.4f}m depth,\n"
    f"Omega Surface                   ends at {zom[j_south]:8.4f}m depth,\n"
    f"Potential Density Surface       ends at {zpd[j_south]:8.4f}m depth,\n"
    f"In-situ density anomaly surface ends at {zda[j_south]:8.4f}m depth."
)

print(
    "Uncomment below to plot neutral trajectory and ANSs along meridional section"
)
# import matplotlib.pyplot as plt
# lat = g["YCvec"]  # latitudes
# fig, ax = plt.subplots()
# ax.plot(lat, -zom, label="omega")
# ax.plot(lat, -zpd, label="potential density")
# ax.plot(lat, -zda, label="in-situ density anomaly")
# ax.plot(lat, -znt, "--k", label="neutral trajectory")
# ax.scatter(lat[j0], -z0)  # reference latitude and depth.
# ax.legend()

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
e_RMS, e_MAV = ntp_epsilon_errors_norms(s, t, z_omega, grid, eos_s_t)
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


# In[Calculate a large-scale potential vorticity on the above specvol anomaly surface]

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
