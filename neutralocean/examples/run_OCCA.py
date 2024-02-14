# In[Imports]

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

# Functions to load and work with rectilinear (lat-lon) OCCA data
from neutralocean.examples.load_OCCA import load_OCCA
from neutralocean.stability import stabilize_ST
from neutralocean.grid.rectilinear import build_grid

# In[Load OCCA data]

g, S, T = load_OCCA()  # S, T arranged as (Longitude, Latitude, Depth)
ni, nj, nk = S.shape
Z = -g["RC"]  # Depth vector (note positive and increasing down)

# make Boussinesq version of the Jackett and McDougall (1995) equation of state
# --- which is what OCCA used --- and its partial derivatives
eos = make_eos("jmd95", g["grav"], g["rho_c"])
eos_s_t = make_eos_s_t("jmd95", g["grav"], g["rho_c"])

# Perturb S, T to ensure static stability everywhere
stabilize_ST(S, T, Z, eos, min_dLRPDdz=1e-6, verbose=False)  # about 30 sec


# Select pinning cast in the equatorial Pacific.
# When these are used for 'pin_cast' and 'pin_p' or 'p_init', the following
# surfaces will intersect cast (i0,j0) at a depth of z0.
i0, j0 = 220, 80
z0 = 1500.0

# Select vertical interpolation method. Options are "linear" or "pchip"
interp_name = "linear"

# Build grid adjacency and distance information for neutralocean functions
grid = build_grid(
    (ni, nj), g["wrap"], g["DXCvec"], g["DYCsc"], g["DYGsc"], g["DXGvec"]
)

# Prepare some default options for potential_surf, anomaly_surf, and omega_surf
opts = {}
opts["grid"] = grid
opts["interp"] = interp_name
opts["vert_dim"] = "Depth_c"
opts["eos"] = (eos, eos_s_t)

# In[Potential Density surfaces]

# Provide reference pressure (actually depth, in Boussinesq) and isovalue.
args = opts.copy()
args["ref"] = 0.0
args["isoval"] = 1027.5
s, t, z, d = potential_surf(S, T, Z, **args)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
args = opts.copy()
args["ref"] = 0.0
args["pin_cast"] = (i0, j0)
args["pin_p"] = z0
s, t, z, d = potential_surf(S, T, Z, **args)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide just the location to intersect `(pin_cast, pin_p)`.
# This takes the reference depth `ref` to match `pin_p`.
# Also illustrate using xarray coordinates for pin_cast.
# Also show how to just give the EOS name and the necessary parameters (g and rho_c)
# for its Boussinesq version, rather than using the pre-made EOS's as above.
args = opts.copy()
args["pin_cast"] = {"Longitude_t": 220.5, "Latitude_t": 0.5}
args["pin_p"] = z0
args["eos"] = "jmd95"
args["grav"] = g["grav"]
args["rho_c"] = g["rho_c"]
s, t, z, d = potential_surf(S, T, Z, **args)
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

args = opts.copy()
args["ref"] = (s0, t0)
args["isoval"] = 0.0
s, t, z, d = anomaly_surf(S, T, Z, **args)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
args = opts.copy()
args["ref"] = (s0, t0)
args["pin_cast"] = (i0, j0)
args["pin_p"] = z0
s, t, z, d = anomaly_surf(S, T, Z, **args)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# Provide just the location to intersect: depth `pin_p` on cast `pin_cast`
# This takes the reference S and T values from that location.
args = opts.copy()
args["pin_cast"] = (i0, j0)
args["pin_p"] = z0
s, t, z, d = anomaly_surf(S, T, Z, **args)
z_delta = z  # save for later
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS']} kg m-4"
)

# In[Omega surfaces]

# Initialize omega surface with a (locally referenced) in-situ density anomaly surface.
# By default, this does as many iterations as
# needed to get the root-mean-square change in the locally referenced potential
# density from one iteration to the next to drop below 10^-7 kg m-3, or a
# maximum of 10 iterations.
args = opts.copy()
args["pin_cast"] = (i0, j0)
args["p_init"] = z_delta
args["p_ml"] = {"bottle_index": 1, "ref_p": 0.0}  # see `mixed_layer` for info
s, t, z, d = omega_surf(S, T, Z, **args)
print(
    f" ** The omega-surface"
    f" initialized from the given in-situ density anomaly surface"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)


# Initialize omega surface by iteratively making Neutral Tangent Plane links
# outwards from the pinning cast.
# Specify higher than default tolerances: Quit iterations when the
# depth change from one iteration to the next has a root-mean-square value less
# than 1e-6 m, and use a tolerance of 1e-7 m in the "vertical solve" that
# updates the vertical position of the surface during each iteration.
# Also, keep wetting through all the iterations.
args = opts.copy()
args["pin_cast"] = (i0, j0)
args["p_init"] = z0
args["TOL_P_CHANGE_RMS"] = 1e-6
args["TOL_LRPD_MAV"] = 0.0  # deactivate this exit threshold
args["TOL_P_SOLVER"] = 1e-7
args["ITER_STOP_WETTING"] = 99  # > ITER_MAX, so don't stop wetting
s, t, z, d = omega_surf(S, T, Z, **args)
s_omega, t_omega, z_omega = s, t, z  # save for later
print(
    f" ** The omega-surface"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)
args_omega = args.copy()  # save for later


# Initialize omega surface with a pre-computed surface.
# In this case, let's continue from the above omega surface, but change the
# pinning location. Since the above omega surface converged (to the default
# tolerances), this omega surface should basically match the above one.
# Indeed, it will do one iteration, find it is below the tolerances, and exit.
i1, j1 = 315, 110  # North Atlantic
args["pin_cast"] = (i1, j1)
args["p_init"] = z_omega  # set init surface from above
s, t, z, d = omega_surf(S, T, Z, **args)
print(
    f" ** The omega-surface"
    f" initialized from an omega surface and pinned to the cast {(i1,j1)} at {z0}m)"
    f" has root-mean-square ϵ neutrality error {d['e_RMS'][-1]} kg m-4"
)


# In[Begin showing more advanced features]

import numpy as np
from neutralocean.mixed_layer import mixed_layer
from neutralocean.ntp import ntp_epsilon_errors, ntp_epsilon_errors_norms
from neutralocean.label import veronis
from neutralocean.lib import _process_casts, xr_to_np
from neutralocean.ppinterp import make_pp
from neutralocean.eos import make_eos_p, vectorize_eos
from neutralocean.traj import ntp_bottle_to_cast, neutral_trajectory
from neutralocean.grid.rectilinear import edgedata_to_maps

# In[Show vertical interpolation]

# Build interpolation function using same interpolation kernel ("linear") as
# omega surface we saved above.
# Allow this to operate across all water columns with one call (kind="u").
# Evaluate the interpolant rather than build its coefficients (out="interp").
# Allow that there could be NaNs in the data (nans=True).
# There will be two numpy arrays of dependent data (S, T), sharing the same
# independent data Z (num_dep_vars=2).
interp_two = make_pp(
    interp_name, kind="u", out="interp", nans=True, num_dep_vars=2
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


# In[Approximate b, the integrating factor, from a pair of omega surfaces]


def calc_lrpd_z(z):
    """
    Calculate ∂(Locally Referenced Potential Density)/dz on the surface `z`,
    lrpd_z = (ρ_S ∂S/∂z + ρ_Θ ∂Θ/∂z).
    """
    z, Snp, Tnp, Znp = (xr_to_np(x) for x in (z, S, T, Z))
    s, t = interp_two(z, Znp, Snp, Tnp, 0)
    ds, dt = interp_two(z, Znp, Snp, Tnp, 1)
    rs, rt = eos_s_t(s, t, z)
    lrpd_z = rs * ds + rt * dt
    return lrpd_z


def calc_b(z, z2, I0):
    """
    Estimate the integrating factor, b, from a pair of surfaces.

    This estimates b from the vertical component of the neutral relation
    ∇γ = b N, which is approximately true for highly accurate approximately
    neutral surfaces such as omega surfaces, where N = ρ_S ∇S + ρ_Θ ∇Θ is the
    dianeutral vector.
    Thus, we calculate b as
      b = (∂γ/∂z) / Nz,
    where Nz = (ρ_S ∂S/∂z + ρ_Θ ∂Θ/∂z)
    and with the approximation
      (∂γ/∂z) = Δγ / Δz = Δγ / (z2 - z),
    where Δγ is the γ density label on the `z2` surface minus that on the `z` surface.
    We determine Δγ by imposing b = 1 at the reference cast indexed by `I0`, so
      b = (z2[I0] - z[I0]) * Nz[I0] / ((z2 - z) * Nz)
    """
    z, z2 = (xr_to_np(x) for x in (z, z2))
    dz = z2 - z
    lrpd_z = calc_lrpd_z(z)
    b = (lrpd_z[I0] * dz[I0]) / (lrpd_z * dz)
    return b


def same_valid_casts(z, z2):
    z, z2 = (xr_to_np(x) for x in (z, z2))
    return np.all(np.isnan(z) == np.isnan(z2))


# Calculate 2nd omega surface that's `ztiny` below the previous one at `pin_cast`.
# Use the same options as for the previous omega surface, but turn off wetting
# to help ensure the two omega surfaces will have exactly the same wet casts.
# Also, using highly converged omega surfaces (eg small TOL_P_CHANGE_RMS) helps
# the b estimate be smooth.
ztiny = 1e-2
if True:
    # Heave the omega surface rigidly down by an amount `ztiny`
    z_init = z_omega + ztiny
else:
    # Heave the omega surface rigidly if LRPD were the vertical coordinate, so
    # as to be at a depth `ztiny` deeper at the reference cast.
    # This is slightly better than the above, unless lrpd_z is very near zero
    # in some casts. Either way, omega_surf will converge to the same surface.
    lrpd_z = calc_lrpd_z(z_omega)
    z_init = z_omega + (ztiny * lrpd_z[i0, j0]) / lrpd_z

args = args_omega

args["p_init"] = z_init
s, t, z, d = omega_surf(S, T, Z, **args)
z_omega_pair = z

if not same_valid_casts(z_omega, z_omega_pair):
    print(
        "Warning: pair of surfaces has different valid casts, "
        "so the estimate of b cannot be trusted. "
        "Make sure wetting is turned off for the second surface, and try "
        "reducing the depth difference between the pair of surfaces."
    )

b = calc_b(z_omega, z_omega_pair, (i0, j0))
print(
    "The integrating factor b for the omega surface varies between "
    f"{np.nanmin(b)} and {np.nanmax(b)}."
)

# In[Veronis Density, used to label an approx neutral surface]
S_ref_cast = S.values[i0, j0]
T_ref_cast = T.values[i0, j0]
rho_v = veronis(z0, S_ref_cast, T_ref_cast, Z, eos="jmd95")
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


def mixedlayer2nan(S, T, Z, z_ml):
    # Mutates S and T!
    for n in np.ndindex(z_ml.shape):
        z = z_ml[n]
        for k in range(nk):
            if Z[k] < z:
                S[(*n, k)] = np.nan
                T[(*n, k)] = np.nan
            else:
                break


mixedlayer2nan(S.values, T.values, Z, z_ml)

args = opts.copy()
args["pin_cast"] = (i0, j0)
args["p_init"] = z0
args["interp"] = "pchip"
args["ITER_MAX"] = 10
args["TOL_P_SOLVER"] = 1e-5
s, t, z, d = omega_surf(S, T, Z, **args)

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
