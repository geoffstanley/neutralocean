# In[Imports]

# Functions to make the Equation of State
from neutralocean.eos import make_eos, make_eos_s_t

# Functions to compute various approximately neutral surfaces
from neutralocean.surface import potential_surf, anomaly_surf, omega_surf

# Functions to load OCCA data
from neutralocean.examples.load_OCCA import load_OCCA

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
eos = make_eos("jmd95", g["grav"], g["ρ_c"])
eos_s_t = make_eos_s_t("jmd95", g["grav"], g["ρ_c"])

# Package up grid distance information for neutralocean functions:
geom = {
    "dist1_iJ": g["DXCvec"],  # Distance [m] in 1st dim centred at (I-1/2, J)
    "dist1_Ij": g["DXGvec"],  # Distance [m] in 1st dim centred at (I, J-1/2)
    "dist2_Ij": g["DYGsc"],  # Distance [m] in 2nd dim centred at (I-1/2, J)
    "dist2_iJ": g["DYCsc"],  # Distance [m] in 2nd dim centred at (I, J-1/2)
}


# In[Potential Density surfaces]

# Provide reference pressure (actually depth, in Boussinesq) and isovalue
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos="jmd95",
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=0.0,
    isoval=1027.5,
    **geom,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos="jmd95",
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=0.0,
    pin_cast=(i0, j0),
    pin_p=z0,
    **geom,
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide just the location to intersect `(pin_cast, pin_p)`.
# This takes the reference depth `ref` to match `pin_p`.
# Also illustrate using xarray coordinates for pin_cast.
# Also use PCHIPs as the vertical interpolants.
# Also provide the equation of state and its partial derivatives w.r.t S and T.
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos=(eos, eos_s_t),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast={"Longitude_t": 180.5, "Latitude_t": 0.5},
    pin_p=z0,
    interp="pchip",
)
print(
    f" ** The potential density surface (referenced to {d['ref']}m)"
    f" intersecting the cast at (180.5 E, 0.5 N) at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# In[Delta surfaces]

# Provide reference salinity and potential temperature values
s0, t0 = 34.5, 4.0
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=(eos, eos_s_t),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=(s0, t0),
    isoval=0.0,
    **geom,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" with isovalue = {d['isoval']}kg m-3"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide pin_cast and pin_p: the reference location and depth that the surface intersects
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=(eos, eos_s_t),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=(s0, t0),
    pin_cast=(i0, j0),
    pin_p=z0,
    **geom,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# Provide just the location to intersect: depth `pin_p` on cast `pin_cast`
# This takes the reference S and T values from that location.
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=(eos, eos_s_t),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    **geom,
)
print(
    f" ** The in-situ density anomaly surface (referenced to {d['ref']})"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" (isovalue = {d['isoval']}kg m-3)"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS']} kg m-4"
)

# In[Omega surfaces]

# Initialize omega surface with a (locally referenced) potential density surface.
# Provide grid distances.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    ITER_MAX=10,
    ITER_START_WETTING=1,
    **geom,
)
print(
    f" ** The omega-surface"
    f" initialized from a potential density surface (referenced to {z0}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS'][-1]} kg m-4"
)

# Initialize omega surface with a (locally referenced) in-situ density anomaly surface.
# Use PCHIP interpolation rather than the default, linear interpolation.
# Remove the mixed layer, calculated internally according to the given parameters --
#   see `mixed_layer` for details on these parameters.
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    ref=(None, None),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    interp="pchip",
    p_ml={"bottle_index": 1, "ref_p": 0.0},
    ITER_MAX=10,
    ITER_START_WETTING=1,
    TOL_P_SOLVER=1e-5,
    **geom,
)
print(
    f" ** The omega-surface"
    f" initialized from an in-situ density anomaly surface (referenced locally to cast {(i0,j0)} at {z0}m)"
    f" intersecting the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has root-mean-square ϵ neutrality error {d['ϵ_RMS'][-1]} kg m-4"
)


# In[Begin showing more advanced features]

import numpy as np
from neutralocean.mixed_layer import mixed_layer
from neutralocean.ntp import ntp_ϵ_errors, ntp_ϵ_errors_norms
from neutralocean.label import veronis_density
from neutralocean.lib import _process_casts, find_first_nan
from neutralocean.interp1d import make_interpolator
from neutralocean.ppinterp import select_ppc
from neutralocean.eos import make_eos_p, vectorize_eos
from neutralocean.traj import ntp_bottle_to_cast, _ntp_bottle_to_cast

# In[Veronis Density, used to label an approx neutral surface]
S_ref_cast = S.values[i0, j0]
T_ref_cast = T.values[i0, j0]
ρ_v = veronis_density(S_ref_cast, T_ref_cast, Z, z0, eos="jmd95")
print(
    f"A surface through the cast indexed by {(i0,j0)} at depth {z0}m"
    f" has Veronis density {ρ_v} kg m-3"
)

# In[Remove mixed layer from an omega surface]

# Pre-compute depth of the mixed layer
z_ml = mixed_layer(S, T, Z, eos)

s, t, z, d = omega_surf(
    S,
    T,
    Z,
    ref=(None, None),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=(eos, eos_s_t),
    ITER_MAX=10,
    ITER_START_WETTING=1,
    TOL_P_SOLVER=1e-5,
    **geom,
    p_ml=z_ml,
)

# potential and anomaly surfaces don't take `p_ml` as an argument, since they
# aren't iterative algorithms, so we can just remove it manually, e.g.
# z[z < z_ml] = np.nan

# In[Neutrality errors on a surface]
ϵx, ϵy = ntp_ϵ_errors(
    s, t, z, eos_s_t, "Longitude_t", geom["dist1_iJ"], geom["dist2_Ij"]
)
ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, "Longitude_t", **geom)
print(f"RMS of ϵ is {ϵ_RMS : 4e} [kg m-4])")

# In[Neutral Tangent Plane bottle to cast]

sB, tB, zB = 35.0, 16.0, 500.0  # Thermodynamic properties of a given Bottle
S1 = S.values[180, 80, :]
T1 = T.values[180, 80, :]
s1, t1, z1 = ntp_bottle_to_cast(sB, tB, zB, S1, T1, Z)

# Or the more manual version:
n_good = find_first_nan(S1)[()]
ppc_fn = select_ppc("linear", "1")
S1ppc, T1ppc = (ppc_fn(Z, C) for C in (S1, T1))
s1, t1, z1 = _ntp_bottle_to_cast(sB, tB, zB, S1ppc, T1ppc, Z, n_good, 1e-4, eos)


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
    eos=(eos, eos_s_t),
    wrap=(True, False),
    vert_dim=-1,
    ref=(s0, t0),
    isoval=0.0,
)


# In[Calculate a large-scale potential vorticity on our surface]

# Create function for partial deriv of equation of state with respect to depth z
eos_z = make_eos_p("jmd95", g["grav"], g["ρ_c"])  # for scalar inputs
eos_z = vectorize_eos(eos_z)  # for nd inputs

# Earth sidereal day period [s]
Earth_day = 86164

# Coriolis param [s-1] on tracer grid
f = 2 * (2 * np.pi / Earth_day) * np.sin(g["YCvec"] * (np.pi / 180))

linterp_dx_utwo = make_interpolator("linear", deriv=1, kind="u", two=True)
sz, tz = linterp_dx_utwo(z, Z, S.values, T.values)  # ∂S/∂Z and ∂T/∂Z, on the surface
rs, rt = eos_s_t(s, t, z)  # ∂ρ/∂S and ∂ρ/∂T, on the surface

# ∂δ/∂z on the surface, where δ is the in-situ density anomaly
δz = rs * sz + rt * tz + (eos_z(s, t, z) - eos_z(s0, t0, z))

# large-scale potential vorticity [m-1 s-1] defined via in-situ density anomaly, on the surface
q = f * δz
