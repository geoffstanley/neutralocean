# %% Imports

# %matplotlib notebook
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from neutralocean.eos.tools import make_eos, make_eos_s_t, make_eos_p

from neutralocean.surface.trad import potential_surf, anomaly_surf
from neutralocean.surface.omega import omega_surf

from neutralocean.mixed_layer import mixed_layer

from neutralocean.ntp import ntp_ϵ_errors_norms
from neutralocean.label import veronis_density
from neutralocean.lib import _process_casts
from neutralocean.interp_ppc import linear_coeffs, val2

from neutralocean.examples.load_OCCA import load_OCCA

# %% Load OCCA data

# import wget
path_neutralocean = "/home/stanley/work/projects-gfd/neutralocean/"  # EDIT AS NEEDED
path_occa = path_neutralocean + "neutralocean/examples/"
# path_occa = "~/work/data/OCCA/"  # EDIT AS NEEDED

# url = 'ftp://mit.ecco-group.org/ecco_for_las/OCCA_1x1_v2/2004-6/annual/'
# url = "https://www.dropbox.com/s/q9hywvjup1mwhc9/DDsalt.0406annclim.nc"
# wget.download(url, path_occa)
# url = "https://www.dropbox.com/s/qr6bivfyk0s06ot/DDtheta.0406annclim.nc"
# wget.download(url, path_occa)

g, S, T = load_OCCA(path_occa)  # S arranged as (Longitude, Latitude, Depth)
ni, nj, nk = S.shape
Z = -g["RC"]  # Depth vector (note positive and increasing down)

# Select pinning cast
i0 = int(ni / 2)
j0 = int(nj / 2)
z0 = 1500.0

# Prepare equation of state functions (in this case, the Boussinesq versions)
eos = make_eos("jmd95", g["grav"], g["ρ_c"])
eos_s_t = make_eos_s_t("jmd95", g["grav"], g["ρ_c"])
eos_z = make_eos_p("jmd95", g["grav"], g["ρ_c"])


# Pre-compute depth of the mixed layer
z_ml = mixed_layer(S, T, Z, eos)

# Pre-compute linear interpolants for S and T in terms of Z
Sppc = linear_coeffs(Z, S)
Tppc = linear_coeffs(Z, T)

# %% Potential Density surface

# Provide reference pressure and isovalue
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos=eos,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=0.0,
    isoval=1027.5,
)

# Provide reference pressure and location for the surface to intersect (pin_cast and pin_p)
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos=eos,
    eos_s_t=eos_s_t,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=0.0,
    pin_cast=(i0, j0),
    pin_p=z0,
)

# Provide just the location to intersect (pin_cast, pin_p).
# This takes the reference pressure ref to match pin_p.
# Also illustrate using xarray coordinates for pin_cast
s, t, z, d = potential_surf(
    S,
    T,
    Z,
    eos=eos,
    eos_s_t=eos_s_t,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast={"Longitude_t": 180.5, "Latitude_t": 0.5},
    pin_p=z0,
)

# Calculate area-weighted epsilon neutrality errors on the surface (these are also given in diagnostics `d`)
dist1_iJ = g["DXCvec"]  # Distance [m] in 1st dim centred at (I-1/2, J)
dist1_Ij = g["DXGvec"]  # Distance [m] in 1st dim centred at (I, J-1/2)
dist2_Ij = g["DYGsc"]  # Distance [m] in 2nd dim centred at (I-1/2, J)
dist2_iJ = g["DYCsc"]  # Distance [m] in 2nd dim centred at (I, J-1/2)
geom = [dist1_iJ, dist1_Ij, dist2_Ij, dist2_iJ]

ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, "Longitude_t")
print(
    f"RMS of ϵ on potential density anomaly surface: {ϵ_RMS : 4e} [kg m-3] all grid distances = 1)"
)

ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, "Longitude_t", *geom)
print(
    f"Area-weighted RMS of ϵ on potential density anomaly surface: {ϵ_RMS : 4e} [kg m-4]"
)

# %% Delta surface
# Provide reference pressure and isovalue
s0, t0 = 34.5, 4.0
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=eos,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    ref=(s0, t0),
    isoval=0.0,
)

# Provide reference pressure and location for the surface to intersect (pin_cast and pin_p)
# and don't ask for diagnostics
s, t, z, _ = anomaly_surf(
    S,
    T,
    Z,
    eos=eos,
    eos_s_t=eos_s_t,
    diags=False,
    vert_dim="Depth_c",
    ref=(s0, t0),
    pin_cast=(i0, j0),
    pin_p=z0,
)

# Provide just the location to intersect (pin_cast, pin_p).
# This takes the reference pressure ref to match pin_p.
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=eos,
    eos_s_t=eos_s_t,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
)

ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, "Longitude_t")
print(
    f"RMS of ϵ on in-situ density anomaly surface: {ϵ_RMS : 4e} [kg m-3] all grid distances = 1)"
)

ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, "Longitude_t", *geom)
print(
    f"Area-weighted RMS of ϵ on in-situ density anomaly surface: {ϵ_RMS : 4e} [kg m-4]"
)


# %% Omega surface

# Initialize omega surface with a (locally referenced) sigma surface
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=eos,
    ITER_MAX=10,
    ITER_START_WETTING=1,
)
print(f'Total time  : {np.sum(d["timer"]) : .4f} sec')
print(f'      bfs time: {np.sum(d["timer_bfs"]) : .4f} sec')
print(f'   matrix time: {np.sum(d["timer_mat"]) : .4f} sec')
print(f'   update time: {np.sum(d["timer_update"]) : .4f} sec')

# Initialize omega surface with a (locally referenced) delta surface.
# Also remove the pre-computed mixed layer.  Could also pass
# p_ml={"bottle_index" : 1, "ref_p" : 0.0}
# for example, to compute mixed layer internally with the given parameters.
# Also provide grid distances.
geomargs = {
    "dist1_iJ": dist1_iJ,  # Distance [m] in 1st dim centred at (I-1/2, J)
    "dist1_Ij": dist1_Ij,  # Distance [m] in 1st dim centred at (I, J-1/2)
    "dist2_Ij": dist2_Ij,  # Distance [m] in 2nd dim centred at (I-1/2, J)
    "dist2_iJ": dist2_iJ,  # Distance [m] in 2nd dim centred at (I, J-1/2)
}
s, t, z, d = omega_surf(
    S,
    T,
    Z,
    ref=(None, None),
    wrap="Longitude_t",
    vert_dim="Depth_c",
    pin_cast=(i0, j0),
    pin_p=z0,
    eos=eos,
    ITER_MAX=10,
    ITER_START_WETTING=1,
    TOL_P_SOLVER=1e-5,
    p_ml=z_ml,
    **geomargs,
)
print(f'Total time  : {np.sum(d["timer"]) : .4f} sec')
print(f'      bfs time: {np.sum(d["timer_bfs"]) : .4f} sec')
print(f'   matrix time: {np.sum(d["timer_mat"]) : .4f} sec')
print(f'   update time: {np.sum(d["timer_update"]) : .4f} sec')

# %% Show figure mapping the depth of the (most recently calculated) surface

fig, ax = plt.subplots()
cs = ax.imshow(z.T, origin="lower")
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label("Depth [m]")
ax.set_title(r"Depth of surface in OCCA")


# %% Neutral Tangent Plane bottle to cast
from neutralocean.ntp import ntp_bottle_to_cast
from neutralocean.interp_ppc import linear_coeffs

sB, tB, zB = 35.0, 16.0, 500.0  # Thermodynamic properties of a given Bottle
S1 = S.values[180, 80, :]
T1 = T.values[180, 80, :]
s1, t1, z1 = ntp_bottle_to_cast(sB, tB, zB, S1, T1, Z)

# Or the more manual version:
from neutralocean.ntp import _ntp_bottle_to_cast
from neutralocean.lib import find_first_nan

n_good = find_first_nan(S1)[()]
S1ppc = linear_coeffs(Z, S1)
T1ppc = linear_coeffs(Z, T1)
s1, t1, z1 = _ntp_bottle_to_cast(sB, tB, zB, S1, T1, Z, S1ppc, T1ppc, n_good, eos, 1e-4)


# %% Veronis Density, used to label an approx neutral surface
S_ref_cast = S.values[i0, j0]
T_ref_cast = T.values[i0, j0]
ρ_v = veronis_density(
    S_ref_cast, T_ref_cast, Z, z0, eos=eos, eos_s_t=eos_s_t
)  # 1027.7700462375435


# %% Work with Numpy arrays instead of xarrays

# Convert S and T from xarray to numpy ndarrays, and make vertical dimension contiguous in memory.
# If not done here in advance, this will be done each time potential_surf, anomaly_surf, or omega_surf is called.
S, T, Z = _process_casts(S, T, Z, "Depth_c")

# %%
s, t, z, d = anomaly_surf(
    S,
    T,
    Z,
    eos=eos,
    eos_s_t=eos_s_t,
    wrap=(True, False),
    vert_dim=-1,
    ref=(s0, t0),
    isoval=0.0,
)


# %% Calculate a large-scale potential vorticity on our surface

# Earth sidereal day period [s]
Earth_day = 86164

# Coriolis param [s-1] on tracer grid
f = 2 * (2 * np.pi / Earth_day) * np.sin(g["YCvec"] * (np.pi / 180))

sz, tz = val2(Z, S, Sppc, T, Tppc, z, 1)  # ∂S/∂Z and ∂T/∂Z, on the surface
rs, rt = eos_s_t(s, t, z)  # ∂ρ/∂S and ∂ρ/∂T, on the surface

# ∂σ/∂z on the surface, where σ is the locally reference potential density.
# σz = rs * sz + rt * tz

# large-scale potential vorticity [m-1 s-1] defined via locally referenced potential density, on the surface
# q = f * σz

# ∂δ/∂z on the surface, where δ is the in-situ density anomaly
δz = rs * sz + rt * tz + (eos_z(s, t, z) - eos_z(s0, t0, z))

# large-scale potential vorticity [m-1 s-1] defined via in-situ density anomaly, on the surface
q = f * δz
