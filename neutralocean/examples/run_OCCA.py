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

# %% Load OCCA data

# *** EDIT `path_occa` AS NEEDED ***
path_occa = path_neutralocean + "/home/stanley/work/projects-gfd/neutralocean/neutralocean/examples/" 

# import wget
# url = 'ftp://mit.ecco-group.org/ecco_for_las/OCCA_1x1_v2/2004-6/annual/'
# url = "https://www.dropbox.com/s/q9hywvjup1mwhc9/DDsalt.0406annclim.nc"
# wget.download(url, path_occa)
# url = "https://www.dropbox.com/s/qr6bivfyk0s06ot/DDtheta.0406annclim.nc"
# wget.download(url, path_occa)

def load_OCCA(OCCA_dir, ts=0):

    # Read grid info from the theta nc file
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "theta")).load()

    # Build our own grid, as the MITgcm does it
    # Pa2db = 1e-4
    deg2rad = np.pi / 180

    g = dict()  # model grid and parameters
    g["ρ_c"] = 1027.5  # A guess. Same as ECCO2
    g["grav"] = 9.81  # A guess. Same as ECCO2
    g["rSphere"] = 6.37e6  # A guess. Same as ECCO2
    g["resx"] = 1  # 1 grid cell per zonal degree
    g["resy"] = 1  # 1 grid cell per meridional degree
    g["wrap"] = (True, False)  # periodic in longitude, not in latitude

    # Lateral coordinates
    g["XCvec"] = np.require(x.Longitude_t.values, dtype=np.float64, requirements="C")
    g["XGvec"] = np.require(x.Longitude_u.values, dtype=np.float64, requirements="C")
    g["YCvec"] = np.require(x.Latitude_t.values, dtype=np.float64, requirements="C")
    g["YGvec"] = np.require(x.Latitude_v.values, dtype=np.float64, requirements="C")

    # g["YCvec"] = g["YCvec"].reshape((1, -1))  # (Lon, Lat)
    # g["YGvec"] = g["YGvec"].reshape((1, -1))  # (Lon, Lat)

    # Lateral distances
    g["DXGvec"] = g["rSphere"] * np.cos(g["YGvec"] * deg2rad) / g["resx"] * deg2rad
    g["DYGsc"] = g["rSphere"] * deg2rad / g["resy"]
    g["DXCvec"] = g["rSphere"] * np.cos(g["YCvec"] * deg2rad) / g["resx"] * deg2rad
    g["DYCsc"] = g["DYGsc"]

    # g["DXCvec"] = g["DXCvec"].reshape((1, -1))  # (Lon, Lat)
    # g["DXGvec"] = g["DXGvec"].reshape((1, -1))  # (Lon, Lat)

    # Vertical coordinate and distances
    g["RC"] = -np.require(x.Depth_c, dtype=np.float64, requirements="C")
    g["DRC"] = np.diff(-g["RC"])

    g["nx"] = g["XCvec"].size
    g["ny"] = g["YCvec"].size
    g["nz"] = g["RC"].size

    # Vertical area of the tracer cells [m^2]
    g["RACvec"] = (g["rSphere"] ** 2 / g["resx"] * deg2rad) * abs(
        np.sin((g["YGvec"] + 1 / g["resy"]) * deg2rad) - np.sin(g["YGvec"] * deg2rad)
    )

    T = x.theta.isel(Time=ts)
    x.close()

    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "salt")).load()
    S = x.salt.isel(Time=ts)
    x.close()

    # # phihyd = Pres / rho_c +  grav * z
    # x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "phihyd")).load()
    # P = x.phihyd.isel(Time=ts)

    # # convert to full in-situ pressure, in [dbar]
    # Z3D = -g["RC"].reshape(tuple(-1 if x == "Depth_c" else 1 for x in P.dims))
    # P = (P + g["grav"] * Z3D) * (g["ρ_c"] * Pa2db)
    # x.close()

    # x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "etan"))
    # η = x.etan.isel(Time=ts)
    # x.close()

    # # Reorder dimensions to ensure individual water columns are float64 and contiguous in memory
    dims = ("Longitude_t", "Latitude_t", "Depth_c")
    # S, T, P = (x.transpose(*dims).astype(np.float64, order="C") for x in (S, T, P))
    S, T = (x.transpose(*dims).astype(np.float64, order="C") for x in (S, T))
    # η = η.transpose(*dims[0:-1]).astype(np.float64, order="C")

    # ATMP = 0.  # Atmospheric Pressure (loading)
    # SAP = 0.  # Standard Atmospheric Pressure

    return g, S, T  # , P, η


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
