import numpy as np
import xarray as xr


def load_OCCA(OCCA_dir, ts=0):

    # Read grid info from the theta nc file
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "theta")).load()

    # Build our own grid, as the MITgcm does it
    Pa2db = 1e-4
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

    # Lateral distances
    g["DXGvec"] = g["rSphere"] * np.cos(g["YGvec"] * deg2rad) / g["resx"] * deg2rad
    g["DYGsc"] = g["rSphere"] * deg2rad / g["resy"]
    g["DXCvec"] = g["rSphere"] * np.cos(g["YCvec"] * deg2rad) / g["resx"] * deg2rad
    g["DYCsc"] = g["DYGsc"]

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

    T = (
        x.theta.isel(Time=ts)
        .transpose("Longitude_t", "Latitude_t", "Depth_c")
        .astype(np.float64, order="C")
    )
    x.close()

    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "salt")).load()
    S = (
        x.salt.isel(Time=ts)
        .transpose("Longitude_t", "Latitude_t", "Depth_c")
        .astype(np.float64, order="C")
    )
    x.close()

    # phihyd = Pres / rho_c +  grav * z
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "phihyd")).load()
    P = (
        x.phihyd.isel(Time=ts)
        .transpose("Longitude_t", "Latitude_t", "Depth_c")
        .astype(np.float64, order="C")
    )
    # convert to full in-situ pressure, in [dbar]
    P = (P - g["grav"] * g["RC"].reshape(1, 1, -1)) * (g["ρ_c"] * Pa2db)
    x.close()

    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "etan"))
    η = (
        x.etan.isel(Time=ts)
        .transpose("Longitude_t", "Latitude_t")
        .astype(np.float64, order="C")
    )
    x.close()

    # ATMP = 0.
    # SAP = 0.

    return g, S, T, P, η
