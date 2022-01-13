import numpy as np
import xarray as xr


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
