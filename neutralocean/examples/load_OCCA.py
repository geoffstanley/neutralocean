import numpy as np
import xarray as xr

import pooch


def load_OCCA():
    """
    Load grid, salinity, and potential temperature for OCCA 2004-2006 average

    Returns
    -------
    g : dict
        Grid information

    S : xarray.DataArray
        Salinity

    T : xarray.DataArray
        Potential temperature
    """

    # Use friendly pooch to download dataset, if it hasn't already done so
    # and cached it locally.
    url_salt = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/RNXA2A/VREV1Z"
    url_theta = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/RNXA2A/APHURG"

    hash_salt = (
        "184ca879e0ed18a078f31160c16ec078701e0f1af4d6de44bd1e70cc200d111c"
    )
    hash_theta = (
        "7fcef4663439e34b13de4b08a373a0e17b83bee2c621ea8ac07a8175fe15368e"
    )

    file_salt = pooch.retrieve(url=url_salt, known_hash=hash_salt)
    file_theta = pooch.retrieve(url=url_theta, known_hash=hash_theta)

    ts = 0  # Select the first time step (January)

    # Read grid info from the theta nc file
    ds = xr.open_dataset(file_theta).load()

    # Build our own grid, as the MITgcm does it
    # Pa2db = 1e-4
    d2r = np.pi / 180

    g = dict()  # model grid and parameters
    g["rho_c"] = 1027.5  # A guess. Same as ECCO2
    g["grav"] = 9.81  # A guess. Same as ECCO2
    g["rSphere"] = 6.37e6  # A guess. Same as ECCO2
    g["resx"] = 1  # 1 grid cell per zonal degree
    g["resy"] = 1  # 1 grid cell per meridional degree
    g["wrap"] = (True, False)  # periodic in longitude, not in latitude

    # Lateral coordinates
    g["XCvec"], g["XGvec"], g["YCvec"], g["YGvec"] = (
        np.require(ds[x].values, dtype=np.double, requirements="C")
        for x in ("Longitude_t", "Longitude_u", "Latitude_t", "Latitude_v")
    )

    # Lateral distances
    g["DXGvec"] = g["rSphere"] * np.cos(g["YGvec"] * d2r) / g["resx"] * d2r
    g["DYGsc"] = g["rSphere"] * d2r / g["resy"]
    g["DXCvec"] = g["rSphere"] * np.cos(g["YCvec"] * d2r) / g["resx"] * d2r
    g["DYCsc"] = g["DYGsc"]

    # Vertical coordinate and distances
    g["RC"] = np.require(-ds.Depth_c, dtype=np.double, requirements="C")
    g["DRC"] = np.diff(-g["RC"])

    g["nx"] = g["XCvec"].size
    g["ny"] = g["YCvec"].size
    g["nz"] = g["RC"].size

    # Vertical area of the tracer cells [m^2]
    g["RACvec"] = (g["rSphere"] ** 2 / g["resx"] * d2r) * abs(
        np.sin((g["YGvec"] + 1 / g["resy"]) * d2r) - np.sin(g["YGvec"] * d2r)
    )

    T = ds.theta.isel(Time=ts)
    ds.close()

    ds = xr.open_dataset(file_salt).load()
    S = ds.salt.isel(Time=ts)
    ds.close()

    # Reorder dimensions to ensure individual water columns are float64 and contiguous in memory
    dims = ("Longitude_t", "Latitude_t", "Depth_c")
    S, T = (x.transpose(*dims).astype(np.double, order="C") for x in (S, T))

    return g, S, T
