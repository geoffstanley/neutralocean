import numpy as np
import xarray as xr

import os
import urllib

try:
    import wget

    have_wget = True
except:
    have_wget = False


def load_OCCA(folder=None):
    """
    Load grid, salinity, and potential temperature for OCCA 2004-2006 average

    Parameters
    ----------
    folder : str, Default None
        Path to folder containing DDsalt.0406annclim.nc and DDtheta.0406annclim.nc
        If None, these .nc files will be downloaded from the internet.

    Returns
    -------
    g : dict
        Grid information

    S : xarray.DataArray
        Salinity

    T : xarray.DataArray
        Potential temperature
    """

    # First check .nc files exist, and download them if not
    if folder is None:
        file_salt, file_theta = download_OCCA()

    ts = 0  # Select the first time step (the only time step in these files)

    # Read grid info from the theta nc file
    x = xr.open_dataset(file_theta).load()

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

    T = x.theta.isel(Time=ts)
    x.close()

    x = xr.open_dataset(file_salt).load()
    S = x.salt.isel(Time=ts)
    x.close()

    # # Reorder dimensions to ensure individual water columns are float64 and contiguous in memory
    dims = ("Longitude_t", "Latitude_t", "Depth_c")
    S, T = (x.transpose(*dims).astype(np.float64, order="C") for x in (S, T))

    # Remove any downloaded files that were temporary
    urllib.request.urlcleanup()

    return g, S, T


def download_OCCA(folder=None):
    """
    Download OCCA annual average S and T .nc files into given folder, or into
    the folder containing this script.  If that folder is not writable, fall-
    back to downloading to a temporary file.
    """

    if folder is None:
        folder = local_folder()

    # The dropbox links can be used for testing... they are faster:
    url_salt = "https://www.dropbox.com/s/q9hywvjup1mwhc9/DDsalt.0406annclim.nc?dl=1"
    url_theta = "https://www.dropbox.com/s/qr6bivfyk0s06ot/DDtheta.0406annclim.nc?dl=1"
    # url_salt = "ftp://mit.ecco-group.org/ecco_for_las/OCCA_1x1_v2/2004-6/annual/DDsalt.0406annclim.nc"
    # url_theta = "ftp://mit.ecco-group.org/ecco_for_las/OCCA_1x1_v2/2004-6/annual/DDtheta.0406annclim.nc"

    file_salt = folder + "DDsalt.0406annclim.nc"
    file_theta = folder + "DDtheta.0406annclim.nc"

    # Check if the files are already there
    if (
        os.access(folder, os.R_OK)
        and os.path.exists(file_salt)
        and os.path.exists(file_theta)
    ):
        return file_salt, file_theta

    # The files aren't there, so download them (using wget if available,
    # otherwise fall-back on urllib)
    if os.access(folder, os.W_OK):

        if not os.path.exists(file_salt):
            print(
                "Attempt download of OCCA annual average salinity (~11mb) "
                f"from {url_salt}"
            )
            if have_wget:
                wget.download(url_salt, file_salt)
            else:
                _, _ = urllib.request.urlretrieve(url_salt, file_salt)

        if not os.path.exists(file_theta):
            print(
                "Attempt download of OCCA annual average potential temperature (~11mb) "
                f"from {url_theta}"
            )
            if have_wget:
                wget.download(url_theta, file_theta)
            else:
                _, _ = urllib.request.urlretrieve(url_salt, file_salt)

    else:
        print(
            "Attempt download of OCCA annual average salinity (~11mb) "
            f"from {url_salt} to a temporary file ... standby ..."
        )
        file_salt, _ = urllib.request.urlretrieve(url_salt)
        print(f"... saved as {file_salt}")

        print(
            "Attempt download of OCCA annual average potential temperature (~11mb) "
            f"from {url_theta} to a temporary file ... standby ..."
        )
        file_theta, _ = urllib.request.urlretrieve(url_theta)
        print(f"... saved as {file_theta}")

    return file_salt, file_theta


def local_folder():
    return os.path.dirname(os.path.abspath(__file__)) + os.path.sep
