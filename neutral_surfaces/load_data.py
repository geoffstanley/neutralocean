import numpy as np
import xarray as xr


class gridC(object):
    def __init__(
        self,
        ρ_c=1027.5,
        grav=9.81,
        rSphere=6.37e6,
        wrap=(True, True),
        XCvec=[],
        XGvec=[],
        YCvec=[],
        YGvec=[],
        RC=[],
        resx=1,
        resy=1,
    ):
        self.ρ_c = ρ_c
        self.grav = grav
        self.rSphere = rSphere
        self.wrap = wrap
        self.XCvec = XCvec
        self.YCvec = YCvec
        self.XGvec = XGvec
        self.YGvec = YGvec
        self.RC = RC

        deg2rad = np.pi / 180

        self.nx = XCvec.size
        self.ny = YCvec.size
        self.nz = RC.size
        self.RACvec = (
            rSphere ** 2
            / resx
            * deg2rad
            * abs(np.sin((YGvec + 1 / resy) * deg2rad) - np.sin(YGvec * deg2rad))
        )  # Vertical area of the tracer cells [m^2]
        self.RAWvec = self.RACvec  # Vertical area of the U cells [m^2]
        self.RASvec = (
            rSphere ** 2
            / resx
            * deg2rad
            * abs(np.sin(YCvec * deg2rad) - np.sin((YCvec - 1 / resy) * deg2rad))
        )  # Vertical area of the V cells [m^2]
        self.RAZvec = (
            rSphere ** 2
            / resx
            * deg2rad
            * abs(np.sin(YCvec * deg2rad) - np.sin((YCvec - 1 / resy) * deg2rad))
        )  # Vertical area of the vorticity cells [m^2]
        self.DXGvec = rSphere * np.cos(YGvec * deg2rad) / resx * deg2rad
        self.DYGsc = rSphere * deg2rad / resy
        self.DXCvec = rSphere * np.cos(YCvec * deg2rad) / resx * deg2rad
        self.DYCsc = self.DYGsc
        self.DRC = np.diff(-RC)

    def XCvecpad(self):
        return np.hstack((self.XCvec, self.XCvec[-1] + (self.XCvec[1] - self.XCvec[0])))

    def YCvecpad(self):
        return np.hstack((self.YCvec, self.YCvec[-1] + (self.YCvec[1] - self.YCvec[0])))


def load_OCCA(OCCA_dir, ts=0):

    # Read grid info from the theta nc file
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "theta"))
    XCvec = np.require(x.Longitude_t.values, dtype=np.float64, requirements="C")
    XGvec = np.require(x.Longitude_u.values, dtype=np.float64, requirements="C")
    YCvec = np.require(x.Latitude_t.values, dtype=np.float64, requirements="C")
    YGvec = np.require(x.Latitude_v.values, dtype=np.float64, requirements="C")
    Depth_c = np.require(x.Depth_c, dtype=np.float64, requirements="C")
    x.close()

    # Build our own grid, as the MITgcm does it
    Pa2db = 1e-4

    g = gridC(
        ρ_c=1027.5,  # A guess. Same as ECCO2
        grav=9.81,  # A guess. Same as ECCO2
        rSphere=6.37e6,  # A guess. Same as ECCO2
        wrap=(True, False),
        XCvec=XCvec,
        XGvec=XGvec,
        YCvec=YCvec,
        YGvec=YGvec,
        RC=-Depth_c,
        resx=1,  # 1 grid cell per zonal degree
        resy=1,  # 1 grid cell per meridional degree
    )

    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "theta"))
    T = x.theta.values[ts]  # (depth, lat, lon)
    T = np.moveaxis(
        T, (0, 1, 2), (2, 1, 0)
    )  # Move vertical axis to end.  (lon, lat, depth)
    T = np.require(T, dtype=np.float64, requirements="C")
    x.close()

    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "salt"))
    S = x.salt.values[ts]  # (depth, lat, lon)
    S = np.moveaxis(
        S, (0, 1, 2), (2, 1, 0)
    )  # Move vertical axis to end.  (lon, lat, depth)
    S = np.require(S, dtype=np.float64, requirements="C")
    x.close()

    # phihyd = Pres / rho_c +  grav * z
    varname = "phihyd"
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "phihyd"))
    P = x.phihyd.values[ts]  # (depth, lat, lon)
    P = np.moveaxis(
        P, (0, 1, 2), (2, 1, 0)
    )  # Move vertical axis to end.  (lon, lat, depth)
    P = np.require(P, dtype=np.float64, requirements="C")
    P = (P - g.grav * g.RC.reshape(1, 1, -1)) * (
        g.ρ_c * Pa2db
    )  # convert to full in-situ pressure, in [dbar]
    x.close()

    varname = "etan"
    x = xr.open_dataset("%sDD%s.0406annclim.nc" % (OCCA_dir, "etan"))
    η = x.etan.values[ts]  # (lat, lon)
    η = η.T  # (lon, lat)
    η = np.require(η, dtype=np.float64, requirements="C")

    # ATMP = 0.
    # SAP = 0.

    return g, S, T, P, η
