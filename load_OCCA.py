# import os
import numpy as np
import netCDF4 as nc

class gridC(object):

    def __init__(self, ρ_c=1027.5, grav=9.81, rSphere=6.37e6, WRAP=(True, True), XCvec=[], XGvec=[], YCvec=[], YGvec=[], RC=[], resx=1, resy=1):
        self.ρ_c = ρ_c
        self.grav = grav
        self.rSphere = rSphere
        self.WRAP = WRAP
        self.XCvec = XCvec
        self.YCvec = YCvec
        self.XGvec = XGvec
        self.YGvec = YGvec
        self.RC = RC

        self.nx = XCvec.size
        self.ny = YCvec.size
        self.nz = RC.size
        self.RACvec = rSphere**2 / resx * deg2rad * abs(np.sin((YGvec + 1/resy)*deg2rad) - np.sin(YGvec*deg2rad))    # Vertical area of the tracer cells [m^2]
        self.RAWvec = RACvec                                                                                         # Vertical area of the U cells [m^2]
        self.RASvec = rSphere**2 / resx * deg2rad * abs(np.sin(YCvec * deg2rad) - np.sin((YCvec - 1/resy)*deg2rad))  # Vertical area of the V cells [m^2]
        self.RAZvec = rSphere**2 / resx * deg2rad * abs(np.sin(YCvec*deg2rad) - np.sin((YCvec - 1/resy)*deg2rad))    # Vertical area of the vorticity cells [m^2]
        self.DXGvec = rSphere * np.cos(YGvec * deg2rad) / resx * deg2rad
        self.DYGsc = rSphere * deg2rad / resy
        self.DXCvec = rSphere * np.cos(YCvec * deg2rad) / resx * deg2rad
        self.DYCsc = DYGsc
        self.DRC = np.diff(-RC)


def load_OCCA(PATH_OCCA, ts=0):

    # Read grid info from the theta nc file
    varname = 'theta'
    ds = nc.Dataset('%sDD%s.0406annclim.nc' % (PATH_OCCA, varname), 'r')
    ds.set_auto_mask(False)
    XCvec = np.float64(ds['Longitude_t'][:])
    XGvec = np.float64(ds['Longitude_u'][:])
    YCvec = np.float64(ds['Latitude_t'][:])
    YGvec = np.float64(ds['Latitude_v'][:])
    Depth_c = np.float64(ds['Depth_c'][:])
    ds.close()

    # Build our own grid, as the MITgcm does it
    Pa2db = 1e-4

    g = gridC(ρ_c = 1027.5,  # A guess. Same as ECCO2
        grav = 9.81,  # A guess. Same as ECCO2
        rSphere = 6.37e6,  # A guess. Same as ECCO2
        WRAP = (True, False),
        XCvec = XCvec,
        XGvec = XGvec,
        YCvec = YCvec,
        YGvec = YGvec,
        RC = -Depth_c,
        resx = 1,  # 1 grid cell per zonal degree
        resy = 1  # 1 grid cell per meridional degree
        )


    varname = 'theta'
    ds = nc.Dataset('%sDD%s.0406annclim.nc' % (PATH_OCCA, varname), 'r')
    ds.set_auto_mask(False)
    T = np.float64(ds[varname][ts])
    T[T == ds[varname].missing_value] = np.NaN
    ds.close()

    varname = 'salt'
    ds = nc.Dataset('%sDD%s.0406annclim.nc' % (PATH_OCCA, varname), 'r')
    ds.set_auto_mask(False)
    S = np.float64(ds[varname][ts])
    S[S == ds[varname].missing_value] = np.NaN
    ds.close()

    # phihyd = Pres / rho_c +  grav * z
    varname = 'phihyd'
    ds = nc.Dataset('%sDD%s.0406annclim.nc' % (PATH_OCCA, varname), 'r')
    ds.set_auto_mask(False)
    P = np.float64(ds[varname][ts])
    P[P == ds[varname].missing_value] = np.NaN
    ds.close()
    P = (P - g.grav * g.RC.reshape(-1,1,1)) * (g.ρ_c * Pa2db)  # convert to full in-situ pressure, in [dbar]

    varname = 'etan'
    ds = nc.Dataset('%sDD%s.0406annclim.nc' % (PATH_OCCA, varname), 'r')
    ds.set_auto_mask(False)
    ETAN = np.float64(ds[varname][ts])
    ETAN[ETAN == ds[varname].missing_value] = np.NaN
    ds.close()

    ATMP = 0.
    SAP = 0.

    return (g, S, T, P, ETAN, ATMP, SAP)
