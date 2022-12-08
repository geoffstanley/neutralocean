import numpy as np


def synthocean(
    shape, pbot=4000.0, SSSSa=0.3, zonally_uniform=False, wrap=(False, False)
):
    """
    Synthetic idealization of the Pacific and Southern Ocean with tuneable parameters

    Parameters
    ----------
    shape : tuple of int
        A three element tuple giving the dimensions of the output.  The elements
        specify the number of points in longitude, latitude, and depth space,
        respectively.

    pbot : float, Default 4000.0
        Value of `P` at the bottommost data point, i.e. `P[-1]`

    SSSSa : float, Default 0.3
        Southern Sea Surface Salinity anomaly: the sea surface salinity is
        increased by `SSSS` in the southernmost grid cells, and by 0 in the
        northernmost grid cells, and linearly in between. The degree to
        which the southern casts are statically unstable can be controlled
        by this parameter. Increase to make more unstable.

    zonally_uniform : bool, Default False
        If `True`, the synthetic ocean is uniform in longitude, and the neutral
        helicity is zero.  If `False`, some zonal structure is created which
        results in non-zero neutral helicity.

    wrap : tuple of bool, Default (False, False)
        Specify periodicity of the lateral dimensions.
        When `wrap[0]` is `False`, `S[0, :, :] = T[0, :, :] = nan`.
        When `wrap[1]` is `False`, `S[:, 0, :] = T[:, 0, :] = nan`.


    Returns
    -------
    S, T : ndarray
        The practical / Absolute Salinity and potential / Conservative
        Temperature as a 3D array.

    P : ndarray
        The pressure / depth as a 1D array.

    g : dict
        The geometry of the grid, including latitude, longitude, cell widths
        and areas.  See code for details.

    """

    ni, nj, nk = shape

    X = np.linspace(-1, 1, ni).reshape((ni, 1, 1))  # longitude (temporarily scaled)
    Ynorth = 60
    Ysouth = -80
    Y = np.linspace(Ysouth, Ynorth, nj).reshape((1, nj, 1))  # latitude

    # Set surface T
    #   Linear model Poly5:
    #   f(x) = p1*x^5 + p2*x^4 + p3*x^3 + p4*x^2 + p5*x + p6
    #   where x is normalized by mean -7 and std 41.14
    #   Coefficients (with 95# confidence bounds):
    p1 = 0.6265  #  (0.3083, 0.9446)
    p2 = 3.269  #  (2.994, 3.545)
    p3 = -3.602  #  (-4.682, -2.522)
    p4 = -18.93  #  (-19.66, -18.19)
    p5 = 5.927  #  (5.108, 6.745)
    p6 = 27.41  #  (27.06, 27.76)
    x = (Y - -7) / 41.14
    Ts = p1 * x ** 5 + p2 * x ** 4 + p3 * x ** 3 + p4 * x ** 2 + p5 * x + p6

    if zonally_uniform:
        Ts = np.tile(Ts, (ni, 1, 1))  # no helicity
    else:
        # add some east-west structure
        Ts = Ts - 1.5 * X

    # contract extreme values into a given range
    Tmin = -1.8
    Tmax = 30.0
    Ts = (Ts - np.min(Ts)) / (np.max(Ts) - np.min(Ts)) * (Tmax - Tmin) + Tmin

    # Set surface S
    #   Linear model Poly4:
    #   f(x) = p1*x^4 + p2*x^3 + p3*x^2 + p4*x + p5
    #   where x is normalized by mean -7 and std 41.14
    #   Coefficients (with 95# confidence bounds):
    p1 = 0.2221  #  (0.1526, 0.2916)
    p2 = -0.09487  #  (-0.1553, -0.03444)
    p3 = -1.355  #  (-1.54, -1.17)
    p4 = -0.126  #  (-0.2438, -0.00815)
    p5 = 35.59  #  (35.51, 35.68)
    x = (Y - -7) / 41.14
    Ss = p1 * x ** 4 + p2 * x ** 3 + p3 * x ** 2 + p4 * x + p5

    if zonally_uniform:
        Ss = np.tile(Ss, (ni, 1, 1))  # no helicity
    else:
        # add some east-west structure (this adds helicity)
        Ss = Ss + (0.4 + 0.5 * X - 0.8 * X ** 2)

    # Tilt surface salinity function, so high SSS in South (if SSSSa > 0)
    Ss = Ss + np.linspace(SSSSa, 0, nj).reshape((1, nj, 1))

    # Set bottom T,S
    # linearly warmer and fresher moving northwards
    Tb = np.linspace(0, 1, nj).reshape((1, nj, 1)) + (Tmin - 0.1)
    Sb = np.linspace(34.7, 34.5, nj).reshape((1, nj, 1))

    # Add depth structure by linear interpolation of surface->bottom T,S data
    S = np.linspace(0, 1, nk).reshape((1, 1, nk)) * (Sb - Ss) + Ss
    T = np.linspace(0, 1, nk).reshape((1, 1, nk)) * (Tb - Ts) + Ts

    # Nonlinear spacing of pressure means dTdp and dSdp will be non-uniform
    P = np.linspace(0, 1, nk) ** 3 * pbot

    # Add walls
    if not wrap[0]:
        S[0, :, :] = np.nan
        T[0, :, :] = np.nan
    if not wrap[1]:
        S[:, 0, :] = np.nan
        T[:, 0, :] = np.nan

    X = (X + 1) * 180  # change longitude, to be [0, 360].

    g = dict()
    g["nx"] = ni  # number of zonal grid points
    g["ny"] = nj  # number of meridional grid points
    g["nz"] = nk  # number of vertical grid points
    g["rSphere"] = 6370000  # Earth radius [m]
    g["resx"] = ni / 360  # Number of tracer cells per zonal degree
    g["resy"] = nj / (Ynorth - Ysouth)  # Number of tracer cells per meridional degree

    # Override the loaded grid variables with ones calculated in accordance with
    # MITgcm's spherical polar grid: See INI_SPHERICAL_POLAR_GRID.F
    # You may wish to do this if your grid data is given in single precision.

    # Create our own spherical-polar grid. Short-cuts have been
    # taken in the formulas, appropriate for a uniform
    # latitude-longitude grid. See INI_SPHERICAL_POLAR_GRID.F
    # if adjustments must be made for non-uniform grid.
    deg2rad = np.pi / 180

    # fmt: off
    g["XGvec"] = 0 + np.arange(g["nx"]) / g["resx"]  # longitude on west edge of tracer cell
    g["YGvec"] = Ysouth + np.arange(g["ny"]) / g["resy"]  # latitude on south edge of tracer cell
    g["XCvec"] = g["XGvec"] + 1 / (2 * g["resx"]) # longitude at centre of tracer cell
    g["YCvec"] = g["YGvec"] + 1 / (2 * g["resy"]) # latitude at centre of tracer cell

    g["DXGvec"] = g["rSphere"] * np.cos(g["YGvec"] * deg2rad) / g["resx"] * deg2rad  # zonal distance of tracer cell at south edge [m]
    g["DYGsc"] = g["rSphere"] * deg2rad / g["resy"]  # meridional distance of tracer cell at west edge [m]
    g["DXCvec"] = g["rSphere"] * np.cos(g["YCvec"] * deg2rad) / g["resx"] * deg2rad  # zonal distance between cell centres of tracer cell and its western neighbour [m]
    g["DYCsc"] = g["DYGsc"]  # meridional distance between cell centres of tracer cell and its southern neighbour [m]

    g["RACvec"] = g["rSphere"] * g["rSphere"] / g["resx"] * deg2rad * abs( np.sin((g["YGvec"] + 1/g["resy"])*deg2rad) - np.sin(g["YGvec"]*deg2rad) )  # area of tracer cell [m2]
    g["RAWvec"] = g["RACvec"]  # area of cell centred on west edge of tracer cell [m2]
    g["RASvec"] = g["rSphere"] * g["rSphere"] / g["resx"] * deg2rad * abs( np.sin(g["YCvec"] * deg2rad) - np.sin((g["YCvec"] - 1/g["resy"])*deg2rad) )  # area of cell centred on south edge of tracer cell [m2]
    g["RAZvec"] = g["rSphere"] * g["rSphere"] / g["resx"] * deg2rad * abs( np.sin(g["YCvec"] * deg2rad) - np.sin((g["YCvec"] - 1/g["resy"])*deg2rad) )  # area of vorticity cell, centred on south-west corner of tracer cell [m2]
    # fmt: on

    return S, T, P, g
