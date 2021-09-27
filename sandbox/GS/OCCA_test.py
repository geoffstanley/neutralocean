#!/usr/bin/env python
# coding: utf-8

# %% Imports

# %matplotlib notebook
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from neutral_surfaces import approx_neutral_surf
from neutral_surfaces.load_data import load_OCCA
from neutral_surfaces.lib import ntp_ϵ_errors_norms

from neutral_surfaces.lib import veronis_density

from neutral_surfaces.eos.eostools import make_eos, make_eos_s_t

# %% Load OCCA data
g, S, T, P, _ = load_OCCA("~/work/data/OCCA/")
Z = -g['RC']

ni, nj, nk = S.shape
nij = ni * nj

# Select pinning cast
i0 = int(ni / 2)
j0 = int(nj / 2)
z0 = 1500.

eos = make_eos('jmd95', g['grav'], g['ρ_c'])
eos_s_t = make_eos_s_t('jmd95', g['grav'], g['ρ_c'])

# %% Neutral Tangent Plane bottle to cast
from neutral_surfaces.lib import ntp_bottle_to_cast, _ntp_bottle_to_cast, find_first_nan
from neutral_surfaces.interp_ppc import linear_coeffs
S1 = S.values[180,80,:]
T1 = T.values[180,80,:]
s0, t0, p0 = ntp_bottle_to_cast(35., 16., 500., S1, T1, Z)

n_good = find_first_nan(S1)[()]
S1ppc = linear_coeffs(Z, S1)
T1ppc = linear_coeffs(Z, T1)
s0, t0, p0 = _ntp_bottle_to_cast(35., 16., 500., S1, T1, Z, S1ppc, T1ppc, n_good, 1e-4, eos)


# %% Veronis Density
S_ref_cast = S.values[i0,j0]
T_ref_cast = T.values[i0,j0]
ρ_v = veronis_density(S_ref_cast, T_ref_cast, Z, 1500., eos=eos, eos_s_t=eos_s_t)  # 1027.7700462375435


# %% Potential Density surface

# Provide reference pressure and  isovalue
s, t, z, d = approx_neutral_surf('sigma', S, T, Z, 
                                 eos='jmd95', grav=g['grav'], rho_c=g['ρ_c'],
                                 wrap="Longitude_t", vert_dim="Depth_c", 
                                 ref=0., isoval=1027.9)

# Provide reference pressure and location for the surface to intersect (pin_loc and pin_p)
s, t, z, d = approx_neutral_surf('sigma', S, T, Z, 
                                 eos=eos, eos_s_t=eos_s_t,
                                 wrap="Longitude_t", vert_dim="Depth_c", 
                                 ref=0., pin_loc=(i0, j0), pin_p=z0)

# Provide just the location to intersect (pin_loc, pin_p). 
# This takes the reference pressure ref to match pin_p.
s, t, z, d = approx_neutral_surf('sigma', S, T, Z, 
                                 eos=eos, eos_s_t=eos_s_t,
                                 wrap="Longitude_t", vert_dim="Depth_c", 
                                 pin_loc=(i0, j0), pin_p=z0)

ϵ_RMS, ϵ_MAV = ntp_ϵ_errors_norms(s, t, z, eos_s_t, g['wrap'])

# %% Delta surface
# s, t, z = delta_surf(S, T, Z, axis=-1, pin=(i0, j0, z0),
#                      tol_p=1e-4, eos='jmd95', grav=g['grav'], rho_c=g['ρ_c'])
# # s, t, z, _ = approx_neutral_surf(
# #     'delta', S, T, Z, axis=-1, tol_p=1e-4, eos='jmd95', grav=g['grav'], rho_c=g['ρ_c'],
# #     pin=(i0, j0, z0)
# #     )
# s_delta, t_delta, z_delta = s, t, z  # save alias

# %% Omega surface

# Providing a point at which to pin the surface (pin_loc and pin_p).  Without
# providing ref, the omega surface iterations will be initialized by a potential
# density surface, referenced to the local pressure (pin_p). 
s, t, z, d = approx_neutral_surf(
    'omega',
    S, T, Z,
    wrap="Longitude_t", 
    vert_dim="Depth_c",
    pin_loc=(i0, j0), pin_p = z0,
    eos='jmd95', grav=g['grav'], rho_c=g['ρ_c'],
    ITER_MAX=10, ITER_START_WETTING=1,
    tol_p=1e-4,
)
# s, t, z, diags = approx_neutral_surf(
#     'omega', S, T, Z, axis=-1, tol_p=1e-4, eos='jmd95', grav=g['grav'], rho_c=g['ρ_c'],
#     pin=(i0, j0, z0), wrap=g['wrap'],
#     ITER_MAX=10
#     )
print(f'Total time  : {np.sum(d["timer"]) : .4f} sec')
print(f'      bfs time: {np.sum(d["timer_bfs"]) : .4f} sec')
print(f' matbuild time: {np.sum(d["timer_matbuild"]) : .4f} sec')
print(f' matsolve time: {np.sum(d["timer_matsolve"]) : .4f} sec')
print(f'   update time: {np.sum(d["timer_update"]) : .4f} sec')

# old tests below:

# Initial surface has log_10(|ϵ|_2) = -2.342477 ..................
# Iter  1 [  0.19 sec] log_10(|ϵ|_2) = -3.972863 by |ϕ|_1 = 2.015180e-02;   29 casts freshly wet; |Δp|_2 = 2.009176e+01
# Iter  2 [  0.21 sec] log_10(|ϵ|_2) = -4.279229 by |ϕ|_1 = 3.621773e-04;  272 casts freshly wet; |Δp|_2 = 1.701764e+00
# Iter  3 [  0.23 sec] log_10(|ϵ|_2) = -4.287579 by |ϕ|_1 = 2.237068e-05;    2 casts freshly wet; |Δp|_2 = 1.468717e-01
# Iter  4 [  0.25 sec] log_10(|ϵ|_2) = -4.287644 by |ϕ|_1 = 9.968479e-07;    0 casts freshly wet; |Δp|_2 = 7.314240e-03
# Iter  5 [  0.24 sec] log_10(|ϵ|_2) = -4.287643 by |ϕ|_1 = 1.449314e-07;    0 casts freshly wet; |Δp|_2 = 2.144395e-03
# Iter  6 [  0.23 sec] log_10(|ϵ|_2) = -4.287643 by |ϕ|_1 = 2.316665e-08;    0 casts freshly wet; |Δp|_2 = 3.548166e-04


# z_omega, s, t, diags = omega_surf(
#     S, T, Z, z_delta, (i0, j0), g['wrap'], axis=-1, ITER_MAX=10, ITER_START_WETTING=np.inf,
#     DIST1_iJ=g['DXCvec'],  # Distance [m] in 1st dimension centred at (I-1/2, J)
#     DIST2_Ij=g['DYCsc'],  # Distance [m] in 2nd dimension centred at (I, J-1/2)
#     DIST2_iJ=g['DYGsc'],  # Distance [m] in 2nd dimension centred at (I-1/2, J)
#     DIST1_Ij=g['DXGvec'],  # Distance [m] in 1st dimension centred at (I, J-1/2)
#     )
# Initial surface has log_10(|ϵ|_2) = -7.723916 ..................
# Iter  1 [  0.20 sec] log_10(|ϵ|_2) = -8.726053 by |ϕ|_1 = 1.040064e-02;    0 casts freshly wet; |Δp|_2 = 2.167515e+01
# Iter  2 [  0.19 sec] log_10(|ϵ|_2) = -9.237233 by |ϕ|_1 = 6.336332e-04;    0 casts freshly wet; |Δp|_2 = 6.089017e+00
# Iter  3 [  0.19 sec] log_10(|ϵ|_2) = -9.338989 by |ϕ|_1 = 6.601303e-05;    0 casts freshly wet; |Δp|_2 = 1.187603e+00
# Iter  4 [  0.19 sec] log_10(|ϵ|_2) = -9.340732 by |ϕ|_1 = 6.353485e-06;    0 casts freshly wet; |Δp|_2 = 2.204033e-01
# Iter  5 [  0.18 sec] log_10(|ϵ|_2) = -9.340763 by |ϕ|_1 = 6.398824e-07;    0 casts freshly wet; |Δp|_2 = 2.975967e-02
# Iter  6 [  0.17 sec] log_10(|ϵ|_2) = -9.340763 by |ϕ|_1 = 6.579950e-08;    0 casts freshly wet; |Δp|_2 = 3.238430e-03
# Note:  10 ** -9.340763 == 4.56285848989746e-10  -- matches Stanley et al (2021) Fig 4.

# %% Show figure

# fig, ax = plt.subplots()
# cs = ax.imshow(z.T, origin="lower")
# # cs = ax.contourf(lon, lat, z_sigma.T)
# cbar = fig.colorbar(cs, ax=ax)
# cbar.set_label("Depth [m]")
# ax.set_title(r"Depth of surface in OCCA")
