import numpy as np
import os
import matplotlib.pyplot as plt

import densjmd95
import fzero
import ppc
import load_data
import neutral_surfaces

PATH_OCCA = os.path.expanduser('~/work/data/OCCA/')
(g, S, T, P, ETAN, ATMP, SAP) = load_data.OCCA(PATH_OCCA)
Z = -g.RC
Sppc = ppc.linterp(Z, S)
Tppc = ppc.linterp(Z, T)


x0 = 180. # reference longitude
y0 = 0.  # reference latitude
i0 = np.searchsorted(g.XCvec, x0) # index to reference longitude
j0 = np.searchsorted(g.YCvec, y0) # index to reference latitude
x0 = g.XCvec[i0] # update
y0 = g.YCvec[j0] # update
z0 = 1500. # target depth for surfaces

# Potential Density -- rough calculation by interpolating density
z_ref = 1500. # reference depth for potential density

σ3 = densjmd95.rho(S, T, z_ref) # 3D potential density

σ_0 = ppc.linterp(Z, σ3[i0,j0,:], z_ref) # isovalue of σ

z_σ = ppc.linterp(σ3, Z, σ_0) # depth of σ isosurface

# fig, ax = plt.subplots()
# pc = ax.pcolormesh(g.XCvecpad(), g.YCvecpad(), z_σ.T, shading='flat')
# fig.colorbar(pc, ax=ax, label='z[σ]')


# Potential Density -- careful calculation by interpolating S and T
(z_σ, s_σ, t_σ) = neutral_surfaces.pot_dens_surf(S, T, Z, 1500., 1500.)  # Warning:  This is very slow!
