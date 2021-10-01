import numpy as np
from scipy.sparse import csc_matrix

# from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky

from time import time


def _omega_matsolve_poisson(
    s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A4, qu, qt, mr, eos_s_t
):
    # Doco from MATLAB, needs updating.

    # OMEGA_MATSOLVE_POISSON  Build & solve the sparse matrix Poisson problem for omega surfaces
    #
    #
    # ϕ = omega_matsolve_poisson(s, t, p, DIST2on1_iJ, DIST1on2_Ij, wrap, A4, qu, qt, m_ref)
    # builds & solves the sparse matrix problem for omega surfaces in Poisson
    # form.
    #
    #
    # --- Input:
    #  s [ni, nj]: practical / Absolute Salinity on the surface()
    #  t [ni, nj]: potential / Conservative Temperature on the surface()
    #  p [ni, nj]: pressure [or depth] on the surface()
    # DIST2on1_iJ [ni, nj]: the area of a grid cell centred at [I-1/2, J]
    #   divided by the distance, squared, from [I-1,J] to [I,J].  Equivalently
    #   this is the grid distance in second dimension divided by grid distance
    #   in first dimension, both centred at [I-1/2, J].
    # DIST1on2_Ij [ni, nj]: the area of a grid cell centred at [I, J-1/2]
    #   divided by the distance, squared, from [I,J-1] to [I,J].  Equivalently
    #   this is the grid distance in first dimension divided by grid distance
    #   in second dimension, both centred at [I, J-1/2].
    # wrap [2 element array]: wrap[i] is true iff the domain is periodic in the
    #                         i'th lateral dimension.
    # A4 [4, ni*nj]: adjacency matrix  [see grid_adjacency.m]
    # qu [ni*nj,1]: the nodes visited by the BFS's in order from 1 to qt [see bfs_conncomp1.m]
    # qt [1,1]: the last valid index of qu [see bfs_conncomp1.m]
    # m_ref [1,1]  : linear index to a reference cast at which ϕ will be zero.
    #
    #
    # --- Output:
    # ϕ [ni, nj]: density perturbation attempting to satisfy the discrete
    #             version of  div grad ϕ = - div ϵ
    #             where ϵ is the neutrality error (see ntp_errors).

    # Author[s] : Geoff Stanley
    # Email     : g.stanley@unsw.edu.au
    # Email     : geoffstanley@gmail.com

    timer_loc = time()

    ni, nj = p.shape

    # The value nij appears in A5 to index neighbours that would go across a
    # non-periodic boundary
    nij = ni * nj

    # --- Build & solve sparse matrix problem
    ϕ = np.full(nij, np.nan, dtype=np.float64)

    # If there is only one water column, there are no equations to solve,
    # and the solution is simply phi = 0 at that water column, and nan elsewhere.
    # Note, qt > 0 (N >= 1) should be guaranteed by omega_surf(), so N <= 1 should
    # imply N == 1.  If qt > 0 weren't guaranteed, this could throw an error.
    N = qt + 1  # Number of water columns
    if N <= 1:  # There are definitely no equations to solve
        ϕ[qu[0]] = 0.0  # Leave this isolated pixel at current pressure
        return ϕ.reshape(ni, nj), 0.0

    # Collect & sort linear indices to all pixels in this region
    # sorting here makes matrix better structured; overall speedup.
    m = np.sort(qu[0 : qt + 1])

    # If both gridding variables are 1, then grid is uniform
    UNIFORM_GRID = (
        isinstance(DIST2on1_iJ, float)
        and DIST2on1_iJ == 1
        and isinstance(DIST1on2_Ij, float)
        and DIST1on2_Ij == 1
    )

    # Begin building D = divergence of ϵ, and L = Laplacian [compact representation]

    # L refers to neighbours in this order [so does A4, except without the 5'th entry]:
    # . 1 .
    # 0 4 3
    # . 2 .
    IM = 0  # (I  ,J-1)
    MJ = 1  # (I-1,J  )
    PJ = 2  # (I+1,J  )
    IP = 3  # (I  ,J+1)
    IJ = 4  # (I  ,J  )
    L = np.zeros((ni, nj, 5))  # pre-alloc space

    # Create views into L
    L_IM = L[:, :, IM]
    L_MJ = L[:, :, MJ]
    L_PJ = L[:, :, PJ]
    L_IP = L[:, :, IP]
    L_IJ = L[:, :, IJ]

    # Aliases
    sm = s
    tm = t
    pm = p

    # --- m = (i, j) & n = (i-1, j),  then also n = (i+1, j) by symmetry
    sn = im1(sm)
    tn = im1(tm)
    pn = im1(pm)
    if not wrap[0]:
        sn[0, :] = np.nan

    # A stripped down version of ntp_errors[s,t,p,1,1,true,false,true]
    vs, vt = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # [vs, vt] = eos_s_t[ 0.5 * (sm + sn), 0.5 * (tm + tn), 1500 ];  # DEV: testing omega software to find potential density surface()
    ϵ = vs * (sm - sn) + vt * (tm - tn)

    bad = np.isnan(ϵ)
    ϵ[bad] = 0.0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = DIST2on1_iJ.copy()
        fac[bad] = 0.0
        ϵ *= fac  # scale ϵ

    D = -ϵ + ip1(ϵ)

    L_IJ[:] = fac + ip1(fac)

    L_MJ[:] = -fac

    L_PJ[:] = -ip1(fac)

    # --- m = (i, j) & n = (i, j-1),  then also n = (i, j+1) by symmetry
    sn = jm1(sm)
    tn = jm1(tm)
    pn = jm1(pm)
    if not wrap[1]:
        sn[:, 0] = np.nan

    # A stripped down version of ntp_errors[s,t,p,1,1,true,false,true]
    (vs, vt) = eos_s_t(0.5 * (sm + sn), 0.5 * (tm + tn), 0.5 * (pm + pn))
    # [vs, vt] = eos_s_t[ 0.5 * (sm + sn), 0.5 * (tm + tn), 1500 ];  # DEV: testing omega software to find potential density surface()

    ϵ = vs * (sm - sn) + vt * (tm - tn)
    bad = np.isnan(ϵ)
    ϵ[bad] = 0.0

    if UNIFORM_GRID:
        fac = np.float64(~bad)  # 0 and 1
    else:
        fac = DIST1on2_Ij.copy()
        fac[bad] = 0.0
        ϵ *= fac  # scale ϵ

    D += -ϵ + jp1(ϵ)

    L_IJ[:] += fac + jp1(fac)

    L_IM[:] = -fac

    L_IP[:] = -jp1(fac)

    # `remap` changes from linear indices for the entire 2D space (0, 1, ..., ni*nj-1) into linear
    # indices for the current connected component (0, 1, ..., N-1)
    # If the domain were doubly periodic, we would want `remap` to be a 2D array
    # of size (ni,nj). However, with a potentially non-periodic domain, we need
    # one more value for `A5` to index into.  Hence we use `remap` as a vector
    # with ni*nj+1 elements, the last one corresponding to non-periodic boundaries.
    # Water columns that are not in this connected component, and dry water columns (i.e. land),
    # and the fake water column for non-periodic boundaries are all left
    # to have a remap value of -1.
    remap = np.full(nij + 1, -1, dtype=int)
    remap[m] = np.arange(N)

    # Pin surface at mr by changing the mr'th equation to be 1 * ϕ[mr] = 0.
    D[mr] = 0.0
    L[mr] = 0.0
    L[mr][IJ] = 1.0

    L = L.reshape((nij, 5))
    D = D.reshape(nij)

    # The above change renders the mr'th column on all rows irrelevant
    # since ϕ[mr] will be zero.  So, we may also set this column to 0
    # which we do here by setting the appropriate links in L to 0. This
    # maintains symmetry of the matrix, enabling the use of a Cholesky solver.
    mrI = np.ravel_multi_index(mr, (ni, nj))  # get linear index for mr
    if A4[mrI, IP] != nij:
        L[A4[mrI, IP], IM] = 0
    if A4[mrI, PJ] != nij:
        L[A4[mrI, PJ], MJ] = 0
    if A4[mrI, MJ] != nij:
        L[A4[mrI, MJ], PJ] = 0
    if A4[mrI, IM] != nij:
        L[A4[mrI, IM], IP] = 0

    # Build the RHS of the matrix problem
    rhs = D[m]

    # Build indices for the rows of the sparse matrix, namely
    # [[0,0,0,0,0], ..., [N-1,N-1,N-1,N-1,N-1]]
    r = np.repeat(range(N), 5).reshape(N, 5)

    # Build indices for the columns of the sparse matrix
    # `remap` changes global indices to local indices for this region, numbered 0, 1, ... N-1
    #c = remap[A5[m]]
    c = np.column_stack((remap[A4[m]], np.arange(N)))

    # Build the values of the sparse matrix
    v = L[m]

    # Prune the entries to
    # (a) ignore connections to adjacent pixels that are dry (including those
    #     that are "adjacent" across a non-periodic boundary), and
    # (b) ignore the upper triangle of the matrix, since cholesky only
    #     accessses the lower triangular part of the matrix
    good = (c >= 0) & (r >= c)

    # DEV: Could try exiting here, and do csc_matrix, spsolve inside main
    # function, so that this can be njit'ed.  But numba doesn't support
    # np.roll as we need it...  (nor ravel_multi_index, but we could just do
    # that one ourselves)
    # return r[good], c[good], v[good], N, rhs, m

    timer_build = time() - timer_loc

    # Build the sparse matrix; with N rows & N columns
    mat = csc_matrix((v[good], (r[good], c[good])), shape=(N, N))

    # Solve the matrix problem
    factor = cholesky(mat)
    ϕ[m] = factor(rhs)

    # spsolve (requires good = (c >= 0) above) is slower than using cholesky
    # ϕ[m] = spsolve(mat, rhs)

    return ϕ.reshape(ni, nj), timer_build


def im1(F):  # G[i,j] == F[i-1,j]
    return np.roll(F, 1, axis=0)


def ip1(F):  # G[i,j] == F[i+1,j]
    return np.roll(F, -1, axis=0)


def jm1(F):  # G[i,j] == F[i,j-1]
    return np.roll(F, 1, axis=1)


def jp1(F):  # G[i,j] == F[i,j+1]
    return np.roll(F, -1, axis=1)
