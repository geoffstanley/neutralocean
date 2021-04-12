import numpy as np

from neutral_surfaces._vertsolve import vertsolve, process_arrays, func_zero


def pot_dens_surf(S, T, P, p_ref, target, axis=-1, tol=1e-4):
    ngood, stp_args = process_arrays(S, T, P)
    if isinstance(target, tuple):
        ptarget = target[-1]
        ind = target[:-1]
        stp_args1 = tuple([arg[ind] for arg in stp_args])
        d0 = func_zero(ptarget, p_ref, 0, stp_args1)
    else:
        d0 = target

    shape = ngood.shape
    p_start = np.broadcast_to(np.nan, shape)
    p_ref = np.broadcast_to(p_ref, shape)
    d0 = np.broadcast_to(d0, shape)

    s, t, p = vertsolve(p_start, p_ref, d0, ngood, stp_args, tol)
    return s, t, p
