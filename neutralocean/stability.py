import numpy as np
from scipy.optimize import minimize

from neutralocean.lib import _process_casts
from neutralocean.ppinterp import valid_range_1


def count_unstable(S, T, P, **kw):
    """
    Count number of statically unstable bottles.

    S, T, P :
        As in `stabilize_ST`

    interp_two : function or None, Default None

        When None, static stability is calculated as the difference of
        potential density between pairs of adjacent bottles on each cast,
        referenced to the average pressure between these two bottles.

        When a function, it is used to evaluate the first derivative of an
        interpolant for each of `S` and `T` as a function of `P`, at each
        bottle's `P` value. The vertical gradient of Locally Referenced
        Potential Density is then calculated by combining these derivatives
        with the partial derivatives of the equation of state with respect to
        `S` and `T` at the local `P`.

    eos : function
        Equation of State for density (not specific volume);
        used when `interp_two` is None.

    eos_s_t : function, Optional

        Partial derivatives of the Equation of State (eos above) with respect 
        to `S` and `T`; used when `interp_two` is given.

    """

    vert_dim = kw.get("vert_dim", -1)
    interp_two = kw.get("interp_two")

    S, T, P = _process_casts(S, T, P, vert_dim)
    nk = S.shape[-1]

    if interp_two is None:
        eos = kw.get("eos")
    else:
        eos_s_t = kw.get("eos_s_t")

    num_unstab = 0
    for k in range(0, nk - 1):
        if interp_two is None:
            p_avg = (P[..., k] + P[..., k + 1]) / 2
            σ1 = eos(S[..., k], T[..., k], p_avg)
            σ2 = eos(S[..., k + 1], T[..., k + 1], p_avg)
            num_unstab += np.sum(σ1 >= σ2)
        else:
            s, t = S[..., k], T[..., k]
            ds, dt = interp_two(P[..., k], P, S, T, 1)
            rs, rt = eos_s_t(s, t, P[..., k])
            lrpd_z = rs * ds + rt * dt
            num_unstab += np.sum(lrpd_z < 0)

    return num_unstab


def stabilize_ST(S, T, P, eos, **kw):
    """
    Mutate S, T to ensure the vertical gradient of Locally Referenced Potential
    Density (LRPD) is greater than a given threshold everywhere.

    Parameters
    ----------
    S, T : ndarray or xarray.DataArray

        practical / Absolute salinity and potential / Conservative
        temperature.

    P : ndarray or xarray.DataArray

        In the non-Boussinesq case, `P` is pressure, sharing the same
        dimensions as `S` and `T`.

        In the Boussinesq case, `P` is the depth and can have the dimensions
        as `S` and `T`, or can be 1D with as many elements as there
        are in the vertical dimension of `S` and `T`.

    eos : function
        Equation of state for density (not specific volume).

        Takes three inputs corresponding to `S`, `T`, and `P`, and outputs the
        density. It should be `@numba.njit` decorated and need not be 
        vectorized -- it will be called many times with scalar inputs.

    min_dLRPDdp : float or 1D array of float, Default 1e-6

        Minimum vertical gradient of LRPD for the mutated `S` and `T`.

        If an array, this is the minimum vertical gradient of LRPD between
        adjacent pairs of bottles in each cast; hence, this should have length
        one less than the length of the vertical dimension of `S` and `T`.

    weight : float, Default 10.0

        Multiplicative weighting factor applied to `S` perturbations, but not
        to `T` perturbations. 
        When `weight > 1`, `T` perturbations are favoured.
        When `weight < 1`, `S` perturbations are favoured.

    vert_dim : int or str

        Specifies which dimension of `S`, `T` (and `P` if more than 1D) is vertical.

        If `S` and `T` are `ndarray`, then `vert_dim` is the `int` indexing
        the vertical dimension of `S` and `T` (e.g. -1 indexes the last
        dimension).

        If `S` and `T` are `xarray.DataArray`, then `vert_dim` is a `str`
        naming the vertical dimension of `S` and `T`.

        Ideally, `vert_dim` is -1.  See `Notes`.

    tol : float, Default 1e-10

        Tolerance for (weighted) `S` and (unweighted) `T` perturbations; 
        passed to scipy's `minimize`.

    method : str, Default "SLSQP"

        Minimization method, passed to scipy's `minimize`.
        
    options : dict, Default {"maxiter" : 1000}
    
        Options passed to the `options` argument of `minimize`.
        
    verbose : bool, Default True
    
        Whether to print information any time a cast is mutated.

    Notes
    -----
    This function is similar to the method of Jackett and McDougall (1995) [1]_.
    However, it uses a fixed weighting between S and T, and it solves the
    optimization problem with non-linear constraints. It also uses a
    slightly different numerical implementation for the vertical gradient of
    LRPD.

    .. [1] Jackett and McDougall, 1995, JAOT 12[4], pp. 381-388

    """

    min_dLRPDdp = kw.pop("min_dLRPDdp", 1e-6)  # or N^2 >= 1e-8, roughly
    weight = kw.pop("weight", 10.0)  # crude approximation of |dρ/dS / dρ/dΘ|
    vert_dim = kw.pop("vert_dim", -1)
    tol = kw.pop("tol", 1e-10)
    method = kw.pop("method", "SLSQP")
    verbose = kw.pop("verbose", True)
    options = kw.pop("options", {"maxiter" : 1000})

    S, T, P = _process_casts(S, T, P, vert_dim)

    hshape = S.shape[0:-1]
    nk = S.shape[-1]

    if np.isscalar(min_dLRPDdp):
        min_dLRPDdp = np.repeat(min_dLRPDdp, nk - 1)

    def f(Δx, w):
        # Sum of squares of Δx, weighting the first half.
        # Expects len(Δx) is an even int
        n = len(Δx) // 2
        ΔS = Δx[0:n]
        ΔT = Δx[n:]
        return np.sum(w * ΔS * ΔS + ΔT * ΔT)

    def jac(Δx, w):
        # Jacobian of f
        n = len(Δx) // 2
        out = 2 * Δx
        out[0:n] *= w
        return out

    def perturb_dLRPDdp_fd_1(Δx, S, T, P, const, eos):
        # Calculate d(LRPD)/dp when S, T are perturbed, using Finite Difference scheme in 1 cast.
        n = len(Δx) // 2
        S = S + Δx[0:n]
        T = T + Δx[n:]
        p_avg = (P[0:-1] + P[1:]) / 2
        σ1 = eos(S[0:-1], T[0:-1], p_avg)
        σ2 = eos(S[1:], T[1:], p_avg)
        dσdp = (σ2 - σ1) / (P[1:] - P[0:-1])
        return dσdp - const

    # TODO: Give analytic Jacobian of the constraint function

    con = dict(type="ineq", fun=perturb_dLRPDdp_fd_1)

    # Make deterministic noise for initial perturbation vector; for all casts
    rng = np.random.default_rng(seed=42)
    x0_nk = rng.normal(0.0, 1e-6, 2 * nk)

    # Loop over each cast
    for c in np.ndindex(hshape):
        k, K = valid_range_1(S[c])
        S1, T1, P1 = (X[c][k:K] for X in (S, T, P))
        min_dLRPDdp_1 = min_dLRPDdp[k : K - 1]
        n = len(S1)  # == k2 - k1

        # Check for at least 2 bottles in this cast, and any unstable pairs
        if n > 1 and any(calc_dLRPDdp_fd_1(S1, T1, P1, eos) < min_dLRPDdp_1):
            con["args"] = (S1, T1, P1, min_dLRPDdp_1, eos)

            res = minimize(
                f,
                x0_nk[0 : 2 * n],
                args=(weight,),
                jac=jac,
                constraints=con,
                method=method,
                tol=tol,
                options=options,
            )
            if res.success:
                ΔS = res.x[0:n]
                ΔT = res.x[n:]
                S[c][k:K] += ΔS
                T[c][k:K] += ΔT
            else:
                raise RuntimeError(f"Cast {c} stabilization failed.")

            if verbose:
                print(
                    f"Perturbed cast {c} by weighted RMS amount {f(res.x, weight)}"
                )


def calc_dLRPDdp_fd_1(S, T, P, eos):
    # Calculate d(LRPD)/dp using Finite Difference scheme in 1 cast.
    p_avg = (P[0:-1] + P[1:]) / 2
    σ1 = eos(S[0:-1], T[0:-1], p_avg)
    σ2 = eos(S[1:], T[1:], p_avg)
    dσdp = (σ2 - σ1) / (P[1:] - P[0:-1])
    return dσdp
