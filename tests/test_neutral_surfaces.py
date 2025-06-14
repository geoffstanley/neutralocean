import numpy as np
import neutralocean as no

grav = 9.81
rho_c = 1027.5
eos = no.load_eos("gsw", "", grav, rho_c)
eos_s_t = no.load_eos("gsw", "_s_t", grav, rho_c)
eos_ufunc = no.vectorize_eos(eos)

# Make a simple ocean dataset
ni, nj, nk = 16, 32, 50
wrap = (False, False)  # non-periodic in both horizontal dimensions
S, T, Z, _ = no.data.synthocean((ni, nj, nk), wrap=wrap)
# Raise the sea-floor in some casts
# Make one profile be land, and three profiles have a shallower bottom
# (the last having just one valid bottle)
S[1, 1, 45:] = T[1, 1, 45:] = np.nan  # deep ocean
S[2, 1, 5:] = T[2, 1, 5:] = np.nan  # shallow ocean
S[3, 1, 1:] = T[3, 1, 1:] = np.nan  # coastal ocean (1 valid bottle)

# Build grid adjacency and distance information for neutralocean functions
grid = no.grid.rectilinear.build_grid((ni, nj), wrap)


def test_potential_surf():
    # Test sigma_surf using a prescribed reference depth and isovalue
    z_ref = 0.0
    isoval = 1.0 / 1027.0
    s, t, z, _ = no.potential_surf(
        S,
        T,
        Z,
        ref=z_ref,
        isoval=isoval,
        eos=eos,
        TOL_P_SOLVER=1e-8,
        diags=False,
    )

    σ = np.ma.masked_invalid(eos_ufunc(s, t, z_ref))
    assert np.ma.allclose(σ, isoval)

    # Calculate surface potential density
    σ_sfc = eos_ufunc(S[:, :, 0], T[:, :, 0], z_ref)

    # Calculate seafloor potential density
    n_good = no.lib.find_first_nan(S)
    S_bot, T_bot = (no.lib.val_at(x, n_good - 1) for x in (S, T))
    σ_bot = eos_ufunc(S_bot, T_bot, z_ref)

    σ = eos_ufunc(s, t, z_ref)

    # check for each cast that
    # potential density on surface nearly matches isovalue,
    # or the surface does not intersect this cast (σ is nan) because of one of
    # three conditions: the cast was land, the surface outcropped, or the surface incropped.
    assert np.all(
        (np.abs(σ - isoval) < 1e-8)
        | (np.isnan(σ) & ((n_good == 0) | (isoval < σ_sfc) | (σ_bot < isoval)))
    )


def test_anomaly_surf():
    # Test delta_surf using prescribed reference values and isovalue
    s_ref, t_ref = 34.5, 4.0
    isoval = 0.0
    s, t, z, _ = no.anomaly_surf(
        S,
        T,
        Z,
        ref=(s_ref, t_ref),
        isoval=isoval,
        eos=eos,
        TOL_P_SOLVER=1e-8,
        diags=False,
    )

    δ = np.ma.masked_invalid(eos_ufunc(s, t, z) - eos_ufunc(s_ref, t_ref, z))
    assert np.ma.allclose(δ, isoval)

    # Calculate surface potential density
    δ_sfc = eos_ufunc(S[:, :, 0], T[:, :, 0], Z[0]) - eos_ufunc(s_ref, t_ref, Z[0])

    # Calculate seafloor potential density
    n_good = no.lib.find_first_nan(S)
    S_bot, T_bot, Z_bot = (no.lib.val_at(x, n_good - 1) for x in (S, T, Z))
    δ_bot = eos_ufunc(S_bot, T_bot, Z_bot) - eos_ufunc(s_ref, t_ref, Z_bot)

    δ = eos_ufunc(s, t, z) - eos_ufunc(s_ref, t_ref, z)

    # check for each cast that
    # in-situ density anomaly on surface nearly matches isovalue,
    # or the surface does not intersect this cast (δ is nan) because of one of
    # three conditions: the cast was land, the surface outcropped, or the surface incropped.
    assert np.all(
        (np.abs(δ - isoval) < 1e-8)
        | (np.isnan(δ) & ((n_good == 0) | (isoval < δ_sfc) | (δ_bot < isoval)))
    )


def test_omega_surf():
    # Test omega_surf, initialized from a potential density surface
    z0 = 2000.0
    i0, j0 = (int(x / 2) for x in (ni, nj))  # ref cast in middle of domain

    # Calculate initial surface as a potential density surface
    s, t, z, _ = no.potential_surf(
        S, T, Z, pin_cast=(i0, j0), pin_p=z0, eos=eos, diags=False
    )

    # Calculate divergence of ϵ on initial surface
    e = no.ntp_epsilon_errors(s, t, z, grid, eos_s_t)
    divg_e_init = no.grid.divergence(e, grid["edges"])

    # Calculate omega surface, and divergence of ϵ on it
    s, t, z, d = no.omega_surf(
        S,
        T,
        Z,
        grid,
        pin_cast=(i0, j0),
        p_init=z,
        eos=eos,
        eos_s_t=eos_s_t,
        diags=True,
    )

    e = no.ntp_epsilon_errors(s, t, z, grid, eos_s_t)
    divg_e = no.grid.divergence(e, grid["edges"])

    # Check ratio of ∇⋅ϵ, between initial and converged surface, is small
    rms = lambda x: np.sqrt(np.nanmean(np.square(x)))
    assert rms(divg_e) / rms(divg_e_init) < 1e-4
