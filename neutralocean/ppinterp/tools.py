import functools as ft
from importlib import import_module


@ft.lru_cache(maxsize=10)
def make_pp(
    interpolant="linear",
    kind="u",
    out="coeffs",
    nans=True,
    num_dep_vars=1,
):
    """Make function for piecewise polynomial interpolation."""

    if interpolant not in ("linear", "pchip"):
        raise ValueError(
            f"Expected `interpolant` in ('lienar', 'pchip'); got {interpolant}"
        )

    if kind not in ("1", "u"):
        raise ValueError(f"Expected `kind` in ('1', 'u'); got {kind}")

    if out not in ("coeffs", "interp"):
        raise ValueError(f"Expected `out` in ('coeffs', 'interp'); got {out}")

    if num_dep_vars not in (1, 2):
        raise ValueError(
            "Expected `num_dep_vars` in (1, 2); got {num_dep_vars}"
        )

    if interpolant == "linear":
        if out == "coeffs":
            if kind == "u":
                fcn_name = "linear_coeffs"
            elif kind == "1":
                # No check of `nans`: the code for linear_coeffs_1 is unaffected
                fcn_name = "linear_coeffs_1"
        else:  # build interpolator
            if kind == "u":
                if num_dep_vars == 1:
                    fcn_name = "linear_interp"
                else:
                    fcn_name = "linear_interp_two"
            elif kind == "1":
                if num_dep_vars == 1:
                    fcn_name = "linear_interp_1"
                else:
                    fcn_name = "linear_interp_1_two"
    elif interpolant == "pchip":
        if out == "coeffs":
            if kind == "u":
                fcn_name = "pchip_coeffs"
            elif kind == "1":
                if nans:
                    fcn_name = "pchip_coeffs_1"
                else:
                    fcn_name = "pchip_coeffs_1_nonan"
        else:  # build interpolator
            if kind == "u":
                if num_dep_vars == 1:
                    fcn_name = "pchip_interp"
                else:
                    fcn_name = "pchip_interp_two"
            elif kind == "1":
                if num_dep_vars == 1:
                    fcn_name = "pchip_interp_1"
                else:
                    fcn_name = "pchip_interp_1_two"

    # Below is equivalent to
    # from neutralocean.ppinterp.`interpolant` import `fcn_name`
    return import_module(
        "neutralocean.ppinterp." + interpolant
    ).__getattribute__(fcn_name)
