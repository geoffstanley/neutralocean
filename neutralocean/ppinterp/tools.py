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

    if out == "coeffs" and num_dep_vars != 1:
        raise ValueError(
            "With `out='coeffs'`, currently only handles `num_dep_vars=1`"
        )

    if nans == False:
        if out == "interp" or kind == "u":
            raise ValueError(
                "With `nans=False`, expected `out='coeffs'` and `kind='1'`."
            )

    # Begin programmatically generating the function name to import
    if kind == "1":
        kind = "_1"
    elif kind == "u":
        kind = ""

    if nans == False and interpolant == "pchip":
        nans = "_nonan"
    else:
        # if interpolant == "linear", just use regular coeffs_1 function.
        nans = ""

    if num_dep_vars == 1:
        num_dep_vars = ""
    elif num_dep_vars == 2:
        num_dep_vars = "_two"

    fcn_name = interpolant + "_" + out + kind + nans + num_dep_vars

    # Below is equivalent to
    # from neutralocean.ppinterp.`interpolant` import `fcn_name`
    return import_module(
        "neutralocean.ppinterp." + interpolant
    ).__getattribute__(fcn_name)
