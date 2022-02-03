from .linear import linear_coeffs, linear_coeffs_1
from .pchip import pchip_coeffs, pchip_coeffs_1


def select_ppc(interpolant="linear", kind="u"):
    """Select function for building piecewise polynomial coefficients."""
    if kind not in ("1", "u"):
        raise ValueError(f"Expected `kind` in ('1', 'u'); got {kind}")

    if interpolant == "linear":
        if kind == "1":
            return linear_coeffs_1
        elif kind == "u":
            return linear_coeffs
    elif interpolant == "pchip":
        if kind == "1":
            return pchip_coeffs_1
        elif kind == "u":
            return pchip_coeffs
    else:
        raise ValueError(
            f"Expected `interpolant` in ('linear', 'pchip'); got {interpolant}"
        )
