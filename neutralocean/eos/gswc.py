"""Backward-compatible alias for the official GSW EOS backend.

Prefer importing from `neutralocean.eos.gsw_official`.
"""

from .gsw_official import rho, rho_s_t, rho_p

__all__ = ["rho", "rho_s_t", "rho_p"]
