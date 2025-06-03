Internal functions
******************

Equation of State
=================
JMD95
-----
.. automodule:: neutralocean.eos.jmd95

.. autofunction:: neutralocean.eos.jmd95.rho

.. autofunction:: neutralocean.eos.jmd95.rho_s_t

.. autofunction:: neutralocean.eos.jmd95.rho_p

JMDFWG06
--------
.. automodule:: neutralocean.eos.jmdfwg06

.. autofunction:: neutralocean.eos.jmdfwg06.rho

.. autofunction:: neutralocean.eos.jmdfwg06.rho_s_t

.. autofunction:: neutralocean.eos.jmdfwg06.rho_p

TEOS-10 GSW
-----------
.. automodule:: neutralocean.eos.gsw

.. autofunction:: neutralocean.eos.gsw.specvol

.. autofunction:: neutralocean.eos.gsw.specvol_s_t

.. autofunction:: neutralocean.eos.gsw.specvol_p

.. autofunction:: neutralocean.eos.gsw.specvol_s_t_ss_st_tt_sp_tp

.. autofunction:: neutralocean.eos.gsw.specvol_s_t_ss_st_tt_sp_tp_sss_sst_stt_ttt_ssp_stp_ttp_spp_tpp

.. autofunction:: neutralocean.eos.gsw.rho


(Vertical) Interpolation using Piecewise Polynomials (PP)
=========================================================
.. autofunction:: neutralocean.ppinterp.pval

.. autofunction:: neutralocean.ppinterp.lib.valid_range

.. autofunction:: neutralocean.ppinterp.lib.valid_range_1

.. autofunction:: neutralocean.ppinterp.lib.valid_range_1_two

.. autofunction:: neutralocean.ppinterp.linear.linear_coeffs

.. autofunction:: neutralocean.ppinterp.linear.linear_interp

.. autofunction:: neutralocean.ppinterp.pchip.pchip_coeffs

.. autofunction:: neutralocean.ppinterp.pchip.pchip_interp

Root finding in 1D
==================
.. automodule:: neutralocean.fzero

.. autofunction:: neutralocean.fzero.brent_guess

.. autofunction:: neutralocean.fzero.brent

.. autofunction:: neutralocean.fzero.guess_to_bounds

Library functions
=================
.. autofunction:: neutralocean.lib.find_first_nan

.. autofunction:: neutralocean.lib.val_at

.. autofunction:: neutralocean.lib.take_fill

.. autofunction:: neutralocean.lib.aggsum

.. autofunction:: neutralocean.lib.xr_to_np