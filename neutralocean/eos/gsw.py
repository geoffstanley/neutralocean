"""
Density of Sea Water using TEOS-10 Gibbs Sea Water [1]_ in pure Python

Functions:

specvol :: computes specific volume from Absolute Salinity, Conservative Temperature and
    pressure

specvol_s_t :: compute the partial derivatives of specific volume th
    respect to Absolute Salinity and Conservative Temperature

specvol_p :: compute the partial derivative of specific volume with
    respect to pressure

Notes:
To make Boussinesq versions of these functions, see 
`neutralocean.eos.tools.make_eos_bsq`.

To make vectorized versions of these functions, see
`neutralocean.eos.tools.vectorize_eos`.

All functions here are derived from `gsw_specvol`, documented below.

.. highlight:: python
.. code-block:: python

    # gsw_specvol                            specific volume (75-term equation)
    #==========================================================================
    # 
    # USAGE:  
    #  specvol = gsw_specvol(SA,CT,p)
    # 
    # DESCRIPTION:
    #  Calculates specific volume from Absolute Salinity, Conservative 
    #  Temperature and pressure, using the computationally-efficient 75-term 
    #  polynomial expression for specific volume (Roquet et al., 2015).
    #
    #  Note that the 75-term equation has been fitted in a restricted range of 
    #  parameter space, and is most accurate inside the "oceanographic funnel" 
    #  described in McDougall et al. (2003).  The GSW library function 
    #  "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if 
    #  some of one's data lies outside this "funnel".  
    #
    # INPUT:
    #  SA  =  Absolute Salinity                                        [ g/kg ]
    #  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
    #  p   =  sea pressure                                             [ dbar ]
    #         ( i.e. absolute pressure - 10.1325 dbar )
    #
    #  SA & CT need to have the same dimensions.
    #  p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are MxN.
    #
    # OUTPUT:
    #  specvol  =  specific volume                                   [ m^3/kg ]
    #
    # AUTHOR: 
    #  Fabien Roquet, Gurvan Madec, Trevor McDougall & Paul Barker
    #                                                      [ help@teos-10.org ]
    #
    # VERSION NUMBER: 3.06 (30th August, 2018)
    # 
    # REFERENCES:
    #  IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of 
    #   seawater - 2010: Calculation and use of thermodynamic properties.  
    #   Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    #   UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  
    # 
    #  McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003: 
    #   Accurate and computationally efficient algorithms for potential 
    #   temperature and density of seawater.  J. Atmos. Ocean. Tech., 20, 
    #   730-741.
    #
    #  Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    #   polynomial expressions for the density and specifc volume of seawater
    #   using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.
    #
    #  This software is available from http://www.TEOS-10.org
    # 
    #==========================================================================

.. [1] McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and 
       the Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127, 
       ISBN 978-0-646-55621-5. 
"""

# Check value computed on 16/10/2022, and agrees with MATLAB function
# gsw_specvol_third_derivatives_st.m in neutral-surfaces toolbox.
#
# gsw.specvol_s_t_ss_st_tt_sp_tp_sss_sst_stt_ttt_ssp_stp_ttp_spp_tpp(35, 2, 100)
# (-7.524044242941373e-07,
#  7.835130137433233e-08,
#  1.0840451733910375e-09,
#  2.747547886004396e-09,
#  1.186819176528437e-08,
#  1.5020065837658017e-11,
#  2.6031411152070758e-11,
#  -1.3470792464764363e-12,
#  -1.6001306157869825e-11,
#  -6.839037913975825e-11,
#  -3.491354850608456e-10,
#  -4.925365428774398e-14,
#  -1.8246324672046492e-13,
#  -9.922059938783903e-13,
#  -6.426641694819895e-16,
#  -1.0860670553447786e-15)


import numpy as np
import numba as nb

tfac = 0.025
pfac = 1e-4

# sfac = 1/(40*(35.16504/35))
sfac = 0.0248826675584615

# deltaSA = 24 g/kg, offset = deltaSA*sfac
offset = 5.971840214030754e-1

# fmt: off
v000 =  1.0769995862e-3 
v001 = -6.0799143809e-5 
v002 =  9.9856169219e-6 
v003 = -1.1309361437e-6 
v004 =  1.0531153080e-7 
v005 = -1.2647261286e-8 
v006 =  1.9613503930e-9 
v010 = -1.5649734675e-5 
v011 =  1.8505765429e-5 
v012 = -1.1736386731e-6 
v013 = -3.6527006553e-7 
v014 =  3.1454099902e-7 
v020 =  2.7762106484e-5 
v021 = -1.1716606853e-5 
v022 =  2.1305028740e-6 
v023 =  2.8695905159e-7 
v030 = -1.6521159259e-5 
v031 =  7.9279656173e-6 
v032 = -4.6132540037e-7 
v040 =  6.9111322702e-6 
v041 = -3.4102187482e-6 
v042 = -6.3352916514e-8 
v050 = -8.0539615540e-7 
v051 =  5.0736766814e-7 
v060 =  2.0543094268e-7
v100 = -3.1038981976e-4 
v101 =  2.4262468747e-5 
v102 = -5.8484432984e-7 
v103 =  3.6310188515e-7 
v104 = -1.1147125423e-7
v110 =  3.5009599764e-5 
v111 = -9.5677088156e-6 
v112 = -5.5699154557e-6 
v113 = -2.7295696237e-7 
v120 = -3.7435842344e-5 
v121 = -2.3678308361e-7 
v122 =  3.9137387080e-7 
v130 =  2.4141479483e-5 
v131 = -3.4558773655e-6 
v132 =  7.7618888092e-9 
v140 = -8.7595873154e-6 
v141 =  1.2956717783e-6 
v150 = -3.3052758900e-7 
v200 =  6.6928067038e-4 
v201 = -3.4792460974e-5 
v202 = -4.8122251597e-6 
v203 =  1.6746303780e-8 
v210 = -4.3592678561e-5 
v211 =  1.1100834765e-5 
v212 =  5.4620748834e-6 
v220 =  3.5907822760e-5 
v221 =  2.9283346295e-6 
v222 = -6.5731104067e-7 
v230 = -1.4353633048e-5 
v231 =  3.1655306078e-7 
v240 =  4.3703680598e-6 
v300 = -8.5047933937e-4 
v301 =  3.7470777305e-5 
v302 =  4.9263106998e-6 
v310 =  3.4532461828e-5 
v311 = -9.8447117844e-6 
v312 = -1.3544185627e-6 
v320 = -1.8698584187e-5 
v321 = -4.8826139200e-7 
v330 =  2.2863324556e-6
v400 =  5.8086069943e-4 
v401 = -1.7322218612e-5 
v402 = -1.7811974727e-6 
v410 = -1.1959409788e-5 
v411 =  2.5909225260e-6 
v420 =  3.8595339244e-6 
v500 = -2.1092370507e-4 
v501 =  3.0927427253e-6 
v510 =  1.3864594581e-6
v600 =  3.1932457305e-5 
# fmt: on


# If ndarray inputs are needed, it is best to use @nb.vectorize.  That is,
# apply `.tools.vectorize_eos`.  A vectorized function specified
# for scalars is about twice as fast as a signatureless njit'ed function
# applied to ndarrays.
@nb.njit
def specvol(SA, CT, p):
    """
    GSW specific volume.

    Parameters
    ----------
    SA : float
        Absolute Salinity [g/kg]
    CT : float
        Conservative Temperature [deg C]
    p : float
        sea pressure (i.e. absolute pressure - 10.1325 dbar)  [dbar]

    Returns
    -------
    specvol : float
        Specific volume [m3 kg-1]
    """
    (x, y, z, _) = _process(SA, CT, p)
    return _specvol(x, y, z)


@nb.njit
def specvol_s_t(SA, CT, p):
    """
    Partial derivatives of GSW specific volume with respect to salinity & temperature

    Parameters
    ----------
    SA : float
        Absolute Salinity [g/kg]
    CT : float
        Conservative Temperature [deg C]
    p : float
        sea pressure (i.e. absolute pressure - 10.1325 dbar)  [dbar]

    Returns
    -------
    s : float
        Partial deriv of specific volume w.r.t. SA [m3 kg-1 / (g/kg)]

    t : float
        Partial deriv of specific volume w.r.t. CT [m3 kg-1 / (deg C)]
    """
    (x, y, z, _) = _process(SA, CT, p)
    s = _s(x, y, z)
    t = _t(x, y, z)

    return (s, t)


@nb.njit
def specvol_p(SA, CT, p):
    """
    Partial derivative of GSW specific volume with respect to pressure

    Parameters
    ----------
    SA : float
        Absolute Salinity [g/kg]
    CT : float
        Conservative Temperature [deg C]
    p : float
        sea pressure (i.e. absolute pressure - 10.1325 dbar)  [dbar]

    Returns
    -------
    p : float
        Partial deriv of specific volume w.r.t. p [m3 kg-1 / (dbar)]
    """
    (x, y, z, _) = _process(SA, CT, p)
    return _p(x, y, z)


@nb.njit
def specvol_s_t_ss_st_tt_sp_tp(SA, CT, p):
    """
    Select partial derivatives of GSW specific volume up to second order

    Parameters
    ----------
    SA : float
        Absolute Salinity [g/kg]
    CT : float
        Conservative Temperature [deg C]
    p : float
        sea pressure (i.e. absolute pressure - 10.1325 dbar)  [dbar]

    Returns
    -------
    s, t, ss, st, tt, sp, tp : float
        Partial derivatives of specific volume w.r.t. `SA`, `CT`, `SA*SA`, `SA*CT`, `CT*CT`,
        `SA*p`, `CT*p`.
    """
    (x, y, z, x2) = _process(SA, CT, p)
    s = _s(x, y, z)
    t = _t(x, y, z)
    ss = _ss(x, y, z, x2)
    st = _st(x, y, z)
    tt = _tt(x, y, z)
    sp = _sp(x, y, z)
    tp = _tp(x, y, z)

    return (s, t, ss, st, tt, sp, tp)


@nb.njit
def specvol_s_t_ss_st_tt_sp_tp_sss_sst_stt_ttt_ssp_stp_ttp_spp_tpp(SA, CT, p):
    """
    Select partial derivatives of GSW specific volume up to third order

    Parameters
    ----------
    SA : float
        Absolute Salinity [g/kg]
    CT : float
        Conservative Temperature [deg C]
    p : float
        sea pressure (i.e. absolute pressure - 10.1325 dbar)  [dbar]

    Returns
    -------
    s, t, ss, st, tt, sp, tp, sss, sst, stt, ttt, ssp, stp, ttp, spp, tpp : float
        Partial derivatives of specific volume.
    """
    (x, y, z, x2) = _process(SA, CT, p)

    s = _s(x, y, z)
    t = _t(x, y, z)
    ss = _ss(x, y, z, x2)
    st = _st(x, y, z)
    tt = _tt(x, y, z)
    sp = _sp(x, y, z)
    tp = _tp(x, y, z)
    sss = _sss(x, y, z, x2)
    sst = _sst(x, y, z, x2)
    stt = _stt(x, y, z)
    ttt = _ttt(x, y, z)
    ssp = _ssp(x, y, z, x2)
    stp = _stp(x, y, z)
    ttp = _ttp(x, y, z)
    spp = _spp(x, y, z)
    tpp = _tpp(x, y, z)

    # fmt: off
    return (s, t, ss, st, tt, sp, tp, sss, sst, stt, ttt, ssp, stp, ttp, spp, tpp)
    # fmt: on


@nb.njit
def _process(SA, CT, p):
    SA = np.maximum(SA, 0)
    x2 = sfac * SA + offset
    x = np.sqrt(x2)
    y = CT * tfac
    z = p * pfac
    return (x, y, z, x2)


""" Begin individual partial derivatives """

# fmt: off
@nb.njit
def _specvol(x, y, z):
    return (v000 + x*(v100 + x*(v200 + x*(v300 + x*(v400 + x*(v500 + x*v600)))))
       + y*(v010 + x*(v110 + x*(v210 + x*(v310 + x*(v410 + x*v510))))
       + y*(v020 + x*(v120 + x*(v220 + x*(v320 + x*v420)))
       + y*(v030 + x*(v130 + x*(v230 + x*v330))
       + y*(v040 + x*(v140 + x* v240)
       + y*(v050 + x* v150
       + y* v060)))))
    + z*(   v001 + x*(v101 + x*(v201 + x*(v301 + x*(v401 + x*v501))))
       + y*(v011 + x*(v111 + x*(v211 + x*(v311 + x*v411)))
       + y*(v021 + x*(v121 + x*(v221 + x*v321))
       + y*(v031 + x*(v131 + x* v231)
       + y*(v041 + x* v141
       + y* v051))))
    + z*(   v002 + x*(v102 + x*(v202 + x*(v302 + x*v402)))
       + y*(v012 + x*(v112 + x*(v212 + x*v312))
       + y*(v022 + x*(v122 + x* v222)
       + y*(v032 + x* v132
       + y* v042)))
    + z*(   v003 + x*(v103 + x* v203)
       + y*(v013 + x* v113
       + y* v023)
    + z*(   v004 + x* v104
       + y* v014
    + z*(   v005
    + z*    v006))))))


@nb.njit
def _s(x, y, z):
    return ( v100 + x*(2*v200 + x*(3*v300 + x*(4*v400 + x*(5*v500 + x*(6*v600)))))
        + y*(v110 + x*(2*v210 + x*(3*v310 + x*(4*v410 + x*(5*v510))))
        + y*(v120 + x*(2*v220 + x*(3*v320 + x*(4*v420)))
        + y*(v130 + x*(2*v230 + x*(3*v330))
        + y*(v140 + x*(2*v240)
        + y* v150))))
    + z*(    v101 + x*(2*v201 + x*(3*v301 + x*(4*v401 + x*(5*v501))))
        + y*(v111 + x*(2*v211 + x*(3*v311 + x*(4*v411)))
        + y*(v121 + x*(2*v221 + x*(3*v321))
        + y*(v131 + x*(2*v231)
        + y* v141)))
    + z*(    v102 + x*(2*v202 + x*(3*v302 + x*(4*v402)))
        + y*(v112 + x*(2*v212 + x*(3*v312))
        + y*(v122 + x*(2*v222)
        + y* v132))
    + z*(    v103 + x*(2*v203)
        + y* v113
    + z*     v104)))) / x * (0.5 * sfac)


@nb.njit
def _t(x, y, z):
    return (   v010 + x*(  v110 + x*(  v210 + x*(  v310 + x*(  v410 + x*v510))))
       + y* (2*v020 + x*(2*v120 + x*(2*v220 + x*(2*v320 + x*(2*v420))))
       + y* (3*v030 + x*(3*v130 + x*(3*v230 + x*(3*v330)))
       + y* (4*v040 + x*(4*v140 + x*(4*v240))
       + y* (5*v050 + x*(5*v150)
       + y* (6*v060)))))
    + z*(   (  v011 + x*(  v111 + x*(  v211 + x*(  v311 + x*  v411)))
       + y* (2*v021 + x*(2*v121 + x*(2*v221 + x*(2*v321)))
       + y* (3*v031 + x*(3*v131 + x*(3*v231))
       + y* (4*v041 + x*(4*v141)
       + y* (5*v051)))))
    + z*(   (  v012 + x*(  v112 + x*(  v212 + x*v312))
       + y* (2*v022 + x*(2*v122 + x*(2*v222))
       + y* (3*v032 + x*(3*v132)
       + y* (4*v042))))
    + z*(   (  v013 + x*v113
       + y* (2*v023))
    + z*(      v014 ))))) * tfac

    
@nb.njit
def _p(x, y, z):
    return (   v001 + x*(v101 + x*(v201 + x*(v301 + x*(v401 + x*v501))))
       + y* (  v011 + x*(v111 + x*(v211 + x*(v311 + x*v411)))
       + y* (  v021 + x*(v121 + x*(v221 + x*v321))
       + y* (  v031 + x*(v131 + x*v231)
       + y* (  v041 + x*v141
       + y*    v051))))
    + z*(    2*v002 + x*(2*v102 + x*(2*v202 + x*(2*v302 + x*(2*v402))))
       + y* (2*v012 + x*(2*v112 + x*(2*v212 + x*(2*v312)))
       + y* (2*v022 + x*(2*v122 + x*(2*v222))
       + y* (2*v032 + x*(2*v132)
       + y* (2*v042))))
    + z*(    3*v003 + x*(3*v103 + x*(3*v203))
       + y* (3*v013 + x*(3*v113)
       + y* (3*v023))
    + z*(    4*v004 + x*(4*v104)
       + y* (4*v014)  
    + z*(    5*v005
    + z*(    6*v006)))))) * pfac
    

@nb.njit
def _ss(x, y, z, x2):
    return ( -0.25*v100 + x2*(0.75*v300 + x*(2.0*v400 + x*(3.75*v500 + x*(6*v600))))
        + y*(-0.25*v110 + x2*(0.75*v310 + x*(2.0*v410 + x*(3.75*v510)))
        + y*(-0.25*v120 + x2*(0.75*v320 + x*(2.0*v420))
        + y*(-0.25*v130 + x2*(0.75*v330)
        + y*(-0.25*v140
        + y*(-0.25*v150)))))
    + z*(    -0.25*v101 + x2*(0.75*v301 + x*(2.0*v401 + x*(3.75*v501)))
        + y*(-0.25*v111 + x2*(0.75*v311 + x*(2.0*v411))
        + y*(-0.25*v121 + x2*(0.75*v321)
        + y*(-0.25*v131
        + y*(-0.25*v141))))
    + z*(    -0.25*v102 + x2*(0.75*v302 + x*(2.0*v402))
        + y*(-0.25*v112 + x2*(0.75*v312)
        + y*(-0.25*v122
        + y*(-0.25*v132)))
    + z*(    -0.25*v103
        + y*(-0.25*v113)
    + z*(    -0.25*v104))))) / (x * x2) * (sfac**2) 
    

@nb.njit
def _st(x, y, z):
    return (  v110 + x*(2*v210 + x*(3*v310 + x*(4*v410 + x*(5*v510))))
       + y*(2*v120 + x*(4*v220 + x*(6*v320 + x*(8*v420)))
       + y*(3*v130 + x*(6*v230 + x*(9*v330))
       + y*(4*v140 + x*(8*v240)
       + y*(5*v150))))
    + z*(     v111 + x*(2*v211 + x*(3*v311 + x*(4*v411)))
       + y*(2*v121 + x*(4*v221 + x*(6*v321))
       + y*(3*v131 + x*(6*v231)
       + y*(4*v141)))
    + z*(     v112 + x*(2*v212 + x*(3*v312))
       + y*(2*v122 + x*(4*v222)
       + y*(3*v132))
    + z*(  v113 )))) / x * (0.5 * tfac * sfac)


@nb.njit
def _tt(x, y, z):
    return (  2*v020 + x*( 2*v120 + x*( 2*v220 + x*(2*v320 + x*(2*v420))))
       + y* ( 6*v030 + x*( 6*v130 + x*( 6*v230 + x*(6*v330)))
       + y* (12*v040 + x*(12*v140 + x*(12*v240))
       + y* (20*v050 + x*(20*v150)
       + y* (30*v060))))
    + z*(     2*v021 + x*( 2*v121 + x*(2*v221 + x*(2*v321)))
       + y* ( 6*v031 + x*( 6*v131 + x*(6*v231))
       + y* (12*v041 + x*(12*v141)
       + y* (20*v051)))
    + z*(     2*v022 + x*(2*v122 + x*(2*v222))
       + y* ( 6*v032 + x*(6*v132)
       + y* (12*v042))
    + z*(     2*v023)))) * (tfac * tfac) 
    

@nb.njit
def _sp(x, y, z):
    return (  v101 + x*(2*v201 + x*(3*v301 + x*(4*v401 + x*(5*v501))))
       + y*(  v111 + x*(2*v211 + x*(3*v311 + x*(4*v411)))
       + y*(  v121 + x*(2*v221 + x*(3*v321))
       + y*(  v131 + x*(2*v231)
       + y*   v141)))
    + z*(   2*v102 + x*(4*v202 + x*(6*v302 + x*(8*v402)))
       + y*(2*v112 + x*(4*v212 + x*(6*v312))
       + y*(2*v122 + x*(4*v222)
       + y*(2*v132)))
    + z*(   3*v103 + x*(6*v203)
       + y*(3*v113)
    + z*(   4*v104)))) / x * (0.5 * pfac * sfac)

    
@nb.njit
def _tp(x, y, z):
    return (   v011 + x*(  v111 + x*(  v211 + x*(  v311 + x*  v411)))
       + y* (2*v021 + x*(2*v121 + x*(2*v221 + x*(2*v321)))
       + y* (3*v031 + x*(3*v131 + x*(3*v231))
       + y* (4*v041 + x*(4*v141)
       + y* (5*v051))))
    + z*(   (2*v012 + x*(2*v112 + x*(2*v212 + x*(2*v312)))
       + y* (4*v022 + x*(4*v122 + x*(4*v222))
       + y* (6*v032 + x*(6*v132)
       + y* (8*v042))))
    + z*(   (3*v013 + x*(3*v113)
       + y* (6*v023))
    + z*(   (4*v014))))) * (tfac * pfac)


@nb.njit
def _pp(x, y, z):
    return (
              2*v002 + x*( 2*v102 + x*(2*v202 + x*(2*v302 + x*(2*v402))))
       + y* ( 2*v012 + x*( 2*v112 + x*(2*v212 + x*(2*v312)))
       + y* ( 2*v022 + x*( 2*v122 + x*(2*v222))
       + y* ( 2*v032 + x*( 2*v132)
       + y* ( 2*v042))))
    + z*(     6*v003 + x*( 6*v103 + x*(6*v203))
       + y* ( 6*v013 + x*( 6*v113)
       + y* ( 6*v023))
    + z*(    12*v004 + x*(12*v104)
       + y* (12*v014)  
    + z*(    20*v005
    + z*(    30*v006))))) * (pfac * pfac)


@nb.njit
def _sss(x, y, z, x2):
    return (
            0.375*v100 + x2*(-0.375*v300 + x2*(1.875*v500 + x*(6.0*v600)))
       + y*(0.375*v110 + x2*(-0.375*v310 + x2*(1.875*v510))
       + y*(0.375*v120 + x2*(-0.375*v320)
       + y*(0.375*v130 + x2*(-0.375*v330)
       + y*(0.375*v140
       + y*(0.375*v150)))))
    + z*(   0.375*v101 + x2*(-0.375*v301 + x2*(1.875*v501))
       + y*(0.375*v111 + x2*(-0.375*v311)
       + y*(0.375*v121 + x2*(-0.375*v321)
       + y*(0.375*v131
       + y*(0.375*v141))))
    + z*(   0.375*v102 + x2*(-0.375*v302)
       + y*(0.375*v112 + x2*(-0.375*v312)
       + y*(0.375*v122
       + y*(0.375*v132)))
    + z*(   0.375*v103
       + y*(0.375*v113))))) / (x * x2 * x2) * (sfac**3)


@nb.njit
def _sst(x, y, z, x2):
    return (
        +   (-0.25*v110 + x2*(0.75*v310 + x*(2.0*v410 + x*(3.75*v510)))
        + y*(-0.50*v120 + x2*(1.50*v320 + x*(4.0*v420))
        + y*(-0.75*v130 + x2*(2.25*v330)
        + y*(-1.00*v140
        + y*(-1.25*v150)))))
    + z*(
        +   (-0.25*v111 + x2*(0.75*v311 + x*(2.0*v411))
        + y*(-0.50*v121 + x2*(1.50*v321)
        + y*(-0.75*v131
        + y*(-1.00*v141))))
    + z*(
        +   (-0.25*v112 + x2*(0.75*v312)
        + y*(-0.50*v122
        + y*(-0.75*v132)))
    + z*(
        +   (-0.25*v113))))) / (x * x2) * (sfac**2 * tfac)
    

@nb.njit
def _stt(x, y, z):
    return ( 2*v120 + x*( 4*v220 + x*( 6*v320 + x*(8*v420)))
       + y*( 6*v130 + x*(12*v230 + x*(18*v330))
       + y*(12*v140 + x*(24*v240)
       + y*(20*v150)))
    + z*(    2*v121 + x*( 4*v221 + x*(6*v321))
       + y*( 6*v131 + x*(12*v231)
       + y*(12*v141))
    + z*(   2*v122 + x*(4*v222)
       + y*(6*v132)))) / x * (tfac * tfac * sfac / 2)


@nb.njit
def _ttt(x, y, z):
    return (   6*v030 + x*( 6*v130 + x*( 6*v230 + x*(6*v330)))
       + y* ( 24*v040 + x*(24*v140 + x*(24*v240))
       + y* ( 60*v050 + x*(60*v150)
       + y* (120*v060)))
    + z*(     6*v031 + x*( 6*v131 + x*(6*v231))
       + y* (24*v041 + x*(24*v141)
       + y* (60*v051))
    + z*(     6*v032 + x*(6*v132)
       + y* (24*v042)))) * (tfac**3)


@nb.njit
def _ssp(x, y, z, x2):
    return (
            -0.25*v101 + x2*(0.75*v301 + x*(2.0*v401 + x*(3.75*v501)))
        + y*(-0.25*v111 + x2*(0.75*v311 + x*(2.0*v411))
        + y*(-0.25*v121 + x2*(0.75*v321)
        + y*(-0.25*v131
        + y*(-0.25*v141))))
    + z*(    -0.50*v102 + x2*(1.50*v302 + x*(4.0*v402))
        + y*(-0.50*v112 + x2*(1.50*v312)
        + y*(-0.50*v122
        + y*(-0.50*v132)))
    + z*(    -0.75*v103
        + y*(-0.75*v113)
    + z*(    -1.00*v104)))) / (x * x2) * (sfac**2 * pfac)
    

@nb.njit
def _stp(x, y, z):
    return (  v111 + x*(2*v211 + x*(3*v311 + x*(4*v411)))
       + y*(2*v121 + x*(4*v221 + x*(6*v321))
       + y*(3*v131 + x*(6*v231)
       + y*(4*v141)))
    + z*(   2*v112 + x*(4*v212 + x*(6*v312))
       + y*(4*v122 + x*(8*v222)
       + y*(6*v132))
    + z*(   3*v113 ))) / x * (0.5 * sfac * tfac * pfac)


@nb.njit
def _ttp(x, y, z):    
    return (  2*v021 + x*( 2*v121 + x*(2*v221 + x*(2*v321)))
       + y* ( 6*v031 + x*( 6*v131 + x*(6*v231))
       + y* (12*v041 + x*(12*v141)
       + y* (20*v051)))
    + z*(     4*v022 + x*( 4*v122 + x*(4*v222))
       + y* (12*v032 + x*(12*v132)
       + y* (24*v042))
    + z*(     6*v023))) * (tfac * tfac * pfac) 


@nb.njit
def _spp(x, y, z):
    return ( 2*v102 + x*( 4*v202 + x*(6*v302 + x*(8*v402)))
       + y*( 2*v112 + x*( 4*v212 + x*(6*v312))
       + y*( 2*v122 + x*( 4*v222)
       + y*( 2*v132)))
    + z*(    6*v103 + x*(12*v203)
       + y*( 6*v113)
    + z*(   12*v104))) / x * (pfac * pfac * sfac / 2)


@nb.njit
def _tpp(x, y, z):
    return (( 2*v012 + x*(2*v112 + x*(2*v212 + x*(2*v312)))
       + y* ( 4*v022 + x*(4*v122 + x*(4*v222))
       + y* ( 6*v032 + x*(6*v132)
       + y* ( 8*v042))))
    + z*(   ( 6*v013 + x*(6*v113)
       + y* (12*v023))
    + z*(   (12*v014)))) * (tfac * pfac * pfac)
    

@nb.njit
def _ppp(x, y, z):
    return (   6*v003 + x*( 6*v103 + x*(6*v203))
       + y* (  6*v013 + x*( 6*v113)
       + y* (  6*v023))
    + z*(     24*v004 + x*(24*v104)
       + y* ( 24*v014)  
    + z*(     60*v005
    + z*(    120*v006)))) * (pfac**3)

# fmt: on
