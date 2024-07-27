"""
In-situ density using polyTEOS10bsq [1]_ approximation to the TEOS-10 Gibbs Sea Water standard [2]_

.. [1] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

.. [2] McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and 
    the Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127, 
    ISBN 978-0-646-55621-5. 
"""

import numpy as np
import numba as nb

# From Roquet et al. (2015) Appendix A.1.
R00 = 4.6494977072e01
R01 = -5.2099962525e00
R02 = 2.2601900708e-01
R03 = 6.4326772569e-02
R04 = 1.5616995503e-02
R05 = -1.7243708991e-03

# Coefficients from NEMO eosbn2.F90
deltaS = 32.0
sfac = 0.875 / 35.16504
tfac = 1.0 / 40.0
zfac = 1.0e-4
EOS000 = 8.0189615746e02
EOS100 = 8.6672408165e02
EOS200 = -1.7864682637e03
EOS300 = 2.0375295546e03
EOS400 = -1.2849161071e03
EOS500 = 4.3227585684e02
EOS600 = -6.0579916612e01
EOS010 = 2.6010145068e01
EOS110 = -6.5281885265e01
EOS210 = 8.1770425108e01
EOS310 = -5.6888046321e01
EOS410 = 1.7681814114e01
EOS510 = -1.9193502195
EOS020 = -3.7074170417e01
EOS120 = 6.1548258127e01
EOS220 = -6.0362551501e01
EOS320 = 2.9130021253e01
EOS420 = -5.4723692739
EOS030 = 2.1661789529e01
EOS130 = -3.3449108469e01
EOS230 = 1.9717078466e01
EOS330 = -3.1742946532
EOS040 = -8.3627885467
EOS140 = 1.1311538584e01
EOS240 = -5.3563304045
EOS050 = 5.4048723791e-01
EOS150 = 4.8169980163e-01
EOS060 = -1.9083568888e-01
EOS001 = 1.9681925209e01
EOS101 = -4.2549998214e01
EOS201 = 5.0774768218e01
EOS301 = -3.0938076334e01
EOS401 = 6.6051753097
EOS011 = -1.3336301113e01
EOS111 = -4.4870114575
EOS211 = 5.0042598061
EOS311 = -6.5399043664e-01
EOS021 = 6.7080479603
EOS121 = 3.5063081279
EOS221 = -1.8795372996
EOS031 = -2.4649669534
EOS131 = -5.5077101279e-01
EOS041 = 5.5927935970e-01
EOS002 = 2.0660924175
EOS102 = -4.9527603989
EOS202 = 2.5019633244
EOS012 = 2.0564311499
EOS112 = -2.1311365518e-01
EOS022 = -1.2419983026
EOS003 = -2.3342758797e-02
EOS103 = -1.8507636718e-02
EOS013 = 3.7969820455e-01
ALP000 = -6.5025362670e-01
ALP100 = 1.6320471316
ALP200 = -2.0442606277
ALP300 = 1.4222011580
ALP400 = -4.4204535284e-01
ALP500 = 4.7983755487e-02
ALP010 = 1.8537085209
ALP110 = -3.0774129064
ALP210 = 3.0181275751
ALP310 = -1.4565010626
ALP410 = 2.7361846370e-01
ALP020 = -1.6246342147
ALP120 = 2.5086831352
ALP220 = -1.4787808849
ALP320 = 2.3807209899e-01
ALP030 = 8.3627885467e-01
ALP130 = -1.1311538584
ALP230 = 5.3563304045e-01
ALP040 = -6.7560904739e-02
ALP140 = -6.0212475204e-02
ALP050 = 2.8625353333e-02
ALP001 = 3.3340752782e-01
ALP101 = 1.1217528644e-01
ALP201 = -1.2510649515e-01
ALP301 = 1.6349760916e-02
ALP011 = -3.3540239802e-01
ALP111 = -1.7531540640e-01
ALP211 = 9.3976864981e-02
ALP021 = 1.8487252150e-01
ALP121 = 4.1307825959e-02
ALP031 = -5.5927935970e-02
ALP002 = -5.1410778748e-02
ALP102 = 5.3278413794e-03
ALP012 = 6.2099915132e-02
ALP003 = -9.4924551138e-03
BET000 = 1.0783203594e01
BET100 = -4.4452095908e01
BET200 = 7.6048755820e01
BET300 = -6.3944280668e01
BET400 = 2.6890441098e01
BET500 = -4.5221697773
BET010 = -8.1219372432e-01
BET110 = 2.0346663041
BET210 = -2.1232895170
BET310 = 8.7994140485e-01
BET410 = -1.1939638360e-01
BET020 = 7.6574242289e-01
BET120 = -1.5019813020
BET220 = 1.0872489522
BET320 = -2.7233429080e-01
BET030 = -4.1615152308e-01
BET130 = 4.9061350869e-01
BET230 = -1.1847737788e-01
BET040 = 1.4073062708e-01
BET140 = -1.3327978879e-01
BET050 = 5.9929880134e-03
BET001 = -5.2937873009e-01
BET101 = 1.2634116779
BET201 = -1.1547328025
BET301 = 3.2870876279e-01
BET011 = -5.5824407214e-02
BET111 = 1.2451933313e-01
BET211 = -2.4409539932e-02
BET021 = 4.3623149752e-02
BET121 = -4.6767901790e-02
BET031 = -6.8523260060e-03
BET002 = -6.1618945251e-02
BET102 = 6.2255521644e-02
BET012 = -2.6514181169e-03
BET003 = -2.3025968587e-04
PEN000 = -9.8409626043
PEN100 = 2.1274999107e01
PEN200 = -2.5387384109e01
PEN300 = 1.5469038167e01
PEN400 = -3.3025876549
PEN010 = 6.6681505563
PEN110 = 2.2435057288
PEN210 = -2.5021299030
PEN310 = 3.2699521832e-01
PEN020 = -3.3540239802
PEN120 = -1.7531540640
PEN220 = 9.3976864981e-01
PEN030 = 1.2324834767
PEN130 = 2.7538550639e-01
PEN040 = -2.7963967985e-01
PEN001 = -1.3773949450
PEN101 = 3.3018402659
PEN201 = -1.6679755496
PEN011 = -1.3709540999
PEN111 = 1.4207577012e-01
PEN021 = 8.2799886843e-01
PEN002 = 1.7507069098e-02
PEN102 = 1.3880727538e-02
PEN012 = -2.8477365341e-01
APE000 = -1.6670376391e-01
APE100 = -5.6087643219e-02
APE200 = 6.2553247576e-02
APE300 = -8.1748804580e-03
APE010 = 1.6770119901e-01
APE110 = 8.7657703198e-02
APE210 = -4.6988432490e-02
APE020 = -9.2436260751e-02
APE120 = -2.0653912979e-02
APE030 = 2.7963967985e-02
APE001 = 3.4273852498e-02
APE101 = -3.5518942529e-03
APE011 = -4.1399943421e-02
APE002 = 7.1193413354e-03
BPE000 = 2.6468936504e-01
BPE100 = -6.3170583896e-01
BPE200 = 5.7736640125e-01
BPE300 = -1.6435438140e-01
BPE010 = 2.7912203607e-02
BPE110 = -6.2259666565e-02
BPE210 = 1.2204769966e-02
BPE020 = -2.1811574876e-02
BPE120 = 2.3383950895e-02
BPE030 = 3.4261630030e-03
BPE001 = 4.1079296834e-02
BPE101 = -4.1503681096e-02
BPE011 = 1.7676120780e-03
BPE002 = 1.7269476440e-04


@nb.njit
def rho_horiz(S, T, Z):
    # Compute the in situ density anomaly from the vertical reference profile of density.
    # Check value from Roquet et al. (2015):
    #   for S=30, T=10, Z=1000, should get 1022.85377
    # Code from eosbn2.F90

    x, y, z = _process(S, T, Z)

    # fmt: off
    n3 = EOS013*y + EOS103*x + EOS003
    n2 = ((EOS022*y
        + EOS112*x + EOS012)*y
        + (EOS202*x + EOS102)*x + EOS002
        )
    n1 = ((((EOS041*y   
        + EOS131*x + EOS031)*y   
        + (EOS221*x + EOS121)*x + EOS021)*y   
        + ((EOS311*x + EOS211)*x + EOS111)*x + EOS011)*y   
        + (((EOS401*x + EOS301)*x + EOS201)*x + EOS101)*x + EOS001
    )
    n0 = ((((((EOS060*y   
        + EOS150*x + EOS050)*y   
        + (EOS240*x + EOS140)*x + EOS040)*y   
        + ((EOS330*x + EOS230)*x + EOS130)*x + EOS030)*y   
        + (((EOS420*x + EOS320)*x + EOS220)*x + EOS120)*x + EOS020)*y   
        + ((((EOS510*x + EOS410)*x + EOS310)*x + EOS210)*x + EOS110)*x + EOS010)*y   
        + (((((EOS600*x + EOS500)*x + EOS400)*x + EOS300)*x + EOS200)*x + EOS100)*x + EOS000
    )
    
    return ((n3 * z + n2) * z + n1) * z + n0
    # fmt: on


@nb.njit
def rho_vert(Z):
    # Calculate the vertical profile of in-situ density.
    # Check value from Roquet et al. (2015):
    #   for Z=1000, should get 4.59763035
    z = Z * zfac
    return (((((R05 * z + R04) * z + R03) * z + R02) * z + R01) * z + R00) * z


@nb.njit
def rho(S, T, Z):
    # Calculate the in-situ density.
    return rho_horiz(S, T, Z) + rho_vert(Z)


@nb.njit
def rho_anom(rho, rho0):
    # Calculate the in-situ density anomaly, (rho - rho0) / rho0.
    r1_rho0 = 1.0 / rho0
    return rho * r1_rho0 - 1.0


@nb.njit
def rho_s_t(S, T, Z):
    # Calculate S and T derivatives of in-situ density.
    x, y, z = _process(S, T, Z)
    rho_S = _s(x, y, z)
    rho_T = _t(x, y, z)

    return rho_S, rho_T


@nb.njit
def rho_z(S, T, Z):
    # Calculate Z derivative of in-situ density.
    x, y, z = _process(S, T, Z)
    return _z(x, y, z)


@nb.njit
def alpha_beta(S, T, Z, rho0):
    # Calculate thermal and haline expansion coefficients using analytic derivative of `rho_horiz`.
    # This is negligably more accurate than `alpha_beta_pre` which uses pre-computed coefficients.
    rho_s, rho_t = rho_s_t(S, T, Z)
    alpha = -rho_t / rho0
    beta = rho_s / rho0
    return alpha, beta


@nb.njit
def alpha_beta_pre(S, T, Z, rho0):
    # Calculate thermal and haline expansion coefficients, using pre-computed coefficients.
    # Code from NEMO eosbn2.F90

    x, y, z = _process(S, T, Z)
    r1_rho0 = 1.0 / rho0

    # fmt: off
    n3 = ALP003
    n2 = ALP012*y + ALP102*x + ALP002
    n1 = (((ALP031*y   
        + ALP121*x + ALP021)*y   
        + (ALP211*x + ALP111)*x + ALP011)*y   
        + ((ALP301*x + ALP201)*x + ALP101)*x + ALP001
    )
    n0 = (((((ALP050*y   
            + ALP140*x + ALP040)*y   
            + (ALP230*x + ALP130)*x + ALP030)*y   
            + ((ALP320*x + ALP220)*x + ALP120)*x + ALP020)*y   
            + (((ALP410*x + ALP310)*x + ALP210)*x + ALP110)*x + ALP010)*y   
            + ((((ALP500*x + ALP400)*x + ALP300)*x + ALP200)*x + ALP100)*x + ALP000
    )
    n  = ((n3 * z + n2) * z + n1) * z + n0
    alpha = n * r1_rho0

    n3 = BET003
    n2 = BET012*y + BET102*x + BET002
    n1 = (((BET031*y   
            + BET121*x + BET021)*y   
            + (BET211*x + BET111)*x + BET011)*y   
            + ((BET301*x + BET201)*x + BET101)*x + BET001
        )
    n0 = (((((BET050*y   
        + BET140*x + BET040)*y   
        + (BET230*x + BET130)*x + BET030)*y   
        + ((BET320*x + BET220)*x + BET120)*x + BET020)*y   
        + (((BET410*x + BET310)*x + BET210)*x + BET110)*x + BET010)*y   
        + ((((BET500*x + BET400)*x + BET300)*x + BET200)*x + BET100)*x + BET000
    )
    n  = ((n3 * z + n2) * z + n1) * z + n0
    beta = n / x * r1_rho0
    # fmt: on

    return alpha, beta


@nb.njit
def _process(S, T, Z):
    z = Z * zfac
    y = T * tfac
    x = np.sqrt(np.abs(S + deltaS) * sfac)
    return x, y, z


""" Begin individual partial derivatives """

# fmt: off
@nb.njit
def _s(x, y, z):
    n3 = EOS103
    n2 = EOS112*y + 2*EOS202*x + EOS102
    n1 = (((
        + EOS131*y   
        + (2*EOS221*x + EOS121))*y   
        + ((3*EOS311*x + 2*EOS211)*x + EOS111))*y   
        + (((4*EOS401*x + 3*EOS301)*x + 2*EOS201)*x + EOS101)
    )
    n0 = (((((EOS150*y   
        + (2*EOS240*x + EOS140))*y   
        + ((3*EOS330*x + 2*EOS230)*x + EOS130))*y   
        + (((4*EOS420*x + 3*EOS320)*x + 2*EOS220)*x + EOS120))*y   
        + ((((5*EOS510*x + 4*EOS410)*x + 3*EOS310)*x + 2*EOS210)*x + EOS110))*y   
        + (((((6*EOS600*x + 5*EOS500)*x + 4*EOS400)*x + 3*EOS300)*x + 2*EOS200)*x + EOS100)
    )
    return (((n3 * z + n2) * z + n1) * z + n0) / x * (0.5 * sfac)


@nb.njit
def _t(x, y, z):
    n3 = EOS013
    n2 = 2*EOS022*y + EOS112*x + EOS012
    n1 = ((((4*EOS041*y
        + 3*EOS131*x + 3*EOS031)*y   
        + (2*EOS221*x + 2*EOS121)*x + 2*EOS021)*y   
        + ((EOS311*x + EOS211)*x + EOS111)*x + EOS011)
    )
    n0 = ((((((6*EOS060*y   
        + 5*EOS150*x + 5*EOS050)*y   
        + (4*EOS240*x + 4*EOS140)*x + 4*EOS040)*y   
        + ((3*EOS330*x + 3*EOS230)*x + 3*EOS130)*x + 3*EOS030)*y   
        + (((2*EOS420*x + 2*EOS320)*x + 2*EOS220)*x + 2*EOS120)*x + 2*EOS020)*y   
        + ((((EOS510*x + EOS410)*x + EOS310)*x + EOS210)*x + EOS110)*x + EOS010)
    )
    return (((n3 * z + n2) * z + n1) * z + n0) * tfac


@nb.njit
def _z(x, y, z):
    n3 = EOS013*y + EOS103*x + EOS003
    n2 = ((EOS022*y
        + EOS112*x + EOS012)*y
        + (EOS202*x + EOS102)*x + EOS002
        )
    n1 = ((((EOS041*y   
        + EOS131*x + EOS031)*y   
        + (EOS221*x + EOS121)*x + EOS021)*y   
        + ((EOS311*x + EOS211)*x + EOS111)*x + EOS011)*y   
        + (((EOS401*x + EOS301)*x + EOS201)*x + EOS101)*x + EOS001
    )
    
    rho_vert_z = ((((6*R05 * z + 5*R04) * z + 4*R03) * z + 3*R02) * z + 2*R01) * z + R00
    rho_horiz_z = ((3*n3 * z + 2*n2) * z + n1)

    return (rho_vert_z + rho_horiz_z) * zfac
# fmt: on
