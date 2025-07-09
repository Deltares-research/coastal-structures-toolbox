# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as vdm
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as vdm_m


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 6.49,
    c_s: float = 0.97,
    c_pl_mult: float = 1.0,
    c_s_mult: float = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers with the Van der Meer (2021) formula.

    For more details see Van der Meer (2021), available here https://doi.org/10.48438/jchs.2021.0008

    Note that code-wise this comes down to the same as the Modified Van der Meer (1988) formula with
    different coefficients and without the H2%/Hs term, hence the chosen implementation.

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 6.49
    c_s : float, optional
        Coefficient for surging waves (-), by default 0.97
    c_pl_mult : float, optional
        Multiplication factor on the coefficient for plunging waves (-), by default 1.0
    c_s_mult : float, optional
        Multiplication factor on the coefficient for surging waves (-), by default 1.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """

    S = vdm_m.calculate_damage_number_S(
        Hs=Hs,
        H2p=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        rho_armour=rho_armour,
        Dn50=Dn50,
        M50=M50,
        c_pl=c_pl * c_pl_mult,
        c_s=c_s * c_s_mult,
    )

    vdm.check_validity_range(
        P=P,
        Hs=Hs,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    c_pl: float = 6.49,
    c_s: float = 0.97,
    c_pl_mult: float = 1.0,
    c_s_mult: float = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers with the Van der Meer (2021) formula.

    For more details see Van der Meer (2021), available here https://doi.org/10.48438/jchs.2021.0008

    Note that code-wise this comes down to the same as the Modified Van der Meer (1988) formula with
    different coefficients and without the H2%/Hs term, hence the chosen implementation.

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 6.49
    c_s : float, optional
        Coefficient for surging waves (-), by default 0.97
    c_pl_mult : float, optional
        Multiplication factor on the coefficient for plunging waves (-), by default 1.0
    c_s_mult : float, optional
        Multiplication factor on the coefficient for surging waves (-), by default 1.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    Dn50 = vdm_m.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        H2p=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        rho_armour=rho_armour,
        S=S,
        c_pl=c_pl * c_pl_mult,
        c_s=c_s * c_s_mult,
    )

    vdm.check_validity_range(
        P=P,
        Hs=Hs,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 6.49,
    c_s: float = 0.97,
    c_pl_mult: float = 1.0,
    c_s_mult: float = 1.0,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for rock armour layers with the Van der Meer (2021) formula.

    For more details see Van der Meer (2021), available here https://doi.org/10.48438/jchs.2021.0008

    Note that code-wise this comes down to the same as the Modified Van der Meer (1988) formula with
    different coefficients and without the H2%/Hs term, hence the chosen implementation.

    Parameters
    ----------
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 6.49
    c_s : float, optional
        Coefficient for surging waves (-), by default 0.97
    c_pl_mult : float, optional
        Multiplication factor on the coefficient for plunging waves (-), by default 1.0
    c_s_mult : float, optional
        Multiplication factor on the coefficient for surging waves (-), by default 1.0
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)
    """

    Hs = vdm_m.calculate_significant_wave_height_Hs(
        ratio_H2p_Hs=1.0,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        rho_armour=rho_armour,
        S=S,
        Dn50=Dn50,
        M50=M50,
        c_pl=c_pl * c_pl_mult,
        c_s=c_s * c_s_mult,
        g=g,
    )

    vdm.check_validity_range(
        P=P,
        Hs=Hs,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )
    return Hs
