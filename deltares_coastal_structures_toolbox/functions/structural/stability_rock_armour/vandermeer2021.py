# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as vdm
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as vdm_m


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
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
        H2p=H2p,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
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
        H2p=H2p,
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
