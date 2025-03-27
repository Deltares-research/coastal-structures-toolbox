# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as hudson

unit_properties = {
    "KD": {
        "trunk_double_layer": 28,
        "trunk_single_layer": 12,
        "head_double_layer": 7,
        "head_single_layer": 5,
    },  # CubiPod manual
    "kt": 1.0,
    "nlayers": [1, 2],
}


def check_validity_range():

    pass


def calculate_unit_mass_M(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine required unit mass M based on Hs for Cubipod unit, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Cubidpod Manual (2016) (retrieved march 2025): https://www.cubipod.com/

    A calculator is also available on the cupipod website

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_double_layer: 28
        trunk_single_layer: 12
        head_double_layer: 7
        head_single_layer: 5
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H used according to Rock Manual by default 1.0
        for concrete armour units

    Returns
    -------
    M: float | npt.NDArray[np.float64]
        Unit mass M (kg)
    """
    M = hudson.calculate_median_rock_mass_M50_no_damage(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    return M


def calculate_significant_wave_height_Hs(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine significant wave height Hs based on M for Cubipod unit, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Cubidpod Manual (2016) (retrieved march 2025): https://www.cubipod.com/

    A calculator is also available on the cupipod website

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_double_layer: 28
        trunk_single_layer: 12
        head_double_layer: 7
        head_single_layer: 5
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
        Note that 1.33 is recommended, and shallower then 1.5
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H used according to Rock Manual by default 1.0
        for concrete armour units

    Returns
    -------
    Hs: float | npt.NDArray[np.float64]
        Significant wave height (m)
    """
    Hs = hudson.calculate_significant_wave_height_Hs_no_damage(
        M50=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    return Hs
