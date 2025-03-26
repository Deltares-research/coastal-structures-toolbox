# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as hudson

unit_properties = {
    "KD": {
        "double_layer_trunk_non_breaking": 7.5,
        "double_layer_trunk_breaking": 6.5,
        "double_layer_head_non_breaking": 5.0,
        "double_layer_head_breaking": None,
    },  # page 591 rockmanual
    "nv": 0.47,
    "kt": 1.10,
    "nlayers": 2,
    "Delta_x_fact_Dn": 1.7,  # although irregular placement
    "Delta_y_fact_Dn": 0.85,
}


def check_validity_range_Hudson1959():
    """No validity ranges provided"""
    pass


def calculate_unit_mass_M_Hudson1959(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine required unit mass M based on Hs for double layer cubes, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591

    For more properties, see also unit_properties

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_non_breaking: 7.5
        trunk_breaking: 6.5
        head_non_breaking: 5.0
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


def calculate_significant_wave_height_Hs_Hudson1959(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine significant wave height Hs based on M for doubles layer cubes, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591

    For more properties, see also unit_properties

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_non_breaking: 7.5
        trunk_breaking: 6.5
        head_non_breaking: 5.0
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
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
