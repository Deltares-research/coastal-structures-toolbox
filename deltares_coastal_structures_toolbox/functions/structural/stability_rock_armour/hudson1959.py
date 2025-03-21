# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics

# import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range():
    """No validity ranges provided"""
    pass


def calculate_hudson1959_no_damage_M50(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    alpha_Hs: float | npt.NDArray[np.float64] = 1.27,
) -> float | npt.NDArray[np.float64]:
    """Calculate the required (no-damage) M50 using the Hudson 1959 approach

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 564

    Note: no limits to the formula have been provided in the paper

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        Stability coefficient, Hudson formula
        Hints (Rock Manual, 2007) for use with H10Percent:
        - KD = 2.0 for breaking waves
        - KD = 4.0 for non-breaking waves
        Breaking waves relate to breaking on foreshore, not to breaking on structure
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H10Percent according to SPM, by default 1.27
        (as per Hudson approach since Shore Protection Manual 1984)

    Returns
    -------
    M50: float | npt.NDArray[np.float64]
        Median rock mass (kg)
    """

    H_use = Hs * alpha_Hs
    Delta = core_physics.calculate_buoyant_density_Delta(rho_armour, rho_water)
    M50 = (rho_armour * H_use**3) / (KD * Delta**3 * cot_alpha)

    return M50


def calculate_hudson1959_no_damage_Hs(
    M50: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    alpha_Hs: float | npt.NDArray[np.float64] = 1.27,
) -> float | npt.NDArray[np.float64]:
    """Calculate the required (no-damage) Hs using the Hudson 1959 approach

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/ page 564

    Note: no limits to the formula have been provided in the paper

    Parameters
    ----------
    M50 : float | npt.NDArray[np.float64]
        Median rock mass (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        Stability coefficient, Hudson formula
        Hints (Rock Manual, 2007) for use with H10Percent:
        - KD = 2.0 for breaking waves
        - KD = 4.0 for non-breaking waves
        Breaking waves relate to breaking on foreshore, not to breaking on structure
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H10Percent according to SPM, by default 1.27
        (as per Hudson approach since Shore Protection Manual 1984)

    Returns
    -------
    Hs: float | npt.NDArray[np.float64]
        Significant wave height (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(rho_armour, rho_water)
    Hs = (Delta / alpha_Hs) * ((M50 * KD * cot_alpha) / rho_armour) ** (1 / 3)

    return Hs
