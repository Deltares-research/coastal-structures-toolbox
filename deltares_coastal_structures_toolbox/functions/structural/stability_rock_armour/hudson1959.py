# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


def check_validity_range():
    """No validity ranges provided"""
    pass


def calculate_median_rock_mass_M50_no_damage(
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


def calculate_significant_wave_height_Hs_no_damage(
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


def lookup_table_damage_factors(
    damage_percentage: float | npt.NDArray[np.float64], rock_type: str = "rough"
) -> float | npt.NDArray[np.float64]:
    """Returns the multiplication value for the 0-damage value of the
    hudson formula, according to SPM1984 and Rock Manual 2007/2012

    Parameters
    ----------
    damage_percentage : float | npt.NDArray[np.float64]
        Percentage of damage required
    rock_type : str, optional
        Type of rock, either "smooth" or "rough", by default "rough"

    Returns
    -------
    float | npt.NDArray[np.float64]
        Multiplication factor

    Raises
    ------
    ValueError
        Unknown rock_type
    """

    percentages = np.array([0, 5, 10, 15, 20, 30, 40, 50, 100])
    mult_factors_rough = np.array([1.0, 1.0, 1.08, 1.14, 1.20, 1.29, 1.41, 1.54, 1.54])
    mult_factors_smooth = np.array([1.0, 1.0, 1.08, 1.19, 1.27, 1.37, 1.47, 1.56, 1.56])

    ind = np.searchsorted(percentages, damage_percentage, "right")
    if rock_type.lower().strip() == "rough":
        factor = mult_factors_rough[ind]
    elif rock_type.lower().strip() == "smooth":
        factor = mult_factors_smooth[ind]
    else:
        raise ValueError("Unknown rock type: {}".format(rock_type))

    return factor


def calculate_median_rock_mass_M50(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    damage_percentage: float | npt.NDArray[np.float64],
    rock_type: str = "rough",
    alpha_Hs: float | npt.NDArray[np.float64] = 1.27,
) -> float | npt.NDArray[np.float64]:
    """Calculate the required M50 based on Hs using the Hudson 1959 approach, including damage percentage

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
    damage_percentage : float | npt.NDArray[np.float64]
        Percentage of displaced units
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H10Percent according to SPM, by default 1.27
        (as per Hudson approach since Shore Protection Manual 1984)

    Returns
    -------
    M50: float | npt.NDArray[np.float64]
        Median rock mass (kg)
    """

    mult_factor = lookup_table_damage_factors(
        damage_percentage=damage_percentage, rock_type=rock_type
    )
    M50 = calculate_median_rock_mass_M50_no_damage(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    M50 = M50 / np.power(mult_factor, 3)

    return M50


def calculate_significant_wave_height_Hs(
    M50: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    damage_percentage: float | npt.NDArray[np.float64],
    rock_type: str = "rough",
    alpha_Hs: float | npt.NDArray[np.float64] = 1.27,
) -> float | npt.NDArray[np.float64]:
    """Calculate the Hs based on M50 using the Hudson 1959 approach, including damage percentage

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 564

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
    damage_percentage : float | npt.NDArray[np.float64]
        Percentage of displaced units
    alpha_Hs : float | npt.NDArray[np.float64], optional
        Factor between Hs and H10Percent according to SPM, by default 1.27
        (as per Hudson approach since Shore Protection Manual 1984)

    Returns
    -------
    Hs: float | npt.NDArray[np.float64]
        Significant wave height (m)
    """

    mult_factor = lookup_table_damage_factors(
        damage_percentage=damage_percentage, rock_type=rock_type
    )
    Hs = calculate_significant_wave_height_Hs_no_damage(
        M50=M50,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    Hs = Hs * mult_factor

    return Hs
