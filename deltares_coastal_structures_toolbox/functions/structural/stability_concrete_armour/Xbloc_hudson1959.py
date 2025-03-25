# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as hudson
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics

unit_properties = {
    "KD": {
        "trunk_breaking": 16,
        "head_breaking": 13,
    },  # numbers are from rock manual
    "kt": 1.40,
    "nlayers": 1,
}


def check_validity_range(
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    seabed_slope_perc: float | npt.NDArray[np.float64] = np.nan,
):
    pass


def calculate_unit_mass_M(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine required unit mass M based on Hs for Xbloc, using Hudson 1959

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
        trunk_breaking: 16
        head_breaking: 13
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
        Note that 1.33 is recommended, and shallower then 1.5
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

    check_validity_range(cot_alpha=cot_alpha)

    return M


def calculate_significant_wave_height_Hs(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine significant wave height Hs based on M for Accropode II, using Hudson 1959

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
        trunk_breaking: 16
        head_breaking: 13
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

    check_validity_range(cot_alpha=cot_alpha)

    return Hs


def xbloc_calculate_unit_volume_V_base(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
) -> float | npt.NDArray[np.float64]:

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )
    V = (Hs / (2.77 * Delta)) ** 3

    return V


def xbloc_calculate_unit_mass_M_base(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
) -> float | npt.NDArray[np.float64]:

    pass


def calculate_correctionfactor_unit_mass_M_by_cotalpha_seabed(
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    correction_factor = 1.0
    if cot_alpha <= 30 and cot_alpha > 20:
        correction_factor = 1.1
    elif cot_alpha <= 20 and cot_alpha > 15:
        correction_factor = 1.25
    elif cot_alpha <= 15 and cot_alpha > 10:
        correction_factor = 1.5
    elif cot_alpha < 10:
        correction_factor = 2.0

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_slopeperc_seabed(
    slope_perc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    cot_alpha = 100 / slope_perc
    correction_factor = calculate_correctionfactor_unit_mass_M_by_cotalpha_seabed(
        cot_alpha=cot_alpha
    )

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_relative_freeboard(
    freeboard: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
):

    rel_freeboard = freeboard / Hs

    correction_factor = 1.0

    if rel_freeboard < 0.5:
        correction_factor = 2.0
    elif rel_freeboard < 1.0:
        correction_factor = 1.5

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_h(
    h: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
):

    rel_h = h / Hs

    correction_factor = 1.0

    if rel_h > 2.5:
        correction_factor = 1.5
    elif rel_h > 3.5:
        correction_factor = 2.0

    return correction_factor


def switch_correction_factor_unit_mass_M_design_event_frequency(
    design_event_occurs_frequently: bool = False,
):
    correction_factor = 1.0
    if design_event_occurs_frequently:
        correction_factor = 1.25

    return correction_factor


def switch_correctionfactor_unit_mass_M_by_core_permeability(
    low_core_permeability: bool = False,
    core_impermeable: bool = False,
):

    correction_factor = 1.0

    if low_core_permeability:
        correction_factor = 1.5
    if core_impermeable:
        correction_factor = 2.0

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_slope_armour(
    cot_alpha: float | npt.NDArray[np.float64],
):

    correction_factor = 1.0
    if cot_alpha > 2 / 3:
        correction_factor = 1.25
    elif cot_alpha > 1 / 2:
        correction_factor = 1.5

    return correction_factor
