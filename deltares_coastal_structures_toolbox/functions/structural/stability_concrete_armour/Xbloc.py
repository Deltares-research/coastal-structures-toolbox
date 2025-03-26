# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

# import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
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


def xbloc_calculate_wave_height_Hs_from_V(
    V: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    total_correction_factor: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate applicable  Hs from Xbloc Volume V

    For more information see Xbloc website and team at https://www.xbloc.com/ and the design manual
    Information in this approach is based on the design manual from 2024, (retrieved march 2025)

    Parameters
    ----------
    V : float | npt.NDArray[np.float64]
        Volume of unit (m3)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    total_correction_factor : float | npt.NDArray[np.float64]
        Total correction factor (all correction factors multiplied) by default 1.0

    Returns
    -------
    Hs: float | npt.NDArray[np.float64]
        Significant wave height (m)
    """
    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )
    V = V / total_correction_factor
    Hs = (V ** (1 / 3)) * 2.77 * Delta

    return Hs


def xbloc_calculate_wave_height_Hs_from_M(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    total_correction_factor: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate applicable  Hs from Xbloc mass M

    For more information see Xbloc website and team at https://www.xbloc.com/ and the design manual
    The information in this approach is based on the design manual from 2024, (retrieved march 2025).
    This approach is limited to a slope of 1:1.33 (or 4:3), expandable by correction factors

    A calculator is also available at the xbloc website.

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Mass of unit (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    total_correction_factor : float | npt.NDArray[np.float64]
        Total correction factor (all correction factors multiplied), by default 1.0

    Returns
    -------
    Hs: float | npt.NDArray[np.float64]
        Significant wave height (m)
    """

    V = M / rho_armour

    Hs = xbloc_calculate_wave_height_Hs_from_V(
        V=V,
        rho_armour=rho_armour,
        rho_water=rho_water,
        total_correction_factor=total_correction_factor,
    )

    return Hs


def xbloc_calculate_unit_volume_V(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    total_correction_factor: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate Xbloc unit volume V

    For more information see Xbloc website and team at https://www.xbloc.com/ and the design manual
    The information in this approach is based on the design manual from 2024, (retrieved march 2025).
    This approach is limited to a slope of 1:1.33 (or 4:3), expandable by correction factors

    A calculator is also available at the xbloc website.

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    total_correction_factor : float | npt.NDArray[np.float64], optional
        Total correction factor (starting point: maximum of calculated factors), by default 1.0
        The design manual states on correction factors: "It should be noted that the factors
        presented should be used with care as these are based
        more on project specific model test experience rather than
        on vast research programs. For the detailed design, physical
        model tests are always recommended."

    Returns
    -------
    V : float | npt.NDArray[np.float64]
        Volume of unit (m3)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    V = total_correction_factor * (Hs / (2.77 * Delta)) ** 3

    return V


def xbloc_calculate_unit_mass_M(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    total_correction_factor: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate Xbloc unit volume V

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    total_correction_factor : float | npt.NDArray[np.float64], optional
        Total correction factor (starting point: maximum of calculated factors), by default 1.0
        The design manual states on correction factors: "It should be noted that the factors
        presented should be used with care as these are based
        more on project specific model test experience rather than
        on vast research programs. For the detailed design, physical
        model tests are always recommended."


    Returns
    -------
    M : float | npt.NDArray[np.float64]
        Mass of unit (kg)
    """

    V = xbloc_calculate_unit_volume_V(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        total_correction_factor=total_correction_factor,
    )

    M = V * rho_armour

    return M


def calculate_correctionfactor_unit_mass_M_by_cotalpha_seabed(
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate correction factor based on seabed slope

    A steep foreshore can lead to adverse wave impact against the armour layer

    The design manual states on correction factors: "It should be noted that the factors
    presented should be used with care as these are based
    more on project specific model test experience rather than
    on vast research programs. For the detailed design, physical
    model tests are always recommended."

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Seabed slope (-)

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """

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


def calculate_correctionfactor_unit_mass_M_by_relative_freeboard(
    rel_freeboard: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate correction factor by relative freeboard

    Parameters
    ----------
    rel_freeboard : float | npt.NDArray[np.float64]
        the freeboard divided by the design wave height (-)

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """
    correction_factor = 1.0

    if rel_freeboard < 0.5:
        correction_factor = 2.0
    elif rel_freeboard < 1.0:
        correction_factor = 1.5

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_rel_depth(
    rel_depth: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate correction factor by relative depth

    Parameters
    ----------
    rel_depth : float | npt.NDArray[np.float64]
        the depth in front of the structure divided by the design wave height (-)

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """

    correction_factor = 1.0

    if rel_depth > 2.5:
        correction_factor = 1.5
    elif rel_depth > 3.5:
        correction_factor = 2.0

    return correction_factor


def switch_correction_factor_unit_mass_M_near_design_event_frequency(
    design_event_occurs_frequently: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Switch for correction factor by near design event frequency

    Parameters
    ----------
    design_event_occurs_frequently : bool, optional
        switch true/false. True in case there is frequently a near-design wave height
        during the lifetime of the structure, by default False

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """

    correction_factor = 1.0
    if design_event_occurs_frequently:
        correction_factor = 1.25

    return correction_factor


def switch_correctionfactor_unit_mass_M_by_core_permeability(
    low_core_permeability: bool = False,
    core_impermeable: bool = False,
):
    """Switch for correction factor by low or impermeable core

    Parameters
    ----------
    low_core_permeability : bool, optional
        For core with low permeability, by default False
    core_impermeable : bool, optional
        For impermeable core, by default False

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """

    correction_factor = 1.0

    if low_core_permeability:
        correction_factor = 1.5
    if core_impermeable:
        correction_factor = 2.0

    return correction_factor


def calculate_correctionfactor_unit_mass_M_by_structure_slope(
    cot_alpha: float | npt.NDArray[np.float64],
):
    """Calculate correction factor for structure slope different then 1:1.33

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)

    Returns
    -------
    correction_factor = float | npt.NDArray[np.float64]
        Correction factor applied on volume or mass
    """

    correction_factor = 1.0
    if cot_alpha > 2 / 3:
        correction_factor = 1.25
    elif cot_alpha > 1 / 2:
        correction_factor = 1.5

    return correction_factor
