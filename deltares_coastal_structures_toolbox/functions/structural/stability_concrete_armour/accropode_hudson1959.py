# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as hudson
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility

unit_properties = {
    "KD": {
        "trunk_non_breaking": 15,
        "trunk_breaking": 12,
        "head_non_breaking": 11.5,
        "head_breaking": 9.5,
        "Note": "Depending on seabed slope, see website concrete layer innovations",
    },  # numbers are from rock manual
    "kt": 1.29,  # as per concretelayer innovations design table information
    "nlayers": 1,
}


def check_validity_range(
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    seabed_slope_perc: float | npt.NDArray[np.float64] = np.nan,
):
    """Check the parameter values vs the validity range for Accropode

    For all parameters supplied, their values are checked versus the range of validity.
    When parameters are nan (by default), they are not checked.

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    seabed_slope_perc : float | npt.NDArray[np.float64], optional
        Seabed slope in percentage, by default np.nan
    """

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of the front-side slope of the structure",
            "Accropodes Hudson 1959",
            cot_alpha,
            1.33,
            1.5,
        )

    if not np.any(np.isnan(seabed_slope_perc)):
        core_utility.check_variable_validity_range(
            "Seabed slope in percentage",
            "Accropode documentation",
            seabed_slope_perc,
            0,
            10,
        )

    return


def calculate_unit_mass_M(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    KD: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = 1.33,
    alpha_Hs: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Determine required unit mass M based on Hs for Accropodes, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591
    Information in this set of functions is also based on the design manual
    (design table 2012, retrieved in march-2025)

    For more properties, see also unit_properties

    More information is available at the Accropode website and from the concrete layer innovations (CLI) team
    https://www.concretelayer.com/en/solutions/technologies/accropode

    A calculator is also available on the CLI website

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_non_breaking: 15
        trunk_breaking: 12
        head_non_breaking: 11.5
        head_breaking: 9.5
        see also functions to calculate KD from seabed slope
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
    """Determine significant wave height Hs based on M for Accropodes, using Hudson 1959

    For more details see: Hudson 1959 and Rock Manual:
    Hudson 1959, available here: https://doi.org/10.1061/JWHEAU.0000142 (or google)
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591
    Information in this set of functions is also based on the design manual
    (design table 2012, retrieved in march-2025)

    For more properties, see also unit_properties

    More information is available at the Accropode website and from the concrete layer innovations (CLI) team
    https://www.concretelayer.com/en/solutions/technologies/accropode

    A calculator is also available on the CLI website

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    KD : float | npt.NDArray[np.float64]
        trunk_non_breaking: 15
        trunk_breaking: 12
        head_non_breaking: 11.5
        head_breaking: 9.5
        see also functions to calculate KD from seabed slope
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


def calculate_KD_breaking_trunk_from_seabed_slope(
    seabed_slope_perc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Returns the KD value based on the seabed slope.
    Only to be applied for breaking condition at the trunk

    For more information, see Concrete Layer Innovations:
    https://www.concretelayer.com/en/solutions/technologies/accropode

    This value is an interpretation of graphical information in the design table
    (design table 2012, retrieved in march-2025)

    Parameters
    ----------
    seabed_slope_perc : float | npt.NDArray[np.float64]
        slope of seabed (%)

    Returns
    -------
    float | npt.NDArray[np.float64]
        KD value
    """
    graph_seabed_slope_perc = np.array([0, 1, 5, 10])  # fitted from design table 2012
    graph_KD = np.array([15, 15, 9.7, 8])

    seabed_slope_KD = np.interp(seabed_slope_perc, graph_seabed_slope_perc, graph_KD)
    seabed_slope_KD = np.floor(seabed_slope_KD / 0.1) * 0.1

    check_validity_range(seabed_slope_perc=seabed_slope_perc)

    return seabed_slope_KD


def calculate_KD_nonbreaking_trunk_from_seabed_slope() -> (
    float | npt.NDArray[np.float64]
):
    """Returns the KD value based on the seabed slope.
    Only to be applied for nonbreaking condition at the trunk.
    This value is fixed at 15, similar to 1% trunk breaking waves value

    For more information, see Concrete Layer Innovations:
    https://www.concretelayer.com/en/solutions/technologies/accropode

    This value is an interpretation of graphical information in the design table
    (design table 2012, retrieved in march-2025)

    Returns
    -------
    float | npt.NDArray[np.float64]
        KD value
    """
    return 15.0
