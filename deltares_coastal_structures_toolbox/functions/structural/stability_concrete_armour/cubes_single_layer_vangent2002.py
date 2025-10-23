# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics

unit_properties = {
    "kt": 1.0,
    "nlayers": 1,
}


def check_validity_range():
    """No validity ranges provided"""
    pass


def calculate_unit_mass_M_start_of_damage(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns: float | npt.NDArray[np.float64] = 2.9,
) -> float | npt.NDArray[np.float64]:
    """Calculate required unit mass M for start of damage using van Gent (2002)

    For more details see Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 594
    And van Gent (2002) with DOI: https://doi.org/10.1680/bcsac.30428.0026

    Note that it is recommended to use a safety factor due to the small difference
    between start of damage and failure.
    Start of damage is determined as an Nod of 0.0
    Failure is determined as an Nod of 0.2

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    Ns : float | npt.NDArray[np.float64], optional
        Stability number, by default 2.9
        Recommended range for start of damage (see van Gent (2002)) 2.9 - 3.0

    Returns
    -------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    Dn = core_physics.check_usage_stabilitynumber(Hs=Hs, Delta=Delta, Ns=Ns)[0]

    M = core_physics.calculate_M50_from_Dn50(Dn50=Dn, rho_rock=rho_armour)

    return M


def calculate_unit_mass_M_failure(
    Hs: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns: float | npt.NDArray[np.float64] = 3.5,  # value according to rockmanual 3.5-3.75
) -> float | npt.NDArray[np.float64]:
    """Calculate required unit mass M for failure using van Gent (2002)

    For more details see Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 594
    And van Gent (2002) with DOI: https://doi.org/10.1680/bcsac.30428.0026

    Note that it is recommended to use a safety factor due to the small difference
    between start of damage and failure.
    Start of damage is determined as an Nod of 0.0
    Failure is determined as an Nod of 0.2

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    Ns : float | npt.NDArray[np.float64], optional
        Stability number, by default 3.5
        Recommended range for start of damage (see van Gent (2002)) 3.5 - 3.75

    Returns
    -------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    """

    M = calculate_unit_mass_M_start_of_damage(
        Hs=Hs, rho_water=rho_water, rho_armour=rho_armour, Ns=Ns
    )

    return M


def calculate_significant_wave_height_Hs_start_of_damage(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns: float | npt.NDArray[np.float64] = 2.9,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave height Hs for start of damage using van Gent (2002)

    For more details see Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 594
    And van Gent (2002) with DOI: https://doi.org/10.1680/bcsac.30428.0026

    Note that it is recommended to use a safety factor due to the small difference
    between start of damage and failure.
    Start of damage is determined as an Nod of 0.0
    Failure is determined as an Nod of 0.2

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    Ns : float | npt.NDArray[np.float64], optional
        Stability number, by default 2.9
        Recommended range for start of damage (see van Gent (2002)) 2.9 - 3.0

    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)

    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )
    Dn = core_physics.calculate_Dn50_from_M50(M50=M, rho_rock=rho_armour)
    Hs = core_physics.check_usage_stabilitynumber(Ns=Ns, Delta=Delta, Dn=Dn)[0]

    return Hs


def calculate_significant_wave_height_Hs_failure(
    M: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns: float | npt.NDArray[np.float64] = 3.5,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave height Hs for failure using van Gent (2002)

    For more details see Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 594
    And van Gent (2002) with DOI: https://doi.org/10.1680/bcsac.30428.0026

    Note that it is recommended to use a safety factor due to the small difference
    between start of damage and failure.
    Start of damage is determined as an Nod of 0.0
    Failure is determined as an Nod of 0.2

    Parameters
    ----------
    M : float | npt.NDArray[np.float64]
        Unit mass M (kg)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    Ns : float | npt.NDArray[np.float64], optional
        Stability number, by default 3.5
        Recommended range for start of damage (see van Gent (2002)) 3.5 - 3.75

    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)

    """

    Hs = calculate_significant_wave_height_Hs_start_of_damage(
        M=M, rho_water=rho_water, rho_armour=rho_armour, Ns=Ns
    )

    return Hs
