# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    ht: float | npt.NDArray[np.float64] = np.nan,
    h: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(ht)) and not np.any(np.isnan(h)):
        core_utility.check_variable_validity_range(
            "Relative toe height (h)",
            "Gerding (1993)",
            ht / h,
            0.4,
            0.9,
        )

    if not np.any(np.isnan(ht)) and not np.any(np.isnan(Dn50)):
        core_utility.check_variable_validity_range(
            "Relative toe height (Dn50)",
            "Gerding (1993)",
            ht / Dn50,
            3,
            25,
        )


def calculate_damage_Nod(
    Hs: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate damage number Nod for toe structures using Gerding 1993

    For more information, please refer to:
    Gerding, E. 1993. Toe structure stability of rubble mound breakwaters, M.Sc. thesis, Delft University of
        Technology, Delft and Delft Hydraulics Report H1874, Delft.
    https://resolver.tudelft.nl/uuid:51af1788-de9f-4ef3-8115-ffefb2e26f76

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    h : float | npt.NDArray[np.float64]
        Water depth in front of the toe (m)
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)

    Returns
    -------
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Nod = ((Hs / (Delta * Dn50)) / (1.6 + 0.24 * (ht / Dn50))) ** (1 / 0.15)

    check_validity(ht=ht, h=h, Dn50=Dn50)

    return Nod


def calculate_nominal_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate nominal diameter Dn50 for toe structures using Gerding 1993

    For more information, please refer to:
    Gerding, E. 1993. Toe structure stability of rubble mound breakwaters, M.Sc. thesis, Delft University of
        Technology, Delft and Delft Hydraulics Report H1874, Delft.
    https://resolver.tudelft.nl/uuid:51af1788-de9f-4ef3-8115-ffefb2e26f76

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    h : float | npt.NDArray[np.float64]
        Water depth in front of the toe (m)
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = ((Hs / Delta) - (Nod**0.15) * 0.24 * ht) / (Nod**0.15 * 1.6)

    check_validity(ht=ht, h=h, Dn50=Dn50)

    return Dn50


def calculate_wave_height_Hs(
    Dn50: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate wave height Hs for toe structures using Gerding 1993

    For more information, please refer to:
    Gerding, E. 1993. Toe structure stability of rubble mound breakwaters, M.Sc. thesis, Delft University of
        Technology, Delft and Delft Hydraulics Report H1874, Delft.
    https://resolver.tudelft.nl/uuid:51af1788-de9f-4ef3-8115-ffefb2e26f76

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    h : float | npt.NDArray[np.float64]
        Water depth in front of the toe (m)
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Hs = (Nod**0.15) * Delta * Dn50 * (1.6 + 0.24 * (ht / Dn50))

    check_validity(ht=ht, h=h, Dn50=Dn50)

    return Hs


def calculate_depth_above_toe_ht(
    Hs: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate depth above toe ht for toe structures using Gerding 1993

    For more information, please refer to:
    Gerding, E. 1993. Toe structure stability of rubble mound breakwaters, M.Sc. thesis, Delft University of
        Technology, Delft and Delft Hydraulics Report H1874, Delft.
    https://resolver.tudelft.nl/uuid:51af1788-de9f-4ef3-8115-ffefb2e26f76

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    h : float | npt.NDArray[np.float64]
        Water depth in front of the toe (m)
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    ht = (Dn50 * ((Hs / (Delta * Dn50 * (Nod**0.15))) - 1.6)) / 0.24

    check_validity(ht=ht, h=h, Dn50=Dn50)

    return ht
