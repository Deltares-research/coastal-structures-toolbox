# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    ht: float | npt.NDArray[np.float64] = np.nan,
    h: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Nod: float | npt.NDArray[np.float64] = np.nan,
    Delta: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Relative toe height (h)",
            "van der Meer (1998)",
            Hs,
            0.01,
            20,
        )

    if not np.any(np.isnan(ht)) and not np.any(np.isnan(h)):
        core_utility.check_variable_validity_range(
            "Relative toe height (Dn50)",
            "van der Meer (1998)",
            ht / h,
            0.4,
            0.9,
        )

    if not np.any(np.isnan(ht)) and not np.any(np.isnan(Dn50)):
        core_utility.check_variable_validity_range(
            "Relative toe height (Dn50)",
            "van der Meer (1998)",
            ht / Dn50,
            3,
            25,
        )

    if (
        not np.any(np.isnan(Nod))
        and not np.any(np.isnan(Dn50))
        and not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Delta))
    ):
        core_utility.check_variable_validity_range(
            "Nod vs Ns",
            "van der Meer (1998)",
            (Nod**-0.15) * (Hs / Delta * Dn50),
            2,
            np.inf,
        )


def calculate_damage_Nod(
    Hs: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate damage number Nod for toe structures using van der Meer 1998

    For more information, please refer to:
    WL|Delft Hydraulics (former Deltares), report number H2458/H3051, June, 1997
    or
    Meer, J. Van der, 1998. “Geometrical design of coastal structures.” Infram publication Nr. 2.

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
    Ns = Hs / (Delta * Dn50)
    Nod = (Ns / (2 + 6.2 * (ht / h) ** 2.7)) ** (1 / 0.15)

    check_validity(Hs=Hs, ht=ht, h=h, Dn50=Dn50, Nod=Nod, Delta=Delta)

    return Nod


def calculate_nominal_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate nominal diameter Dn50 for toe structures using van der Meer 1998

    For more information, please refer to:
    WL|Delft Hydraulics (former Deltares), report number H2458/H3051, June, 1997
    or
    Meer, J. Van der, 1998. “Geometrical design of coastal structures.” Infram publication Nr. 2.

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

    Dn50 = Hs / ((2 + 6.2 * (ht / h) ** 2.7) * Nod**0.15 * Delta)

    check_validity(Hs=Hs, ht=ht, h=h, Dn50=Dn50, Nod=Nod, Delta=Delta)

    return Dn50


def calculate_wave_height_Hs(
    Dn50: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate wave height Hs for toe structures using van der Meer 1998

    For more information, please refer to:
    WL|Delft Hydraulics (former Deltares), report number H2458/H3051, June, 1997
    or
    Meer, J. Van der, 1998. “Geometrical design of coastal structures.” Infram publication Nr. 2.

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

    Hs = ((2 + 6.2 * (ht / h) ** 2.7) * Delta * Dn50) / Nod**-0.15

    check_validity(Hs=Hs, ht=ht, h=h, Dn50=Dn50, Nod=Nod, Delta=Delta)

    return Hs


def calculate_depth_above_toe_ht(
    Hs: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """calculate depth above toe ht for toe structures using van der Meer 1998

    For more information, please refer to:
    WL|Delft Hydraulics (former Deltares), report number H2458/H3051, June, 1997
    or
    Meer, J. Van der, 1998. “Geometrical design of coastal structures.” Infram publication Nr. 2.

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

    ht = (((Hs * Nod**-0.15) / (Delta * Dn50)) - 2) / 6.2
    if ht > 0:
        ht = ht ** (1 / 2.7) * h
    else:
        ht = np.nan

    check_validity(Hs=Hs, ht=ht, h=h, Dn50=Dn50, Nod=Nod, Delta=Delta)

    return ht
