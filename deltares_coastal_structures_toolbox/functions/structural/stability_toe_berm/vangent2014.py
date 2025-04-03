# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    tt: float | npt.NDArray[np.float64] = np.nan,
    ht: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_armour_slope: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(cot_alpha_armour_slope)):
        core_utility.check_variable_validity_range(
            "Armour slope cot_alpha_armour_slope",
            "van Gent & van der Werf (2014)",
            cot_alpha_armour_slope,
            1.5,
            4,
        )

    if not np.any(np.isnan(cot_alpha_armour_slope)):
        core_utility.check_variable_validity_range(
            "Armour slope cot_alpha_armour_slope, closer to 1:4 more inaccuracies are found",
            "van Gent & van der Werf (2014)",
            cot_alpha_armour_slope,
            1.5,
            2,
        )

    if not np.any(np.isnan(ht)) and not np.any(np.isnan(tt)):
        h = ht + tt
        core_utility.check_variable_validity_range(
            "Relative toe height tt/h",
            "van Gent & van der Werf (2014)",
            tt / h,
            0.1,
            0.3,
        )

    if (
        not np.any(np.isnan(ht))
        and not np.any(np.isnan(tt))
        and not np.any(np.isnan(Hs))
    ):
        h = ht + tt
        core_utility.check_variable_validity_range(
            "Relative water depth h/Hs",
            "van Gent & van der Werf (2014)",
            h / Hs,
            1.2,
            4.5,
        )


def calculate_damage_Nod(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    tt: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    cot_alpha_armour_slope: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    # no arrays, only single values
    c1 = 0.032
    c2 = 0.3
    c3 = 1.0
    c4 = 3.0
    c5 = 1.0

    u_delta = calculate_velocity_u_delta(Hs=Hs, Tmm10=Tmm10, ht=ht, g=g)
    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )
    Nod = (
        (c1 * (Bt / Hs) ** c2)
        * ((tt / Hs) ** c3)
        * ((Hs / (Delta * Dn50)) ** c4)
        * ((u_delta / np.sqrt(g * Hs)) ** c5)
    )

    check_validity(Hs=Hs, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Nod


def calculate_nominal_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    tt: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    cot_alpha_armour_slope: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    u_delta = calculate_velocity_u_delta(Hs=Hs, Tmm10=Tmm10, ht=ht, g=g)
    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = (
        0.32
        * (Hs / (Delta * Nod ** (1 / 3)))
        * (Bt / Hs) ** 0.1
        * (tt / Hs) ** (1 / 3)
        * (u_delta / np.sqrt(g * Hs)) ** (1 / 3)
    )

    check_validity(Hs=Hs, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Dn50


def calculate_velocity_u_delta(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
):
    """estimate characteristic orbital velocity above the toe structure

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Wave period determined from spectrum (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe structure
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    u_delta : float | npt.NDArray[np.float64]
        Characteristic orbital velocity (m/s)
    """
    Lmm10 = (g / (2 * np.pi)) * Tmm10**2
    k = (2 * np.pi) / Lmm10
    u_delta = ((np.pi * Hs) / Tmm10) * (1 / np.sinh(k * ht))

    return u_delta
