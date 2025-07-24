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
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    tt: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    cot_alpha_armour_slope: float | npt.NDArray[np.float64],
    g: float = 9.81,
    c1: float = 0.032,
    c2: float = 0.3,
    c3: float = 1.0,
    c4: float = 3.0,
    c5: float = 1.0,
) -> float | npt.NDArray[np.float64]:
    """calculate damage number Nod for toe structure using van Gent and van der Werf (2014)

    For more information, please refer to:
    Van Gent, M.R.A. and I.M. van der Werf. 2014. Rock toe stability of rubble mound breakwaters,
        Coastal Engineering, Vol. 83, pp. 166-176, Elsevier.
    http://dx.doi.org/10.1016/j.coastaleng.2013.10.012

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Mean energy wave period or spectral wave period (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    tt : float | npt.NDArray[np.float64]
        Height of toe structure (m)
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
    cot_alpha_armour_slope : float | npt.NDArray[np.float64]
        Slope above structure (not used in formula, only in checks) (-)
    g : float, optional
        Gravitational acceleration, by default 9.81
    c1 : float, optional
        Coefficient in the toe stability formula, by default 0.032
    c2 : float, optional
        Coefficient in the toe stability formula, by default 0.3
    c3 : float, optional
        Coefficient in the toe stability formula, by default 1.0
    c4 : float, optional
        Coefficient in the toe stability formula, by default 3.0
    c5 : float, optional
        Coefficient in the toe stability formula, by default 1.0

    Returns
    -------
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    """

    u_delta = calculate_velocity_u_delta(Hs=Hm0, Tmm10=Tmm10, ht=ht, g=g)
    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )
    Nod = (
        (c1 * (Bt / Hm0) ** c2)
        * ((tt / Hm0) ** c3)
        * ((Hm0 / (Delta * Dn50)) ** c4)
        * ((u_delta / np.sqrt(g * Hm0)) ** c5)
    )

    check_validity(Hs=Hm0, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Nod


def calculate_nominal_diameter_Dn50(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    tt: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    cot_alpha_armour_slope: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """calculate nominal diameter Dn50 for toe structure using van Gent and van der Werf (2014)

    For more information, please refer to:
    Van Gent, M.R.A. and I.M. van der Werf. 2014. Rock toe stability of rubble mound breakwaters,
        Coastal Engineering, Vol. 83, pp. 166-176, Elsevier.
    http://dx.doi.org/10.1016/j.coastaleng.2013.10.012

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Mean energy wave period or spectral wave period (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    tt : float | npt.NDArray[np.float64]
        Height of toe structure (m)
    Nod : float | npt.NDArray[np.float64]
        Damage parameter (-)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
    cot_alpha_armour_slope : float | npt.NDArray[np.float64]
        Slope above structure (not used in formula, only in checks) (-)
    g : float, optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    """

    u_delta = calculate_velocity_u_delta(Hs=Hm0, Tmm10=Tmm10, ht=ht, g=g)
    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = (
        0.32
        * (Hm0 / (Delta * Nod ** (1 / 3)))
        * (Bt / Hm0) ** 0.1
        * (tt / Hm0) ** (1 / 3)
        * (u_delta / np.sqrt(g * Hm0)) ** (1 / 3)
    )

    check_validity(Hs=Hm0, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Dn50


def calculate_velocity_u_delta(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    g: float = 9.81,
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
    g : float, optional
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
