# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    h: float | npt.NDArray[np.float64] = np.nan,
    ht: float | npt.NDArray[np.float64] = np.nan,
    Bt: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    m: float | npt.NDArray[np.float64] = np.nan,
    Nod: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_armour_slope: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
):
    """Check the parameter values vs the validity range of the Etemad-Shahidi et al. (2021) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified by
    Etemad-Shahidi et al. (2021) in Table 3. When parameters are nan (by default), they are not checked.

    For more details see Etemad-Shahidi et al. (2021), available here: https://doi.org/10.1016/j.coastaleng.2020.103835

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    h : float | npt.NDArray[np.float64], optional
        Water depth (m), by default np.nan
    ht : float | npt.NDArray[np.float64], optional
        Water depth above the toe (m), by default np.nan
    Bt : float | npt.NDArray[np.float64], optional
        Width of toe structure (m), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    m : float | npt.NDArray[np.float64], optional
        Tangent of the foreshore slope, by default np.nan
    Nod : float | npt.NDArray[np.float64], optional
        Damage parameter (-), by default np.nan
    cot_alpha_armour_slope : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side armour slope of the structure (-), by default np.nan
    rho_armour : float | npt.NDArray[np.float64], optional
        Armour rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    """

    if not np.any(np.isnan(cot_alpha_armour_slope)):
        core_utility.check_variable_validity_range(
            "Armour slope cot_alpha_armour_slope",
            "Etemad-Shahidi et al. (2021)",
            cot_alpha_armour_slope,
            1.5,
            2.0,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hs, Tmm10=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0",
            "Etemad-Shahidi et al. (2021)",
            smm10,
            0.009,
            0.061,
        )

    if not np.any(np.isnan(h)) and not np.any(np.isnan(ht)):
        core_utility.check_variable_validity_range(
            "Relative water depth above the toe ht/h",
            "Etemad-Shahidi et al. (2021)",
            ht / h,
            0.39,
            0.88,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(ht)):
        core_utility.check_variable_validity_range(
            "Relative water depth above the toe ht/Hs",
            "Etemad-Shahidi et al. (2021)",
            ht / Hs,
            0.48,
            2.58,
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(ht)):
        core_utility.check_variable_validity_range(
            "Relative water depth above the toe ht/Dn50",
            "Etemad-Shahidi et al. (2021)",
            ht / Dn50,
            1.97,
            23.40,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(ht)):
        core_utility.check_variable_validity_range(
            "Relative toe width Bt/Hs",
            "Etemad-Shahidi et al. (2021)",
            Bt / Hs,
            0.17,
            1.92,
        )

    if not np.any(np.isnan(h)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Relative water depth h/Hs",
            "Etemad-Shahidi et al. (2021)",
            h / Hs,
            1.22,
            3.0,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Dn50)):
        Ns = core_physics.calculate_stability_number_Ns(
            H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Etemad-Shahidi et al. (2021)",
            Ns,
            1.58,
            10.0,
        )

    if not np.any(np.isnan(rho_armour)):
        Delta = core_physics.calculate_buoyant_density_Delta(
            rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Buoyant density Delta",
            "Etemad-Shahidi et al. (2021)",
            Delta,
            1.65,
            1.75,
        )

    if not np.any(np.isnan(m)):
        core_utility.check_variable_validity_range(
            "Foreshore slope m",
            "Etemad-Shahidi et al. (2021)",
            m,
            0.02,
            0.10,
        )

    if not np.any(np.isnan(Nod)):
        core_utility.check_variable_validity_range(
            "Damage parameter Nod",
            "Etemad-Shahidi et al. (2021)",
            Nod,
            0.5,
            3.79,
        )

    return


def calculate_damage_Nod(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    m: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    g: float = 9.81,
    c1: float = 1.2,
    c2: float = 11.2,
    c3: float = 7.0 / 4.0,
    c4: float = 1.0 / 6.0,
    c5: float = 2.0 / 5.0,
    c6: float = -1.0 / 10.0,
    c7: float = 3.7,
) -> float | npt.NDArray[np.float64]:

    smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10, g=g)

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs,
        D=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    Nod = np.power(
        (Ns - c1)
        * (1.0 / c2)
        * np.power(ht / h, -c3)
        * np.power(smm10, -c4)
        * np.power(Bt / Hs, -c6)
        * (1.0 / (1.0 - c7 * m)),
        1.0 / c5,
    )

    check_validity(Hs=Hs, ht=ht)

    return Nod


def calculate_nominal_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    m: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    g: float = 9.81,
    c1: float = 1.2,
    c2: float = 11.2,
    c3: float = 7.0 / 4.0,
    c4: float = 1.0 / 6.0,
    c5: float = 2.0 / 5.0,
    c6: float = -1.0 / 10.0,
    c7: float = 3.7,
) -> float | npt.NDArray[np.float64]:

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10, g=g)

    Dn50 = (Hs / Delta) * np.power(
        c1
        + c2
        * np.power(ht / h, c3)
        * np.power(smm10, c4)
        * np.power(Nod, c5)
        * np.power(Bt / Hs, c6)
        * (1.0 - c7 * m),
        -1.0,
    )

    check_validity(Hs=Hs, ht=ht)

    return Dn50


def calculate_significant_wave_height_Hs(
    Tmm10: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
    m: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    g: float = 9.81,
    c1: float = 1.2,
    c2: float = 11.2,
    c3: float = 7.0 / 4.0,
    c4: float = 1.0 / 6.0,
    c5: float = 2.0 / 5.0,
    c6: float = -1.0 / 10.0,
    c7: float = 3.7,
    smm10_init: float = 0.03,
    max_iter: int = 1000,
):

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    n_iter = 0
    Hs_init = smm10_init * np.power(Tmm10, 2) * g / (2 * np.pi)

    Hs = Hs_init
    smm10 = smm10_init

    Hs_diff = np.inf
    Hs_prev = Hs_init

    while Hs_diff > 1e-3 and n_iter < max_iter:

        Hs = (
            Delta
            * Dn50
            * (
                c1
                + c2
                * np.power(ht / h, c3)
                * np.power(smm10, c4)
                * np.power(Nod, c5)
                * np.power(Bt / Hs, c6)
                * (1.0 - c7 * m)
            )
        )

        Hs_diff = np.abs(Hs - Hs_prev)
        Hs_prev = Hs

        smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10, g=g)

    return Hs
