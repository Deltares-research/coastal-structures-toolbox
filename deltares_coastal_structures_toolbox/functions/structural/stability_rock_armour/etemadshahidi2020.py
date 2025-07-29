# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


# TODO refer to https://doi.org/10.1016/j.coastaleng.2020.103655 and https://doi.org/10.1016/j.coastaleng.2022.104142


def check_validity_range(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
) -> None:

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0",
            "Jumelet et al., 2024",
            smm10,
            0.009,
            0.057,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
            H=Hs, T=Tmm10, cot_alpha=cot_alpha
        )
        core_utility.check_variable_validity_range(
            "Irribarren number ksi_m-1,0",
            "Jumelet et al., 2024",
            ksi_mm10,
            0.42,
            1.66,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Jumelet et al., 2024",
            cot_alpha,
            6.0,
            10.0,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Jumelet et al., 2024",
            N_waves,
            250,
            20000,
        )
    return


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:

    # TODO ref eq. 17a & 17b

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
    )

    Cp = calculate_permeability_coefficient_Cp(Dn50=Dn50, Dn50_core=Dn50_core)

    S_s = (
        np.power(0.26, 6.0)
        * np.power(Ns, 6.0)
        * np.power(Cp, -6.0)
        * np.power(N_waves, 0.6)
        * np.power(ksi_mm10, 2.0)
    )
    S_pl = (
        np.power(0.22, 6.0)
        * np.power(Ns, 6.0)
        * np.power(Cp, -6.0)
        * np.power(N_waves, 0.6)
        * np.power(ksi_mm10, 7.0 / 2.0)
    )

    S = np.where(
        ksi_mm10 >= 1.8,
        S_s,
        S_pl,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:

    # TODO ref eq. 10a & 10b

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Cp = calculate_permeability_coefficient_Cp(Dn50=Dn50, Dn50_core=Dn50_core)

    Dn50_s = (
        (Hs / Delta)
        * (1.0 / 3.9)
        * (1.0 / Cp)
        * np.power(N_waves, 1.0 / 10.0)
        * np.power(S, -1.0 / 6.0)
        * np.power(ksi_mm10, 1.0 / 3.0)
    )
    Dn50_pl = (
        (Hs / Delta)
        * (1.0 / 4.5)
        * (1.0 / Cp)
        * np.power(N_waves, 1.0 / 10.0)
        * np.power(S, -1.0 / 6.0)
        * np.power(ksi_mm10, 7.0 / 12.0)
    )

    Dn50 = np.where(
        ksi_mm10 >= 1.8,
        Dn50_s,
        Dn50_pl,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    # TODO ref eq. 18a & 18b

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Cp = calculate_permeability_coefficient_Cp(Dn50=Dn50, Dn50_core=Dn50_core)

    Hs_s = np.power(
        S
        * np.power(0.26, -6.0)
        * (2 * np.pi / g)
        * np.power(Cp, 6.0)
        * np.power(Delta * Dn50, 6.0)
        * np.power(Tmm10, -2.0)
        * np.power(cot_alpha, 2.0)
        * np.power(N_waves, -0.6),
        1.0 / 5.0,
    )

    Hs_pl = np.power(
        S
        * np.power(0.22, -6.0)
        * np.power(2 * np.pi / g, 7.0 / 4.0)
        * np.power(Cp, 6.0)
        * np.power(Delta * Dn50, 6.0)
        * np.power(Tmm10, -7.0 / 2.0)
        * np.power(cot_alpha, 7.0 / 2.0)
        * np.power(N_waves, -0.6),
        4.0 / 17.0,
    )

    ksi_mm10_s = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_s, T=Tmm10, cot_alpha=cot_alpha
    )

    ksi_mm10_pl = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_pl, T=Tmm10, cot_alpha=cot_alpha
    )

    Hs = np.where(
        ((ksi_mm10_pl < 1.8) & (ksi_mm10_s < 1.8)),
        Hs_pl,
        Hs_s,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )
    return Hs


def calculate_permeability_coefficient_Cp(
    Dn50: float | npt.NDArray[np.float64],
    Dn50_core: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    Cp = np.power(1 + np.power(Dn50_core / Dn50, 3.0 / 10.0), 3.0 / 5.0)

    return Cp
