# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.vangent2002 as vangent2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


def check_validity_range(
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Rc_rear: float | npt.NDArray[np.float64] = np.nan,
    cot_phi: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    Dn50_rear: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float | npt.NDArray[np.float64] = np.nan,
    B_c: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    z1p: float | npt.NDArray[np.float64] = np.nan,
    S: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
) -> None:

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hs, Tmm10)
        core_utility.check_variable_validity_range(
            "sm-1,0", "Van Gent & Pozueta (2004)", smm10, 0.019, 0.036
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "N_waves", "Van Gent & Pozueta (2004)", N_waves, 0, 4000
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc/Hs", "Van Gent & Pozueta (2004)", Rc / Hs, 0.3, 2.0
        )

    if not np.any(np.isnan(Rc_rear)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc_rear/Hs", "Van Gent & Pozueta (2004)", Rc_rear / Hs, 0.3, 6.0
        )

    if not np.any(np.isnan(B_c)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "B_c/Hs", "Van Gent & Pozueta (2004)", B_c / Hs, 1.3, 1.6
        )

    if (
        not np.any(np.isnan(Rc))
        and not np.any(np.isnan(Hs))
        and not np.any(np.isnan(z1p))
        and not np.any(np.isnan(gamma_f))
    ):
        core_utility.check_variable_validity_range(
            "(z1% - Rc)/(gamma_f*Hs)",
            "Van Gent & Pozueta (2004)",
            (z1p - Rc) / (gamma_f * Hs),
            0.0,
            1.4,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Dn50_rear))
        and not np.any(np.isnan(rho_rock))
        and not np.any(np.isnan(rho_water))
    ):
        Ns = core_physics.calculate_stability_number_Ns(
            H=Hs, D=Dn50_rear, rho_rock=rho_rock, rho_water=rho_water
        )

        core_utility.check_variable_validity_range(
            "Hs/(Delta*Dn50_rear)",
            "Van Gent & Pozueta (2004)",
            Ns,
            5.5,
            8.5,
        )

    if not np.any(np.isnan(cot_phi)):
        core_utility.check_variable_validity_range(
            "cot phi", "Van Gent & Pozueta (2004)", cot_phi, 2.0, 4.0
        )

    if not np.any(np.isnan(S)):
        core_utility.check_variable_validity_range(
            "S", "Van Gent & Pozueta (2004)", S, 2.0, 50.0
        )

    return


def calculate_damage_number_S(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
    cs: float = np.nan,
) -> float | npt.NDArray[np.float64]:

    z1p = vangent2001.calculate_wave_runup_height_z1p(
        Hs=Hs, Tmm10=Tmm10, gamma=gamma_f, cot_alpha=cot_alpha
    )

    u1p = vangent2002.calculate_maximum_wave_overtopping_velocity_uXp(
        Hs=Hs, zXp=z1p, Rc=Rc, Bc=Bc, gamma_f=gamma_f, gamma_f_Crest=gamma_f_Crest
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_rock)

    S = (
        cs
        * np.power((u1p * Tmm10 / (np.sqrt(Delta) * Dn50)), 6.0)
        * np.power(cot_phi, -2.5)
        * (1 + 10 * np.exp(-Rc_rear / Hs))
        * np.sqrt(N_waves)
    )

    check_validity_range(
        Rc=Rc,
        Rc_rear=Rc_rear,
        cot_phi=cot_phi,
        gamma_f=gamma_f,
        Dn50_rear=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        B_c=Bc,
        Hs=Hs,
        Tmm10=Tmm10,
        z1p=z1p,
        S=S,
        N_waves=N_waves,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:

    z1p = vangent2001.calculate_wave_runup_height_z1p(
        Hs=Hs, Tmm10=Tmm10, gamma=gamma_f, cot_alpha=cot_alpha
    )

    u1p = vangent2002.calculate_maximum_wave_overtopping_velocity_uXp(
        Hs=Hs, zXp=z1p, Rc=Rc, Bc=Bc, gamma_f=gamma_f, gamma_f_Crest=gamma_f_Crest
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = (
        0.008
        * np.power(S / np.sqrt(N_waves), -(1.0 / 6.0))
        * (u1p * Tmm10 / np.sqrt(Delta))
        * np.power(cot_phi, -(2.5 / 6.0))
        * np.power(1 + 10 * np.exp(-Rc_rear / Hs), (1.0 / 6.0))
    )

    check_validity_range(
        Rc=Rc,
        Rc_rear=Rc_rear,
        cot_phi=cot_phi,
        gamma_f=gamma_f,
        Dn50_rear=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        B_c=Bc,
        Hs=Hs,
        Tmm10=Tmm10,
        z1p=z1p,
        S=S,
        N_waves=N_waves,
    )

    return Dn50
