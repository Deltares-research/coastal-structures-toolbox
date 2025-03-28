# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as wave_runup_eurotop2018
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    use_best_fit: bool = False,
):

    q_ET2_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        gamma_star=gamma_star,
        use_best_fit=use_best_fit,
    )
    q_ET2 = q_ET2_diml * np.sqrt(9.81 * Hm0**3)

    return q_ET2


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    use_best_fit: bool = False,
):
    """
    EuroTOP2 formule 2018
    eq 5.10, 5.11
    """

    if use_best_fit:
        c1 = 0.023
        c2 = 2.7
        c3 = 0.09
        c4 = 1.5
    else:
        c1 = 0.026
        c2 = 2.5
        c3 = 0.1035
        c4 = 1.35

    # Check if this is the same as runup TAW2002, if so then delete here and use TAW2002
    cot_alpha_average, _, L_berm = determine_average_slope(
        Hm0, B_berm, cot_alpha_down, cot_alpha_up, Tmm10, beta, gamma_f, db
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha_average
    )

    # gamma_b = iteration_procedure_gamma_b(
    #     Hm0=Hm0,
    #     Tmm10=Tmm10,
    #     beta=beta,
    #     cot_alpha_average=cot_alpha_average,
    #     B_berm=B_berm,
    #     L_berm=L_berm,
    #     db=db,
    #     gamma_f=gamma_f,
    # )

    # ru_gamma_beta = wave_runup_eurotop2018.ru_ET2_gamma_oblique_waves(beta, gamma_f)

    # Ru2p_i1 = wave_runup_eurotop2018.ru_EuroTOP2(
    #     Hm0,
    #     cot_alpha_average,
    #     steepness_s0_mm10,
    #     gamma_b=1.0,
    #     gamma_f=gamma_f,
    #     gamma_beta=ru_gamma_beta,
    # )

    # ru_gamma_b = calculate_influence_berm_gamma_b(B_berm, L_berm, db, Hm0, Ru2p_i1)

    # # TODO Check of dit niet ru_EuroTOP2() ipv ru_TAW() moet zijn
    # Ru2p_i2 = ru_TAW(
    #     Hm0,
    #     cot_alpha_average,
    #     steepness_s0_mm10,
    #     gamma_b=ru_gamma_b,
    #     gamma_f=gamma_f,
    #     gamma_beta=ru_gamma_beta,
    # )

    gamma_b = calculate_influence_berm_gamma_b(B_berm, L_berm, db, Hm0, Ru2p=0.0)

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta, gamma_f)

    q_diml_eq510 = (
        c1
        / np.sqrt(cot_alpha_average)
        * ksi_mm10
        * gamma_b
        * np.exp(
            -1
            * np.power(
                c2
                * (Rc / Hm0)
                * (1 / (ksi_mm10 * gamma_b * gamma_f * gamma_beta * gamma_v)),
                1.3,
            )
        )
    )
    q_diml_eq511 = c3 * np.exp(
        -1 * np.power(c4 * (Rc / Hm0) * (1 / (gamma_f * gamma_beta * gamma_star)), 1.3)
    )

    q_ET2_diml = np.min([q_diml_eq510, q_diml_eq511], axis=0)
    max_reached = np.min([q_diml_eq510, q_diml_eq511], axis=0) == q_diml_eq511

    return q_ET2_diml, max_reached


def calculate_influence_berm_gamma_b(Bberm, Lberm, db, Hm0, Ru2p):
    """
    ET2 eq 5.40, 5.41, 5.42
    """

    # TODO check if this is the same as TAW2002, if so then delete here and use TAW2002
    x = Hm0.copy()
    x = x * 2
    rdb = 0.5 - 0.5 * np.cos(np.pi * db / x)
    rdb_neg = 0.5 - 0.5 * np.cos(np.pi * db / Ru2p)
    rB = Bberm / Lberm

    gamma_b = 1 - rB * (1 - rdb)
    gamma_b[db < 0] = 1 - rB[db < 0] * (1 - rdb_neg[db < 0])
    gamma_b[db > 2 * Hm0] = 1.0
    gamma_b[db < -Ru2p] = 1.0
    gamma_b[Bberm == 0] = 1.0

    gamma_b[gamma_b < 0.6] = 0.6
    gamma_b[gamma_b > 1.0] = 1.0

    outside_validity = False

    if db < 0:
        if np.abs(db) < Ru2p:
            x = Ru2p
        else:
            outside_validity = True
    else:
        if db < 2 * Hm0:
            x = 2 * Hm0
        else:
            outside_validity = True

    if outside_validity:
        rdb = 1
    else:
        rdb = 0.5 - 0.5 * np.cos(np.pi * db / x)

    rB = Bberm / Lberm
    gamma_b = 1 - rB * (1 - rdb)

    if gamma_b < 0.6:
        gamma_b = 0.6
    elif gamma_b > 1.0:
        gamma_b = 1.0
    return gamma_b


def determine_average_slope(
    Hm0, Bberm, cotad, cotau, steepness_s0_mm10, beta, gamma_f, db
):
    # TODO check if this is the same as TAW2002, if so then delete here and use TAW2002
    Lslope_iter1 = 1.5 * Hm0 * cotad + Bberm + 1.5 * Hm0 * cotau
    tana_iter1 = 3 * Hm0 / (Lslope_iter1 - Bberm)

    Lberm = 1.0 * Hm0 * cotad + Bberm + 1.0 * Hm0 * cotau

    ru_gamma_beta = wave_runup_eurotop2018.ru_ET2_gamma_oblique_waves(beta, gamma_f)
    # Invloed berm niet meenemen in run-up
    # ru_gamma_b = ot_ET2_gamma_berm(Bberm, Lberm, db, Hm0, Ru2p=2 * Hm0)
    ru_gamma_b = 1.0

    z2p = wave_runup_taw2002.calculate_wave_runup_height_z2p(
        Hm0,
        tana_iter1,
        steepness_s0_mm10,
        gamma_b=ru_gamma_b,
        gamma_f=gamma_f,
        gamma_beta=ru_gamma_beta,
    )
    Lslope_iter2 = 1.5 * Hm0 * cotad + Bberm + z2p * cotau
    tana_iter2 = (1.5 * Hm0 + z2p) / (Lslope_iter2 - Bberm)

    slope_tana = tana_iter2
    return slope_tana, z2p, Lberm


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_crit: float = 0.6,
    c_gamma_beta_smooth: float = 0.0033,
    c_gamma_beta_rough: float = 0.0063,
    max_angle: float = 80.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor gamma_beta is determined using the EurOtop (2018) eq. 5.29 for smooth slopes and eq. 6.9
    for rough slopes.

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_crit : float, optional
        Critical value for the influence factor dividing smooth (higher) and rough (lower) slopes, by default 0.6
    c_gamma_beta_smooth : float, optional
        Coefficient for wave runup on smooth slopes, by default 0.0022
    c_gamma_beta_rough : float, optional
        Coefficient for wave runup on rough slopes, by default 0.0063
    max_angle : float, optional
        Maximum angle of wave incidence, by default 80.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    gamma_beta = wave_runup_eurotop2018.calculate_influence_oblique_waves_gamma_beta(
        beta=beta,
        gamma_f=gamma_f,
        gamma_f_crit=gamma_f_crit,
        c_gamma_beta_smooth=c_gamma_beta_smooth,
        c_gamma_beta_rough=c_gamma_beta_rough,
        max_angle=max_angle,
    )

    return gamma_beta
