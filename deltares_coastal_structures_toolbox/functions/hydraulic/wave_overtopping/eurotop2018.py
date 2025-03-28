# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as wave_runup_eurotop2018
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
):

    q_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        gamma_star=gamma_star,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(9.81 * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
):
    """
    EuroTOP2 formule 2018
    eq 5.10, 5.11
    """

    if use_best_fit:
        c1 = 2.7
        c2 = 0.09
        c3 = 1.5
        c4 = 0.023
    else:
        c1 = 2.5
        c2 = 0.1035
        c3 = 1.35
        c4 = 0.026

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
        )

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_eurotop2018.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
            use_best_fit=use_best_fit,
        )

        cot_alpha = wave_runup_taw2002.determine_average_slope(
            Hm0=Hm0,
            z2p=z2p_for_slope,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    L_berm = wave_runup_taw2002.calculate_berm_length(
        Hm0=Hm0, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up, B_berm=B_berm
    )

    gamma_b = wave_runup_eurotop2018.iteration_procedure_gamma_b(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha_average=cot_alpha,
        B_berm=B_berm,
        L_berm=L_berm,
        db=db,
        gamma_f=gamma_f,
        gamma_beta=gamma_beta,
    )

    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    q_diml_eq510 = (
        c4
        / np.sqrt(1.0 / cot_alpha)
        * ksi_mm10
        * gamma_b
        * np.exp(
            -1.0
            * np.power(
                c1
                * (Rc / Hm0)
                * (1.0 / (ksi_mm10 * gamma_b * gamma_f_adj * gamma_beta * gamma_v)),
                1.3,
            )
        )
    )
    q_diml_eq511 = c2 * np.exp(
        -1.0
        * np.power(
            c3 * (Rc / Hm0) * (1.0 / (gamma_f_adj * gamma_beta * gamma_star)), 1.3
        )
    )

    q_diml = np.min([q_diml_eq510, q_diml_eq511], axis=0)
    max_reached = np.min([q_diml_eq510, q_diml_eq511], axis=0) == q_diml_eq511

    return q_diml, max_reached


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
