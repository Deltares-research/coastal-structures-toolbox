# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


def calculate_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    use_best_fit: bool = False,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:
    """
    TAW eq 3a & 3b (EurOtop I eq 5.3)
    """

    z2p_diml, _ = calculate_dimensionless_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        B_berm=B_berm,
        db=db,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        use_best_fit=use_best_fit,
    )

    z2p = z2p_diml * Hm0

    return z2p


def calculate_dimensionless_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """
    TAW eq 3a & 3b (EurOtop I eq 5.3)
    """

    if use_best_fit:
        c1 = 1.65
        c2 = 4.0
        c3 = 1.5
    else:
        c1 = 1.75
        c2 = 4.3
        c3 = 1.6

    if check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        cot_alpha, _, _ = determine_average_slope(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    gamma_f_adj = calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    z2p_diml_eq3a = c1 * gamma_b * gamma_f_adj * gamma_beta * ksi_mm10
    z2p_diml_eq3b = (
        1.0 * gamma_b * gamma_f_adj * gamma_beta * (c2 - c3 / np.sqrt(ksi_mm10))
    )  # TODO check difference TAW/ET1!

    z2p_diml = np.min([z2p_diml_eq3a, z2p_diml_eq3b], axis=0)
    max_reached = np.min([z2p_diml_eq3a, z2p_diml_eq3b], axis=0) == z2p_diml_eq3b

    return z2p_diml, max_reached


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    TAW eq 8
    """

    beta_calc = np.where(beta < 0, np.abs(beta), beta)
    beta_calc = np.where(beta_calc > 80, 80, beta_calc)

    gamma_beta = 1 - 0.0022 * beta_calc

    return gamma_beta


def calculate_adjusted_influence_roughness_gamma_f(
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    Breakwat User Manual eq 3.62
    """

    gamma_f_adj = np.where(
        (gamma_b * ksi_mm10 >= 1.8) & (gamma_b * ksi_mm10 <= 10.0),
        ((1 - gamma_f) / (10.0 - 1.8)) * ksi_mm10
        + gamma_f
        + (gamma_f - 1.0) * (1.8 / (10.0 - 1.8)),
        gamma_f,
    )

    return gamma_f_adj


# def determine_average_slope(
#     Hm0: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     beta: float | npt.NDArray[np.float64],
#     cot_alpha_down: float | npt.NDArray[np.float64],
#     cot_alpha_up: float | npt.NDArray[np.float64],
#     B_berm: float | npt.NDArray[np.float64],
#     dh: float | npt.NDArray[np.float64],
#     gamma_f: float | npt.NDArray[np.float64],
# ) -> float | npt.NDArray[np.float64]:

#     L_slope_iter1 = 1.5 * Hm0 * cot_alpha_down + B_berm + 1.5 * Hm0 * cot_alpha_up
#     tan_alpha_average_iter1 = 3 * Hm0 / (L_slope_iter1 - B_berm)

#     # TODO L_berm is unrelated, move to separate function
#     L_berm = 1.0 * Hm0 * cot_alpha_down + B_berm + 1.0 * Hm0 * cot_alpha_up

#     # do not account for berm influence in wave runup
#     # TODO double check this
#     gamma_b = 1.0

#     z2p = calculate_wave_runup_height_z2p(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         cot_alpha=1.0 / tan_alpha_average_iter1,
#         gamma_b=gamma_b,
#         gamma_f=gamma_f,
#     )
#     L_slope_iter2 = (
#         np.maximum((1.5 * Hm0 - dh) * cot_alpha_down, 0.0)
#         + B_berm
#         + np.maximum((z2p - dh) * cot_alpha_down, 0.0)
#     )
#     tan_alpha_average_iter2 = (1.5 * Hm0 + z2p) / (L_slope_iter2 - B_berm)

#     # If both slopes are equal, do not modify that value
#     cot_alpha_average = np.where(
#         cot_alpha_down == cot_alpha_down, cot_alpha_down, 1.0 / tan_alpha_average_iter2
#     )

#     return cot_alpha_average, z2p, L_berm


# def determine_average_slope_EurOtopI(
def determine_average_slope(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    # This is the procedure as described in section 5.3.4 of the EurOtop (2007) manual

    L_slope_iter1 = 1.5 * Hm0 * cot_alpha_down + B_berm + 1.5 * Hm0 * cot_alpha_up
    tan_alpha_average_iter1 = 3 * Hm0 / (L_slope_iter1 - B_berm)

    # TODO L_berm is unrelated, move to separate function
    L_berm = 1.0 * Hm0 * cot_alpha_down + B_berm + 1.0 * Hm0 * cot_alpha_up

    # do not account for berm influence in wave runup
    # TODO double check this
    gamma_b = 1.0

    z2p = calculate_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=1.0 / tan_alpha_average_iter1,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
    )
    L_slope_iter2 = 1.5 * Hm0 * cot_alpha_down + B_berm + z2p * cot_alpha_up
    tan_alpha_average_iter2 = (1.5 * Hm0 + z2p) / (L_slope_iter2 - B_berm)

    cot_alpha_average = 1.0 / tan_alpha_average_iter2
    return cot_alpha_average, z2p, L_berm


def check_composite_slope(
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:

    if np.all(np.isnan(cot_alpha)) and np.all(np.isnan(cot_alpha_down)):
        raise ValueError(
            "Either a single (cot_alpha) or composite (cot_alpha_down & cot_alpha_up) slope should be provided"
        )

    if (
        np.all(np.isnan(cot_alpha))
        and not np.all(np.isnan(cot_alpha_down))
        and not np.all(np.isnan(cot_alpha_up))
    ):
        is_composite_slope = True
    else:
        is_composite_slope = False
    return is_composite_slope
