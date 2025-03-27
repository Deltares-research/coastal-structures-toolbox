# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt


# def calculate_wave_runup_height_z2p(
#     Hm0: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     beta: float | npt.NDArray[np.float64],
#     gamma_b: float | npt.NDArray[np.float64] = np.nan,
#     gamma_f: float | npt.NDArray[np.float64] = 1.0,
#     B_berm: float | npt.NDArray[np.float64] = 0.0,
#     db: float | npt.NDArray[np.float64] = 0.0,
#     cot_alpha: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
#     use_best_fit: bool = False,
# ) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

#     z2p_diml, _ = calculate_dimensionless_wave_runup_height_z2p(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         B_berm=B_berm,
#         db=db,
#         cot_alpha=cot_alpha,
#         cot_alpha_down=cot_alpha_down,
#         cot_alpha_up=cot_alpha_up,
#         gamma_b=gamma_b,
#         gamma_f=gamma_f,
#         use_best_fit=use_best_fit,
#     )

#     z2p = z2p_diml * Hm0

#     return z2p


# def calculate_dimensionless_wave_runup_height_z2p(
#     Hm0: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     beta: float | npt.NDArray[np.float64],
#     gamma_b: float | npt.NDArray[np.float64] = np.nan,
#     gamma_f: float | npt.NDArray[np.float64] = 1.0,
#     B_berm: float | npt.NDArray[np.float64] = 0.0,
#     db: float | npt.NDArray[np.float64] = 0.0,
#     cot_alpha: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
#     use_best_fit: bool = False,
# ) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

#     if use_best_fit:
#         c1 = 1.65
#         c2 = 4.0
#         c3 = 1.5
#         c4 = 1.0
#     else:
#         c1 = 1.75
#         c2 = 4.3
#         c3 = 1.5
#         c4 = 1.07

#     return


def ru_EuroTOP2(
    Hm0, slope_tana, steepness_s0_mm10, gamma_b=1.0, gamma_f=1.0, gamma_beta=1.0
):
    """
    ET2 eq 5.1, 5.2
    """

    ksi_mm10 = slope_tana / np.sqrt(steepness_s0_mm10)

    Ru2p_diml_eq51 = 1.65 * gamma_b * gamma_f * gamma_beta * ksi_mm10
    Ru2p_diml_eq52 = (
        1.0 * gamma_f * gamma_beta * (4 - 1.5 / (np.sqrt(gamma_b * ksi_mm10)))
    )

    Ru2p = np.min([Ru2p_diml_eq51, Ru2p_diml_eq52], axis=0) * Hm0

    return Ru2p


def ru_ET2_gamma_oblique_waves(beta, gamma_f):
    """
    ET2 eq 5.29 and 6.9
    """

    beta[beta < 0] = np.abs(beta[beta < 0])

    gamma_beta = beta.copy()
    gamma_beta[beta > 80] = 80

    # Dikes
    gamma_beta[gamma_f > 0.60] = 1 - 0.0022 * gamma_beta[gamma_f > 0.60]

    # Rubble mound structures
    gamma_beta[gamma_f <= 0.60] = 1 - 0.0063 * gamma_beta[gamma_f <= 0.60]

    if beta < 0:
        beta = np.abs(beta)

    if beta > 80:
        beta = 80

    if gamma_f > 0.6:
        gamma_beta = 1 - 0.0022 * beta
    else:
        gamma_beta = 1 - 0.0063 * beta
    return gamma_beta
