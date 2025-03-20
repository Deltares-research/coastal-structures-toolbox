# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np


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
