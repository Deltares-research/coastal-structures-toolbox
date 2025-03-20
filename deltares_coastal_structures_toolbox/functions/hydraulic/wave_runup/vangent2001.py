# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


def calculate_wave_runup_height_z1p(
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    c0: float = 0,
    c1: float = 0,
) -> float | npt.NDArray[np.float64]:

    if np.all(np.isnan(Hm0)) and np.all(np.isnan(Hs)):
        raise ValueError("Either Hm0 or Hs should be provided")

    if np.all(np.isnan(Hm0)) and not np.any(np.isnan(Hs)):
        # Use coefficients for Hs instead of Hm0
        c0 = 1.45
        c1 = 5.1

    # TODO implement c0/c1 values for Hm0, if they exist

    z1p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z1p


def calculate_wave_runup_height_z2p(
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    c0: float = 1.45,
    c1: float = 3.8,
) -> float:

    if np.isnan(Hm0) and np.isnan(Hs):
        raise ValueError("Either Hm0 or Hs should be provided")

    if np.isnan(Hm0) and not np.isnan(Hs):
        # Use coefficients for Hs instead of Hm0
        c0 = 1.35
        c1 = 4.7

    z2p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z2p


def calculate_wave_runup_height_z10p(
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    c0: float = 0,
    c1: float = 0,
) -> float | npt.NDArray[np.float64]:

    if np.isnan(Hm0) and np.isnan(Hs):
        raise ValueError("Either Hm0 or Hs should be provided")

    if np.isnan(Hm0) and not np.isnan(Hs):
        # Use coefficients for Hs instead of Hm0
        c0 = 1.1
        c1 = 4.0

    # TODO implement c0/c1 values for Hm0, if they exist

    z10p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z10p


def calculate_wave_runup_height_zXp(
    H: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    c0: float,
    c1: float,
) -> float | npt.NDArray[np.float64]:

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(H, Tmm10, cot_alpha)

    p = 0.5 * c1 / c0

    zXp_a = c0 * ksi_mm10 * gamma * H

    c2 = 0.25 * np.power(c1, 2) / c0
    zXp_b = (c1 - c2 / ksi_mm10) * gamma * H

    zXp = np.where(ksi_mm10 < p, zXp_a, zXp_b)

    return zXp
