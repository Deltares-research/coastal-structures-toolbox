# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

import numpy as np
import numpy.typing as npt


def calculate_wave_steepness_s(
    H: float | npt.NDArray[np.float64],
    T: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave steepness

    Determines the wave steepness based on the deep water wave lenght corresponding to the wave period supplied.

    Parameters
    ----------
    H : float | npt.NDArray[np.float64]
        Wave height (m)
    T : float | npt.NDArray[np.float64]
        Wave period Tm-1,0 (s)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Deep water Wave steepness s (-)
    """

    s = (2 * np.pi / g) * H / np.power(T, 2)
    return s


def calculate_Irribarren_number_ksi(
    H: float | npt.NDArray[np.float64],
    T: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    s = calculate_wave_steepness_s(H, T)
    ksi = (1.0 / cot_alpha) / np.sqrt(s)
    return ksi


def calculate_critical_Irribarren_number_ksi_mc(c_pl, c_s, P, cot_alpha):

    ksi_mc = np.power(
        (c_pl / c_s) * np.power(P, 0.31) * np.sqrt(1.0 / cot_alpha),
        1.0 / (P + 0.5),
    )
    return ksi_mc


def calculate_stability_number_Ns(
    H: float | npt.NDArray[np.float64],
    D: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    Delta = calculate_buoyant_density_Delta(rho_rock, rho_water)
    Ns = H / (Delta * D)
    return Ns


def calculate_buoyant_density_Delta(
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    Delta = (rho_rock - rho_water) / rho_water
    return Delta


def calculate_M50_from_Dn50(
    Dn50: float | npt.NDArray[np.float64], rho_rock: float = 2650
) -> float | npt.NDArray[np.float64]:

    M50 = rho_rock * np.power(Dn50, 3)
    return M50


def calculate_Dn50_from_M50(
    M50: float | npt.NDArray[np.float64], rho_rock: float = 2650
) -> float | npt.NDArray[np.float64]:

    Dn50 = np.power(M50 / rho_rock, 1 / 3)
    return Dn50


def check_usage_Dn50_or_M50(
    Dn50: float | npt.NDArray[np.float64],
    M50: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    if np.all(np.isnan(Dn50)) and np.all(np.isnan(M50)):
        raise ValueError("Either Dn50 or M50 should be provided")

    if np.all(np.isnan(Dn50)) and not np.all(np.isnan(M50)):
        if np.all(np.isnan(rho_armour)):
            warnings.warn(
                "rho_armour is not provided, the default value of 2650 kg/m3 is used."
            )
            Dn50 = calculate_Dn50_from_M50(M50)
        else:
            Dn50 = calculate_Dn50_from_M50(M50, rho_rock=rho_armour)
    return Dn50
