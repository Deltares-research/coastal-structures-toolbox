# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_wave_toolbox.cores.core_dispersion as dispersion


def calculate_forces(
    HD: float | npt.NDArray[np.float64],  # design wave height (Hmax)
    Tmax: float | npt.NDArray[np.float64],  # max wave period
    beta: float | npt.NDArray[np.float64],  # wave angle wrt normal
    h_s: float | npt.NDArray[np.float64],  # water depth at site
    d: float | npt.NDArray[np.float64],  # water depth above toe
    rho_water: float | npt.NDArray[np.float64],
    hacc: (
        float | npt.NDArray[np.float64]
    ),  # distance from sea water level to base of caisson
    Rc: float | npt.NDArray[np.float64],  # Crest freeboard above SWL
    B_up: float | npt.NDArray[np.float64],  # Width of upright section of caisson
    g: float | npt.NDArray[np.float64] = 9.81,
    return_dict: bool = False,
) -> float | npt.NDArray[np.float64] | dict:

    # beta, if not 0 should be rotated by 15 degrees towards the normal of the breakwater according to Goda (2000)
    beta = np.abs(beta)
    if beta >= 15:
        beta_use = beta - 15
    elif beta < 15:
        beta_use = 0
    cos_beta = np.cos(np.deg2rad(beta_use))

    etastar = calculate_etastar(HD=HD, cos_beta=cos_beta)

    L = calculate_local_wavelength(T=Tmax, h=h_s, g=g)  # local wave length

    # alpha factors
    alpha_1 = (
        0.6 + 0.5 * (((4 * np.pi * h_s) / L) / (np.sinh(((4 * np.pi * h_s) / L)))) ** 2
    )

    alpha_21 = ((h_s - d) / (3 * h_s)) * ((HD / d) ** 2)
    alpha_22 = (2 * d) / HD
    alpha_2 = np.minimum(alpha_21, alpha_22)

    alpha_3 = 1 - (hacc / h_s) * (1 - (1 / (np.cosh(2 * np.pi * h_s / L))))

    # pressures
    p1 = 0.5 * (1 + cos_beta) * (alpha_1 + alpha_2 * cos_beta**2) * rho_water * g * HD
    p2 = p1 / (np.cosh(2 * np.pi * h_s / L))
    p3 = alpha_3 * p1
    if etastar > Rc:
        p4 = p1 * (1 - (Rc / etastar))
    else:
        p4 = 0

    hstar_c = np.minimum(etastar, Rc)

    pu = 0.5 * (1 + cos_beta) * alpha_1 * alpha_3 * rho_water * 9.81 * HD

    FH = 0.5 * (p1 + p3) * hacc + 0.5 * (p1 + p4) * hstar_c
    MH = (
        (1 / 6) * (2 * np.pi + p3) * hacc**2
        + 0.5 * (p1 + p4) * hacc * hstar_c
        + (1 / 6) * (p1 + 2 * p4) * hstar_c**2
    )
    FU = 0.5 * pu * B_up
    MU = (2 / 3) * FU * B_up

    if not return_dict:
        return FH, FU, MH, MU
    else:
        all_results = dict()
        return all_results


def calculate_local_wavelength(
    T: float | npt.NDArray[np.float64],  # max wave period
    h: float | npt.NDArray[np.float64],  # water depth at site
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    # implementation of disper handles arrays for T, but not for h, hence this implementation
    if isinstance(h, float):
        k = dispersion.disper(w=((2 * np.pi) / T), h=h, g=g)
    elif len(h) > 1 and isinstance(T, float):
        k = np.array([])
        for hsub in h:
            k = np.append(k, dispersion.disper(w=((2 * np.pi) / T), h=hsub, g=g))
    elif len(h) > 1 and len(h) == len(T):
        k = np.array([])
        for hsub, Tsub in zip(h, T):
            k = np.append(k, dispersion.disper(w=((2 * np.pi) / Tsub), h=hsub, g=g))

    L = (2 * np.pi) / k

    return L


def calculate_etastar(
    HD: float | npt.NDArray[np.float64],
    cos_beta: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    return 0.75 * (1 + cos_beta) * HD


def calculate_p1():
    pass
