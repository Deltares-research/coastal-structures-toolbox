# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    P: float | npt.NDArray[np.float64] = np.nan,
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
) -> None:

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Dn50)):
        core_utility.check_variable_validity_range(
            "Relative crest height Rc / Dn50",
            "Kramer & Burcharth, 2004",
            Rc / Dn50,
            -3.0,
            2.0,
        )

    return


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns_init: float = 2.0,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
) -> float | npt.NDArray[np.float64]:

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    n_iter = 0

    Dn50_diff = np.inf
    Dn50_prev = np.inf
    Dn50 = Hs / (Delta * Ns_init)

    while np.max(Dn50_diff) > tolerance and n_iter < max_iter:
        n_iter += 1

        Dn50 = (1.0 / 1.36) * ((Hs / Delta) - Rc * (0.06 * (Rc / Dn50) - 0.23))

        Dn50_diff = np.abs(Dn50 - Dn50_prev)
        Dn50_prev = Dn50

    check_validity_range(
        Hs=Hs,
        Rc=Rc,
        Dn50=Dn50,
        rho_armour=rho_armour,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    Rc: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Hs = (0.06 * np.power(Rc / Dn50, 2.0) - 0.23 * Rc / Dn50 + 1.36) * Delta * Dn50

    check_validity_range(
        Hs=Hs,
        Rc=Rc,
        Dn50=Dn50,
        rho_armour=rho_armour,
    )
    return Hs
