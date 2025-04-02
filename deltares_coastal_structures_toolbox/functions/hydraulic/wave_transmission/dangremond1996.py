# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics

# import deltares_wave_toolbox.cores.core_dispersion as dispersion


def check_validity(
    Kt: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(Kt)):
        core_utility.check_variable_validity_range(
            "Transmission coefficient Kt", "D'Angremond et al. (1996)", Kt, 0.075, 0.8
        )


def calculate_wave_transmission_Kt_permeable(
    Hsi: float | npt.NDArray[np.float64],
    Tpi: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    C1: float | npt.NDArray[np.float64] = 0.64,
) -> float | npt.NDArray[np.float64]:

    ksi_op = core_physics.calculate_Irribarren_number_ksi(Hsi, Tpi, cot_alpha=cot_alpha)

    Kt = -0.4 * (Rc / Hsi) + (B / Hsi) ** -0.31 * (1 - np.exp(-0.5 * ksi_op)) * C1

    check_validity(Kt)

    Kt = np.minimum(0.8, Kt)
    Kt = np.maximum(0.075, Kt)

    return Kt


def calculate_wave_transmission_Kt_impermeable(
    Hsi: float | npt.NDArray[np.float64],
    Tpi: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    C1: float | npt.NDArray[np.float64] = 0.80,
) -> float | npt.NDArray[np.float64]:

    Kt = calculate_wave_transmission_Kt_permeable(
        Rc=Rc, Hsi=Hsi, Tpi=Tpi, B=B, cot_alpha=cot_alpha, C1=C1
    )

    return Kt
