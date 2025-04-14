# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt


def calculate_maximum_wave_overtopping_velocity_uXp(
    Hs: float | npt.NDArray[np.float64],
    zXp: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    cu1: float = 1.7,
    cu2: float = 0.1,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    uXp = (
        np.sqrt(g * Hs)
        * cu1
        * np.sqrt(gamma_f_Crest)
        * np.sqrt((zXp - Rc) / (gamma_f * Hs))
        / (1 + cu2 * Bc / Hs)
    )

    return uXp


def invert_for_zXp(
    Hs: float | npt.NDArray[np.float64],
    uXp: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    cu1: float = 1.7,
    cu2: float = 0.1,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    zXp = (
        np.power(
            (uXp / np.sqrt(g * Hs))
            * (1.0 / (cu1 * np.sqrt(gamma_f_Crest)))
            * (1.0 + cu2 * Bc / Hs),
            2.0,
        )
        * Hs
        * gamma_f
        + Rc
    )

    return zXp
