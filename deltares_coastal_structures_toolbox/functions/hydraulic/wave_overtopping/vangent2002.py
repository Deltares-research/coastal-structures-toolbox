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
    """Calculate the maximum wave overtopping velocity following Van Gent (2002).

    For a given wave runup height with an exceedance probability, the maximum wave overtopping velocity with the
    same exceedance probability is calculated.

    For more details see Van Gent (2002), available here: https://doi.org/10.1142/9789812791306_0185 or here:
    https://www.researchgate.net/publication/259260272_Wave_overtopping_events_at_dikes

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    zXp : float | npt.NDArray[np.float64]
        The wave runup height z (m)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Bc : float | npt.NDArray[np.float64]
        Width of the crest of the structure (m)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_Crest : float | npt.NDArray[np.float64]
        Influence factor for surface roughness on the crest of the structure (-)
    cu1 : float, optional
        Coefficient, by default 1.7
    cu2 : float, optional
        Coefficient, by default 0.1
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The maximum wave overtopping velocity u (m/s)
    """

    uXp = (
        np.sqrt(g * Hs)
        * cu1
        * np.sqrt(gamma_f_Crest)
        * np.sqrt((zXp - Rc) / (gamma_f * Hs))
        / (1 + cu2 * Bc / Hs)
    )

    return uXp


def _invert_for_zXp(
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
