# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt


def check_validity():
    "No range provided"
    pass


def calculate_wave_transmission_Kt(
    Hsi: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    alpha_1: float | npt.NDArray[np.float64] = 2.2,
    beta_1: float | npt.NDArray[np.float64] = 0.4,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave transmission coefficient Kt using Goda et al (1967) for caisson structures

    For caisson structures
    Y., Takeda, H. and Moriya, Y. (1967). Laboratory investigation of wave transmission
        over breakwaters. Rep. port and Harbour Res. Inst., 13 (from Seelig 1979).

    Parameters
    ----------
    Hsi : float | npt.NDArray[np.float64]
        Incident significant wave height (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard, vertical distance from SWL to crest of structure
    alpha_1 : float | npt.NDArray[np.float64], optional
        coefficient depending on structure type, by default 2.2
        vertical (caisson) breakwater: 2.2
        vertical wall (no crest width): 1.8
    beta_1 : float | npt.NDArray[np.float64], optional
        coefficient depending on structure type, by default 0.4
        vertical (caisson) breakwater: 0.4
        vertical wall (no crest width): 0.1

    Returns
    -------
    Kt : float | npt.NDArray[np.float64]
        Wave Transmission Coefficient (-)
    """

    check_is_array = [
        isinstance(Rc, np.ndarray),
        isinstance(Hsi, np.ndarray),
    ]

    if not any(check_is_array):
        # no arrays, only single values
        Rc_over_Hsi = Rc / Hsi

        if Rc_over_Hsi <= -1 * alpha_1 - beta_1:
            Kt = 1
        elif Rc_over_Hsi <= alpha_1 - beta_1:
            Kt = 0.5 * (1 - np.sin((np.pi / 2) * ((Rc_over_Hsi + beta_1) / (alpha_1))))
        else:
            Kt = 0.03

    else:
        # arrays, so vectorize this function
        vfunc = np.vectorize(calculate_wave_transmission_Kt)
        Kt = vfunc(Hsi, Rc, alpha_1, beta_1)

    return Kt
