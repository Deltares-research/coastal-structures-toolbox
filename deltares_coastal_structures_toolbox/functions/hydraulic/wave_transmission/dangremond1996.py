# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics

# import deltares_wave_toolbox.cores.core_dispersion as dispersion


def check_validity(
    Kt: float | npt.NDArray[np.float64] = np.nan,
    sop: float | npt.NDArray[np.float64] = np.nan,
    Hsi_over_h: float | npt.NDArray[np.float64] = np.nan,
    Rc_over_Hsi: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(Kt)):
        core_utility.check_variable_validity_range(
            "Transmission coefficient Kt", "D'Angremond et al. (1996)", Kt, 0.075, 0.8
        )

    if not np.any(np.isnan(sop)):
        core_utility.check_variable_validity_range(
            "Wave steepness sop", "D'Angremond et al. (1996)", sop, 0, 0.06
        )

    if not np.any(np.isnan(Hsi_over_h)):
        core_utility.check_variable_validity_range(
            "Relative water depth Hsi/h",
            "D'Angremond et al. (1996)",
            Hsi_over_h,
            0,
            0.54,
        )

    if not np.any(np.isnan(Rc_over_Hsi)):
        core_utility.check_variable_validity_range(
            "Relative crest level Rc/Hsi",
            "D'Angremond et al. (1996)",
            Rc_over_Hsi,
            -2.5,
            2.5,
        )


def calculate_wave_transmission_Kt_permeable(
    Hsi: float | npt.NDArray[np.float64],
    Tpi: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    C1: float | npt.NDArray[np.float64] = 0.64,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave transmission coefficient Kt using D'Angremond et al 1996

    For permeable structures

    Background: d’Angremond, K.; van der Meer, J.W.; de Jong, R.J.Wave Transmission at Low-Crested Structures.
        In Coastal Engineering 1996; American Society of Civil Engineers: New York, NY, USA, 1997; pp. 2418–2427.
    https://doi.org/10.9753/icce.v25.%p

    Parameters
    ----------
    Hsi : float | npt.NDArray[np.float64]
        Incident significant wave height (m)
    Tpi : float | npt.NDArray[np.float64]
        Incident peak wave period (s)
    h : float | npt.NDArray[np.float64]
        Water level in front of the structure
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard, vertical distance from SWL to top of crest
    B : float | npt.NDArray[np.float64]
        Width of structure at crest level
    cot_alpha : float | npt.NDArray[np.float64]
        Slope of front slope
    C1 : float | npt.NDArray[np.float64], optional
        Constant (calibration) value, by default 0.64

    Returns
    -------
    Kt : float | npt.NDArray[np.float64]
        Wave Transmission Coefficient (-)
    """
    ksi_op = core_physics.calculate_Irribarren_number_ksi(Hsi, Tpi, cot_alpha=cot_alpha)
    sop = core_physics.calculate_wave_steepness_s(H=Hsi, T=Tpi)

    Kt = -0.4 * (Rc / Hsi) + ((B / Hsi) ** -0.31) * (1 - np.exp(-0.5 * ksi_op)) * C1

    check_validity(Kt, sop=sop, Hsi_over_h=Hsi / h, Rc_over_Hsi=Rc / Hsi)

    Kt = np.minimum(0.8, Kt)
    Kt = np.maximum(0.075, Kt)

    return Kt


def calculate_wave_transmission_Kt_impermeable(
    Hsi: float | npt.NDArray[np.float64],
    Tpi: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    C1: float | npt.NDArray[np.float64] = 0.80,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave transmission coefficient Kt using D'Angremond et al 1996

    For impermeable structures

    Background: d’Angremond, K.; van der Meer, J.W.; de Jong, R.J.Wave Transmission at Low-Crested Structures.
        In Coastal Engineering 1996; American Society of Civil Engineers: New York, NY, USA, 1997; pp. 2418–2427.
    https://doi.org/10.9753/icce.v25.%p

    Parameters
    ----------
    Hsi : float | npt.NDArray[np.float64]
        Incident significant wave height (m)
    Tpi : float | npt.NDArray[np.float64]
        Incident peak wave period (s)
    h : float | npt.NDArray[np.float64]
        Water level in front of the structure
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard, vertical distance from SWL to top of crest
    B : float | npt.NDArray[np.float64]
        Width of structure at crest level
    cot_alpha : float | npt.NDArray[np.float64]
        Slope of front slope
    C1 : float | npt.NDArray[np.float64], optional
        Constant (calibration) value, by default 0.80

    Returns
    -------
    Kt : float | npt.NDArray[np.float64]
        Wave Transmission Coefficient (-)
    """

    Kt = calculate_wave_transmission_Kt_permeable(
        Rc=Rc, Hsi=Hsi, Tpi=Tpi, h=h, B=B, cot_alpha=cot_alpha, C1=C1
    )

    return Kt
