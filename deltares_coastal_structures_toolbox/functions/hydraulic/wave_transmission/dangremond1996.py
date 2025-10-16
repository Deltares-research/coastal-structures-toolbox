# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    Kt: float | npt.NDArray[np.float64] = np.nan,
    s0p: float | npt.NDArray[np.float64] = np.nan,
    Hsi_over_h: float | npt.NDArray[np.float64] = np.nan,
    Rc_over_Hsi: float | npt.NDArray[np.float64] = np.nan,
):
    """Check the parameter values vs the validity range as defined in D'Angremond et al. (1996).

    For all parameters supplied, their values are checked versus the range of validity specified by
    D'Angremond et al. (1996). When parameters are nan (by default), they are not checked.

    Background: d’Angremond, K.; van der Meer, J.W.; de Jong, R.J.Wave Transmission at Low-Crested Structures.
        In Coastal Engineering 1996; American Society of Civil Engineers: New York, NY, USA, 1997; pp. 2418–2427.
    https://doi.org/10.9753/icce.v25.%p

    Parameters
    ----------
    Kt : float | npt.NDArray[np.float64], optional
        Wave Transmission Coefficient (-), by default np.nan
    s0p : float | npt.NDArray[np.float64], optional
        Wave steepness (-), by default np.nan
    Hsi_over_h : float | npt.NDArray[np.float64], optional
        Relative water depth Hsi/h (-), by default np.nan
    Rc_over_Hsi : float | npt.NDArray[np.float64], optional
        Relative crest level Rc/Hsi (-), by default np.nan
    """

    if not np.any(np.isnan(Kt)):
        core_utility.check_variable_validity_range(
            "Transmission coefficient Kt", "D'Angremond et al. (1996)", Kt, 0.075, 0.8
        )

    if not np.any(np.isnan(s0p)):
        core_utility.check_variable_validity_range(
            "Wave steepness s0p (dataset limit)",
            "D'Angremond et al. (1996)",
            s0p,
            0,
            0.06,
        )

    if not np.any(np.isnan(Hsi_over_h)):
        core_utility.check_variable_validity_range(
            "Relative water depth Hsi/h (dataset limit)",
            "D'Angremond et al. (1996)",
            Hsi_over_h,
            0,
            0.54,
        )

    if not np.any(np.isnan(Rc_over_Hsi)):
        core_utility.check_variable_validity_range(
            "Relative crest level Rc/Hsi (dataset limit)",
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
    do_validity_check: bool = True,
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
    ksi_0p = core_physics.calculate_Iribarren_number_ksi(Hsi, Tpi, cot_alpha=cot_alpha)
    s0p = core_physics.calculate_wave_steepness_s(H=Hsi, T=Tpi)

    Kt = -0.4 * (Rc / Hsi) + ((B / Hsi) ** -0.31) * (1 - np.exp(-0.5 * ksi_0p)) * C1

    if do_validity_check:
        check_validity(Kt, s0p=s0p, Hsi_over_h=Hsi / h, Rc_over_Hsi=Rc / Hsi)

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
