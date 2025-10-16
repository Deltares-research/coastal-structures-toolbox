# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.dangremond1996 as dangremond1996


def check_validity(
    s0p: float | npt.NDArray[np.float64] = np.nan,
    ksi_0p: float | npt.NDArray[np.float64] = np.nan,
):
    """Check the parameter values vs the validity range as defined in Briganti (2003).

    For all parameters supplied, their values are checked versus the range of validity
    specified by Briganti (2003). When parameters are nan (by default), they are not checked.

    Parameters
    ----------
    s0p : float | npt.NDArray[np.float64], optional
        Wave steepness s0p (-), by default np.nan
    ksi_0p : float | npt.NDArray[np.float64], optional
        Iribarren number ksi_0p (-), by default np.nan
    """

    if not np.any(np.isnan(s0p)):
        core_utility.check_variable_validity_range(
            "Wave steepness s0p", "Briganti (2003)", s0p, 0.005, 0.07
        )

    if not np.any(np.isnan(ksi_0p)):
        core_utility.check_variable_validity_range(
            "Surf similarity parameter ksi_0p", "Briganti (2003)", ksi_0p, 0.5, 10.0
        )

    return


def calculate_wave_transmission_Kt(
    Hsi: float | npt.NDArray[np.float64],
    Tpi: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate wave transmission coefficient Kt using Briganti (2003)

    For permeable structures

    See: Briganti, R.,J.W. Van der Meer, M. Buccino and M.Calabrese (2003),’Wave transmission
        behind low-crested structures’. Proceedings of Coastal Structures 2003, Portland,
        USA, p. 580-592
    http://dx.doi.org/10.1061/40733(147)48

    Note that for structures with B/Hsi < 10 this approach is equal to D'Angremond (1996)

    Parameters
    ----------
    Hsi : float | npt.NDArray[np.float64]
        Incident significant wave height (m)
    Tpi : float | npt.NDArray[np.float64]
        Incident peak wave period (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard, vertical distance from SWL to top of crest
    B : float | npt.NDArray[np.float64]
        Width of structure at crest level
    cot_alpha : float | npt.NDArray[np.float64]
        Slope of front slope

    Returns
    -------
    Kt : float | npt.NDArray[np.float64]
        Wave Transmission Coefficient (-)
    """

    check_is_array = [
        isinstance(B, np.ndarray),
        isinstance(Hsi, np.ndarray),
    ]

    if not any(check_is_array):
        # no arrays, only single values
        B_over_Hsi = B / Hsi
        ksi_0p = core_physics.calculate_Iribarren_number_ksi(
            H=Hsi, T=Tpi, cot_alpha=cot_alpha
        )
        s0p = core_physics.calculate_wave_steepness_s(H=Hsi, T=Tpi)
        if B_over_Hsi < 10:
            Kt = dangremond1996.calculate_wave_transmission_Kt_permeable(
                Hsi=Hsi,
                Tpi=Tpi,
                B=B,
                h=np.nan,
                Rc=Rc,
                cot_alpha=cot_alpha,
                do_validity_check=False,
            )
        else:

            Kt = -0.35 * (Rc / Hsi) + (0.51 * (B / Hsi) ** -0.65) * (
                1 - np.exp(-0.41 * ksi_0p)
            )

        # limit value
        Ktu = -0.006 * (B / Hsi) + 0.93
        Kt = np.minimum(Kt, Ktu)
        Kt = np.maximum(0.05, Kt)

        check_validity(s0p=s0p, ksi_0p=ksi_0p)
    else:
        # arrays, so vectorize this function
        vfunc = np.vectorize(calculate_wave_transmission_Kt)
        Kt = vfunc(Hsi, Tpi, Rc, B, cot_alpha)

    return Kt
