# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.dangremond1996 as dangremond1996

# import deltares_wave_toolbox.cores.core_dispersion as dispersion


def check_validity(
    sop: float | npt.NDArray[np.float64] = np.nan,
    ksiop: float | npt.NDArray[np.float64] = np.nan,
):

    if not np.any(np.isnan(sop)):
        core_utility.check_variable_validity_range(
            "Wave steepness sop", "Briganti (2003)", sop, 0.005, 0.07
        )

    if not np.any(np.isnan(ksiop)):
        core_utility.check_variable_validity_range(
            "Surf similarity parameter ksiop", "Briganti (2003)", ksiop, 0.5, 10.0
        )


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
        ksi_op = core_physics.calculate_Irribarren_number_ksi(
            H=Hsi, T=Tpi, cot_alpha=cot_alpha
        )
        sop = core_physics.calculate_wave_steepness_s(H=Hsi, T=Tpi)
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
                1 - np.exp(-0.41 * ksi_op)
            )

        # limit value
        Ktu = -0.006 * (B / Hsi) + 0.93
        Kt = np.minimum(Kt, Ktu)
        Kt = np.maximum(0.05, Kt)

        check_validity(sop=sop, ksiop=ksi_op)
    else:
        # arrays, so vectorize this function
        vfunc = np.vectorize(calculate_wave_transmission_Kt)
        Kt = vfunc(Hsi, Tpi, Rc, B, cot_alpha)

    return Kt
