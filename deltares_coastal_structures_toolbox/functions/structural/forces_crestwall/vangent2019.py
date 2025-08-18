import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as runup


def check_validity_FH2p(Rc, Ac, Hs):
    pass


def calculate_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_beta: float | npt.NDArray[np.float64] = 0.5,
) -> tuple[np.float64 | float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculates influence factor on 2% runup level
    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Incident wave angle measured wrt normal of the dike
    c_beta : float | npt.NDArray[np.float64]
        Coefficient, 0.5 default

    Returns
    -------
    tuple[np.float64 | float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        influence factor gamma_beta
    """

    gamma_beta = (1 - c_beta) * np.cos(beta) ** 2 + c_beta

    return gamma_beta


def calculate_z2p(Hm0, Tmm10, cot_alpha, gamma_f, gamma_beta=1, c0=1.45, c1=5.0):

    gamma = gamma_f * gamma_beta

    ru2p = runup.calculate_wave_runup_height_z2p(
        Hm0=Hm0, Tmm10=Tmm10, cot_alpha=cot_alpha, gamma=gamma, c0=c0, c1=c1
    )

    return ru2p


def calculate_FH2p_perpendicular_from_Hm0():
    pass


def calculate_FH2p_perpendicular_from_z2p(
    Hs: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Hwall: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

    check_validity_FH2p()

    FH2p = rho_water * g * Hwall * (z2p - Ac)

    FH2p = np.min(FH2p, 0)

    return FH2p


def calculate_FV2p_perpendicular(
    z2p: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Bwall: float | npt.NDArray[np.float64],
    CFV: float | npt.NDArray[np.float64],
    Fb: float | npt.NDArray[np.float64],  # Level base plate relative to water level [m]
    g: float | npt.NDArray[np.float64] = 9.81,  # gravitational acceleration [m2/s]
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

    FV2p = CFV * rho_water * g * Bwall * (z2p - 0.75 * Ac) * (1 - (np.sqrt(Fb / Ac)))

    FV2p = np.min(FV2p, 0)

    return FV2p


def calculate_FH01p_perpendicular(
    FH2p: float | npt.NDArray[np.float64],
):

    FH01p = 1.6 * FH2p

    return FH01p


def calculate_FV01p_perpendicular(
    FV2p: float | npt.NDArray[np.float64],
    sop: float | npt.NDArray[np.float64],
):

    FV01p = (2.88 - 32 * sop) * FV2p

    return FV01p


def check_FV1p(
    sop: float | npt.NDArray[np.float64],
):

    pass
