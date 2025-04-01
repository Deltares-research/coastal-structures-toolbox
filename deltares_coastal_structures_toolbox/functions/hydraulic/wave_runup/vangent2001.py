# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


def calculate_wave_runup_height_z1p(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    c0: float = 1.45,
    c1: float = 5.1,
) -> float | npt.NDArray[np.float64]:
    """Calculate the wave runup height with a 1% probability of exceedance z1% with the Van Gent (2001) formula.

    The wave runup height is calculated using the Van Gent (2001) formula. Van Gent (2001) provides the coefficients
    to calculate the z2%, and Van Gent (2007) lists coefficients for the z1% and z10%.

    For more details see Van Gent (2001), available here: https://doi.org/10.1061/(ASCE)0733-950X(2001)127:5(254)
    or here: https://www.researchgate.net/publication/245293002_Wave_Run-Up_on_Dikes_with_Shallow_Foreshores

    And Van Gent (2007), available here: https://doi.org/10.1142/9789814282024_0002 or here:
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor gamma = gamma_f * gamma_beta (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    c0 : float, optional
        Coefficient in wave runup formula (-), by default 1.45
    c1 : float, optional
        Coefficient in wave runup formula (-), by default 5.1

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 1% exceedance wave runup height z1% (m)
    """

    z1p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z1p


def calculate_wave_runup_height_z2p(
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    c0: float = 1.35,
    c1: float = 4.7,
) -> float:
    """Calculate the wave runup height with a 2% probability of exceedance z2% with the Van Gent (2001) formula.

    The wave runup height is calculated using the Van Gent (2001) formula. Van Gent (2001) provides the coefficients
    to calculate the z2%, and Van Gent (2007) lists coefficients for the z1% and z10%.

    For more details see Van Gent (2001), available here: https://doi.org/10.1061/(ASCE)0733-950X(2001)127:5(254)
    or here: https://www.researchgate.net/publication/245293002_Wave_Run-Up_on_Dikes_with_Shallow_Foreshores

    And Van Gent (2007), available here: https://doi.org/10.1142/9789814282024_0002 or here:
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Note that in Van Gent (2001) various values for the coefficients c0 and c1 are given for different wave height
    metrics (Hm0 and Hs), and including or excluding long waves. Here, only the coefficients including long waves
    are implemented. This function can be called supplying either Hm0 or Hs.

    Parameters
    ----------
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor gamma = gamma_f * gamma_beta (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    Hm0 : float | npt.NDArray[np.float64], optional
        Spectral significant wave height (m), by default np.nan
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    c0 : float, optional
        Coefficient in wave runup formula (-), by default 1.35
    c1 : float, optional
        Coefficient in wave runup formula (-), by default 4.7

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 2% exceedance wave runup height z2% (m)

    Raises
    ------
    ValueError
        Raise an error when both or neither Hm0 and Hs are provided.
    """

    if (np.isnan(Hm0) and np.isnan(Hs)) or (not np.isnan(Hs) and not np.isnan(Hm0)):
        raise ValueError("Either Hm0 or Hs should be provided")

    if np.isnan(Hs) and not np.isnan(Hm0):
        # Use coefficients for Hm0 instead of Hs
        c0 = 1.45
        c1 = 3.8

    z2p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z2p


def calculate_wave_runup_height_z10p(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    c0: float = 1.1,
    c1: float = 4.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the wave runup height with a 10% probability of exceedance z10% with the Van Gent (2001) formula.

    The wave runup height is calculated using the Van Gent (2001) formula. Van Gent (2001) provides the coefficients
    to calculate the z2%, and Van Gent (2007) lists coefficients for the z1% and z10%.

    For more details see Van Gent (2001), available here: https://doi.org/10.1061/(ASCE)0733-950X(2001)127:5(254)
    or here: https://www.researchgate.net/publication/245293002_Wave_Run-Up_on_Dikes_with_Shallow_Foreshores

    And Van Gent (2007), available here: https://doi.org/10.1142/9789814282024_0002 or here:
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor gamma = gamma_f * gamma_beta (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    c0 : float, optional
        Coefficient in wave runup formula (-), by default 1.1
    c1 : float, optional
        Coefficient in wave runup formula (-), by default 4.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 10% exceedance wave runup height z10% (m)
    """

    z10p = calculate_wave_runup_height_zXp(Hs, Tmm10, gamma, cot_alpha, c0, c1)

    return z10p


def calculate_wave_runup_height_zXp(
    H: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    c0: float,
    c1: float,
) -> float | npt.NDArray[np.float64]:
    """Calculate the wave runup height with the Van Gent (2001) formula.

    The wave runup height is calculated using the Van Gent (2001) formula. Van Gent (2001) provides the coefficients
    to calculate the z2%, and Van Gent (2007) lists coefficients for the z1% and z10%.

    For more details see Van Gent (2001), available here: https://doi.org/10.1061/(ASCE)0733-950X(2001)127:5(254)
    or here: https://www.researchgate.net/publication/245293002_Wave_Run-Up_on_Dikes_with_Shallow_Foreshores

    And Van Gent (2007), available here: https://doi.org/10.1142/9789814282024_0002 or here:
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    H : float | npt.NDArray[np.float64]
        Wave height, either the Hs or the Hm0 depending on the coefficients used (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor gamma = gamma_f * gamma_beta (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    c0 : float
        Coefficient in wave runup formula (-)
    c1 : float
        Coefficient in wave runup formula (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The wave runup height z (m)
    """

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(H, Tmm10, cot_alpha)

    p = 0.5 * c1 / c0

    zXp_a = c0 * ksi_mm10 * gamma * H

    c2 = 0.25 * np.power(c1, 2) / c0
    zXp_b = (c1 - c2 / ksi_mm10) * gamma * H

    zXp = np.where(ksi_mm10 < p, zXp_a, zXp_b)

    return zXp
