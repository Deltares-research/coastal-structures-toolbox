# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    h: float | npt.NDArray[np.float64] = np.nan,
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Hm0_HF: float | npt.NDArray[np.float64] = np.nan,
    Hm0_LF: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    Tmm10_HF: float | npt.NDArray[np.float64] = np.nan,
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    sigma_theta: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range as defined in De Ridder et al. (2026).

    For all parameters supplied, their values are checked versus the range of test conditions specified in Table 3
    (De Ridder et al., 2026). When parameters are nan (by default), they are not checked.

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    h : float | npt.NDArray[np.float64], optional
        Water depth at the toe of the structure (m), by default np.nan
    Hm0 : float | npt.NDArray[np.float64], optional
        Significant spectral wave height (m), by default np.nan
    Hm0_HF : float | npt.NDArray[np.float64], optional
        High frequency significant spectral wave height (m), by default np.nan
    Hm0_LF : float | npt.NDArray[np.float64], optional
        Low frequency significant spectral wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    Tmm10_HF : float | npt.NDArray[np.float64], optional
        High frequency spectral wave period Tm-1,0 (s), by default np.nan
    Rc : float | npt.NDArray[np.float64], optional
        Freeboard of the structure (m), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    sigma_theta : float | npt.NDArray[np.float64], optional
        Directional spreading in deegrees, by default np.nan
    """

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0", "De Ridder et al. (2024)", smm10, 0.003, 0.041
        )

    if not np.any(np.isnan(Hm0_HF)) and not np.any(np.isnan(Tmm10_HF)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0_HF, Tmm10_HF)
        core_utility.check_variable_validity_range(
            "Short wave steepness sm-1,0_HF",
            "De Ridder et al. (2024)",
            smm10,
            0.012,
            0.049,
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "Rc/Hm0", "De Ridder et al. (2024)", Rc / Hm0, 0.98, 1.94
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "Dn50/Hm0", "De Ridder et al. (2024)", Dn50 / Hm0, 0.18, 0.35
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Hm0_LF)):
        core_utility.check_variable_validity_range(
            "Hm0_LF/Hm0", "De Ridder et al. (2024)", Hm0_LF / Hm0, 0.15, 0.55
        )

    if not np.any(np.isnan(h)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "h/Hm0", "De Ridder et al. (2024)", h / Hm0, 1.41, 2.15
        )

    return


def calculate_crest_freeboard_discharge_q_eq28(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the crest freeboard given a q for a rubble mound breakwater
    following equation 24 in  De Ridder et al. (2026).

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Crest freeboard Rc (m)
    """
    Rc_diml = calculate_dimensionless_crest_freeboard_discharge_q_eq28(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        q=q,
    )

    Rc = Rc_diml * Hm0

    return Rc


def calculate_crest_freeboard_discharge_q_eq32(
    Hm0_HF: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    Hm0_LF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    g: float = 9.81,
    theta: float | npt.NDArray[np.float64] = 0.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the crest freeboard given a q for a rubble mound breakwater
    following equation 32 in  De Ridder et al. (2026).

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0_HF : float | npt.NDArray[np.float64]
        Significant spectral wave height of short waves (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    Hm0_LF : float | npt.NDArray[np.float64]
        Low-frequency wave height (m)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    theta : float | npt.NDArray[np.float64], optional
        Wave direction w.r.t to the structure (degrees), by default 0.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        Crest freeboard Rc (m)
    """
    gamma_beta = calculate_gamma_beta(theta=theta)
    Rc = (
        -(np.log(q / np.sqrt(g * np.power(Hm0_HF, 3))) - np.log(0.05))
        / (10.40 * np.power(smm10_HF, 0.50) * (1 / (gamma_f * gamma_beta)))
        * Hm0_HF
    ) + 0.97 * Hm0_LF

    return Rc


def calculate_dimensionless_crest_freeboard_discharge_q_eq28(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless crest freeboard given a q for a rubble mound breakwater
    following equation 28 in  De Ridder et al. (2026).

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Dimensionless crest freeboard Rc (-)
    """

    Rc_diml = (
        -(np.log(q / np.sqrt(g * np.power(Hm0, 3))) - np.log(0.16))
        / (6.93 * np.power(smm10_HF, 0.36))
        * gamma_f
    )
    return Rc_diml


def calculate_overtopping_discharge_q_eq28(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for a rubble mound breakwater following equation 28 in
     De Ridder et al. (2026).

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Mean wave overtopping discharge q (m^3/s/m)
    """

    q = calculate_dimensionless_overtopping_discharge_eq28(
        Hm0, smm10_HF, gamma_f, Rc
    ) * np.sqrt(g * np.power(Hm0, 3))
    return q


def calculate_overtopping_discharge_q_eq32(
    Hm0_HF: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    Hm0_LF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for a rubble mound breakwater following equation 32 in
     De Ridder et al. (2026).

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0_HF : float | npt.NDArray[np.float64]
        Significant spectral wave height of short waves (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    Hm0_LF : float | npt.NDArray[np.float64]
        Low-frequency wave height (m)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Mean wave overtopping discharge q (m^3/s/m)
    """

    q = calculate_dimensionless_overtopping_discharge_eq32(
        Hm0_HF, smm10_HF, Hm0_LF, gamma_f, Rc
    ) * np.sqrt(g * np.power(Hm0_HF, 3))
    return q


def calculate_dimensionless_overtopping_discharge_eq28(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for a rubble mound breakwater following
    De Ridder et al. (2026) using equation 28.

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
    """

    q_dimensionless = 0.16 * np.exp(
        -6.93 * (Rc / (Hm0 * gamma_f)) * np.power(smm10_HF, 0.36)
    )
    check_validity_range(
        Hm0=Hm0,
        Tmm10_HF=np.sqrt((2 * np.pi * Hm0) / smm10_HF / 9.81),
        Rc=Rc,
    )
    return q_dimensionless


def calculate_gamma_beta(
    theta: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for the wave direction as described in Van Gent & van der Werf, 2019

    Parameters:
        theta (float | npt.NDArray[np.float64]):
        Wave direction

    Returns:
        float | npt.NDArray[np.float64]: _description_
    """
    gamma_beta = 0.65 * np.cos(np.deg2rad(np.abs(theta))) ** 2 + 0.35
    return gamma_beta


def calculate_dimensionless_overtopping_discharge_eq32(
    Hm0_HF: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    Hm0_LF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    theta: float | npt.NDArray[np.float64] = 0.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for a rubble mound breakwater following
    De Ridder et al. (2026) using equation 32.

    For more details see De Ridder et al. (2026), available here https://doi.org/10.1016/j.coastaleng.2026.105039

    Parameters
    ----------
    Hm0_HF : float | npt.NDArray[np.float64]
        Significant spectral wave height of short waves (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    Hm0_LF : float | npt.NDArray[np.float64]
        Low-frequency wave height (m)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    theta : float | npt.NDArray[np.float64], optional
        Wave direction w.r.t to the structure (degrees), by default 0.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
    """

    gamma_beta = calculate_gamma_beta(theta=theta)

    q_dimensionless = 0.05 * np.exp(
        -10.40
        * ((Rc - 0.97 * Hm0_LF) / Hm0_HF)
        * (1 / (gamma_f * gamma_beta))
        * np.power(smm10_HF, 0.50)
    )

    check_validity_range(
        Hm0=Hm0_HF,
        Tmm10_HF=np.sqrt((2 * np.pi * Hm0_HF) / smm10_HF / 9.81),
        Rc=Rc,
        Hm0_LF=Hm0_LF,
    )

    return q_dimensionless
