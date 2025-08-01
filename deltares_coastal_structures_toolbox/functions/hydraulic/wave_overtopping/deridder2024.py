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
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    m_foreshore_slope: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range as defined in De Ridder et al. (2024).

    For all parameters supplied, their values are checked versus the range of test conditions specified in Table 3
    (De Ridder et al., 2024). When parameters are nan (by default), they are not checked.

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2024.104626

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
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    m_foreshore_slope : float | npt.NDArray[np.float64], optional
        (Tangent of the) slope of the foreshore (-), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    """

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0", "De Ridder et al. (2024)", smm10, 0.001, 0.040
        )

    if not np.any(np.isnan(Hm0_HF)) and not np.any(np.isnan(Tmm10_HF)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0_HF, Tmm10_HF)
        core_utility.check_variable_validity_range(
            "Short wave steepness sm-1,0_HF",
            "De Ridder et al. (2024)",
            smm10,
            0.004,
            0.047,
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        Ksi_smm10 = core_physics.calculate_Irribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "Irribarren number Ksi_m-1,0",
            "De Ridder et al. (2024)",
            Ksi_smm10,
            0.05,
            1.11,
        )

    if not np.any(np.isnan(m_foreshore_slope)):
        core_utility.check_variable_validity_range(
            "Foreshore slope m",
            "De Ridder et al. (2024)",
            m_foreshore_slope,
            1.0 / 100.0,
            1.0 / 20.0,
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "Rc/Hm0", "De Ridder et al. (2024)", Rc / Hm0, 0.80, 3.72
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "Dn50/Hm0", "De Ridder et al. (2024)", Dn50 / Hm0, 0.12, 0.89
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Hm0_LF)):
        core_utility.check_variable_validity_range(
            "Hm0_LF/Hm0", "De Ridder et al. (2024)", Hm0_LF / Hm0, 0.10, 0.81
        )

    if not np.any(np.isnan(h)) and not np.any(np.isnan(Hm0)):
        core_utility.check_variable_validity_range(
            "h/Hm0", "De Ridder et al. (2024)", h / Hm0, 0.57, 4.95
        )

    return


def calculate_overtopping_discharge_q_eq24(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for a rubble mound breakwater following equation 24 in  De Ridder et al. (2024).

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2024.104626

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

    q = calculate_dimensionless_overtopping_discharge_eq24(
        Hm0, smm10_HF, gamma_f, Rc
    ) * np.sqrt(g * np.power(Hm0, 3))
    return q

def calculate_overtopping_discharge_q_eq26(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for a rubble mound breakwater following equation 26 in De Ridder et al. (2024).

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2024.104626

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
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

    q = calculate_dimensionless_overtopping_discharge_eq26(
        Hm0, smm10_HF, gamma_f, Rc
    ) * np.sqrt(g * np.power(Hm0, 3))
    return q

def calculate_dimensionless_overtopping_discharge_eq24(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for a rubble mound breakwater following
    De Ridder et al. (2024) using equation 24.

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2024.104626

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

    q_dimensionless = 0.74 * np.exp(
        -8.51 * (Rc / Hm0 * gamma_f) * np.power(smm10_HF, 0.32)
    )

    check_validity_range(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    return q_dimensionless

def calculate_dimensionless_overtopping_discharge_eq26(
    Hm0: float | npt.NDArray[np.float64],
    smm10_HF: float | npt.NDArray[np.float64],
    Hm0_LF: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for a rubble mound breakwater following
    De Ridder et al. (2024) using equation 26.

    For more details see De Ridder et al. (2024), available here https://doi.org/10.1016/j.coastaleng.2024.104626

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    smm10_HF : float | npt.NDArray[np.float64]
        Wave steepness sm-1,0 based on the deep water wave length corresponding
        to the high frequency spectral wave period Tm-1,0,HF(-)
    Hm0_LF : float | npt.NDArray[np.float64]
        Low-frequency wave height (m)
    gamma_f : float | npt.NDArray[np.float64]
        Reduction factor for wave overtopping due to friction (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
    """

    q_dimensionless = 0.50 * np.exp(
        -7.91 * ((Rc-0.21*Hm0_LF) / Hm0 * gamma_f) * np.power(smm10_HF, 0.30)
    )

    check_validity_range(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
        Hm0_LF=Hm0_LF,
    )

    return q_dimensionless
