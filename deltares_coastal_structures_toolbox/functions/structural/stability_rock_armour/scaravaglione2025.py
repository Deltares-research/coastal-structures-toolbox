# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    P: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
) -> None:
    """Check the parameter values vs the validity range of the stability formula as defined in
    Scaravaglione et al. (2025).

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    Scaravaglione et al. (2025). When parameters are nan (by default), they are not checked.

    For more details see Scaravaglione et al. (2025), available here https://doi.org/10.1016/j.coastaleng.2024.104657

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64], optional
        Significant spectral wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    P : float | npt.NDArray[np.float64], optional
        Notional permeability coefficient (-), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal (armour) rock diameter (m), by default np.nan
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal core rock diameter (m), by default np.nan
    rho_armour : float | npt.NDArray[np.float64], optional
        Armour rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    """

    if not np.any(np.isnan(P)):
        core_utility.check_variable_validity_range(
            "Notional permeability P",
            "Scaravaglione et al. (2025)",
            P,
            0.37,
            0.55,
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Dn50)):
        Ns = core_physics.calculate_stability_number_Ns(
            H=Hm0, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Scaravaglione et al. (2025)",
            Ns,
            1.47,
            3.44,
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        s_mm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness s_m-1,0",
            "Scaravaglione et al. (2025)",
            s_mm10,
            0.020,
            0.049,
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
            H=Hm0, T=Tmm10, cot_alpha=cot_alpha
        )
        core_utility.check_variable_validity_range(
            "Iribarren number ksi_m-1,0",
            "Scaravaglione et al. (2025)",
            ksi_mm10,
            1.3,
            15.0,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Scaravaglione et al. (2025)",
            cot_alpha,
            2.0,
            2.0,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Scaravaglione et al. (2025)",
            N_waves,
            652,
            4064,
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(Dn50_core)):
        core_utility.check_variable_validity_range(
            "Ratio core to armour Dn50_core / Dn50",
            "Scaravaglione et al. (2025)",
            Dn50_core / Dn50,
            0.21,
            0.71,
        )

    return


def calculate_damage_number_S(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    rho_core: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    c_VGnew: float = 3.3,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers in shallow water with
    the Scaravaglione et al. (2025) formula.

    For more details see Scaravaglione et al. (2025), available here https://doi.org/10.1016/j.coastaleng.2024.104657

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    rho_core : float | npt.NDArray[np.float64]
        Core rock density (kg/m^3)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal (armour) rock diameter (m), by default np.nan
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal core rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median (armour) rock mass (kg), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median core rock mass (kg), by default np.nan
    c_VGnew : float, optional
        Coefficient (-), by default 3.3
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50=Dn50, M50=M50, rho_rock=rho_armour)
    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_rock=rho_core
    )

    s_mm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hm0, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
    )

    S = np.power(
        (1 / c_VGnew)
        * np.power(s_mm10, -0.1)
        * np.sqrt(1 / cot_alpha)
        * Ns
        * (1.0 / (1 + Dn50_core / Dn50)),
        5,
    ) * np.sqrt(N_waves)

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        rho_armour=rho_armour,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    rho_core: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    c_VGnew: float = 3.3,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers in shallow water
    with the Scaravaglione et al. (2025) formula.

    For more details see Scaravaglione et al. (2025), available here https://doi.org/10.1016/j.coastaleng.2024.104657

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Significant spectral wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    rho_core : float | npt.NDArray[np.float64]
        Core rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal core rock diameter (m), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median core rock mass (kg), by default np.nan
    c_VGnew : float, optional
        Coefficient (-), by default 3.3
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_rock=rho_core
    )

    s_mm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    Dn50 = (Hm0 / Delta) * (1 / c_VGnew) * np.power(S / np.sqrt(N_waves), -0.2) * (
        1 / np.sqrt(cot_alpha)
    ) * np.power(s_mm10, -0.1) - Dn50_core

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        rho_armour=rho_armour,
    )
    return Dn50


def calculate_significant_wave_height_Hm0(
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    rho_core: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    c_VGnew: float = 3.3,
    g: float = 9.81,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant spectral wave height Hm0 for rock armour
    layers in shallow water with the Scaravaglione et al. (2025) formula.

    For more details see Scaravaglione et al. (2025), available here https://doi.org/10.1016/j.coastaleng.2024.104657

    Parameters
    ----------
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    rho_core : float | npt.NDArray[np.float64]
        Core rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal (armour) rock diameter (m), by default np.nan
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal core rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median (armour) rock mass (kg), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median core rock mass (kg), by default np.nan
    c_VGnew : float, optional
        Coefficient (-), by default 3.3
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant spectral wave height Hm0 (m)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_armour)

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_rock=rho_core
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    Hm0 = np.power(
        Delta
        * Dn50
        * c_VGnew
        * np.sqrt(cot_alpha)
        * (1 + Dn50_core / Dn50)
        * np.power(2 * np.pi / (g * np.power(Tmm10, 2)), 0.1)
        * np.power(S / np.sqrt(N_waves), 0.2),
        1.0 / 0.9,
    )

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        rho_armour=rho_armour,
    )
    return Hm0
