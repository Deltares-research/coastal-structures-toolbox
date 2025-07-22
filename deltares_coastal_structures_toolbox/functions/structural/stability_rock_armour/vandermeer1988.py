# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    H2p: float | npt.NDArray[np.float64] = np.nan,
    Tp: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    P: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Van der Meer (1988) formula

    For all parameters supplied, their values are checked versus the range of test conditions specified by
    Van der Meer (1988). When parameters are nan (by default), they are not checked.

    For more details see Van der Meer (1988), available here
    https://resolver.tudelft.nl/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    H2p : float | npt.NDArray[np.float64], optional
        Wave height exceeded by 2% of waves H2% (m), by default np.nan
    Tp : float | npt.NDArray[np.float64], optional
        Peak wave period (s), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    P : float | npt.NDArray[np.float64], optional
        Notional permeability coefficient (-), by default np.nan
    rho_armour : float | npt.NDArray[np.float64], optional
        Armour rock density (kg/m^3), by default np.nan
    """

    if not np.any(np.isnan(P)):
        core_utility.check_variable_validity_range(
            "Notional permeability P", "Van der Meer (1988)", P, 0.1, 0.6
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Van der Meer (1988)",
            cot_alpha,
            1.1,
            7.0,
        )

    # TODO double check this: in formula only Tm is used, not Tp
    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tp)):
        s0p = core_physics.calculate_wave_steepness_s(H=Hs, T=Tp)
        core_utility.check_variable_validity_range(
            "Wave steepness s0p", "Van der Meer (1988)", s0p, 0.005, 0.06
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves", "Van der Meer (1988)", N_waves, 0, 7500
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(H2p)):
        core_utility.check_variable_validity_range(
            "Ratio H2%/Hs", "Van der Meer (1988)", H2p / Hs, 1.10, 1.40
        )

    if not np.any(np.isnan(rho_armour)):
        core_utility.check_variable_validity_range(
            "Density of armour layer rock rho_armour",
            "Van der Meer (1988)",
            rho_armour,
            2000,
            3100,
        )

    return


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
    Tm: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 8.7,
    c_s: float = 1.4,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers with the Van der Meer (1988) formula.

    For more details see Van der Meer (1988), available here
    https://resolver.tudelft.nl/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4

    Note that for cot_alpha >= 4.0, the formula for plunging waves is used as mentioned in Van der Meer (1993),
    available here (see Section 4.2): https://resolver.tudelft.nl/uuid:5a09837f-65b3-4ecf-92f1-aa3e6dc56d47

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    H2p : float | npt.NDArray[np.float64]
        Wave height exceeded by 2% of waves H2% (m)
    Tm : float | npt.NDArray[np.float64]
        Mean wave period (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 8.7
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.4
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    ksi_0m = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tm, cot_alpha=cot_alpha
    )

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl=c_pl, c_s=c_s, P=P, cot_alpha=cot_alpha
    )

    gamma_N = calculate_correction_term_gamma_N(N_waves=N_waves)

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
    )

    # Plunging waves
    S_pl = (
        np.power(
            Ns * (H2p / Hs) * (1 / c_pl) * np.power(P, -0.18) * np.power(ksi_0m, 0.5),
            5,
        )
        * (1 / gamma_N)
        * np.sqrt(N_waves)
    )

    # Surging waves
    S_s = (
        np.power(
            Ns
            * (H2p / Hs)
            * (1 / c_s)
            * np.power(P, 0.13)
            * (1 / np.sqrt(cot_alpha))
            * np.power(ksi_0m, -P),
            5,
        )
        * (1 / gamma_N)
        * np.sqrt(N_waves)
    )

    S = np.where((ksi_0m < ksi_mc) | (cot_alpha >= 4.0), S_pl, S_s)

    check_validity_range(
        P=P,
        Hs=Hs,
        cot_alpha=cot_alpha,
        H2p=H2p,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
    Tm: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    c_pl: float = 8.7,
    c_s: float = 1.4,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers with the Van der Meer (1988) formula.

    For more details see Van der Meer (1988), available here
    https://resolver.tudelft.nl/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4

    Note that for cot_alpha >= 4.0, the formula for plunging waves is used as mentioned in Van der Meer (1993),
    available here (see Section 4.2): https://resolver.tudelft.nl/uuid:5a09837f-65b3-4ecf-92f1-aa3e6dc56d47

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    H2p : float | npt.NDArray[np.float64]
        Wave height exceeded by 2% of waves H2% (m)
    Tm : float | npt.NDArray[np.float64]
        Mean wave period (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 8.7
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.4
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    ksi_0m = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tm, cot_alpha=cot_alpha
    )

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl=c_pl, c_s=c_s, P=P, cot_alpha=cot_alpha
    )

    gamma_N = calculate_correction_term_gamma_N(N_waves=N_waves)

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    # Plunging waves
    Dn50_pl = (
        (Hs / Delta)
        * (H2p / Hs)
        * (1 / c_pl)
        * np.power(P, -0.18)
        * np.power(gamma_N * S / np.sqrt(N_waves), -0.2)
        * np.power(ksi_0m, 0.5)
    )

    # Surging waves
    Dn50_s = (
        (Hs / Delta)
        * (H2p / Hs)
        * (1 / c_s)
        * np.power(P, 0.13)
        * np.power(gamma_N * S / np.sqrt(N_waves), -0.2)
        * (1 / np.sqrt(cot_alpha))
        * np.power(ksi_0m, -P)
    )

    Dn50 = np.where((ksi_0m < ksi_mc) | (cot_alpha >= 4.0), Dn50_pl, Dn50_s)

    check_validity_range(
        P=P,
        Hs=Hs,
        cot_alpha=cot_alpha,
        H2p=H2p,
        rho_armour=rho_armour,
        N_waves=N_waves,
    )

    return Dn50


def calculate_significant_wave_height_Hs(
    ratio_H2p_Hs: float | npt.NDArray[np.float64],
    Tm: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 8.7,
    c_s: float = 1.4,
    g: float = 9.81,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for rock armour layers with the Van der Meer (1988) formula.

    For more details see Van der Meer (1988), available here
    https://resolver.tudelft.nl/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4

    Note that for cot_alpha >= 4.0, the formula for plunging waves is used as mentioned in Van der Meer (1993),
    available here (see Section 4.2): https://resolver.tudelft.nl/uuid:5a09837f-65b3-4ecf-92f1-aa3e6dc56d47

    Parameters
    ----------
    ratio_H2p_Hs : float | npt.NDArray[np.float64]
        Ratio between the Wave height exceeded by 2% of waves H2% and the significant
        wave height Hs, H2% / Hs (-)
    Tm : float | npt.NDArray[np.float64]
        Mean wave period (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    P : float | npt.NDArray[np.float64]
        Notional permeability coefficient (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient for plunging waves (-), by default 8.7
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.4
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl=c_pl, c_s=c_s, P=P, cot_alpha=cot_alpha
    )

    gamma_N = calculate_correction_term_gamma_N(N_waves=N_waves)

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    # Plunging waves
    Hs_pl = np.power(
        c_pl
        * np.power(P, 0.18)
        * np.power(gamma_N * S / np.sqrt(N_waves), 0.2)
        * np.power(1.0 / cot_alpha, -0.5)
        * np.power((2 * np.pi / g) * (1.0 / np.power(Tm, 2)), 0.25)
        * Delta
        * Dn50
        * (1 / ratio_H2p_Hs),
        1.0 / 0.75,
    )

    # Surging waves
    Hs_s = np.power(
        c_s
        * np.power(P, -0.13)
        * np.power(gamma_N * S / np.sqrt(N_waves), 0.2)
        * np.power(1.0 / cot_alpha, P)
        * np.power((2 * np.pi / g) * (1.0 / np.power(Tm, 2)), -0.5 * P)
        * np.power(1.0 / cot_alpha, -0.5)
        * Delta
        * Dn50
        * (1 / ratio_H2p_Hs),
        1.0 / (1.0 + 0.5 * P),
    )

    ksi_0m_pl = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_pl, T=Tm, cot_alpha=cot_alpha
    )

    ksi_0m_s = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_s, T=Tm, cot_alpha=cot_alpha
    )

    Hs = np.where(
        ((ksi_0m_pl < ksi_mc) & (ksi_0m_s < ksi_mc)) | (cot_alpha >= 4.0), Hs_pl, Hs_s
    )

    check_validity_range(
        Hs=Hs,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        rho_armour=rho_armour,
    )
    return Hs


def calculate_correction_term_gamma_N(
    N_waves: int | npt.NDArray[np.int32],
) -> float | npt.NDArray[np.float64]:
    """Calculate the correction term gamma_N for the number of waves for rock armour layers with the
    Van der Meer (1988) formula.

    For more details see Van der Meer (1988), available here
    https://resolver.tudelft.nl/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4

    Parameters
    ----------
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The correction term for the number of waves gamma_N (-)
    """

    gamma_N_few = np.sqrt(1000 / N_waves)

    gamma_N_many = np.sqrt(N_waves) / (
        np.sqrt(5000) * (1.3 * (1 - np.exp(-0.00031 * N_waves)))
    )

    gamma_N_tmp = np.where(N_waves < 1000, gamma_N_few, 1.0)
    gamma_N = np.where(N_waves > 5000, gamma_N_many, gamma_N_tmp)

    return gamma_N
