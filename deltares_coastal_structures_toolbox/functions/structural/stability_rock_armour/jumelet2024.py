# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Jumelet et al. (2024) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    Table 4 in Jumelet et al. (2024). When parameters are nan (by default), they are not checked.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104418

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    """

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0",
            "Jumelet et al., 2024",
            smm10,
            0.009,
            0.057,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
            H=Hs, T=Tmm10, cot_alpha=cot_alpha
        )
        core_utility.check_variable_validity_range(
            "Iribarren number ksi_m-1,0",
            "Jumelet et al., 2024",
            ksi_mm10,
            0.42,
            1.66,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Jumelet et al., 2024",
            cot_alpha,
            6.0,
            10.0,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Jumelet et al., 2024",
            N_waves,
            250,
            20000,
        )
    return


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 4.3,
    rho_water: float = 1025.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers with the Jumelet et al. (2024) formula.

    Here, eq. 8 from Jumelet et al. (2024) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104418

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient in the stability formula, by default 4.3
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50=Dn50, M50=M50, rho_rock=rho_armour)

    ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
    )

    Fp = calculate_fraction_plunging_waves_Fp(ksi_mm10=ksi_mm10)

    N_plunging = np.round(Fp * N_waves)

    S = np.power(Ns * ksi_mm10 * np.power(N_plunging, 0.1) * (1.0 / c_pl), 5.0)

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    c_pl: float = 4.3,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers with the Jumelet et al. (2024) formula.

    Here, eq. 8 from Jumelet et al. (2024) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104418

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    c_pl : float, optional
        Coefficient in the stability formula, by default 4.3

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Fp = calculate_fraction_plunging_waves_Fp(ksi_mm10=ksi_mm10)

    N_plunging = np.round(Fp * N_waves)

    Dn50 = (
        (Hs / Delta)
        * ksi_mm10
        * np.power(N_plunging, 0.1)
        * (1.0 / c_pl)
        * np.power(S, -0.2)
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 4.3,
    g: float = 9.81,
    smm10_init: float = 0.03,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for rock armour layers with the Jumelet et al. (2024) formula.

    Here, eq. 8 from Jumelet et al. (2024) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104418

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
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    c_pl : float, optional
        Coefficient in the stability formula, by default 4.3
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    smm10_init : float, optional
        Initial wave steepness sm-1,0 (-) for the iterative solution, by default 0.03
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tolerance : float, optional
        Tolerance for convergence of the iterative solution, by default 1e-3

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50=Dn50, M50=M50, rho_rock=rho_armour)

    n_iter = 0
    Hs_init = smm10_init * np.power(Tmm10, 2) * g / (2 * np.pi)
    ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
        H=Hs_init, T=Tmm10, cot_alpha=cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Hs_diff = np.inf
    Hs_prev = Hs_init

    while np.max(Hs_diff) > tolerance and n_iter < max_iter:
        n_iter += 1

        Fp = calculate_fraction_plunging_waves_Fp(ksi_mm10=ksi_mm10)

        N_plunging = np.round(Fp * N_waves)

        Hs = np.power(
            c_pl
            * np.power(S, 0.2)
            * np.power(N_plunging, -0.1)
            * cot_alpha
            * np.sqrt((2 * np.pi / g) * np.power(Tmm10, -2))
            * Delta
            * Dn50,
            2.0,
        )

        Hs_diff = np.abs(Hs - Hs_prev)
        Hs_prev = Hs

        ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
            H=Hs, T=Tmm10, cot_alpha=cot_alpha
        )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
    )
    return Hs


def calculate_fraction_plunging_waves_Fp(
    ksi_mm10: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the fraction of plunging waves Fp

    Here, eq. 6 from Jumelet et al. (2024) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104418

    Parameters
    ----------
    ksi_mm10 : float | npt.NDArray[np.float64]
        The Iribarren number based on the spectral wave period Tm-1,0 (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The fraction of plunging waves Fp, [-]
    """

    Fp = -1.7 * np.power(ksi_mm10, 2) + 3.2 * ksi_mm10 - 0.5

    return Fp
