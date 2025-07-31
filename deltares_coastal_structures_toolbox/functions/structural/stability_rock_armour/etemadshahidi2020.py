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
    P: float | npt.NDArray[np.float64] = np.nan,
    S: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
) -> None:
    """Check the parameter values vs the validity range of the Etemad-Shahidi et al. (2020) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    Table 1 of Etemad-Shahidi et al. (2020). When parameters are nan (by default), they are not checked.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103655
    and the corresponding corrigendum: https://doi.org/10.1016/j.coastaleng.2022.104142

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
    P : float | npt.NDArray[np.float64], optional
        Notional permeability coefficient (-), by default np.nan
    S : float | npt.NDArray[np.float64], optional
        Damage number (-), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal rock diameter of the core material (m), by default np.nan
    rho_armour : float | npt.NDArray[np.float64], optional
        Armour rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    """

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hs, T=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0",
            "Etemad-Shahidi et al., 2020",
            smm10,
            0.003,
            0.088,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
            H=Hs, T=Tmm10, cot_alpha=cot_alpha
        )
        core_utility.check_variable_validity_range(
            "Irribarren number ksi_m-1,0",
            "Etemad-Shahidi et al., 2020",
            ksi_mm10,
            0.65,
            8.18,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Etemad-Shahidi et al., 2020",
            cot_alpha,
            1.5,
            6.0,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Etemad-Shahidi et al., 2020",
            N_waves,
            500,
            5000,
        )

    if not np.any(np.isnan(rho_armour)):
        Delta = core_physics.calculate_buoyant_density_Delta(
            rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Buoyant density Delta",
            "Etemad-Shahidi et al., 2020",
            Delta,
            0.92,
            2.05,
        )

    if not np.any(np.isnan(P)):
        Delta = core_physics.calculate_buoyant_density_Delta(
            rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Permeability P",
            "Etemad-Shahidi et al., 2020",
            P,
            0.1,
            0.6,
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(Dn50_core)):
        core_utility.check_variable_validity_range(
            "Relative core diameter Dn50_core / Dn50",
            "Etemad-Shahidi et al., 2020",
            Dn50_core / Dn50,
            0.0,
            1.0,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Dn50))
        and not np.any(np.isnan(rho_armour))
    ):
        Ns = core_physics.calculate_stability_number_Ns(
            H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
        )
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Etemad-Shahidi et al., 2020",
            Ns,
            1.0,
            4.3,
        )

    if not np.any(np.isnan(S)):
        core_utility.check_variable_validity_range(
            "Damage number S",
            "Etemad-Shahidi et al., 2020",
            S,
            2.0,
            12.0,
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
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
    m: float | npt.NDArray[np.float64] = np.nan,
    use_depth_limited_version: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers with the Etemad-Shahidi et al. (2020) formula.

    Here, eqs. 17a & 17b from Etemad-Shahidi et al. (2020) are implemented. The user can also choose to
    use the depth limited version of the formula (eqs. 12a & 12b) by enabling that option and providing
    the foreshore slope.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103655
    and the corresponding corrigendum: https://doi.org/10.1016/j.coastaleng.2022.104142

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
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal rock diameter of the core material (m), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median rock mass of the core material (kg), by default np.nan
    rho_core : float | npt.NDArray[np.float64], optional
        Core rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    m : float | npt.NDArray[np.float64], optional
        Tangent of the foreshore slope (-), by default np.nan
    use_depth_limited_version : bool, optional
        Use depth limited version of the formula, by default False

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)

    Raises
    ------
    ValueError
        If depth limited version is used and m is not provided.
    """

    if use_depth_limited_version:
        if np.any(np.isnan(m)):
            raise ValueError("For the depth limited version, m should be provided")
        else:
            influence_depth_limitation = 1.0 - 3.0 * m
    else:
        influence_depth_limitation = 1.0

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=rho_water
    )

    Cp = calculate_permeability_coefficient_Cp(Dn50=Dn50, Dn50_core=Dn50_core)

    S_s = (
        np.power(1.0 / 3.9, 6.0)
        * np.power(Ns, 6.0)
        * np.power(Cp, -6.0)
        * np.power(N_waves, 0.6)
        * np.power(ksi_mm10, 2.0)
        * np.power(influence_depth_limitation, -6.0)
    )
    S_pl = (
        np.power(1.0 / 4.5, 6.0)
        * np.power(Ns, 6.0)
        * np.power(Cp, -6.0)
        * np.power(N_waves, 0.6)
        * np.power(ksi_mm10, 7.0 / 2.0)
        * np.power(influence_depth_limitation, -6.0)
    )

    S = np.where(
        ksi_mm10 >= 1.8,
        S_s,
        S_pl,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        S=S,
        rho_armour=rho_armour,
        rho_water=rho_water,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
    m: float | npt.NDArray[np.float64] = np.nan,
    use_depth_limited_version: bool = False,
    Cp_init: float = 0.5,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers with the Etemad-Shahidi et al. (2020) formula.

    Here, eqs. 10a & 10b from Etemad-Shahidi et al. (2020) are implemented. The user can also choose to
    use the depth limited version of the formula (eqs. 12a & 12b) by enabling that option and providing
    the foreshore slope.

    Note that, due to the dependency of Cp on Dn50, an iterative solution is necessary.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103655
    and the corresponding corrigendum: https://doi.org/10.1016/j.coastaleng.2022.104142

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
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal rock diameter of the core material (m), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median rock mass of the core material (kg), by default np.nan
    rho_core : float | npt.NDArray[np.float64], optional
        Core rock density (kg/m^3), by default np.nan
    m : float | npt.NDArray[np.float64], optional
        Tangent of the foreshore slope (-), by default np.nan
    use_depth_limited_version : bool, optional
        Use depth limited version of the formula, by default False
    Cp_init : float, optional
        Initial coefficient of permeability Cp (-) for the iterative solution, by default 0.5
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tolerance : float, optional
        Tolerance for convergence of the iterative solution, by default 1e-5

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)

    Raises
    ------
    ValueError
        If depth limited version is used and m is not provided.
    """

    if use_depth_limited_version:
        if np.any(np.isnan(m)):
            raise ValueError("For the depth limited version, m should be provided")
        else:
            influence_depth_limitation = 1.0 - 3.0 * m
    else:
        influence_depth_limitation = 1.0

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hs, T=Tmm10, cot_alpha=cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    n_iter_s = 0

    Dn50_s_diff = np.inf
    Dn50_s_prev = np.inf
    Cp_s = Cp_init

    while np.max(Dn50_s_diff) > tolerance and n_iter_s < max_iter:
        n_iter_s += 1

        Dn50_s = (
            (Hs / Delta)
            * (1.0 / 3.9)
            * (1.0 / Cp_s)
            * np.power(N_waves, 1.0 / 10.0)
            * np.power(S, -1.0 / 6.0)
            * np.power(ksi_mm10, 1.0 / 3.0)
            * np.power(influence_depth_limitation, -1.0)
        )

        Cp_s = calculate_permeability_coefficient_Cp(Dn50=Dn50_s, Dn50_core=Dn50_core)
        Dn50_s_diff = np.abs(Dn50_s - Dn50_s_prev)
        Dn50_s_prev = Dn50_s

    n_iter_pl = 0

    Dn50_pl_diff = np.inf
    Dn50_pl_prev = np.inf
    Cp_pl = Cp_init

    while np.max(Dn50_pl_diff) > tolerance and n_iter_pl < max_iter:
        n_iter_pl += 1

        Dn50_pl = (
            (Hs / Delta)
            * (1.0 / 4.5)
            * (1.0 / Cp_pl)
            * np.power(N_waves, 1.0 / 10.0)
            * np.power(S, -1.0 / 6.0)
            * np.power(ksi_mm10, 7.0 / 12.0)
            * np.power(influence_depth_limitation, -1.0)
        )

        Cp_pl = calculate_permeability_coefficient_Cp(Dn50=Dn50_pl, Dn50_core=Dn50_core)
        Dn50_pl_diff = np.abs(Dn50_pl - Dn50_pl_prev)
        Dn50_pl_prev = Dn50_pl

    Dn50 = np.where(
        ksi_mm10 >= 1.8,
        Dn50_s,
        Dn50_pl,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        S=S,
        rho_armour=rho_armour,
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
    Dn50_core: float | npt.NDArray[np.float64] = np.nan,
    M50_core: float | npt.NDArray[np.float64] = np.nan,
    rho_core: float | npt.NDArray[np.float64] = np.nan,
    m: float | npt.NDArray[np.float64] = np.nan,
    use_depth_limited_version: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for rock armour layers with the Etemad-Shahidi et al. (2020)
    formula.

    Here, eqs. 18a & 18b from Etemad-Shahidi et al. (2020) are implemented. The user can also choose to
    use the depth limited version of the formula (eqs. 12a & 12b) by enabling that option and providing
    the foreshore slope.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103655
    and the corresponding corrigendum: https://doi.org/10.1016/j.coastaleng.2022.104142

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
    Dn50_core : float | npt.NDArray[np.float64], optional
        Nominal rock diameter of the core material (m), by default np.nan
    M50_core : float | npt.NDArray[np.float64], optional
        Median rock mass of the core material (kg), by default np.nan
    rho_core : float | npt.NDArray[np.float64], optional
        Core rock density (kg/m^3), by default np.nan
    m : float | npt.NDArray[np.float64], optional
        Tangent of the foreshore slope (-), by default np.nan
    use_depth_limited_version : bool, optional
        Use depth limited version of the formula, by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)

    Raises
    ------
    ValueError
        If depth limited version is used and m is not provided.
    """

    if use_depth_limited_version:
        if np.any(np.isnan(m)):
            raise ValueError("For the depth limited version, m should be provided")
        else:
            influence_depth_limitation = 1.0 - 3.0 * m
    else:
        influence_depth_limitation = 1.0

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Dn50_core = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50_core, M50=M50_core, rho_armour=rho_core
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Cp = calculate_permeability_coefficient_Cp(Dn50=Dn50, Dn50_core=Dn50_core)

    Hs_s = np.power(
        S
        * np.power(1.0 / 3.9, -6.0)
        * (2 * np.pi / g)
        * np.power(Cp, 6.0)
        * np.power(Delta * Dn50, 6.0)
        * np.power(Tmm10, -2.0)
        * np.power(cot_alpha, 2.0)
        * np.power(N_waves, -0.6)
        * np.power(influence_depth_limitation, 6.0),
        1.0 / 5.0,
    )

    Hs_pl = np.power(
        S
        * np.power(1.0 / 4.5, -6.0)
        * np.power(2 * np.pi / g, 7.0 / 4.0)
        * np.power(Cp, 6.0)
        * np.power(Delta * Dn50, 6.0)
        * np.power(Tmm10, -7.0 / 2.0)
        * np.power(cot_alpha, 7.0 / 2.0)
        * np.power(N_waves, -0.6)
        * np.power(influence_depth_limitation, 6.0),
        4.0 / 17.0,
    )

    ksi_mm10_s = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_s, T=Tmm10, cot_alpha=cot_alpha
    )

    ksi_mm10_pl = core_physics.calculate_Irribarren_number_ksi(
        H=Hs_pl, T=Tmm10, cot_alpha=cot_alpha
    )

    Hs = np.where(
        ((ksi_mm10_pl < 1.8) & (ksi_mm10_s < 1.8)),
        Hs_pl,
        Hs_s,
    )

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Dn50=Dn50,
        Dn50_core=Dn50_core,
        S=S,
        rho_armour=rho_armour,
    )
    return Hs


def calculate_permeability_coefficient_Cp(
    Dn50: float | npt.NDArray[np.float64],
    Dn50_core: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the coefficient of permeability Cp following Etemad-Shahidi et al. (2020).

    Here, eq. 11 from Etemad-Shahidi et al. (2020) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103655
    and the corresponding corrigendum: https://doi.org/10.1016/j.coastaleng.2022.104142

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal rock diameter (m)
    Dn50_core : float | npt.NDArray[np.float64]
        Nominal rock diameter of the core material (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The coefficient of permeability Cp (-)
    """

    Cp = np.power(1 + np.power(Dn50_core / Dn50, 3.0 / 10.0), 3.0 / 5.0)

    return Cp
