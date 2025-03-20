# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hs: float | npt.NDArray[np.float64] = np.nan,
    H2p: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    P: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    rho_armour: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Modified Van der Meer formula as defined in
    Van Gent et al. (2003).

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    (Van Gent et al., 2003). When parameters are nan (by default), they are not checked.

    For more details see Van Gent et al. (2003), available here https://doi.org/10.1061/40733(147)9 or here
    https://www.researchgate.net/publication/259258688_Stability_of_Rock_Slopes_with_Shallow_Foreshores


    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    H2p : float | npt.NDArray[np.float64], optional
        Wave height exceeded by 2% of waves H2% (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    P : float | npt.NDArray[np.float64], optional
        Notional permeability coefficient (-), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    rho_armour : float | npt.NDArray[np.float64], optional
        Armour rock density (kg/m^3), by default np.nan
    """

    if not np.any(np.isnan(P)):
        core_utility.check_variable_validity_range(
            "Notional permeability P",
            "Modified Van der Meer (Van Gent et al., 2003)",
            P,
            0.1,
            0.6,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Dn50)):
        Ns = core_physics.calculate_stability_number_Ns(Hs, Dn50, rho_armour, 1025)
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Modified Van der Meer (Van Gent et al., 2003)",
            Ns,
            0.5,
            4.5,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(Hs, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Modified Van der Meer (Van Gent et al., 2003)",
            ksi_mm10,
            1.3,
            15.0,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Modified Van der Meer (Van Gent et al., 2003)",
            cot_alpha,
            2.0,
            4.0,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Modified Van der Meer (Van Gent et al., 2003)",
            N_waves,
            0,
            3000,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(H2p)):
        core_utility.check_variable_validity_range(
            "Ratio H2%/Hs",
            "Modified Van der Meer (Van Gent et al., 2003)",
            H2p / Hs,
            1.20,
            1.40,
        )

    return


def calculate_damage_number_S(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 8.4,
    c_s: float = 1.3,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock armour layers with the Modified Van der Meer
    formula by Van Gent et al. (2003).

    For more details see Van Gent et al. (2003), available here https://doi.org/10.1061/40733(147)9 or here
    https://www.researchgate.net/publication/259258688_Stability_of_Rock_Slopes_with_Shallow_Foreshores

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    H2p : float | npt.NDArray[np.float64]
        Wave height exceeded by 2% of waves H2% (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
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
        Coefficient for plunging waves (-), by default 8.4
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.3

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """
    # TODO implement calculating H2% with Battjes-Groenendijk (see dwt, where it's already implemented)

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_armour)

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(Hs, Tmm10, cot_alpha)

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl, c_s, P, cot_alpha
    )

    Ns = core_physics.calculate_stability_number_Ns(
        H=Hs, D=Dn50, rho_rock=rho_armour, rho_water=1025
    )

    # Plunging waves
    S_pl = np.power(
        (1 / c_pl) * np.power(P, -0.18) * np.power(ksi_mm10, 0.5) * Ns * (H2p / Hs),
        5,
    ) * np.sqrt(N_waves)

    # Surging waves
    S_s = np.power(
        (1 / c_s)
        * np.power(P, 0.13)
        * np.power(ksi_mm10, -P)
        * np.sqrt(1 / cot_alpha)
        * Ns
        * (H2p / Hs),
        5,
    ) * np.sqrt(N_waves)

    # TODO check: it seems that the very gentle slopes (cot_alpha > 3.5) are not implemented in BREAKWAT
    # TODO => what should we do?
    # # Very gentle slopes
    # ksi_mcc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
    #     c_pl, c_s, P, 3.5
    # )

    # S_s_cot_alpha_3p5 = np.power(
    #     (1 / c_pl)
    #     * np.power(P, -0.18)
    #     * np.power(ksi_mcc, 0.5)
    #     * np.power(ksi_mcc / ksi_mm10, 0.5)
    #     * Ns
    #     * (H2p / Hs),
    #     5,
    # ) * np.sqrt(N_waves)

    # S_s_combined = np.where(cot_alpha > 3.5, S_s_cot_alpha_3p5, S_s)

    # S = np.where(ksi_mm10 < ksi_mc, S_pl, S_s_combined)

    S = np.where(ksi_mm10 < ksi_mc, S_pl, S_s)

    check_validity_range(
        Hs=Hs,
        H2p=H2p,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        Dn50=Dn50,
        rho_armour=rho_armour,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    H2p: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    c_pl: float = 8.4,
    c_s: float = 1.3,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for rock armour layers with the Modified Van der Meer
    formula by Van Gent et al. (2003).

    For more details see Van Gent et al. (2003), available here https://doi.org/10.1061/40733(147)9 or here
    https://www.researchgate.net/publication/259258688_Stability_of_Rock_Slopes_with_Shallow_Foreshores

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    H2p : float | npt.NDArray[np.float64]
        Wave height exceeded by 2% of waves H2% (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
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
        Coefficient for plunging waves (-), by default 8.4
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.3

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(Hs, Tmm10, cot_alpha)

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl, c_s, P, cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    # Plunging waves
    Dn50_pl = (
        (Hs / Delta)
        * (H2p / Hs)
        * (1 / c_pl)
        * np.power(P, -0.18)
        * np.power(S / np.sqrt(N_waves), -0.2)
        * np.power(ksi_mm10, 0.5)
    )

    # Surging waves
    Dn50_s = (
        (Hs / Delta)
        * (H2p / Hs)
        * (1 / c_s)
        * np.power(P, 0.13)
        * np.power(S / np.sqrt(N_waves), -0.2)
        * (1 / np.sqrt(cot_alpha))
        * np.power(ksi_mm10, -P)
    )

    # Very gentle slopes
    ksi_mcc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl, c_s, P, 3.5
    )

    Dn50_s_cot_alpha_3p5 = (
        (Hs / Delta)
        * (H2p / Hs)
        * (1 / c_pl)
        * np.power(P, -0.18)
        * np.power(S / np.sqrt(N_waves), -0.2)
        * np.power(ksi_mcc, 0.5)
        * np.power(ksi_mcc / ksi_mm10, 0.5)
    )
    Dn50_s_combined = np.where(cot_alpha > 3.5, Dn50_s_cot_alpha_3p5, Dn50_s)

    Dn50 = np.where(ksi_mm10 < ksi_mc, Dn50_pl, Dn50_s_combined)

    check_validity_range(
        Hs=Hs,
        H2p=H2p,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        Dn50=Dn50,
        rho_armour=rho_armour,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    ratio_H2p_Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    P: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    c_pl: float = 8.4,
    c_s: float = 1.3,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for rock armour layers with the Modified Van der Meer
    formula by Van Gent et al. (2003).

    For more details see Van Gent et al. (2003), available here https://doi.org/10.1061/40733(147)9 or here
    https://www.researchgate.net/publication/259258688_Stability_of_Rock_Slopes_with_Shallow_Foreshores

    Parameters
    ----------
    ratio_H2p_Hs : float | npt.NDArray[np.float64]
        Ratio between the Wave height exceeded by 2% of waves H2% and the significant
        wave height Hs, H2% / Hs (-)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
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
        Coefficient for plunging waves (-), by default 8.4
    c_s : float, optional
        Coefficient for surging waves (-), by default 1.3
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_armour)

    ksi_mc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
        c_pl, c_s, P, cot_alpha
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    # Plunging waves
    Hs_pl = np.power(
        c_pl
        * np.power(P, 0.18)
        * np.power(S / np.sqrt(N_waves), 0.2)
        * np.power(1.0 / cot_alpha, -0.5)
        * np.power((2 * np.pi / g) * (1.0 / np.power(Tmm10, 2)), 0.25)
        * Delta
        * Dn50
        * (1 / ratio_H2p_Hs),
        1.0 / 0.75,
    )

    # Surging waves
    Hs_s = np.power(
        c_s
        * np.power(P, -0.13)
        * np.power(S / np.sqrt(N_waves), 0.2)
        * np.power(1.0 / cot_alpha, P)
        * np.power((2 * np.pi / g) * (1.0 / np.power(Tmm10, 2)), -0.5 * P)
        * np.power(1.0 / cot_alpha, -0.5)
        * Delta
        * Dn50
        * (1 / ratio_H2p_Hs),
        1.0 / (1.0 + 0.5 * P),
    )

    # TODO check: it seems that the very gentle slopes (cot_alpha > 3.5) are not implemented in BREAKWAT
    # TODO => what should we do?
    # # Very gentle slopes
    # ksi_mcc = core_physics.calculate_critical_Irribarren_number_ksi_mc(
    #     c_pl, c_s, P, 3.5
    # )

    # Hs_s_cot_alpha_3p5 = np.power(
    #     c_pl
    #     * np.power(P, 0.18)
    #     * np.power(S / np.sqrt(N_waves), 0.2)
    #     * np.power(ksi_mcc, -1.0)
    #     * np.power(1.0 / cot_alpha, 0.5)
    #     * np.power((2 * np.pi / g) * (1.0 / np.power(Tmm10, 2)), -0.25)
    #     * Delta
    #     * Dn50
    #     * (1 / ratio_H2p_Hs),
    #     1.0 / 1.25,
    # )

    ksi_mm10_pl = core_physics.calculate_Irribarren_number_ksi(Hs_pl, Tmm10, cot_alpha)

    ksi_mm10_s = core_physics.calculate_Irribarren_number_ksi(Hs_s, Tmm10, cot_alpha)

    # Hs_s_combined = np.where(cot_alpha > 3.5, Hs_s_cot_alpha_3p5, Hs_s)

    # Hs = np.where((ksi_mm10_pl < ksi_mc) & (ksi_mm10_s < ksi_mc), Hs_pl, Hs_s_combined)

    Hs = np.where((ksi_mm10_pl < ksi_mc) & (ksi_mm10_s < ksi_mc), Hs_pl, Hs_s)

    check_validity_range(
        Hs=Hs,
        Tmm10=Tmm10,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        P=P,
        Dn50=Dn50,
        rho_armour=rho_armour,
    )
    return Hs
