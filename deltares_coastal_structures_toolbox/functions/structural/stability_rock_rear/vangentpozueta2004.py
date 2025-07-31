# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.vangent2002_velocity as vg2002_velocity
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


def check_validity_range(
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Rc_rear: float | npt.NDArray[np.float64] = np.nan,
    cot_phi: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float | npt.NDArray[np.float64] = np.nan,
    Bc: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    z1p: float | npt.NDArray[np.float64] = np.nan,
    S: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
) -> None:
    """Check the parameter values vs the validity range as defined in Van Gent & Pozueta (2004).

    For all parameters supplied, their values are checked versus the range of test conditions specified in Table 2
    (Van Gent & Pozueta, 2004). When parameters are nan (by default), they are not checked.

    For more details see Van Gent & Pozueta (2004), available here https://doi.org/10.1142/9789812701916_0281 or here:
    https://www.researchgate.net/publication/259260766_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES

    Parameters
    ----------
    Rc : float | npt.NDArray[np.float64], optional
        Freeboard of the structure (m), by default np.nan
    Rc_rear : float | npt.NDArray[np.float64], optional
        Vertical distance between still-water level and the crest at the rear side (m), by default np.nan
    cot_phi : float | npt.NDArray[np.float64], optional
        Cotangent of the rear-side slope of the structure (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    rho_rock : float | npt.NDArray[np.float64], optional
        Rock density (kg/m^3), by default np.nan
    rho_water : float | npt.NDArray[np.float64], optional
        Water density (kg/m^3), by default np.nan
    Bc : float | npt.NDArray[np.float64], optional
        Width of the crest of the structure (m), by default np.nan
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    z1p : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    S : float | npt.NDArray[np.float64], optional
        Damage number (-), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    """

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hs, Tmm10)
        core_utility.check_variable_validity_range(
            "sm-1,0", "Van Gent & Pozueta (2004)", smm10, 0.019, 0.036
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "N_waves", "Van Gent & Pozueta (2004)", N_waves, 0, 4000
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc/Hs", "Van Gent & Pozueta (2004)", Rc / Hs, 0.3, 2.0
        )

    if not np.any(np.isnan(Rc_rear)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc_rear/Hs", "Van Gent & Pozueta (2004)", Rc_rear / Hs, 0.3, 6.0
        )

    if not np.any(np.isnan(Bc)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "B_c/Hs", "Van Gent & Pozueta (2004)", Bc / Hs, 1.3, 1.6
        )

    if (
        not np.any(np.isnan(Rc))
        and not np.any(np.isnan(Hs))
        and not np.any(np.isnan(z1p))
        and not np.any(np.isnan(gamma_f))
    ):
        core_utility.check_variable_validity_range(
            "(z1% - Rc)/(gamma_f*Hs)",
            "Van Gent & Pozueta (2004)",
            (z1p - Rc) / (gamma_f * Hs),
            0.0,
            1.4,
        )

    if (
        not np.any(np.isnan(Hs))
        and not np.any(np.isnan(Dn50))
        and not np.any(np.isnan(rho_rock))
        and not np.any(np.isnan(rho_water))
    ):
        Ns = core_physics.calculate_stability_number_Ns(
            H=Hs, D=Dn50, rho_rock=rho_rock, rho_water=rho_water
        )

        core_utility.check_variable_validity_range(
            "Hs/(Delta*Dn50_rear)",
            "Van Gent & Pozueta (2004)",
            Ns,
            5.5,
            8.5,
        )

    if not np.any(np.isnan(cot_phi)):
        core_utility.check_variable_validity_range(
            "cot phi", "Van Gent & Pozueta (2004)", cot_phi, 2.0, 4.0
        )

    if not np.any(np.isnan(S)):
        core_utility.check_variable_validity_range(
            "S", "Van Gent & Pozueta (2004)", S, 2.0, 50.0
        )

    return


def calculate_damage_number_S(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
    cs: float = np.power(0.008, 6.0),
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock at the rear side of a rubble mound structure following
    Van Gent & Pozueta (2004).

    For more details see Van Gent & Pozueta (2004), available here https://doi.org/10.1142/9789812701916_0281 or here:
    https://www.researchgate.net/publication/259260766_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    cot_phi : float | npt.NDArray[np.float64]
        Cotangent of the rear-side slope of the structure (-)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_Crest : float | npt.NDArray[np.float64]
        Influence factor for surface roughness on the crest of the structure (-)
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Rc_rear : float | npt.NDArray[np.float64]
        Vertical distance between still-water level and the crest at the rear side (m)
    Bc : float | npt.NDArray[np.float64]
        Width of the crest of the structure (m)
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    rho_rock : float | npt.NDArray[np.float64], optional
        Rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    cs : float, optional
        Coefficient, by default np.power(0.008, 6.0)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)
    """

    z1p = vangent2001.calculate_wave_runup_height_z1p(
        Hs=Hs, Tmm10=Tmm10, gamma=gamma_f, cot_alpha=cot_alpha
    )

    u1p = vg2002_velocity.calculate_maximum_wave_overtopping_velocity_uXp(
        Hs=Hs, zXp=z1p, Rc=Rc, Bc=Bc, gamma_f=gamma_f, gamma_f_Crest=gamma_f_Crest
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_rock)

    S = (
        cs
        * np.power((u1p * Tmm10 / (np.sqrt(Delta) * Dn50)), 6.0)
        * np.power(cot_phi, -2.5)
        * (1 + 10 * np.exp(-Rc_rear / Hs))
        * np.sqrt(N_waves)
    )

    check_validity_range(
        Rc=Rc,
        Rc_rear=Rc_rear,
        cot_phi=cot_phi,
        gamma_f=gamma_f,
        Dn50=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        Bc=Bc,
        Hs=Hs,
        Tmm10=Tmm10,
        z1p=z1p,
        S=S,
        N_waves=N_waves,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    rho_water: float = 1025.0,
    cs: float = np.power(0.008, 6.0),
) -> float | npt.NDArray[np.float64]:
    """Calculate the minimum Dn50 for armour at the rear side of a rubble mound structure following
    Van Gent & Pozueta (2004).

    For more details see Van Gent & Pozueta (2004), available here https://doi.org/10.1142/9789812701916_0281 or here:
    https://www.researchgate.net/publication/259260766_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    cot_phi : float | npt.NDArray[np.float64]
        Cotangent of the rear-side slope of the structure (-)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_Crest : float | npt.NDArray[np.float64]
        Influence factor for surface roughness on the crest of the structure (-)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Rc_rear : float | npt.NDArray[np.float64]
        Vertical distance between still-water level and the crest at the rear side (m)
    Bc : float | npt.NDArray[np.float64]
        Width of the crest of the structure (m)
    rho_rock : float | npt.NDArray[np.float64]
        Rock density (kg/m^3)
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    cs : float, optional
        Coefficient, by default np.power(0.008, 6.0)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The median nominal rock diameter Dn50 (m)
    """

    z1p = vangent2001.calculate_wave_runup_height_z1p(
        Hs=Hs, Tmm10=Tmm10, gamma=gamma_f, cot_alpha=cot_alpha
    )

    u1p = vg2002_velocity.calculate_maximum_wave_overtopping_velocity_uXp(
        Hs=Hs, zXp=z1p, Rc=Rc, Bc=Bc, gamma_f=gamma_f, gamma_f_Crest=gamma_f_Crest
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = (
        np.power(cs, 1.0 / 6.0)
        * np.power(S / np.sqrt(N_waves), -(1.0 / 6.0))
        * (u1p * Tmm10 / np.sqrt(Delta))
        * np.power(cot_phi, -(2.5 / 6.0))
        * np.power(1 + 10 * np.exp(-Rc_rear / Hs), (1.0 / 6.0))
    )

    check_validity_range(
        Rc=Rc,
        Rc_rear=Rc_rear,
        cot_phi=cot_phi,
        gamma_f=gamma_f,
        Dn50=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        Bc=Bc,
        Hs=Hs,
        Tmm10=Tmm10,
        z1p=z1p,
        S=S,
        N_waves=N_waves,
    )

    return Dn50


def calculate_maximum_significant_wave_height_Hs(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_Crest: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    Bc: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
    rho_water: float = 1025.0,
    cs: float = np.power(0.008, 6.0),
    tolerance: float = 1e-4,
    max_iterations: int = 10000,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum allowable Hs for armour at the rear side of a rubble mound structure following
    Van Gent & Pozueta (2004).

    For more details see Van Gent & Pozueta (2004), available here https://doi.org/10.1142/9789812701916_0281 or here:
    https://www.researchgate.net/publication/259260766_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    cot_phi : float | npt.NDArray[np.float64]
        Cotangent of the rear-side slope of the structure (-)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_Crest : float | npt.NDArray[np.float64]
        Influence factor for surface roughness on the crest of the structure (-)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Rc_rear : float | npt.NDArray[np.float64]
        Vertical distance between still-water level and the crest at the rear side (m)
    Bc : float | npt.NDArray[np.float64]
        Width of the crest of the structure (m)
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    rho_rock : float | npt.NDArray[np.float64], optional
        Rock density (kg/m^3), by default np.nan
    rho_water : float, optional
        Water density (kg/m^3), by default 1025.0
    cs : float, optional
        Coefficient, by default np.power(0.008, 6.0)
    tolerance : float, optional
        Tolerance in the iteration to Hs (m), by default 1e-4
    max_iterations : int, optional
        Maximum number of iterations, by default 10000

    Returns
    -------
    float | npt.NDArray[np.float64]
        The maximum allowable significant wave height Hs (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_rock)

    Hs_i1 = (Rc + 0.5) / (2.552 * gamma_f)  # (Rc + 0.01) / (2.552 * gamma_f)
    Hs_i0 = Hs_i1 + np.inf
    iteration = 0

    while iteration < max_iterations and np.abs(Hs_i1 - Hs_i0) > tolerance:

        iteration += 1
        Hs_i0 = Hs_i1

        # calculate u1% using inverted vGent&Pozueta (2004) formula
        u1p = _invert_for_u1p(
            cot_phi=cot_phi,
            S=S,
            Hs=Hs_i0,
            Tmm10=Tmm10,
            Rc_rear=Rc_rear,
            N_waves=N_waves,
            Dn50=Dn50,
            Delta=Delta,
            cs=cs,
        )

        # calculate z1% using inverted u1% formula
        z1p = vg2002_velocity._invert_for_zXp(
            Hs=Hs_i0,
            uXp=u1p,
            Rc=Rc,
            Bc=Bc,
            gamma_f=gamma_f,
            gamma_f_Crest=gamma_f_Crest,
        )

        # calculate next Hs iteration using inverted z1% formula
        Hs_i1 = vangent2001._invert_for_Hs(
            Hs_i0=Hs_i0,
            z1p=z1p,
            Tmm10=Tmm10,
            gamma=gamma_f,
            cot_alpha=cot_alpha,
        )

    if iteration >= max_iterations:
        warnings.warn("Maximum number of iterations reached, convergence not achieved")

    Hs = Hs_i1

    check_validity_range(
        Rc=Rc,
        Rc_rear=Rc_rear,
        cot_phi=cot_phi,
        gamma_f=gamma_f,
        Dn50=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        Bc=Bc,
        Hs=Hs,
        Tmm10=Tmm10,
        z1p=z1p,
        S=S,
        N_waves=N_waves,
    )

    return Hs


def _invert_for_u1p(
    cot_phi: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc_rear: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    Delta: float | npt.NDArray[np.float64] = np.nan,
    cs: float = np.power(0.008, 6.0),
) -> float | npt.NDArray[np.float64]:

    u1p = (
        np.power(
            (1.0 / cs)
            * (S / np.sqrt(N_waves))
            * np.power(cot_phi, 2.5)
            * (1.0 / (1 + 10 * np.exp(-Rc_rear / Hs))),
            1.0 / 6.0,
        )
        * np.sqrt(Delta)
        * Dn50
        / Tmm10
    )

    return u1p
