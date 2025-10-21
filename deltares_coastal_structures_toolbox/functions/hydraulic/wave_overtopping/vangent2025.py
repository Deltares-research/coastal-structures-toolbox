# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
) -> None:

    # TODO adjust validity ranges to based on the paper
    # TODO check of alleen ranges nodig zijn uit table 1, of is dat maar een deel van de data?
    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0", "TAW (2002)", smm10, 0.0, 0.07
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_smm10 = core_physics.calculate_Iribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "Iribarren number ksi_m-1,0",
            "TAW (2002)",
            ksi_smm10,
            0.0,
            7.0,
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
        and not np.any(np.isnan(gamma_b))
    ):
        ksi_smm10 = core_physics.calculate_Iribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "gamma_b * ksi_m-1,0",
            "TAW (2002)",
            gamma_b * ksi_smm10,
            0.5,
            10.0,
        )

    if not np.any(np.isnan(cot_alpha_down)):
        core_utility.check_variable_validity_range(
            "Cotangent alpha lower slope", "TAW (2002)", cot_alpha_down, 1.0, 7.0
        )

    if not np.any(np.isnan(cot_alpha_up)):
        core_utility.check_variable_validity_range(
            "Cotangent alpha upper slope", "TAW (2002)", cot_alpha_up, 1.0, 7.0
        )

    if not np.any(np.isnan(beta)):
        core_utility.check_variable_validity_range(
            "Incident wave angle beta", "TAW (2002)", beta, 0.0, 90.0
        )

    if not np.any(np.isnan(gamma_b)):
        core_utility.check_variable_validity_range(
            "Influence factor berm gamma_b", "TAW (2002)", gamma_b, 0.4, 1.0
        )

    if not np.any(np.isnan(gamma_f)):
        core_utility.check_variable_validity_range(
            "Influence factor roughness gamma_f", "TAW (2002)", gamma_f, 0.4, 1.0
        )

    if not np.any(np.isnan(gamma_beta)):
        core_utility.check_variable_validity_range(
            "Influence factor oblique waves gamma_beta",
            "TAW (2002)",
            gamma_beta,
            0.4,
            1.0,
        )

    if not np.any(np.isnan(gamma_v)):
        core_utility.check_variable_validity_range(
            "Influence factor vertical wall gamma_v", "TAW (2002)", gamma_v, 0.4, 1.0
        )

    return


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    g: float = 9.81,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the mean wave overtopping discharge q with the Van Gent et al. (2025) formula.

    The mean wave overtopping discharge q (m^3/s/m) is calculated using the Van Gent et al. (2025) formula.
    Here, eq. B1 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    design_calculation : bool, optional
        Use the 95% confidence level for design calculations, by default True
    include_influence_wind : bool, optional
        Include influence of wind on wave overtopping, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        Mean wave overtopping discharge q (m^3/s/m) and a boolean indicating
        whether the maximum value formula was used
    """

    q_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        Rc=Rc,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=db,
        Dn50=Dn50,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
        design_calculation=design_calculation,
        include_influence_wind=include_influence_wind,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless mean wave overtopping discharge q with the Van Gent et al. (2025) formula.

    The dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-) is calculated using the
    Van Gent et al. (2025) formula. Here, eq. B1 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    design_calculation : bool, optional
        Use the 95% confidence level for design calculations, by default True
    include_influence_wind : bool, optional
        Include influence of wind on wave overtopping, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
        and a boolean indicating whether the maximum value formula was used

    Raises
    ------
    ValueError
        Raise an error if gamma_f is not provided and Dn50 is not provided so it cannot be calculated.
    """

    if np.isnan(gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    if np.isnan(gamma_b):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)
        gamma_b = calculate_influence_berm_gamma_b(
            Hm0=Hm0,
            smm10=smm10,
            Ac=Ac,
            B_berm=B_berm,
            BL=Ac - db,
        )

    if np.isnan(gamma_f):
        if np.isnan(Dn50):
            raise ValueError("Either gamma_f or Dn50 should be provided")
        smm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)
        gamma_f = calculate_influence_friction_gamma_f(Dn50=Dn50, Hm0=Hm0, smm10=smm10)

    q_diml_eqB1 = (
        6.8
        * np.power(cot_alpha, -1.0)
        * np.exp(
            -5.0
            * (Rc - 0.4 * Hm0_swell)
            / (gamma_f * gamma_b * gamma_v * gamma_beta * ksi_mm10 * Hm0)
        )
    )

    q_diml_max = q_diml_max_equation(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        ksi_mm10=ksi_mm10,
        cot_alpha=cot_alpha,
        Rc=Rc,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
    )

    q_diml = np.min([q_diml_eqB1, q_diml_max], axis=0)
    max_reached = np.min([q_diml_eqB1, q_diml_max], axis=0) == q_diml_max

    if include_influence_wind:
        gamma_w = calculate_influence_wind_gamma_w(
            Rc=Rc,
            Ac=Ac,
            Hm0=Hm0,
            q_diml=q_diml,
        )
        q_diml = q_diml * gamma_w

    if design_calculation:
        q_diml = np.power(q_diml, 0.857)

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return q_diml, max_reached


def q_diml_max_equation(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    gamma_v: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64],
    c2: float = 0.8,
    c3: float = -2.5,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum dimensionless mean wave overtopping discharge q with the Van Gent et al. (2025) formula.

    The maximum value for the dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-) is calculated
    using the Van Gent et al. (2025) formula. Here, eq. B2 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    ksi_mm10 : float | npt.NDArray[np.float64]
        The Iribarren number based on the spectral wave period Tm-1,0 (-)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_b : float | npt.NDArray[np.float64]
        Influence factor for a berm (-)
    gamma_v : float | npt.NDArray[np.float64]
        Influence factor for a crest wall (-)
    gamma_beta : float | npt.NDArray[np.float64]
        Influence factor for oblique wave incidence (-)
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.8
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default -2.5

    Returns
    -------
    float | npt.NDArray[np.float64]
        Maximum value of the dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
    """

    q_diml_max = (
        c2
        * np.power(cot_alpha, -1.0)
        * np.exp(
            c3
            * (Rc - 0.4 * Hm0_swell)
            / (
                gamma_f
                * gamma_b
                * gamma_v
                * gamma_beta
                * np.power(ksi_mm10, 0.24)
                * Hm0
            )
        )
    )

    return q_diml_max


def calculate_influence_friction_gamma_f(
    Dn50: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    smm10: float | npt.NDArray[np.float64],
    c_f1: float = 0.70,
    c_f2: float = 0.05,
    smm10_lim: float = 0.012,
) -> float | npt.NDArray[np.float64]:
    """Calculate influence factor for surface roughness gamma_f

    The influence factor gamma_f is determined using Van Gent et al. (2025) eq. B3a for sm-1,0 >= smm10__lim
     (0.012 by default) and eq. B3b for sm-1,0 < smm10__lim.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal rock diameter (m)
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    smm10 : float | npt.NDArray[np.float64]
        Deep water wave steepness based on the spectral wave period Tm-1,0 (-)
    c_f1 : float, optional
        Coefficient in the gamma_f formula, by default 0.70
    c_f2 : float, optional
        Coefficient in the gamma_f formula, by default 0.05
    smm10_lim : float, optional
        Limit for the deep water wave steepness (-), by default 0.012

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for surface roughness gamma_f (-)
    """

    gamma_f1 = 1.0 - c_f1 * np.power(Dn50 / Hm0, c_f2)

    gamma_f = np.where(
        smm10 >= smm10_lim,
        gamma_f1,
        gamma_f1 + 12.0 * (smm10_lim - smm10) * (1.0 - gamma_f1),
    )
    return gamma_f


def calculate_influence_berm_gamma_b(
    Hm0: float | npt.NDArray[np.float64],
    smm10: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    BL: float | npt.NDArray[np.float64],
    c_b1: float = 18.0,
    c_b2: float = 1.3,
    c_b3: float = 0.34,
    c_b4: float = 0.2,
) -> float | npt.NDArray[np.float64]:
    """Calculate influence factor for a berm gamma_b

    The influence factor gamma_b is determined using Van Gent et al. (2025) eq. B4.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    smm10 : float | npt.NDArray[np.float64]
        The deep water wave steepness based on the spectral wave period Tm-1,0 (-)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    BL : float | npt.NDArray[np.float64]
        Vertical distance of the berm relative to the crest of the armour Ac (m)
    c_b1 : float, optional
        Coefficient in the gamma_b formula, by default 18.0
    c_b2 : float, optional
        Coefficient in the gamma_b formula, by default 1.3
    c_b3 : float, optional
        Coefficient in the gamma_b formula, by default 0.34
    c_b4 : float, optional
        Coefficient in the gamma_b formula, by default 0.2

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a berm gamma_b (-)
    """

    gamma_b = 1.0 - c_b1 * np.power(smm10 * B_berm / Hm0, c_b2) * (
        1.0 - c_b3 * np.power(BL / (smm10 * Ac), c_b4)
    )

    return gamma_b


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_beta: float = 0.35,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor gamma_beta is determined using Van Gent et al. (2025) eq. B6

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    c_beta : float, optional
        Coefficient in the gamma_beta formula, by default 0.35

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    # TODO check with Marcel: what to do for Hm0-deep/htoe >= 1.0?

    gamma_beta = (1 - c_beta) * np.power(np.cos(np.radians(beta)), 2) + c_beta

    return gamma_beta


def calculate_influence_crest_wall_gamma_v(
    cot_alpha: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    c_v: float = 0.45,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for a crest wall gamma_v

    The influence factor gamma_v is determined using Van Gent et al. (2025) eq. B5a for cot_alpha <= 4.0
    and B5b for cot_alpha > 4.0.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    c_v : float, optional
        Coefficient in the gamma_v formula, by default 0.45

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a crest wall gamma_v (-)
    """

    gamma_v = np.where(
        cot_alpha <= 4.0,
        1.0 + c_v * (Rc - Ac) / Rc,
        1.0 + 0.1125 * cot_alpha * (Rc - Ac) / Rc,
    )

    return gamma_v


def calculate_influence_wind_gamma_w(
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    q_diml: float | npt.NDArray[np.float64],
    c_w1: float = 0.075,
    c_w2: float = -0.3,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for wind gamma_w

    The influence factor gamma_w is determined using Van Gent et al. (2025) eq. B7

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    hc : float | npt.NDArray[np.float64]
        protruding part of a crest wall, hc = Rc - Ac (m)
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    q_diml : float | npt.NDArray[np.float64]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for wind gamma_w (-)
    """

    hc = Rc - Ac
    gamma_w = 1.0 + c_w1 * (hc / Hm0) * np.power(q_diml, c_w2)

    return gamma_w


def calculate_crest_freeboard_Rc(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the crest freeboard Rc with the Van Gent et al. (2025) formula.

    The crest freeboard Rc (m) is calculated using the Van Gent et al. (2025) formula.
    Here, eq. B1 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    design_calculation : bool, optional
        Use the 95% confidence level for design calculations, by default True
    include_influence_wind : bool, optional
        Include influence of wind on wave overtopping, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The crest freeboard of the structure Rc (m) and a boolean indicating
        whether the maximum value formula was used
    """

    Rc_diml, max_reached = calculate_dimensionless_crest_freeboard(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        q=q,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=db,
        Dn50=Dn50,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
        design_calculation=design_calculation,
        include_influence_wind=include_influence_wind,
    )

    Rc = Rc_diml * Hm0

    return Rc, max_reached


def calculate_dimensionless_crest_freeboard(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    g: float = 9.81,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless crest freeboard Rc/Hm0 with the Van Gent et al. (2025) formula.

    The dimensionless crest freeboard Rc/Hm0 (-) is calculated using the Van Gent et al. (2025) formula.
    Here, eq. B1 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81
    design_calculation : bool, optional
        Use the 95% confidence level for design calculations, by default True
    include_influence_wind : bool, optional
        Include influence of wind on wave overtopping, by default False
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tolerance : float, optional
        Tolerance for convergence of the iterative solution, by default 1e-5

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The dimensionless crest freeboard of the structure Rc/Hm0 (-) and a boolean indicating
        whether the maximum value formula was used

    Raises
    ------
    ValueError
        Raise an error if gamma_f is not provided and Dn50 is not provided so it cannot be calculated.
    """

    if np.isnan(gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    ksi_mm10 = core_physics.calculate_Iribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    if np.isnan(gamma_b):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        gamma_b = calculate_influence_berm_gamma_b(
            Hm0=Hm0,
            smm10=smm10,
            Ac=Ac,
            B_berm=B_berm,
            BL=Ac - db,
        )

    if np.isnan(gamma_f):
        if np.isnan(Dn50):
            raise ValueError("Either gamma_f or Dn50 should be provided")
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        gamma_f = calculate_influence_friction_gamma_f(Dn50=Dn50, Hm0=Hm0, smm10=smm10)

    q_diml = q / np.sqrt(g * Hm0**3)

    if design_calculation:
        q_diml = np.power(q_diml, 1.0 / 0.857)

    Rc_diml = (
        np.log((1.0 / 6.8) * cot_alpha * q_diml)
        * (-1.0 / 5.0)
        * gamma_f
        * gamma_b
        * gamma_v
        * gamma_beta
        * ksi_mm10
        + 0.4 * Hm0_swell / Hm0
    )

    if include_influence_wind:
        Rc_diff = np.inf
        n_iter = 0
        while np.max(Rc_diff) > tolerance and n_iter < max_iter:
            n_iter += 1

            gamma_w = calculate_influence_wind_gamma_w(
                Rc=Rc_diml * Hm0,
                Ac=Ac,
                Hm0=Hm0,
                q_diml=q_diml,
            )
            q_diml_iter = q_diml / gamma_w

            Rc_diml_prev = Rc_diml

            Rc_diml = (
                np.log((1.0 / 6.8) * cot_alpha * q_diml_iter)
                * (-1.0 / 5.0)
                * gamma_f
                * gamma_b
                * gamma_v
                * gamma_beta
                * ksi_mm10
                + 0.4 * Hm0_swell / Hm0
            )

            Rc_diff = np.abs(Rc_diml - Rc_diml_prev)

    Rc_diml_max = Rc_diml_max_equation(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        q=q,
        cot_alpha=cot_alpha,
        ksi_mm10=ksi_mm10,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
    )

    Rc_diml = np.min([Rc_diml, Rc_diml_max], axis=0)
    max_reached = np.min([Rc_diml, Rc_diml_max], axis=0) == Rc_diml_max

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return Rc_diml, max_reached


def Rc_diml_max_equation(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    gamma_v: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64],
    c2: float = 0.8,
    c3: float = -2.5,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum dimensionless crest freeboard Rc/Hm0 with the Van Gent et al. (2025) formula.

    The maximum value for the dimensionless crest freeboard Rc/Hm0 (-) is calculated
    using the Van Gent et al. (2025) formula. Here, eq. B2 from Van Gent et al. (2025) is implemented.

    For more details, see: https://doi.org/10.59490/jchs.2025.0048

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Hm0_swell : float | npt.NDArray[np.float64]
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    ksi_mm10 : float | npt.NDArray[np.float64]
        _description_
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_b : float | npt.NDArray[np.float64]
        Influence factor for a berm (-)
    gamma_v : float | npt.NDArray[np.float64]
        Influence factor for a crest wall (-)
    gamma_beta : float | npt.NDArray[np.float64]
        Influence factor for oblique wave incidence (-)
    c2 : float, optional
        _description_, by default 0.8
    c3 : float, optional
        _description_, by default -2.5
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The maximum value of the dimensionless crest freeboard of the structure Rc/Hm0 (-)
    """

    Rc_diml_max = (
        np.log((1.0 / c2) * cot_alpha * q / np.sqrt(g * Hm0**3))
        * (1.0 / c3)
        * gamma_f
        * gamma_b
        * gamma_v
        * gamma_beta
        * np.power(ksi_mm10, 0.24)
        + 0.4 * Hm0_swell / Hm0
    )

    return Rc_diml_max
