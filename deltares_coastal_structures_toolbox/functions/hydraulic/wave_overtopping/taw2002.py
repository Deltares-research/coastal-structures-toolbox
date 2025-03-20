# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as taw2002


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
    """_summary_

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64], optional
        Spectral significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm, by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a wave wall (-), by default np.nan
    """

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
        ksi_smm10 = core_physics.calculate_Irribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "Irribarren number ksi_m-1,0",
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
        ksi_smm10 = core_physics.calculate_Irribarren_number_ksi(Hm0, Tmm10, cot_alpha)
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
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the mean wave overtopping discharge q with the TAW (2002) formula.

    TODO: fill out properly

    TAW (2002) formula for MEAN through data (eqs 24 and 25 from TAW: best fit, not for design)
    output = DIMENSIONLESS mean overtopping discharge (q/sqrt(g*Hm0^3))

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    cot_alpha_down : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    cot_alpha_up : float | npt.NDArray[np.float64]
        Cotangent of the upper part of the front-side slope of the structure (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a wave wall (-), by default 1.0
    sigma : float | npt.NDArray[np.float64], optional
        Apply sigma standard deviations to the best fit coefficients, by default 0
    use_best_fit : bool, optional
        Use the coefficients of the best fit instead of the more conservative design variant, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        Mean wave overtopping discharge q (m^3/s/m) and a boolean indicating
        whether the maximum value formula was used
    """

    q_TAW_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        sigma=sigma,
        use_best_fit=use_best_fit,
    )
    q_TAW = q_TAW_diml * np.sqrt(9.81 * Hm0**3)

    return q_TAW, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """_summary_

    TODO: fill out properly

    TAW (2002) formula for MEAN through data (eqs 24 and 25 from TAW: best fit, not for design)
    output = DIMENSIONLESS mean overtopping discharge (q/sqrt(g*Hm0^3))

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    cot_alpha_down : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    cot_alpha_up : float | npt.NDArray[np.float64]
        Cotangent of the upper part of the front-side slope of the structure (-)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a wave wall (-), by default 1.0
    sigma : float | npt.NDArray[np.float64], optional
        Apply sigma standard deviations to the best fit coefficients, by default 0
    use_best_fit : bool, optional
        Use the coefficients of the best fit instead of the more conservative design variant, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
        and a boolean indicating whether the maximum value formula was used
    """

    if use_best_fit:
        c1 = 4.75
        c2 = 2.6
    else:
        c1 = 4.3
        c2 = 2.3

    if sigma == 0:
        cor1 = 0
        cor2 = 0
    else:
        if use_best_fit:
            cor1 = 0.5 * sigma
            cor2 = 0.35 * sigma
        else:
            warnings.warn(
                (
                    "Sigma is only applicable to the best fit coefficients! The design values of the coefficients "
                    "alreaddy account for uncertainty with conservative coefficient values."
                )
            )

    cot_alpha_average, _, L_berm = taw2002.determine_average_slope(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
    )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        Hm0, Tmm10, cot_alpha_average
    )

    gamma_b = iteration_procedure_gamma_b(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_average=cot_alpha_average,
        B_berm=B_berm,
        L_berm=L_berm,
        db=db,
        gamma_f=gamma_f,
    )

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta)

    q_diml_eq24 = (
        (0.067 / np.sqrt(1.0 / cot_alpha_average))
        * ksi_mm10
        * gamma_b
        * np.exp(
            -1
            * (c1 + cor1)
            * (Rc / Hm0)
            * (1 / (ksi_mm10 * gamma_b * gamma_f * gamma_beta * gamma_v))
        )
    )
    q_diml_eq25 = 0.2 * np.exp(
        -1 * (c2 + cor2) * (Rc / Hm0) * (1 / (gamma_f * gamma_beta))
    )

    q_diml_TAW = np.min([q_diml_eq24, q_diml_eq25], axis=0)
    max_reached = np.min([q_diml_eq24, q_diml_eq25], axis=0) == q_diml_eq25

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha_average,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return q_diml_TAW, max_reached


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    TODO: fill out properly
    TAW eq 9

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    beta_calc = np.where(beta < 0, np.abs(beta), beta)
    beta_calc = np.where(beta_calc > 80, 80, beta_calc)

    gamma_beta = 1 - 0.0033 * beta_calc
    return gamma_beta


def calculate_influence_berm_gamma_b(
    Hm0: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    L_berm: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    TODO: fill out properly
    TAW eq 10, 11 & 12

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    z2p : float | npt.NDArray[np.float64]
        Wave runup height exceeded by 2% of waves z2% (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    L_berm : float | npt.NDArray[np.float64]
        Berm length of the structure (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a berm gamma_b (-)
    """

    x = np.where(db < 0, z2p, 2 * Hm0)
    rdh = np.where(
        (db > 2 * Hm0) | (db < -z2p), 1.0, 0.5 - 0.5 * np.cos(np.pi * db / x)
    )
    rB = B_berm / L_berm

    gamma_b = np.max([np.min([1.0 - rB * (1.0 - rdh), 1.0]), 0.6])
    return gamma_b


def iteration_procedure_gamma_b(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_average: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    L_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """_summary_

    TODO: fill out properly
    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    cot_alpha_average : float | npt.NDArray[np.float64]
        Cotangent of the average front-side slope of the structure (-)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    L_berm : float | npt.NDArray[np.float64]
        Berm length of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a berm gamma_b (-)
    """

    if B_berm == 0.0 or L_berm == 0.0:
        gamma_b = 1.0
    else:
        z2p_i1 = taw2002.calculate_wave_runup_height_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha=cot_alpha_average,
            gamma_b=1.0,
            gamma_f=gamma_f,
        )

        gamma_b_runup = calculate_influence_berm_gamma_b(
            Hm0=Hm0, z2p=z2p_i1, db=db, B_berm=B_berm, L_berm=L_berm
        )

        z2p_i2 = taw2002.calculate_wave_runup_height_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha=cot_alpha_average,
            gamma_b=gamma_b_runup,
            gamma_f=gamma_f,
        )

        gamma_b = calculate_influence_berm_gamma_b(
            Hm0=Hm0, z2p=z2p_i2, db=db, B_berm=B_berm, L_berm=L_berm
        )
    return gamma_b


def calculate_influence_vertical_wall_gamma_v(
    alpha_wall_deg: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    TODO: fill out properly
    TAW eq 16 (alpha_wall_deg in degrees!)

    Parameters
    ----------
    alpha_wall_deg : float | npt.NDArray[np.float64]
        Slope of the (near) vertical wave wall (degrees)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a wave wall gamma_v (-)
    """

    gamma_v = np.where(alpha_wall_deg < 45, 1.0, 1.35 - 0.0078 * alpha_wall_deg)
    return gamma_v
