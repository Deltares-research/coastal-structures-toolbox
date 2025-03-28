# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

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
    """Check the parameter values vs the validity range of the TAW (2002) wave overtopping formula

    For all parameters supplied, their values are checked versus the range of test conditions specified by
    TAW (2002) in the table on pages 39-40. When parameters are nan (by default), they are not checked.

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

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
    Rc: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the mean wave overtopping discharge q with the TAW (2002) formula.

    The mean wave overtopping discharge q (m^3/s/m) is calculated using the TAW (2002) formulas. Here eqs. 22 and 23
    from TAW (2002) are implemented for design calculations and eqs. 24 and 25 for best fit calculations (using the
    option best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm gamma_b (-), by default np.nan
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

    q_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_beta=gamma_beta,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        sigma=sigma,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(9.81 * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless mean wave overtopping discharge q with the TAW (2002) formula.

    The dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-) is calculated using the TAW (2002) formulas.
    Here eqs. 22 and 23 from TAW (2002) are implemented for design calculations and eqs. 24 and 25 for best fit
    calculations (using the option best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm gamma_b (-), by default np.nan
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

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_taw2002.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
        )

        cot_alpha = wave_runup_taw2002.determine_average_slope(
            Hm0=Hm0,
            z2p=z2p_for_slope,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    if np.isnan(gamma_b):
        L_berm = wave_runup_taw2002.calculate_berm_length(
            Hm0=Hm0,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
        )

        gamma_b = wave_runup_taw2002.iteration_procedure_gamma_b(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_average=cot_alpha,
            B_berm=B_berm,
            L_berm=L_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
        )

    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    q_diml_eq24 = (
        (0.067 / np.sqrt(1.0 / cot_alpha))
        * ksi_mm10
        * gamma_b
        * np.exp(
            -1.0
            * (c1 + cor1)
            * (Rc / Hm0)
            * (1.0 / (ksi_mm10 * gamma_b * gamma_f_adj * gamma_beta * gamma_v))
        )
    )
    q_diml_eq25 = 0.2 * np.exp(
        -1.0 * (c2 + cor2) * (Rc / Hm0) * (1.0 / (gamma_f_adj * gamma_beta))
    )

    q_diml = np.min([q_diml_eq24, q_diml_eq25], axis=0)
    max_reached = np.min([q_diml_eq24, q_diml_eq25], axis=0) == q_diml_eq25

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return q_diml, max_reached


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_gamma_beta: float = 0.0033,
    max_angle: float = 80.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor for oblique wave incidence gamma_beta (-) on wave overtopping is calculated using
    eq. 9 from TAW (2002). Note that this uses the implementation for wave runup, but changes the coefficient
    to the value used for wave overtopping

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    c_gamma_beta : float, optional
        Coefficient for wave overtopping, by default 0.0033
    max_angle : float, optional
        Maximum angle of wave incidence, by default 80.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """
    gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
        beta=beta, c_gamma_beta=c_gamma_beta, max_angle=max_angle
    )

    return gamma_beta


def calculate_influence_wave_wall_gamma_v(
    alpha_wall_deg: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for wave walls gamma_v

    The influence factor for wave walls gamma_v (-) on wave overtopping is calculated using eq. 16 from TAW (2002).

    Parameters
    ----------
    alpha_wall_deg : float | npt.NDArray[np.float64]
        Slope of the (near) vertical wave wall (degrees)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for a wave wall gamma_v (-)
    """

    gamma_v = np.where(alpha_wall_deg <= 45, 1.0, 1.35 - 0.0078 * alpha_wall_deg)

    return gamma_v


def calculate_crest_freeboard_Rc(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the crest freeboard Rc with the TAW (2002) formula.

    The crest freeboard Rc (m) is calculated using the TAW (2002) formulas. Here eqs. 22 and 23 from TAW (2002)
    are implemented for design calculations and eqs. 24 and 25 for best fit calculations (using the option
    best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm gamma_b (-), by default np.nan
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
        The crest freeboard of the structure Rc (m)
    """

    Rc_diml, max_reached = calculate_dimensionless_crest_freeboard(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        q=q,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        sigma=sigma,
        use_best_fit=use_best_fit,
    )

    Rc = Rc_diml * Hm0

    return Rc, max_reached


def calculate_dimensionless_crest_freeboard(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    sigma: float | npt.NDArray[np.float64] = 0,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless crest freeboard Rc/Hm0 with the TAW (2002) formula.

    The dimensionless crest freeboard Rc/Hm0 (-) is calculated using the TAW (2002) formulas. Here eqs. 22
    and 23 from TAW (2002) are implemented for design calculations and eqs. 24 and 25 for best fit calculations
    (using the option best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm gamma_b (-), by default np.nan
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
        The dimensionless crest freeboard of the structure Rc/Hm0 (-)
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

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_taw2002.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
        )

        cot_alpha = wave_runup_taw2002.determine_average_slope(
            Hm0=Hm0,
            z2p=z2p_for_slope,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    if np.isnan(gamma_b):
        L_berm = wave_runup_taw2002.calculate_berm_length(
            Hm0=Hm0,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
        )

        gamma_b = wave_runup_taw2002.iteration_procedure_gamma_b(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_average=cot_alpha,
            B_berm=B_berm,
            L_berm=L_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
        )

    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    Rc_diml_eq24 = (
        np.log(
            (1.0 / 0.067)
            * np.sqrt(1.0 / cot_alpha)
            * (1.0 / gamma_b)
            * (1.0 / ksi_mm10)
            * q
            / np.sqrt(9.81 * Hm0**3)
        )
        * (-1.0 / (c1 + cor1))
        * ksi_mm10
        * gamma_b
        * gamma_f_adj
        * gamma_beta
        * gamma_v
    )

    Rc_diml_eq25 = (
        np.log(5 * q / np.sqrt(9.81 * Hm0**3))
        * (-1.0 / (c2 + cor2))
        * gamma_f_adj
        * gamma_beta
    )

    Rc_diml = np.min([Rc_diml_eq24, Rc_diml_eq25], axis=0)
    max_reached = np.min([Rc_diml_eq24, Rc_diml_eq25], axis=0) == Rc_diml_eq25

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return Rc_diml, max_reached
