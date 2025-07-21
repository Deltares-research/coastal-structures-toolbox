# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as wave_runup_eurotop2018
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def check_validity_range_rubble_mound(
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
) -> None:

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent alpha",
            "EurOtop (2018) - rubble mound",
            cot_alpha,
            4.0 / 3.0,
            2.0,
        )

    return


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 2.5,
    c2: float = 0.1035,
    c3: float = 1.35,
    c4: float = 0.026,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the mean wave overtopping discharge q with the EurOtop (2018) formula.

    The mean wave overtopping discharge q (m^3/s/m) is calculated using the EurOtop (2018) formulas. Here eqs. 5.12
    and 5.13 from EurOtop (2018) are implemented for design calculations and eqs. 5.10 and 5.11 for best fit
    calculations (using the option best_fit=True).

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default 1.0
    gamma_star : float | npt.NDArray[np.float64], optional
        _description_, by default 1.0
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    c1 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.5
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.1035
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 1.35
    c4 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.026
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

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
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        gamma_star=gamma_star,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 2.5,
    c2: float = 0.1035,
    c3: float = 1.35,
    c4: float = 0.026,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless mean wave overtopping discharge q with the EurOtop (2018) formula.

    The mean wave overtopping discharge q/sqrt(g*Hm0^3) (-) is calculated using the EurOtop (2018) formulas.
    Here eqs. 5.12 and 5.13 from EurOtop (2018) are implemented for design calculations and eqs. 5.10 and 5.11
    for best fit calculations (using the option best_fit=True).

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default 1.0
    gamma_star : float | npt.NDArray[np.float64], optional
        _description_, by default 1.0
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    c1 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.5
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.1035
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 1.35
    c4 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.026
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
        and a boolean indicating whether the maximum value formula was used
    """

    c1, c2, c3, c4 = check_best_fit(
        c1=c1, c2=c2, c3=c3, c4=c4, use_best_fit=use_best_fit
    )

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
        )

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_eurotop2018.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
            use_best_fit=use_best_fit,
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

    L_berm = wave_runup_taw2002.calculate_berm_length(
        Hm0=Hm0,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
    )

    gamma_b = wave_runup_eurotop2018.iteration_procedure_gamma_b(
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

    q_diml_eq510 = (
        (c4 / np.sqrt(1.0 / cot_alpha))
        * ksi_mm10
        * gamma_b
        * np.exp(
            -1.0
            * np.power(
                c1
                * (Rc / Hm0)
                * (1.0 / (ksi_mm10 * gamma_b * gamma_f_adj * gamma_beta * gamma_v)),
                1.3,
            )
        )
    )
    q_diml_max = q_diml_max_equation(
        Hm0=Hm0,
        Rc=Rc,
        gamma_beta=gamma_beta,
        gamma_f=gamma_f,
        gamma_star=gamma_star,
        c2=c2,
        c3=c3,
    )

    q_diml = np.min([q_diml_eq510, q_diml_max], axis=0)
    max_reached = np.min([q_diml_eq510, q_diml_max], axis=0) == q_diml_max

    return q_diml, max_reached


def q_diml_max_equation(
    Hm0: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    c2: float = 0.1035,
    c3: float = 1.35,
):

    q_diml_max = c2 * np.exp(
        -1.0
        * np.power(c3 * (Rc / Hm0) * (1.0 / (gamma_f * gamma_beta * gamma_star)), 1.3)
    )

    return q_diml_max


def check_best_fit(
    c1: float, c2: float, c3: float, c4: float, use_best_fit: bool
) -> tuple[float, float, float, float]:
    """Check whether best fit coefficients need to be used

    If so, return the best fit coefficients, otherwise return the input coefficients

    Parameters
    ----------
    c1 : float
        Coefficient in wave overtopping formula (-)
    c2 : float
        Coefficient in wave overtopping formula (-)
    c3 : float
        Coefficient in wave overtopping formula (-)
    c4 : float
        Coefficient in wave overtopping formula (-)
    use_best_fit : bool
        Switch to either use best fit values for the coefficients (true) or the design values (false)

    Returns
    -------
    tuple[float, float, float, float]
        Coefficients c1, c2, c3 and c4 in the wave runup formula (-)
    """
    if use_best_fit:
        c1 = 2.7
        c2 = 0.09
        c3 = 1.5
        c4 = 0.023

    return c1, c2, c3, c4


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_crit: float = 0.6,
    c_gamma_beta_smooth: float = 0.0033,
    c_gamma_beta_rough: float = 0.0063,
    max_angle: float = 80.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor gamma_beta is determined using the EurOtop (2018) eq. 5.29 for smooth slopes and eq. 6.9
    for rough slopes.

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_f_crit : float, optional
        Critical value for the influence factor dividing smooth (higher) and rough (lower) slopes, by default 0.6
    c_gamma_beta_smooth : float, optional
        Coefficient for wave runup on smooth slopes, by default 0.0022
    c_gamma_beta_rough : float, optional
        Coefficient for wave runup on rough slopes, by default 0.0063
    max_angle : float, optional
        Maximum angle of wave incidence, by default 80.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    gamma_beta = wave_runup_eurotop2018.calculate_influence_oblique_waves_gamma_beta(
        beta=beta,
        gamma_f=gamma_f,
        gamma_f_crit=gamma_f_crit,
        c_gamma_beta_smooth=c_gamma_beta_smooth,
        c_gamma_beta_rough=c_gamma_beta_rough,
        max_angle=max_angle,
    )

    return gamma_beta


def calculate_crest_freeboard_Rc(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 2.5,
    c2: float = 0.1035,
    c3: float = 1.35,
    c4: float = 0.026,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the crest freeboard Rc with the EurOtop (2018) formula.

    The crest freeboard Rc (m) is calculated using the EurOtop (2018) formulas. Here eqs. 5.12 and 5.13 from
    EurOtop (2018) are implemented for design calculations and eqs. 5.10 and 5.11 for best fit calculations
    (using the option best_fit=True).

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default 1.0
    gamma_star : float | npt.NDArray[np.float64], optional
        _description_, by default 1.0
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    c1 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.5
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.1035
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 1.35
    c4 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.026
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The crest freeboard of the structure Rc (m) and a boolean indicating
        whether the maximum value formula was used
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
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        gamma_star=gamma_star,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        use_best_fit=use_best_fit,
        g=g,
    )

    Rc = Rc_diml * Hm0

    return Rc, max_reached


def calculate_dimensionless_crest_freeboard(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_v: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 2.5,
    c2: float = 0.1035,
    c3: float = 1.35,
    c4: float = 0.026,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless crest freeboard Rc/Hm0 with the EurOtop (2018) formula.

    The dimensionless crest freeboard Rc/Hm0 (-) is calculated using the EurOtop (2018) formulas. Here eqs.
    5.12 and 5.13 from EurOtop (2018) are implemented for design calculations and eqs. 5.10 and 5.11 for best
    fit calculations (using the option best_fit=True).

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    gamma_v : float | npt.NDArray[np.float64], optional
        Influence factor for a crest wall (-), by default 1.0
    gamma_star : float | npt.NDArray[np.float64], optional
        _description_, by default 1.0
    B_berm : float | npt.NDArray[np.float64], optional
        Berm width of the structure (m), by default 0.0
    db : float | npt.NDArray[np.float64], optional
        Berm height of the structure (m), by default 0.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan
    c1 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.5
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.1035
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 1.35
    c4 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.026
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The dimensionless crest freeboard of the structure Rc/Hm0 (-) and a boolean indicating
        whether the maximum value formula was used
    """

    c1, c2, c3, c4 = check_best_fit(
        c1=c1, c2=c2, c3=c3, c4=c4, use_best_fit=use_best_fit
    )

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
        )

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_eurotop2018.iteration_procedure_z2p(
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

    L_berm = wave_runup_taw2002.calculate_berm_length(
        Hm0=Hm0,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
    )

    gamma_b = wave_runup_eurotop2018.iteration_procedure_gamma_b(
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

    Rc_diml_eq510 = (
        np.power(
            -np.log(
                (q / np.sqrt(g * np.power(Hm0, 3)))
                * np.sqrt(1.0 / cot_alpha)
                * (1.0 / c4)
                * (1.0 / (gamma_b * ksi_mm10))
            ),
            1.0 / 1.3,
        )
        * (1.0 / c1)
        * ksi_mm10
        * gamma_b
        * gamma_f_adj
        * gamma_beta
        * gamma_v
    )

    Rc_diml_max = Rc_diml_max_equation(
        Hm0=Hm0,
        q=q,
        gamma_beta=gamma_beta,
        gamma_f=gamma_f_adj,
        gamma_star=gamma_star,
        c2=c2,
        c3=c3,
        g=g,
    )

    Rc_diml = np.min([Rc_diml_eq510, Rc_diml_max], axis=0)
    max_reached = np.min([Rc_diml_eq510, Rc_diml_max], axis=0) == Rc_diml_max

    return Rc_diml, max_reached


def Rc_diml_max_equation(
    Hm0: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    gamma_star: float | npt.NDArray[np.float64] = 1.0,
    c2: float = 0.1035,
    c3: float = 1.35,
    g: float = 9.81,
):

    Rc_diml_max = (
        np.power(-np.log((q / np.sqrt(g * np.power(Hm0, 3))) * (1.0 / c2)), 1.0 / 1.3)
        * (1.0 / c3)
        * gamma_f
        * gamma_beta
        * gamma_star
    )

    return Rc_diml_max


def calculate_overtopping_discharge_q_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.1035,
    c3: float = 1.35,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    q_diml = calculate_dimensionless_overtopping_discharge_q_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        c2=c2,
        c3=c3,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q


def calculate_dimensionless_overtopping_discharge_q_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.1035,
    c3: float = 1.35,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:

    _, c2, c3, _ = check_best_fit(
        c1=np.nan, c2=c2, c3=c3, c4=np.nan, use_best_fit=use_best_fit
    )

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
        )

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_eurotop2018.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
            use_best_fit=use_best_fit,
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

    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    q_diml = q_diml_max_equation(
        Hm0=Hm0,
        Rc=Rc,
        gamma_beta=gamma_beta,
        gamma_f=gamma_f_adj,
        c2=c2,
        c3=c3,
    )

    check_validity_range_rubble_mound(
        cot_alpha=cot_alpha,
    )

    return q_diml


def calculate_crest_freeboard_Rc_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.1035,
    c3: float = 1.35,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    Rc_diml, max_reached = calculate_dimensionless_crest_freeboard_rubble_mound(
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
        c2=c2,
        c3=c3,
        use_best_fit=use_best_fit,
        g=g,
    )

    Rc = Rc_diml * Hm0

    return Rc, max_reached


def calculate_dimensionless_crest_freeboard_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.1035,
    c3: float = 1.35,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

    _, c2, c3, _ = check_best_fit(
        c1=np.nan, c2=c2, c3=c3, c4=np.nan, use_best_fit=use_best_fit
    )

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
        )

    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_eurotop2018.iteration_procedure_z2p(
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

    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    Rc_diml = Rc_diml_max_equation(
        Hm0=Hm0,
        q=q,
        gamma_beta=gamma_beta,
        gamma_f=gamma_f_adj,
        c2=c2,
        c3=c3,
        g=g,
    )

    check_validity_range_rubble_mound(
        cot_alpha=cot_alpha,
    )

    return Rc_diml
