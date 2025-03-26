# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the TAW (2002) wave runup formulas

    For all parameters supplied, their values are checked versus the range of test conditions specified by
    TAW (2002) in the table on pages 39-40. When parameters are nan (by default), they are not checked.

    For more details see TAW (2002), available here:
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s), by default np.nan
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64]
        Influence factor for a berm, by default np.nan
    gamma_beta : float | npt.NDArray[np.float64]
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_v : float | npt.NDArray[np.float64]
        Influence factor for a wave wall (-), by default np.nan
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-), by default np.nan
    """

    if not np.any(np.isnan(gamma_b)):
        core_utility.check_variable_validity_range(
            "Influence factor for a berm gamma_b", "TAW (2002)", gamma_b, 0.6, 1.0
        )

    if not np.any(np.isnan(gamma_beta)):
        core_utility.check_variable_validity_range(
            "Influence factor for oblique wave incidence gamma_beta",
            "TAW (2002)",
            gamma_beta,
            0.7,
            1.0,
        )

    if not np.any(np.isnan(gamma_f)):
        core_utility.check_variable_validity_range(
            "Influence factor for surface roughness gamma_f",
            "TAW (2002)",
            gamma_f,
            0.5,
            1.0,
        )

    if not np.any(np.isnan(gamma_v)):
        core_utility.check_variable_validity_range(
            "The influence factor for a wave wall gamma_v",
            "TAW (2002)",
            gamma_v,
            0.65,
            1.0,
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness smm10", "TAW (2002)", smm10, 0.001, 0.07
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
            H=Hm0, T=Tmm10, cot_alpha=cot_alpha
        )
        core_utility.check_variable_validity_range(
            "Irribarren number ksi_mm10", "TAW (2002)", ksi_mm10, 0.4, 20.0
        )

    return


def calculate_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm, by default 1.0
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
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
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        _description_
    """
    # TAW eq 3a & 3b (EurOtop I eq 5.3)

    z2p_diml, _ = calculate_dimensionless_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        B_berm=B_berm,
        db=db,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        use_best_fit=use_best_fit,
    )

    z2p = z2p_diml * Hm0

    return z2p


def calculate_dimensionless_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm, by default 1.0
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
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
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        _description_
    """
    # TAW eq 3a & 3b (EurOtop I eq 5.3)

    if use_best_fit:
        c1 = 1.65
        c2 = 4.0
        c3 = 1.5
    else:
        c1 = 1.75
        c2 = 4.3
        c3 = 1.6

    if check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        cot_alpha, _, _ = determine_average_slope(
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
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    gamma_f_adj = calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    z2p_diml_eq3a = c1 * gamma_b * gamma_f_adj * gamma_beta * ksi_mm10
    z2p_diml_eq3b = (
        1.0 * gamma_b * gamma_f_adj * gamma_beta * (c2 - c3 / np.sqrt(ksi_mm10))
    )  # TODO check difference TAW/ET1!

    z2p_diml = np.min([z2p_diml_eq3a, z2p_diml_eq3b], axis=0)
    max_reached = np.min([z2p_diml_eq3a, z2p_diml_eq3b], axis=0) == z2p_diml_eq3b

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=1.0,
        cot_alpha=cot_alpha,
    )

    return z2p_diml, max_reached


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """
    # TAW eq 8

    beta_calc = np.where(beta < 0, np.abs(beta), beta)
    beta_calc = np.where(beta_calc > 80, 80, beta_calc)

    gamma_beta = 1 - 0.0022 * beta_calc

    return gamma_beta


def calculate_adjusted_influence_roughness_gamma_f(
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    gamma_b : float | npt.NDArray[np.float64]
        Influence factor for a berm
    ksi_mm10 : float | npt.NDArray[np.float64]
        _description_

    Returns
    -------
    float | npt.NDArray[np.float64]
        The adjusted influence factor for surface roughness gamma_f (-)
    """
    # Breakwat User Manual eq 3.62

    gamma_f_adj = np.where(
        (gamma_b * ksi_mm10 >= 1.8) & (gamma_b * ksi_mm10 <= 10.0),
        ((1 - gamma_f) / (10.0 - 1.8)) * ksi_mm10
        + gamma_f
        + (gamma_f - 1.0) * (1.8 / (10.0 - 1.8)),
        gamma_f,
    )

    return gamma_f_adj


# def determine_average_slope_EurOtopI(
def determine_average_slope(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """_summary_

    _extended_summary_

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
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        _description_
    """

    # This is the procedure as described in section 5.3.4 of the EurOtop (2007) manual

    L_slope_iter1 = (
        (1.5 * Hm0 - db) * cot_alpha_down + B_berm + (1.5 * Hm0 + db) * cot_alpha_up
    )
    tan_alpha_average_iter1 = 3 * Hm0 / (L_slope_iter1 - B_berm)

    # TODO L_berm is unrelated, move to separate function
    L_berm = 1.0 * Hm0 * cot_alpha_down + B_berm + 1.0 * Hm0 * cot_alpha_up

    # do not account for berm influence in wave runup
    # TODO double check this
    gamma_b = 1.0

    z2p = calculate_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=1.0 / tan_alpha_average_iter1,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
    )
    # TODO the -db / +db terms come from BREAKWAT -> double check with sheet Alex
    L_slope_iter2 = (
        (1.5 * Hm0 - db) * cot_alpha_down + B_berm + (z2p + db) * cot_alpha_up
    )
    tan_alpha_average_iter2 = (1.5 * Hm0 + z2p) / (L_slope_iter2 - B_berm)

    cot_alpha_average = 1.0 / tan_alpha_average_iter2
    return cot_alpha_average, z2p, L_berm


def check_composite_slope(
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
) -> bool:
    """Check whether the structure has a composite slope

    This function checks whether the structure has a composite slope, i.e. the lower and upper part of the front-side
    of the structure have different slopes. If so, it returns true, if not it returns false.

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        Cotangent of the lower part of the front-side slope of the structure (-), by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        Cotangent of the upper part of the front-side slope of the structure (-), by default np.nan

    Returns
    -------
    bool
        True if the structure has a composite slope, false if upper and lower slopes are equal

    Raises
    ------
    ValueError
        Raise error when no slopes are provided
    """

    if np.all(np.isnan(cot_alpha)) and np.all(np.isnan(cot_alpha_down)):
        raise ValueError(
            "Either a single (cot_alpha) or composite (cot_alpha_down & cot_alpha_up) slope should be provided"
        )

    if (
        np.all(np.isnan(cot_alpha))
        and not np.all(np.isnan(cot_alpha_down))
        and not np.all(np.isnan(cot_alpha_up))
    ):
        is_composite_slope = True
    else:
        is_composite_slope = False
    return is_composite_slope
