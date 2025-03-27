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
    """Check the parameter values vs the validity range of the TAW (2002) wave runup formula

    For all parameters supplied, their values are checked versus the range of test conditions specified by
    TAW (2002) in the table on pages 39-40. When parameters are nan (by default), they are not checked.

    For more details see TAW (2002), available here (in Dutch):
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
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the wave runup height with a 2% probability of exceedance z2% with the TAW (2002) formula.

    The 2% exceedance wave runup height z2% (m) is calculated using the TAW (2002) formulas. Here eqs. 3a and 3b from
    TAW (2002) are implemented for design calculations and eqs. 5a and 5b for best fit calculations (using the option
    best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm, by default np.nan
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
        The 2% exceedance wave runup height z2% (m) and a boolean indicating
        whether the maximum value formula was used
    """
    # TODO include reference to EurOtop I (2007) and its equations in the docstring?

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
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless wave runup height with a 2% probability of exceedance z2%/Hm0
    with the TAW (2002) formula.

    The dimensionless 2% exceedance wave runup height z2%/Hm0 (-) is calculated using the TAW (2002) formulas.
    Here eqs. 3a and 3b from TAW (2002) are implemented for design calculations and eqs. 5a and 5b for best fit
    calculations (using the option best_fit=True).

    For more details see TAW (2002), available here (in Dutch):
    https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@97617/technisch-rapport-golfoploop/

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm, by default np.nan
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
        The dimensionless 2% exceedance wave runup height z2%/Hm0 (-) and a boolean indicating
        whether the maximum value formula was used
    """
    # TODO include reference to EurOtop I (2007) and its equations in the docstring?

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
        z2p_for_slope = iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
        )

        cot_alpha = determine_average_slope(
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
        L_berm = calculate_berm_length(
            Hm0=Hm0,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
        )

        gamma_b = iteration_procedure_gamma_b(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha_average=cot_alpha,
            B_berm=B_berm,
            L_berm=L_berm,
            db=db,
            gamma_f=gamma_f,
        )

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    gamma_f_adj = calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
    )

    z2p_diml_eq3a = c1 * gamma_b * gamma_f_adj * gamma_beta * ksi_mm10
    z2p_diml_eq3b = (
        1.0 * gamma_f_adj * gamma_beta * (c2 - c3 / np.sqrt(ksi_mm10))
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


def determine_average_slope(
    Hm0: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Determine the average slope of the front-side of the structure in case of composite slopes

    For structures with composite slopes (i.e. the lower and upper part of the front-side of the structure have
    different slopes), the average slope of the front-side of the structure is determined. This is done following
    the iterative procedure described in Section 2.3 of TAW (2002).

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    z2p : float | npt.NDArray[np.float64]
        The 2% exceedance wave runup height z2% (m)
    cot_alpha_down : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    cot_alpha_up : float | npt.NDArray[np.float64]
        Cotangent of the upper part of the front-side slope of the structure (-)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        Average cotangent of the front-side slope of the structure cot_alpha_average (-)
    """

    L_slope = (1.5 * Hm0 - db) * cot_alpha_down + B_berm + (z2p + db) * cot_alpha_up
    tan_alpha_average = (1.5 * Hm0 + z2p) / (L_slope - B_berm)

    cot_alpha_average = 1.0 / tan_alpha_average

    return cot_alpha_average


def iteration_procedure_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    tolerance: float = 1e-4,
    max_iter: int = 1000,
) -> float | npt.NDArray[np.float64]:
    """Iterative procedure to determine the 2% exceedance wave runup height z2%

    This iterative procedure to determine the 2% exceedance wave runup height z2% (m) is used in the determination
    of the average slope of the front-side of the structure in case of composite slopes.

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
    tolerance : float, optional
        Maximum allowable tolerance for the z2% iterative procedure, by default 1e-4
    max_iter : int, optional
        Maximum number of iterations in the z2% iterative procedure, by default 1000

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 2% exceedance wave runup height z2% (m)
    """

    n_iter = 0
    z2p_estimate_i1 = 1.5 * Hm0
    z2p_estimate_i0 = z2p_estimate_i1 + 2 * tolerance

    while n_iter <= max_iter and abs(z2p_estimate_i1 - z2p_estimate_i0) > tolerance:
        z2p_estimate_i0 = z2p_estimate_i1

        L_berm = calculate_berm_length(
            Hm0=Hm0,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
        )

        gamma_b = calculate_influence_berm_gamma_b(
            Hm0=Hm0, z2p=z2p_estimate_i0, db=db, B_berm=B_berm, L_berm=L_berm
        )

        cot_alpha_average = determine_average_slope(
            Hm0=Hm0,
            z2p=z2p_estimate_i0,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
        )

        z2p_estimate_i1 = calculate_wave_runup_height_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            beta=beta,
            cot_alpha=cot_alpha_average,
            gamma_b=gamma_b,
            gamma_f=gamma_f,
        )

        n_iter += 1

    z2p = z2p_estimate_i1

    return z2p


def calculate_berm_length(
    Hm0: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the berm length of the structure

    Calculate the berm length of the structure L_berm (m) as is needed for the determination of the influence
    factor for berms in eq. 11 (TAW, 2002)

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    cot_alpha_down : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    cot_alpha_up : float | npt.NDArray[np.float64]
        Cotangent of the upper part of the front-side slope of the structure (-)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)

    Returns
    -------
    float | npt.NDArray[np.float64]
        Berm length of the structure L_berm (m)
    """

    L_berm = 1.0 * Hm0 * cot_alpha_down + B_berm + 1.0 * Hm0 * cot_alpha_up

    return L_berm


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
    """Iterative procedure to determine the influence factor for a berm gamma_b

    Iteratively determine the influence factor for a berm gamma_b (-) (TAW, 2002), as in some cases the value of
    gamma_b is dependant on the z2%.

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
        z2p_i1 = calculate_wave_runup_height_z2p(
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

        z2p_i2 = calculate_wave_runup_height_z2p(
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


def calculate_influence_berm_gamma_b(
    Hm0: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    L_berm: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for a berm gamma_b

    The influence factor for a berm gamma_b (-) on wave runup is calculated using eqs. 10, 11, 12 and 13
    from TAW (2002).

    Note that the actual the recommended procedure to determine gamma_b is iterative and implemented in
    iteration_procedure_gamma_b()

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


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_gamma_beta: float = 0.0022,
    max_angle: float = 80.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor for oblique wave incidence gamma_beta (-) on wave runup is calculated using
    eq. 8 from TAW (2002). Note that this implementation can also be used for wave overtopping by
    changing the c_gamma_beta to 0.0033

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    c_gamma_beta : float, optional
        Coefficient for wave runup, by default 0.0022
    max_angle : float, optional
        Maximum angle of wave incidence, by default 80.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    beta_calc = np.where(beta < 0, np.abs(beta), beta)
    beta_calc = np.where(beta_calc > max_angle, max_angle, beta_calc)

    gamma_beta = 1 - c_gamma_beta * beta_calc

    return gamma_beta


def calculate_adjusted_influence_roughness_gamma_f(
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Calculate adjusted influence factor for surface roughness gamma_f

    In case of longer waves, slope roughness has a smaller effect on the wave runup height. This is reflected in an
    adjusted value of the influence factor, as described in the last paragraph of Section 2.7 in TAW (2002).

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

    gamma_f_adj = np.where(
        (gamma_b * ksi_mm10 >= 1.8) & (gamma_b * ksi_mm10 <= 10.0),
        ((1 - gamma_f) / (10.0 - 1.8)) * ksi_mm10
        + gamma_f
        + (gamma_f - 1.0) * (1.8 / (10.0 - 1.8)),
        gamma_f,
    )

    return gamma_f_adj
