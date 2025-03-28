# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 1.75,
    c2: float = 4.3,
    c3: float = 1.5,
    c4: float = 1.07,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the wave runup height with a 2% probability of exceedance z2% with the EurOtop (2018) formula.

    The 2% exceedance wave runup height z2% (m) is calculated using the EurOtop (2018) formulas. Here eqs. 5.4
    and 5.5 from EurOtop (2018) are implemented for design calculations and eqs. 5.1 and 5.2 for best fit
    calculations (using the option best_fit=True). Note that an erratat has been published regarding eqs. 5.2
    and 5.5.

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
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
    c1 : float, optional
        Coefficient in wave runup formula (-), by default 1.75
    c2 : float, optional
        Coefficient in wave runup formula (-), by default 4.3
    c3 : float, optional
        Coefficient in wave runup formula (-), by default 1.5
    c4 : float, optional
        Coefficient in wave runup formula (-), by default 1.07
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The 2% exceedance wave runup height z2% (m) and a boolean indicating
        whether the maximum value formula was used
    """

    z2p_diml, max_reached = calculate_dimensionless_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        B_berm=B_berm,
        db=db,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_beta=gamma_beta,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        use_best_fit=use_best_fit,
    )

    z2p = z2p_diml * Hm0

    return z2p, max_reached


def calculate_dimensionless_wave_runup_height_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c1: float = 1.75,
    c2: float = 4.3,
    c3: float = 1.5,
    c4: float = 1.07,
    use_best_fit: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """Calculate the dimensionless wave runup height with a 2% probability of exceedance z2%/Hm0 with the EurOtop (2018)
    formula.

    The dimensionless 2% exceedance wave runup height z2%/Hm0 (-) is calculated using the EurOtop (2018) formulas.
    Here eqs. 5.4 and 5.5 from EurOtop (2018) are implemented for design calculations and eqs. 5.1 and 5.2 for best
    fit calculations (using the option best_fit=True). Note that an erratat has been published regarding eqs. 5.2
    and 5.5.

    For more details see EurOtop (2018) and the errata of November 2019, available here:
    https://www.overtopping-manual.com/assets/downloads/EurOtop_II_2018_Final_version.pdf

    https://www.overtopping-manual.com/assets/downloads/Errata_EurOtop_2018_Nov_2019.pdf


    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        Influence factor for a berm (-), by default np.nan
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
    c1 : float, optional
        Coefficient in wave runup formula (-), by default 1.75
    c2 : float, optional
        Coefficient in wave runup formula (-), by default 4.3
    c3 : float, optional
        Coefficient in wave runup formula (-), by default 1.5
    c4 : float, optional
        Coefficient in wave runup formula (-), by default 1.07
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        The 2% exceedance wave runup height z2% (m) and a boolean indicating
        whether the maximum value formula was used
    """

    if use_best_fit:
        c1 = 1.65
        c2 = 4.0
        c3 = 1.5
        c4 = 1.0

    gamma_beta = calculate_influence_oblique_waves_gamma_beta(
        beta=beta,
        gamma_f=gamma_f,
    )

    z2p_diml, max_reached = (
        wave_runup_taw2002.calculate_dimensionless_wave_runup_height_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            gamma_beta=gamma_beta,
            gamma_b=gamma_b,
            gamma_f=gamma_f,
            B_berm=B_berm,
            db=db,
            cot_alpha=cot_alpha,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
        )
    )

    return z2p_diml, max_reached


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_f_crit: float = 0.6,
    c_gamma_beta_smooth: float = 0.0022,
    c_gamma_beta_rough: float = 0.0063,
    max_angle: float = 80.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor gamma_beta is determined using the EurOtop (2018) eq. 5.28 for smooth slopes and eq. 6.9
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

    if gamma_f > gamma_f_crit:
        # Structure slope is smooth
        c_gamma_beta = c_gamma_beta_smooth
    else:
        # Structure slope is rough
        c_gamma_beta = c_gamma_beta_rough

    gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
        beta=beta,
        c_gamma_beta=c_gamma_beta,
        max_angle=max_angle,
    )

    return gamma_beta
