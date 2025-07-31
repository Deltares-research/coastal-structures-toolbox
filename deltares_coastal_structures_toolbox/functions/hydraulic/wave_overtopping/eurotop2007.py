# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_overtopping_discharge_q_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for simple rubble mound slopes with the EurOtop (2007) formula.

    The mean wave overtopping discharge q (m^3/s/m) is calculated using the EurOtop (2007) formulas.
    Here eq. 6.5 from EurOtop (2007) is implemented for design calculations and eq. 6.6 for best fit calculations
    (using the option best_fit=True).

    For more details see EurOtop (2007), available here:
    https://www.overtopping-manual.com/assets/downloads/EAK-K073_EurOtop_2007.pdf

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
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.2
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.3
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The mean wave overtopping discharge q (m^3/s/m)
    """

    q_diml = calculate_dimensionless_overtopping_discharge_q_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        Rc=Rc,
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
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for simple rubble mound slopes with the
    EurOtop (2007) formula.

    The dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-) is calculated using the EurOtop (2007)
    formulas. Here eq. 6.5 from EurOtop (2007) is implemented for design calculations and eq. 6.6 for best fit
    calculations (using the option best_fit=True).

    For more details see EurOtop (2007), available here:
    https://www.overtopping-manual.com/assets/downloads/EAK-K073_EurOtop_2007.pdf

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
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.2
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.3
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    float | npt.NDArray[np.float64]
        The dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-)
    """

    (
        c2,
        c3,
    ) = check_best_fit(c2=c2, c3=c3, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    # TODO Check if we need gamma_f or gamma_f_adj here
    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    q_diml = wave_overtopping_taw2002.q_diml_max_equation(
        Hm0=Hm0, Rc=Rc, c2=c2, c3=c3, gamma_beta=gamma_beta, gamma_f=gamma_f_adj
    )

    return q_diml


def check_best_fit(c2: float, c3: float, use_best_fit: bool) -> tuple[float, float]:
    """Check whether best fit coefficients need to be used

    If so, return the best fit coefficients, otherwise return the input coefficients

    Parameters
    ----------
    c2 : float
        Coefficient in wave overtopping formula (-)
    c3 : float
        Coefficient in wave overtopping formula (-)
    use_best_fit : bool
        Switch to either use best fit values for the coefficients (true) or the design values (false)

    Returns
    -------
    tuple[float, float]
        Coefficients c2 and c3 in the wave runup formula (-)
    """
    if use_best_fit:
        c2 = 0.2
        c3 = 2.6

    return c2, c3


def calculate_crest_freeboard_Rc_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Calculate the crest freeboard Rc for simple rubble mound slopes with the EurOtop (2007) formula.

    The crest freeboard Rc/Hm0 (-) is calculated using the EurOtop (2007) formulas.
    Here eq. 6.5 from EurOtop (2007) is implemented for design calculations and eq. 6.6 for best fit
    calculations (using the option best_fit=True).

    For more details see EurOtop (2007), available here:
    https://www.overtopping-manual.com/assets/downloads/EAK-K073_EurOtop_2007.pdf

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
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.2
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.3
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    float | npt.NDArray[np.float64]
        The crest freeboard of the structure Rc (m)
    """

    Rc_diml = calculate_dimensionless_crest_freeboard_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        q=q,
        gamma_f=gamma_f,
        c2=c2,
        c3=c3,
        use_best_fit=use_best_fit,
    )

    Rc = Rc_diml * Hm0

    return Rc


def calculate_dimensionless_crest_freeboard_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless crest freeboard Rc/Hm0 for simple rubble mound slopes with the
    EurOtop (2007) formula.

    The dimensionless crest freeboard Rc/Hm0 (-) is calculated using the EurOtop (2007) formulas.
    Here eq. 6.5 from EurOtop (2007) is implemented for design calculations and eq. 6.6 for best fit
    calculations (using the option best_fit=True).

    For more details see EurOtop (2007), available here:
    https://www.overtopping-manual.com/assets/downloads/EAK-K073_EurOtop_2007.pdf

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
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 1.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        Cotangent of the front-side slope of the structure (-), by default np.nan
    c2 : float, optional
        Coefficient in wave overtopping formula (-), by default 0.2
    c3 : float, optional
        Coefficient in wave overtopping formula (-), by default 2.3
    use_best_fit : bool, optional
        Switch to either use best fit values for the coefficients (true) or the design values (false), by default False

    Returns
    -------
    float | npt.NDArray[np.float64]
        The dimensionless crest freeboard of the structure Rc/Hm0 (-)
    """

    (
        c2,
        c3,
    ) = check_best_fit(c2=c2, c3=c3, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    # TODO Check if we need gamma_f or gamma_f_adj here
    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    Rc_diml = wave_overtopping_taw2002.Rc_diml_max_equation(
        Hm0=Hm0, q=q, c2=c2, c3=c3, gamma_beta=gamma_beta, gamma_f=gamma_f_adj
    )

    return Rc_diml
