# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    Rc: float | npt.NDArray[np.float64] = np.nan,
    q_diml: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Van Gent (2021) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    Table 2 in Van Gent (2021). When parameters are nan (by default), they are not checked.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64], optional
        Spectral significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    Rc : float | npt.NDArray[np.float64], optional
        Crest freeboard of the structure (m), by default np.nan
    q_diml : float | npt.NDArray[np.float64], optional
        Dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3) (-), by default np.nan
    """

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0", "Van Gent (2021)", smm10, 0.015, 0.041
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Rc)):
        core_utility.check_variable_validity_range(
            "Dimensionless freeboard Rc/Hm0",
            "Van Gent (2021)",
            Rc / Hm0,
            1.2,
            2.9,
        )

    if not np.any(np.isnan(q_diml)):
        core_utility.check_variable_validity_range(
            "Dimensionless overtopping discharge q",
            "Van Gent (2021)",
            q_diml,
            0.0,
            0.0012,
        )

    if not np.any(np.isnan(beta)):
        core_utility.check_variable_validity_range(
            "Incident wave angle beta", "Van Gent (2021)", beta, 0.0, 75.0
        )

    return


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    Hm0_swell: float | npt.NDArray[np.float64] = np.nan,
    c: float = 1.0,
    c_swell: float = 0.4,
    short_crested_waves: bool = True,
    crossing_seas: bool = False,
    parapet: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for caisson breakwaters with the Van Gent (2021) formula.

    The mean wave overtopping discharge q (m^3/s/m) for caisson breakwaters is calculated using the Van Gent (2021)
    formula. Here, eq. 11 from Van Gent (2021) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    Hm0_swell : float | npt.NDArray[np.float64], optional
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m),
        by default np.nan
    c : float, optional
        Exponent in the wave overtopping formula, by default 1.0
    c_swell : float, optional
        Coefficient for the effective freeboard reduction due to swell (only active for crossing_seas = True),
        by default 0.4
    short_crested_waves : bool, optional
        Use coefficient for short-crested waves (else long-crested), by default True
    crossing_seas : bool, optional
        Calculation for crossing seas where Hm0_swell has an influence, by default False
    parapet : bool, optional
        Indicate the presence of a recurved parapet / bullnose / recurved wave return wall, by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The mean wave overtopping discharge q (m^3/s/m)
    """

    q_diml = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Rc=Rc,
        beta=beta,
        gamma_beta=gamma_beta,
        c=c,
        c_swell=c_swell,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    Hm0_swell: float | npt.NDArray[np.float64] = np.nan,
    c: float = 1.0,
    c_swell: float = 0.4,
    short_crested_waves: bool = True,
    crossing_seas: bool = False,
    parapet: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless mean wave overtopping discharge q for caisson breakwaters with the
    Van Gent (2021) formula.

    The dimensionless mean wave overtopping discharge q (m^3/s/m) for caisson breakwaters is calculated using the
    Van Gent (2021) formula. Here, eq. 11 from Van Gent (2021) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    Hm0_swell : float | npt.NDArray[np.float64], optional
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m),
        by default np.nan
    c : float, optional
        Exponent in the wave overtopping formula, by default 1.0
    c_swell : float, optional
        Coefficient for the effective freeboard reduction due to swell (only active for crossing_seas = True),
        by default 0.4
    short_crested_waves : bool, optional
        Use coefficient for short-crested waves (else long-crested), by default True
    crossing_seas : bool, optional
        Calculation for crossing seas where Hm0_swell has an influence, by default False
    parapet : bool, optional
        Indicate the presence of a recurved parapet / bullnose / recurved wave return wall, by default False

    Returns
    -------
    float | npt.NDArray[np.float64]
        The dimensionless mean wave overtopping discharge q/sqrt(g*Hm0^3)

    Raises
    ------
    ValueError
        When Hm0_swell is not provided in the case of crossing seas.
    """

    a, b, c_beta, gamma_p = _determine_coefficients(
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    if not crossing_seas:
        c_swell = 0.0
    else:
        if np.isnan(Hm0_swell):
            raise ValueError(
                "Hm0_swell must be provided when calculating wave overtopping for crossing seas."
            )

    if np.isnan(gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta,
            gamma_p=gamma_p,
            c_beta=c_beta,
        )

    q_diml = a * np.exp(
        -(b / (gamma_beta * gamma_p)) * np.power((Rc - c_swell * Hm0_swell) / Hm0, c)
    )

    check_validity_range(
        Hm0=Hm0,
        beta=beta,
        Rc=Rc,
        q_diml=q_diml,
    )

    return q_diml


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    gamma_p: float,
    c_beta: float,
) -> float | npt.NDArray[np.float64]:
    """Calculate the influence factor for oblique wave incidence gamma_beta

    The influence factor gamma_beta is determined using Van Gent (2021) eq. 10

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    gamma_p : float
        Influence factor for a recurved parapet / bullnose / recurved wave return wall (-)
    c_beta : float
        Coefficient in the gamma_beta formula

    Returns
    -------
    float | npt.NDArray[np.float64]
        The influence factor for oblique wave incidence gamma_beta (-)
    """

    gamma_beta = (1 - c_beta / gamma_p) * np.power(
        np.cos(np.radians(beta)), 2
    ) + c_beta / gamma_p

    return gamma_beta


def calculate_crest_freeboard_Rc(
    Hm0: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    Hm0_swell: float | npt.NDArray[np.float64] = np.nan,
    c: float = 1.0,
    c_swell: float = 0.4,
    short_crested_waves: bool = True,
    crossing_seas: bool = False,
    parapet: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the crest freeboard Rc for caisson breakwaters with the Van Gent (2021) formula.

    The crest freeboard Rc (m) of a caisson breakwater is calculated using the Van Gent (2021) formula.
    Here, eq. 11 from Van Gent (2021) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    Hm0_swell : float | npt.NDArray[np.float64], optional
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m),
        by default np.nan
    c : float, optional
        Exponent in the wave overtopping formula, by default 1.0
    c_swell : float, optional
        Coefficient for the effective freeboard reduction due to swell (only active for crossing_seas = True),
        by default 0.4
    short_crested_waves : bool, optional
        Use coefficient for short-crested waves (else long-crested), by default True
    crossing_seas : bool, optional
        Calculation for crossing seas where Hm0_swell has an influence, by default False
    parapet : bool, optional
        Indicate the presence of a recurved parapet / bullnose / recurved wave return wall, by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The crest freeboard of the structure Rc (m)
    """

    Rc_diml = calculate_dimensionless_crest_freeboard(
        Hm0=Hm0,
        q=q,
        beta=beta,
        gamma_beta=gamma_beta,
        Hm0_swell=Hm0_swell,
        c=c,
        c_swell=c_swell,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
        g=g,
    )

    Rc = Rc_diml * Hm0

    return Rc


def calculate_dimensionless_crest_freeboard(
    Hm0: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    Hm0_swell: float | npt.NDArray[np.float64] = np.nan,
    c: float = 1.0,
    c_swell: float = 0.4,
    short_crested_waves: bool = True,
    crossing_seas: bool = False,
    parapet: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the dimensionless crest freeboard Rc/Hm0 for caisson breakwaters with the Van Gent (2021) formula.

    The dimensionless crest freeboard Rc/Hm0 (-) of a caisson breakwater is calculated using the Van Gent (2021)
    formula. Here, eq. 11 from Van Gent (2021) is implemented.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2020.103834

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    q : float | npt.NDArray[np.float64]
        Mean wave overtopping discharge (m^3/s/m)
    beta : float | npt.NDArray[np.float64], optional
        Angle of wave incidence (degrees), by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default np.nan
    Hm0_swell : float | npt.NDArray[np.float64], optional
        Spectral significant wave height of swell or infragravity waves in case of a second wave field (m),
        by default np.nan
    c : float, optional
        Exponent in the wave overtopping formula, by default 1.0
    c_swell : float, optional
        Coefficient for the effective freeboard reduction due to swell (only active for crossing_seas = True),
        by default 0.4
    short_crested_waves : bool, optional
        Use coefficient for short-crested waves (else long-crested), by default True
    crossing_seas : bool, optional
        Calculation for crossing seas where Hm0_swell has an influence, by default False
    parapet : bool, optional
        Indicate the presence of a recurved parapet / bullnose / recurved wave return wall, by default False
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        The dimensionless crest freeboard of the structure Rc/Hm0 (-)

    Raises
    ------
    ValueError
        When Hm0_swell is not provided in the case of crossing seas.
    """

    a, b, c_beta, gamma_p = _determine_coefficients(
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    if not crossing_seas:
        c_swell = 0.0
    else:
        if np.isnan(Hm0_swell):
            raise ValueError(
                "Hm0_swell must be provided when calculating wave overtopping for crossing seas."
            )

    if np.isnan(gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(
            beta=beta,
            gamma_p=gamma_p,
            c_beta=c_beta,
        )

    Rc_diml = (
        np.power(
            np.log((q / np.sqrt(g * np.power(Hm0, 3.0))) * (1.0 / a))
            * (-gamma_beta * gamma_p / b),
            1.0 / c,
        )
        + c_swell * Hm0_swell / Hm0
    )

    check_validity_range(
        Hm0=Hm0,
        beta=beta,
        Rc=Rc_diml * Hm0,
    )

    return Rc_diml


def _determine_coefficients(
    c: float,
    short_crested_waves,
    crossing_seas,
    parapet,
):

    match c:
        case 1.0:
            if short_crested_waves | crossing_seas:
                a = 0.2
                b = 3.9
                c_beta = 0.75

                if parapet:
                    gamma_p = 0.89
                else:
                    gamma_p = 1.0
            else:
                a = 0.2
                b = 4.0
                c_beta = 0.8

                if parapet:
                    raise ValueError(
                        "Coefficient values for long-crested waves and a parapet have not been derived in"
                        + " Van Gent (2021)."
                    )
                else:
                    gamma_p = 1.0
        case 1.3:
            if short_crested_waves | crossing_seas:
                a = 0.047
                b = 2.57
                c_beta = 0.70

                if parapet:
                    gamma_p = 0.85
                else:
                    gamma_p = 1.0
            else:
                a = 0.047
                b = 2.7
                c_beta = 0.8

                if parapet:
                    raise ValueError(
                        "Coefficient values for long-crested waves and a parapet have not been derived in"
                        + " Van Gent (2021)."
                    )
                else:
                    gamma_p = 1.0
        case _:
            raise ValueError(
                "Invalid value for the exponent of the overtopping formula c. Supported values are 1.0 and 1.3."
            )

    return a, b, c_beta, gamma_p
