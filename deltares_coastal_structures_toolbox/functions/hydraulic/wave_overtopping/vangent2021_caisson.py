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
    """Calculate the mean wave overtopping discharge q for caisson breakwaters with the Van Gent (2021) formula."""

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
    """Calculate the dimensionless mean wave overtopping discharge q for caisson
    breakwaters with the Van Gent (2021) formula."""

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
    # """Calculate the influence factor for oblique wave incidence gamma_beta"""

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
