import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


def check_validity(
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Ac: float | npt.NDArray[np.float64] = np.nan,
    Hm0: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range as defined in Van Gent & Van der Werf (2019).

    For all parameters supplied, their values are checked versus the range of test conditions specified in Table 2
    (Van Gent & Van der Werf, 2019). When parameters are nan (by default), they are not checked.

    Parameters
    ----------
    Rc : float | npt.NDArray[np.float64], optional
        Crest freeboard of the structure (m), by default np.nan
    Ac : float | npt.NDArray[np.float64], optional
        Armour crest freeboard of the structure (m), by default np.nan
    Hm0 : float | npt.NDArray[np.float64], optional
        Spectral significant wave height (m), by default np.nan
    """

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Rc))
        and not np.any(np.isnan(Ac))
    ):
        core_utility.check_variable_validity_range(
            "(Rc-Ac)/Hm0",
            "Van Gent and Van der Werf 2019",
            ((Rc - Ac) / Hm0),
            0.26,
            0.77,
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Ac)):
        core_utility.check_variable_validity_range(
            "Rc/Ac", "Van Gent and Van der Werf 2019", (Rc / Ac), 1.27, 1.55
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Rc)):
        core_utility.check_variable_validity_range(
            "Rc/Hm0", "Van Gent and Van der Werf 2019", (Rc / Hm0), 0.79, 2.18
        )

    return


def calculate_influence_oblique_waves_force_gamma_F_beta(
    z2p: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_A: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    gamma_F_beta = (
        (0.5 * np.power(np.cos(np.deg2rad(beta)), 2.0) + 0.5) * z2p - gamma_A * Ac
    ) / (z2p - gamma_A * Ac)

    return gamma_F_beta


def calculate_influence_oblique_waves_horizontal_force_gamma_FH_beta(
    z2p: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_A: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:

    gamma_FH_beta = calculate_influence_oblique_waves_force_gamma_F_beta(
        z2p=z2p, Ac=Ac, beta=beta, gamma_A=gamma_A
    )

    return gamma_FH_beta


def calculate_influence_oblique_waves_vertical_force_gamma_FV_beta(
    z2p: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    gamma_A: float | npt.NDArray[np.float64] = 0.75,
) -> float | npt.NDArray[np.float64]:

    gamma_FV_beta = calculate_influence_oblique_waves_force_gamma_F_beta(
        z2p=z2p, Ac=Ac, beta=beta, gamma_A=gamma_A
    )

    return gamma_FV_beta


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_beta: float | npt.NDArray[np.float64] = 0.5,
) -> float | npt.NDArray[np.float64]:
    """Calculates influence factor for oblique incident waves on wave runup level

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    c_beta : float | npt.NDArray[np.float64]
        Coefficient, 0.5 default

    Returns
    -------
    float | npt.NDArray[np.float64]
        influence factor for wave runup gamma_beta
    """

    gamma_beta = (1 - c_beta) * np.cos(beta) ** 2 + c_beta

    return gamma_beta


def calculate_FH2p_perpendicular(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Hwall: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 0.45,
    c0: float | npt.NDArray[np.float64] = 1.45,
    c1: float | npt.NDArray[np.float64] = 5.0,
    rho_water: float | npt.NDArray[np.float64] = 1025.0,
    g: float | npt.NDArray[np.float64] = 9.81,
    cFH: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the 2% exceedance horizontal force on the crest wall with the Van Gent and van der Werf (2019) method.

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    Hwall : float | npt.NDArray[np.float64]
        Height of the crest wall (m)
    gamma_beta : float | npt.NDArray[np.float64], optional
        Influence factor for oblique wave incidence (-), by default 1.0
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for surface roughness (-), by default 0.45
    c0 : float | npt.NDArray[np.float64], optional
        Coefficient in wave runup formula (-), by default 1.45
    c1 : float | npt.NDArray[np.float64], optional
        Coefficient in wave runup formula (-), by default 5.0
    rho_water : float | npt.NDArray[np.float64], optional
        Water density (kg/m^3), by default 1025.0
    g : float | npt.NDArray[np.float64], optional
        Gravitational constant (m/s^2), by default 9.81
    cFH : float | npt.NDArray[np.float64], optional
        Coefficient in horizontal force formula (-), by default 1.0

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 2% exceedance horizontal force on the crest wall FH2% (N/m)
    """

    gamma = gamma_f * gamma_beta

    z2p = vangent2001.calculate_wave_runup_height_zXp(
        H=Hm0, Tmm10=Tmm10, cot_alpha=cot_alpha, gamma=gamma, c0=c0, c1=c1
    )

    FH2p = calculate_FH2p_perpendicular_from_z2p(
        Hm0=Hm0, z2p=z2p, rho_water=rho_water, Ac=Ac, Rc=Rc, Hwall=Hwall, g=g, cFH=cFH
    )

    check_validity(Hm0=Hm0, Ac=Ac, Rc=Rc)

    return FH2p


def calculate_FH2p_perpendicular_from_z2p(
    Hm0: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Hwall: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64] = 1025.0,
    g: float | npt.NDArray[np.float64] = 9.81,
    cFH: float | npt.NDArray[np.float64] = 1.0,
) -> float | npt.NDArray[np.float64]:

    check_validity(Hm0=Hm0, Ac=Ac, Rc=Rc)

    # Formula 13 paper
    FH2p = cFH * rho_water * g * Hwall * (z2p - Ac)

    FH2p = np.min(FH2p, 0)

    check_validity(Hm0=Hm0, Ac=Ac, Rc=Rc)

    return FH2p


def calculate_FH2p_oblique(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Hwall: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 0.45,
    c0: float | npt.NDArray[np.float64] = 1.45,
    c1: float | npt.NDArray[np.float64] = 5.0,
    cFH: float | npt.NDArray[np.float64] = 1.0,
    rho_water: float | npt.NDArray[np.float64] = 1025.0,
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    gamma = gamma_f * gamma_beta

    z2p = vangent2001.calculate_wave_runup_height_zXp(
        H=Hm0, Tmm10=Tmm10, cot_alpha=cot_alpha, gamma=gamma, c0=c0, c1=c1
    )

    FH2p_perpendicular = calculate_FH2p_perpendicular_from_z2p(
        Hm0=Hm0, z2p=z2p, rho_water=rho_water, Ac=Ac, Rc=Rc, Hwall=Hwall, g=g, cFH=cFH
    )

    gamma_FH_beta = calculate_influence_oblique_waves_horizontal_force_gamma_FH_beta(
        z2p=z2p, Ac=Ac, beta=cot_alpha
    )

    FH2p_oblique = gamma_FH_beta * FH2p_perpendicular

    check_validity(Hm0=Hm0, Ac=Ac, Rc=Rc)

    return FH2p_oblique


def calculate_FV2p_perpendicular(
    z2p: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Bwall: float | npt.NDArray[np.float64],
    Fb: float | npt.NDArray[np.float64],  # Level base plate relative to water level [m]
    cFV: float | npt.NDArray[np.float64] = 0.4,
    cFb: float | npt.NDArray[np.float64] = 0.5,
    g: float | npt.NDArray[np.float64] = 9.81,  # gravitational acceleration [m2/s]
) -> float | npt.NDArray[np.float64]:

    FV2p = (
        cFV * rho_water * g * Bwall * (z2p - 0.75 * Ac) * (1 - (np.power(Fb / Ac, cFb)))
    )

    FV2p = np.min(FV2p, 0)

    return FV2p


def calculate_FV2p_oblique(
    Ac: float | npt.NDArray[np.float64],
    Bwall: float | npt.NDArray[np.float64],
    Fb: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 0.45,
    c0: float | npt.NDArray[np.float64] = 1.45,
    c1: float | npt.NDArray[np.float64] = 5.0,
    cFV: float | npt.NDArray[np.float64] = 0.4,
    rho_water: float | npt.NDArray[np.float64] = 1025.0,
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    gamma = gamma_f * gamma_beta

    z2p = vangent2001.calculate_wave_runup_height_zXp(
        H=Hm0, Tmm10=Tmm10, cot_alpha=cot_alpha, gamma=gamma, c0=c0, c1=c1
    )

    FV2p_perpendicular = calculate_FV2p_perpendicular(
        z2p=z2p, rho_water=rho_water, Ac=Ac, Bwall=Bwall, cFV=cFV, Fb=Fb, g=g
    )

    gamma_FV_beta = calculate_influence_oblique_waves_vertical_force_gamma_FV_beta(
        z2p=z2p, Ac=Ac, beta=cot_alpha
    )

    FV2p_oblique = gamma_FV_beta * FV2p_perpendicular

    return FV2p_oblique


def calculate_FH01p_perpendicular(
    FH2p: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    FH01p = 1.6 * FH2p

    return FH01p


def calculate_FV01p_perpendicular(
    FV2p: float | npt.NDArray[np.float64],
    sop: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    FV01p = (2.88 - 32 * sop) * FV2p

    return FV01p
