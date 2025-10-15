import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


def check_validity_FH2p(
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
            " ", "Van Gent and Van der Werf 2019", (Rc / Hm0), 0.79, 2.18
        )

    return


def calculate_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_beta: float | npt.NDArray[np.float64] = 0.5,
) -> tuple[np.float64 | float | npt.NDArray[np.float64]]:
    """Calculates influence factor for oblique incident waves on level

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Incident wave angle in degrees measured wrt normal of the dike
    c_beta : float | npt.NDArray[np.float64]
        Coefficient, 0.5 default

    Returns
    -------
    tuple[np.float64 | float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        influence factor gamma_beta
    """

    gamma_beta = (1 - c_beta) * np.cos(beta) ** 2 + c_beta

    return gamma_beta


def calculate_z2p(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64] = 1.0,
    gamma_f: float | npt.NDArray[np.float64] = 0.45,
    c0: float | npt.NDArray[np.float64] = 1.45,
    c1: float | npt.NDArray[np.float64] = 5.0,
):
    """Calculate the wave runup height with the Van Gent (2001) formula, as referenced in the Van Gent
    and van der Werf (2019) paper

    For more details see Van Gent (2001), available here: https://doi.org/10.1061/(ASCE)0733-950X(2001)127:5(254)
    or here: https://www.researchgate.net/publication/245293002_Wave_Run-Up_on_Dikes_with_Shallow_Foreshores
    The coefficients c0 and c1 as well as gamma_f are according to the above paper and as used in the Van Gent
    and van der Werf (2019) paper

    For more details also see Van Gent and van der Werf (2019), available here:
    https://doi.org/10.1016/j.coastaleng.2019.04.001
    or here:
    https://www.researchgate.net/publication/332744221_Influence_of_oblique_wave_attack_on_wave_overtopping_and_forces_on_rubble_mound_breakwater_crest_walls

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        _description_Spectral wave period Tm-1,0 (s)
    cot_alpha : _tyfloat | npt.NDArray[np.float64]pe_
        Cotangent of the front-side slope of the structure (-)
    gamma_f : float | npt.NDArray[np.float64], optional
        Influence factor for roughness, by default 0.45
    gamma_beta : float | npt.NDArray[np.float64], optional
        Incluence factor of obliquely incident waves, by default 1
    c0 : float | npt.NDArray[np.float64], optional
        Coefficient in wave runup formula (-), by default 1.45 (short waves, Hm0 & Tmm10)
    c1 : float | npt.NDArray[np.float64], optional
        Coefficient in wave runup formula (-), by default 5.0  (short waves, Hm0 & Tmm10)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The 2% exceedance wave runup height z2% (m)
    """

    gamma = gamma_f * gamma_beta

    ru2p = vangent2001.calculate_wave_runup_height_zXp(
        H=Hm0, Tmm10=Tmm10, cot_alpha=cot_alpha, gamma=gamma, c0=c0, c1=c1
    )

    return ru2p


def calculate_FH2p_perpendicular_from_Hm0():
    pass


def calculate_FH2p_perpendicular_from_z2p(
    Hm0: float | npt.NDArray[np.float64],
    z2p: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Hwall: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
    cFH: float | npt.NDArray[np.float64] = 1.0,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

    check_validity_FH2p(Hm0=Hm0, Ac=Ac, Rc=Rc)

    # Formula 13 paper
    FH2p = cFH * rho_water * g * Hwall * (z2p - Ac)

    FH2p = np.min(FH2p, 0)

    return FH2p


def calculate_FV2p_perpendicular(
    z2p: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Bwall: float | npt.NDArray[np.float64],
    CFV: float | npt.NDArray[np.float64],
    Fb: float | npt.NDArray[np.float64],  # Level base plate relative to water level [m]
    g: float | npt.NDArray[np.float64] = 9.81,  # gravitational acceleration [m2/s]
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

    FV2p = CFV * rho_water * g * Bwall * (z2p - 0.75 * Ac) * (1 - (np.sqrt(Fb / Ac)))

    FV2p = np.min(FV2p, 0)

    return FV2p


def calculate_FH01p_perpendicular(
    FH2p: float | npt.NDArray[np.float64],
):

    FH01p = 1.6 * FH2p

    return FH01p


def calculate_FV01p_perpendicular(
    FV2p: float | npt.NDArray[np.float64],
    sop: float | npt.NDArray[np.float64],
):

    FV01p = (2.88 - 32 * sop) * FV2p

    return FV01p


def check_FV1p(
    sop: float | npt.NDArray[np.float64],
):

    pass
