# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


def check_validity_range(
    beta: float | npt.NDArray[np.float64] = np.nan,  # wave angle wrt normal
    cota_seabed: float | npt.NDArray[np.float64] = np.nan,  # seabed slope
    Tmax: float | npt.NDArray[np.float64] = np.nan,  # wave period
    tup: (
        float | npt.NDArray[np.float64]
    ) = np.nan,  # distance caisson heel to centre of gravity
    Bup: float | npt.NDArray[np.float64] = np.nan,  # Width of caisson upright section
    h_s: float | npt.NDArray[np.float64] = np.nan,  # water depth at site
    B1: float | npt.NDArray[np.float64] = np.nan,  # width toe berm
):

    all_is_valid = []
    if not np.any(np.isnan(beta)):
        out = core_utility.check_variable_validity_range(
            "Angle of wave incidence", "Goda (2000)", beta, 0.0, 90.0
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(cota_seabed)):
        out = core_utility.check_variable_validity_range(
            "Seabed slope", "Goda (2000)", cota_seabed, 10.0, 100.0
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(Tmax)):
        out = core_utility.check_variable_validity_range(
            "Wave period", "Goda (2000)", Tmax, 0.0, 30.0
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(tup)) and not np.any(np.isnan(Bup)):
        out = core_utility.check_variable_validity_range(
            "Centre of gravity inside of caisson", "Goda (2000)", tup / Bup, 0.0, 1.0
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(h_s)):
        out = core_utility.check_variable_validity_range(
            "Water depth at site",
            "Goda (2000)",
            h_s,
            0.0,
            np.inf,  # no upper limit provided
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(B1)):
        out = core_utility.check_variable_validity_range(
            "Width of toe berm", "Goda (2000)", B1, 0.0, np.inf
        )
        all_is_valid.append(out)

    if not np.any(np.isnan(Bup)):
        out = core_utility.check_variable_validity_range(
            "Width of caisson upright section", "Goda (2000)", Bup, 0.0, np.inf
        )
        all_is_valid.append(out)

    return all_is_valid


def check_impulsive_breaking(
    beta: float | npt.NDArray[np.float64] = np.nan,  # wave angle wrt normal
    B1: float | npt.NDArray[np.float64] = np.nan,  # width toe berm
    L: float | npt.NDArray[np.float64] = np.nan,  # local wave length
    cota_seabed: float | npt.NDArray[np.float64] = np.nan,  # seabed slope
    offshore_wave_steepness: (
        float | npt.NDArray[np.float64]
    ) = np.nan,  # Hoffshore/Loffshoredeep
    Rc: float | npt.NDArray[np.float64] = np.nan,  # Crest freeboard
    Hsi: float | npt.NDArray[np.float64] = np.nan,  # Incident wave height
    d: float | npt.NDArray[np.float64] = np.nan,  # water depth above toe berm
    h_s: float | npt.NDArray[np.float64] = np.nan,  # water depth at site
):

    all_is_nans = [
        np.isnan(beta),
        np.isnan(B1),
        np.isnan(L),
        np.isnan(cota_seabed),
        np.isnan(offshore_wave_steepness),
        np.isnan(Rc),
        np.isnan(Hsi),
    ]

    if not any(all_is_nans):
        if (
            beta < 20
            and B1 / L < 0.02
            and cota_seabed < 50
            and offshore_wave_steepness < 0.03
            and Rc / Hsi > 0.3
        ):

            core_utility.check_variable_validity_range(
                "Impulsive breaking condition 1", "Goda (2000)", True, False, False
            )
            return True

    all_is_nans = [
        np.isnan(beta),
        np.isnan(B1),
        np.isnan(L),
        np.isnan(d),
        np.isnan(h_s),
        np.isnan(Rc),
        np.isnan(Hsi),
    ]

    if not any(all_is_nans):
        if (
            beta < 20
            and 0.02 < B1 / L
            and B1 / L < 0.3
            and d / h_s < 0.6
            and Rc / Hsi > 0.3
        ):
            core_utility.check_variable_validity_range(
                "Impulsive breaking condition 2", "Goda (2000)", True, False, False
            )
            return True

    return False


def calculate_forces_and_reactions(
    HD: float | npt.NDArray[np.float64],
    Hsi: float | npt.NDArray[np.float64],
    Tmax: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    h_s: float | npt.NDArray[np.float64],
    d: float | npt.NDArray[np.float64],
    B1: float | npt.NDArray[np.float64],
    cota_seabed: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    hacc: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bup: float | npt.NDArray[np.float64],
    rho_fill_above_SWL: float | npt.NDArray[np.float64] = 2400,
    rho_fill_below_SWL: float | npt.NDArray[np.float64] = 2100,
    offshore_wave_steepness: float | npt.NDArray[np.float64] = np.nan,
    muf: float | npt.NDArray[np.float64] = 0.6,
    tup_over_Bup: float | npt.NDArray[np.float64] = 0.5,
    g: float | npt.NDArray[np.float64] = 9.81,
    return_dict: bool = False,
) -> (
    tuple[
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
    ]
    | dict
):
    """Calculate wave induced pressures and forces on caisson and its reaction forces

    Calculation of forces, moments and reactions of waves on caisson structures according to Goda (1985, 2000)
    See:
    Goda, Y., 1985. “Random seas and design of maritime structures.” University of Tokyo Press.,
        Japan. ISBN 0-86008-369-1.
    or
    Goda, Y., 2000. “Random seas and design of maritime structures.” In P.L. Liu (ed) Advanced
        Series on Ocean Engineering, Vol. 15, World Scientific, Singapore, 444 pp. (2nd ed.).

    Note that this set of equations is valid for not impulsively breaking waves

    Parameters
    ----------
    HD : float | npt.NDArray[np.float64]
        design wave height (Hmax) (m)
    Hsi : float | npt.NDArray[np.float64]
        incident wave height (Hs) (m)
    Tmax : float | npt.NDArray[np.float64]
        Maximum wave period (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h_s : float | npt.NDArray[np.float64]
        water depth at site (m)
    d : float | npt.NDArray[np.float64]
        water depth above toe berm (m)
    B1 : float | npt.NDArray[np.float64]
        Width of toe berm (top of toe berm) (m)
    cota_seabed : float | npt.NDArray[np.float64]
        Slope of seabed approaching caisson (-)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    hacc : float | npt.NDArray[np.float64]
        Distance between lowest part of caisson to water level (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard (m)
    Bup : float | npt.NDArray[np.float64]
        Width of upright section of caisson (m)
    rho_fill_above_SWL
        Density of fill material above water level, by default 2400 (example by Goda) (kg/m^3)
    rho_fill_below_SWL
        Density of fill material above water level, by default 2100 (example by Goda) (kg/m^3)
    offshore_wave_steepness
        Offshore wave steepness, used for impulse breaking check (-)
    muf
        Friction factor of bottom caisson with rubble mound (-)
    tup_over_Bup
        Ratio between width of caisson and centre of mass caisson, by default 0.5 (= caisson middle) (-)
    g : float | npt.NDArray[np.float64]
        Gravitational acceleration, by default 9.81 (m/s^2)
    return_dict : bool
        Return a dictionary with all results

    Returns
    -------

    FH, FU, MH, MU, p1, pe, pu, SF_sliding, SF_overturning


    FH : float | npt.NDArray[np.float64]
        Wave induced horizontal force (N/m1)
    FU : float | npt.NDArray[np.float64]
        Wave induced uplift  force (N/m1)
    MH : float | npt.NDArray[np.float64]
        Wave induced horizontal moment (N*m/m1)
    MU : float | npt.NDArray[np.float64]
        Wave induced uplift moment (N*m/m1)
    p1 : float | npt.NDArray[np.float64]
        Wave induced pressure around still water level (N/m^2)
    pe : float | npt.NDArray[np.float64]
        Bearing pressure at heel of caisson (N/m^2)
    pu : float | npt.NDArray[np.float64]
        Wave induced uplift  pressure (N/m^2)
    impulsive_breaking : float | npt.NDArray[np.float64]
        Result of impulsive breaking check. Goda formula is NOT applicable for impulsive breaking waves
    SF_sliding
        Safety factor for sliding
    SF_overturning
        Safety factor for overturning

        OR

    all_results : dict
        All inputs, results and intermediate results

    """

    (
        FH,
        FU,
        MH,
        MU,
        p1,
        p2,
        p3,
        p4,
        hstar_c,
        pu,
        h_5Hs,
        etastar,
        L,
        alpha_1,
        alpha_2,
        alpha_3,
        impulsive_breaking,
    ) = calculate_pressures_and_forces(
        HD=HD,
        Hsi=Hsi,
        Tmax=Tmax,
        beta=beta,
        h_s=h_s,
        d=d,
        cota_seabed=cota_seabed,
        rho_water=rho_water,
        hacc=hacc,
        Rc=Rc,
        Bup=Bup,
        B1=B1,
        g=g,
        offshore_wave_steepness=offshore_wave_steepness,
    )

    # Bearing pressure
    tup = tup_over_Bup * Bup
    Wup = calculate_Wup(
        hacc=hacc,
        Rc=Rc,
        Bup=Bup,
        rho_fill_below_SWL=rho_fill_below_SWL,
        rho_water=rho_water,
        rho_fill_above_SWL=rho_fill_above_SWL,
        g=g,
    )

    pe, Me, We, te = calculate_bearing_pressures(
        Wup=Wup, tup=tup, MH=MH, MU=MU, FU=FU, Bup=Bup
    )

    # Safety factors
    SF_sliding, SF_overturning = calculate_safety_factors(
        FU=FU, FH=FH, Wup=Wup, muf=muf, tup=tup, MH=MH, MU=MU
    )

    # check validity range
    is_valid = check_validity_range(
        beta=beta, cota_seabed=cota_seabed, Tmax=Tmax, tup=tup, Bup=Bup, h_s=h_s
    )

    if not return_dict:
        return (
            FH,
            FU,
            MH,
            MU,
            p1,
            pe,
            pu,
            impulsive_breaking,
            SF_sliding,
            SF_overturning,
        )
    else:
        all_results = {
            "SF_overturning": SF_overturning,
            "SF_sliding": SF_sliding,
            "tup": tup,
            "Wup": Wup,
            "te": te,
            "We": We,
            "Me": Me,
            "pe": pe,
            "FH": FH,
            "MH": MH,
            "FU": FU,
            "MU": MU,
            "hstar_c": hstar_c,
            "h_5Hs": h_5Hs,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "pu": pu,
            "etastar": etastar,
            "alpha1": alpha_1,
            "alpha2": alpha_2,
            "alpha3": alpha_3,
            "HD": HD,
            "Hsi": Hsi,
            "Tmax": Tmax,
            "beta": beta,
            "h_s": h_s,
            "d": d,
            "L": L,
            "cota_seabed": cota_seabed,
            "rho_water": rho_water,
            "hacc": hacc,
            "Rc": Rc,
            "B_up": Bup,
            "g": g,
            "impulsive_breaking": impulsive_breaking,
            "is_valid": is_valid,
        }

        return all_results


def calculate_pressures_and_forces(
    HD: float | npt.NDArray[np.float64],
    Hsi: float | npt.NDArray[np.float64],
    Tmax: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    h_s: float | npt.NDArray[np.float64],
    d: float | npt.NDArray[np.float64],
    B1: float | npt.NDArray[np.float64],
    cota_seabed: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    hacc: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bup: float | npt.NDArray[np.float64],
    offshore_wave_steepness: float | npt.NDArray[np.float64] = np.nan,
    g: float | npt.NDArray[np.float64] = 9.81,
    return_dict: bool = False,
) -> (
    tuple[
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
        bool | npt.NDArray[np.float64],
    ]
    | dict
):
    """Calculate wave induced pressures and forces on caisson

    Calculation of forces, moments and reactions of waves on caisson structures according to Goda (1985, 2000)
    See:
    Goda, Y., 1985. “Random seas and design of maritime structures.” University of Tokyo Press.,
        Japan. ISBN 0-86008-369-1.
    or
    Goda, Y., 2000. “Random seas and design of maritime structures.” In P.L. Liu (ed) Advanced
        Series on Ocean Engineering, Vol. 15, World Scientific, Singapore, 444 pp. (2nd ed.).

    Note that this set of equations is valid for not impulsively breaking waves

    Parameters
    ----------
    HD : float | npt.NDArray[np.float64]
        design wave height (Hmax) (m)
    Hsi : float | npt.NDArray[np.float64]
        incident wave height (Hs) (m)
    Tmax : float | npt.NDArray[np.float64]
        Maximum wave period (s)
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h_s : float | npt.NDArray[np.float64]
        water depth at site (m)
    d : float | npt.NDArray[np.float64]
        water depth above toe berm (m)
    B1 : float | npt.NDArray[np.float64]
        Width of toe berm (top of toe berm) (m)
    cota_seabed : float | npt.NDArray[np.float64]
        Slope of seabed approaching caisson (-)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    hacc : float | npt.NDArray[np.float64]
        Distance between lowest part of caisson to water level (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard (m)
    Bup : float | npt.NDArray[np.float64]
        Width of upright section of caisson (m)
    g : float | npt.NDArray[np.float64]
        Gravitational acceleration, by default 9.81 (m/s^2)
    return_dict : bool
        Return a dictionary with all results

    Returns
    -------
    Wup : float | npt.NDArray[np.float64]
        Weight of upright section of caisson (N)
    FH : float | npt.NDArray[np.float64]
        Wave induced horizontal force (N/m1)
    FU : float | npt.NDArray[np.float64]
        Wave induced uplift  force (N/m1)
    MH : float | npt.NDArray[np.float64]
        Wave induced horizontal moment (N*m/m1)
    MU : float | npt.NDArray[np.float64]
        Wave induced uplift moment (N*m/m1)
    p1 : float | npt.NDArray[np.float64]
        Wave induced pressure around still water level (N/m^2)
    p2 : float | npt.NDArray[np.float64]
        Wave induced pressure at seafloor (N/m^2)
    p3 : float | npt.NDArray[np.float64]
        Wave induced pressure at bottom of caisson(N/m^2)
    p4 : float | npt.NDArray[np.float64]
        Wave induced pressure at top of caisson (N/m^2)
    hstar_c : float | npt.NDArray[np.float64]
        Wave induced pressure (N/m^2)
    pu : float | npt.NDArray[np.float64]
        Wave induced uplift  pressure (N/m^2)
    h_5Hs : float | npt.NDArray[np.float64]
        Water depth at distance of 5 Hs from caisson (m)
    etastar : float | npt.NDArray[np.float64]
        Elevation to which the pressure is exerted (m)
    L : float | npt.NDArray[np.float64]
        Local wave length
    alpha_1 : float | npt.NDArray[np.float64]
        Alpha factor
    alpha_2 : float | npt.NDArray[np.float64]
        Alpha factor
    alpha_3 : float | npt.NDArray[np.float64]
        Alpha factor
    impulsive_breaking : float | npt.NDArray[np.float64]
        Result of impulsive breaking check. Goda formula is NOT applicable for impulsive breaking waves

        OR

    all_results : dict
        All inputs, results and intermediate results

    """

    # beta, if not 0 should be rotated by 15 degrees towards the normal of the breakwater according to Goda (2000)
    beta = np.abs(beta)
    if beta >= 15:
        beta_use = beta - 15
    elif beta < 15:
        beta_use = 0
    cos_beta = np.cos(np.deg2rad(beta_use))

    h_5Hs = h_s + (Hsi * 5) / cota_seabed

    etastar = 0.75 * (1 + cos_beta) * HD

    L = core_physics.calculate_local_wavelength(T=Tmax, h=h_s, g=g)  # local wave length

    # alpha factors
    alpha_1 = (
        0.6 + 0.5 * ((4 * np.pi * h_s / L) / (np.sinh((4 * np.pi * h_s / L)))) ** 2
    )

    alpha_21 = ((h_5Hs - d) / (3 * h_5Hs)) * ((HD / d) ** 2)
    alpha_22 = (2 * d) / HD
    alpha_2 = np.minimum(alpha_21, alpha_22)

    alpha_3 = 1 - (hacc / h_s) * (1 - (1 / (np.cosh(2 * np.pi * h_s / L))))
    # 1 - (hacc / h_s) * (1 - (1 / (np.cosh(2 * np.pi * h_s / L))))

    # pressures
    p1 = 0.5 * (1 + cos_beta) * (alpha_1 + alpha_2 * cos_beta**2) * rho_water * g * HD
    p2 = p1 / (np.cosh(2 * np.pi * h_s / L))
    p3 = alpha_3 * p1
    if etastar > Rc:
        p4 = p1 * (1 - (Rc / etastar))
    else:
        p4 = 0

    hstar_c = np.minimum(etastar, Rc)

    pu = 0.5 * (1 + cos_beta) * alpha_1 * alpha_3 * rho_water * g * HD

    # Forces and Moments
    FH = 0.5 * (p1 + p3) * hacc + 0.5 * (p1 + p4) * hstar_c

    MH = (
        (1 / 6) * (2 * p1 + p3) * hacc**2
        + 0.5 * (p1 + p4) * hacc * hstar_c
        + (1 / 6) * (p1 + 2 * p4) * hstar_c**2
    )
    FU = 0.5 * pu * Bup
    MU = (2 / 3) * FU * Bup

    impulsive_breaking = False
    if check_impulsive_breaking(
        beta=beta,
        B1=B1,
        cota_seabed=cota_seabed,
        d=d,
        h_s=h_s,
        Hsi=Hsi,
        L=L,
        offshore_wave_steepness=offshore_wave_steepness,
        Rc=Rc,
    ):
        impulsive_breaking = True

    if not return_dict:
        return (
            FH,
            FU,
            MH,
            MU,
            p1,
            p2,
            p3,
            p4,
            hstar_c,
            pu,
            h_5Hs,
            etastar,
            L,
            alpha_1,
            alpha_2,
            alpha_3,
            impulsive_breaking,
        )
    else:
        # all_results = dict()
        all_results = {
            "FH": FH,
            "MH": MH,
            "FU": FU,
            "MU": MU,
            "hstar_c": hstar_c,
            "h_5Hs": h_5Hs,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "pu": pu,
            "etastar": etastar,
            "alpha1": alpha_1,
            "alpha2": alpha_2,
            "alpha3": alpha_3,
            "HD": HD,
            "Hsi": Hsi,
            "Tmax": Tmax,
            "beta": beta,
            "h_s": h_s,
            "d": d,
            "L": L,
            "cota_seabed": cota_seabed,
            "rho_water": rho_water,
            "hacc": hacc,
            "Rc": Rc,
            "Bup": Bup,
            "g": g,
            "impulsive_breaking": impulsive_breaking,
        }

        return all_results


def calculate_Wup(
    hacc: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Bup: float | npt.NDArray[np.float64],
    rho_fill_below_SWL: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_fill_above_SWL: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate weight of upright section

    Parameters
    ----------
    hacc : float | npt.NDArray[np.float64]
        height of caisson below water level
    Rc : float | npt.NDArray[np.float64]
        crest freeboard distance top caisson to water level
    Bup : float | npt.NDArray[np.float64]
        Width of upright section
    rho_fill_below_SWL : float | npt.NDArray[np.float64]
        Density of fill placed below water level
    rho_water : float | npt.NDArray[np.float64]
        Density of water
    rho_fill_above_SWL : float | npt.NDArray[np.float64]
        Density of fill placed above water level
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Wup : float | npt.NDArray[np.float64]
        Weight of upright section of caisson
    """

    Wup = (hacc * (rho_fill_below_SWL - rho_water) + Rc * rho_fill_above_SWL) * Bup * g

    return Wup


def calculate_safety_factors(
    FU: float | npt.NDArray[np.float64],
    FH: float | npt.NDArray[np.float64],
    Wup: float | npt.NDArray[np.float64],
    muf: float | npt.NDArray[np.float64],
    tup: float | npt.NDArray[np.float64],
    MH: float | npt.NDArray[np.float64],
    MU: float | npt.NDArray[np.float64],
) -> tuple[
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
]:
    """Calculate safety factors based on pre-calculated forces and moments

    Calculation of forces, moments and reactions of waves on caisson structures according to Goda (1985, 2000)
    See:
    Goda, Y., 1985. “Random seas and design of maritime structures.” University of Tokyo Press.,
        Japan. ISBN 0-86008-369-1.
    or
    Goda, Y., 2000. “Random seas and design of maritime structures.” In P.L. Liu (ed) Advanced
        Series on Ocean Engineering, Vol. 15, World Scientific, Singapore, 444 pp. (2nd ed.).

    Note that this set of equations is valid for not impulsively breaking waves

    Parameters
    ----------
    FU : float | npt.NDArray[np.float64]
        Uplift force
    FH : float | npt.NDArray[np.float64]
        Horizontal force
    Wup : float | npt.NDArray[np.float64]
        Weight of upright section of caisson
    muf : float | npt.NDArray[np.float64]
        Friction factor between caisson and underlayer
    tup : float | npt.NDArray[np.float64]
        Distance center of gravity and heel of caisson
    MH : float | npt.NDArray[np.float64]
        Moment due to horizontal wave pressure
    MU : float | npt.NDArray[np.float64]
        Moment due to uplift pressure

    Returns
    -------
    tuple[ float | npt.NDArray[np.float64], float | npt.NDArray[np.float64], ]
        Safety factors for sliding and overturning
    """

    # Safety factors
    SF_sliding = muf * (Wup - FU) / FH
    SF_overturning = ((Wup * tup) - MU) / MH

    return SF_sliding, SF_overturning


def calculate_bearing_pressures(
    Wup: float | npt.NDArray[np.float64],
    tup: float | npt.NDArray[np.float64],
    MH: float | npt.NDArray[np.float64],
    MU: float | npt.NDArray[np.float64],
    FU: float | npt.NDArray[np.float64],
    Bup: float | npt.NDArray[np.float64],
) -> tuple[
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
]:
    """Calculate pressures on foundation using results from other functions

    Calculation of forces, moments and reactions of waves on caisson structures according to Goda (1985, 2000)
    See:
    Goda, Y., 1985. “Random seas and design of maritime structures.” University of Tokyo Press.,
        Japan. ISBN 0-86008-369-1.
    or
    Goda, Y., 2000. “Random seas and design of maritime structures.” In P.L. Liu (ed) Advanced
        Series on Ocean Engineering, Vol. 15, World Scientific, Singapore, 444 pp. (2nd ed.).

    Note that this set of equations is valid for not impulsively breaking waves

    Parameters
    ----------
    Wup : float | npt.NDArray[np.float64]
        Weight of upright section of caisson
    tup : float | npt.NDArray[np.float64]
        Horizontal distance from heel of caisson to center of gravity
    MH : float | npt.NDArray[np.float64]
        Moment as result of horizontal forces
    MU : float | npt.NDArray[np.float64]
        Moment as result of uplift forces
    FU : float | npt.NDArray[np.float64]
        Uplift force
    Bup : float | npt.NDArray[np.float64]
        Width of upright section of caisson

    Returns
    -------
    pe : float | npt.NDArray[np.float64]
        Bearing pressure at heel
    Me : float | npt.NDArray[np.float64]
        Eccentricity moment
    We : float | npt.NDArray[np.float64]
        Effective weight of caisson
    te : float | npt.NDArray[np.float64]
        Effective arm

    """

    Me = Wup * tup - MU - MH
    We = Wup - FU
    te = Me / We

    if te <= 1 / 3 * Bup:
        pe = (2 * We) / (3 * te)
    else:
        pe = (2 * We / Bup) * (2 - 3 * (te / Bup))

    return pe, Me, We, te
