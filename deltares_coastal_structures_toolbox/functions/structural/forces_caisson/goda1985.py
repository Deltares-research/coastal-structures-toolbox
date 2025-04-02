# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

# import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
# import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_wave_toolbox.cores.core_dispersion as dispersion


def check_validity_range():

    # tup larger then Bup
    pass


def check_impulsive_breaking():
    pass


def calculate_forces_and_reactions(
    HD: float | npt.NDArray[np.float64],  # design wave height (Hmax)
    Hsi: float | npt.NDArray[np.float64],  # incident wave height (Hs)
    Tmax: float | npt.NDArray[np.float64],  # max wave period
    beta: float | npt.NDArray[np.float64],  # wave angle wrt normal
    h_s: float | npt.NDArray[np.float64],  # water depth at site
    d: float | npt.NDArray[np.float64],  # water depth above toe berm
    cota_seabed: float | npt.NDArray[np.float64],  # Seabed slope near structure toe
    rho_water: float | npt.NDArray[np.float64],
    hacc: (
        float | npt.NDArray[np.float64]
    ),  # distance from sea water level to base of caisson
    Rc: float | npt.NDArray[np.float64],  # Crest freeboard above SWL
    B_up: float | npt.NDArray[np.float64],  # Width of upright section of caisson
    rho_fill_above_SWL: (
        float | npt.NDArray[np.float64]
    ) = 2400,  # density of fill material above water level, here taken as concrete
    rho_fill_below_SWL: (
        float | npt.NDArray[np.float64]
    ) = 2100,  # density of fill material below water level
    muf: float | npt.NDArray[np.float64] = 0.6,  # Friction factor with rubble mound
    t_upoverB_up: (
        float | npt.NDArray[np.float64]
    ) = 0.5,  # Ratio between width of caisson and centre of mass caisson
    g: float | npt.NDArray[np.float64] = 9.81,
    return_dict: bool = False,
) -> float | npt.NDArray[np.float64] | dict:

    # beta, if not 0 should be rotated by 15 degrees towards the normal of the breakwater according to Goda (2000)
    beta = np.abs(beta)
    if beta >= 15:
        beta_use = beta - 15
    elif beta < 15:
        beta_use = 0
    cos_beta = np.cos(np.deg2rad(beta_use))

    h_5Hs = h_s + (Hsi * 5) / cota_seabed

    etastar = 0.75 * (1 + cos_beta) * HD

    L = calculate_local_wavelength(T=Tmax, h=h_s, g=g)  # local wave length
    L0 = (g / 2 * np.pi) * Tmax**2

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
    FU = 0.5 * pu * B_up
    MU = (2 / 3) * FU * B_up

    # Bearing pressure
    tup = t_upoverB_up * B_up
    Wup = calculate_Wup(
        hacc=hacc,
        Rc=Rc,
        B_up=B_up,
        rho_fill_below_SWL=rho_fill_below_SWL,
        rho_water=rho_water,
        rho_fill_above_SWL=rho_fill_above_SWL,
        g=g,
    )

    pe, Me, We, te = calculate_bearing_pressures(
        Wup=Wup, tup=tup, MH=MH, MU=MU, FU=FU, B_up=B_up
    )

    # Safety factors
    SF_sliding, SF_overturning = calculate_safety_factors(
        FU=FU, FH=FH, Wup=Wup, muf=muf, tup=tup, MH=MH, MU=MU
    )

    if not return_dict:
        return FH, FU, MH, MU
    else:
        # all_results = dict()
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
            "B_up": B_up,
            "g": g,
        }

        return all_results


def calculate_Wup(
    hacc: float | npt.NDArray[np.float64],  # height below water level
    Rc: float | npt.NDArray[np.float64],  # distance top caisson to water level
    B_up: float | npt.NDArray[np.float64],  # Width of upright section
    rho_fill_below_SWL: float | npt.NDArray[np.float64],  # rho of fill under water
    rho_water: float | npt.NDArray[np.float64],  # rho of water
    rho_fill_above_SWL: float | npt.NDArray[np.float64],  # rho of fill above water
    g: float | npt.NDArray[np.float64] = 9.81,  # gravity
):

    Wup = (hacc * (rho_fill_below_SWL - rho_water) + Rc * rho_fill_above_SWL) * B_up * g

    return Wup


def calculate_safety_factors(
    FU: float | npt.NDArray[np.float64],  # Force uplift
    FH: float | npt.NDArray[np.float64],  # Force Horizontal
    Wup: float | npt.NDArray[np.float64],  # Weight upright section
    muf: (
        float | npt.NDArray[np.float64]
    ),  # Friction factor between caisson and underlayer
    tup: (
        float | npt.NDArray[np.float64]
    ),  # Distance center of gravity and heel of caisson
    MH: float | npt.NDArray[np.float64],  # Moment due to horizontal wave pressure
    MU: float | npt.NDArray[np.float64],  # Moment due to uplift pressure
):

    # Safety factors
    SF_sliding = muf * (Wup - FU) / FH
    SF_overturning = ((Wup * tup) - MU) / MH

    return SF_sliding, SF_overturning


def calculate_bearing_pressures(
    Wup: float | npt.NDArray[np.float64],  # Weight upright section
    tup: (
        float | npt.NDArray[np.float64]
    ),  # Distance center of gravity and heel of caisson
    MH: float | npt.NDArray[np.float64],  # Moment due to horizontal wave pressure
    MU: float | npt.NDArray[np.float64],  # Moment due to uplift pressure
    FU: float | npt.NDArray[np.float64],  # Force uplift
    B_up: float | npt.NDArray[np.float64],  # Width of upright section
):

    Me = Wup * tup - MU - MH
    We = Wup - FU
    te = Me / We

    if te <= 1 / 3 * B_up:
        pe = (2 * We) / (3 * te)
    else:
        pe = (2 * We / B_up) * (2 - 3 * (te / B_up))

    return pe, Me, We, te


def calculate_local_wavelength(
    T: float | npt.NDArray[np.float64],  # wave period
    h: float | npt.NDArray[np.float64],  # water depth
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:

    # implementation of disper handles arrays for T, but not for h, hence this implementation
    if isinstance(h, float):
        k = dispersion.disper(w=((2 * np.pi) / T), h=h, g=g)
    elif len(h) > 1 and isinstance(T, float):
        k = np.array([])
        for hsub in h:
            k = np.append(k, dispersion.disper(w=((2 * np.pi) / T), h=hsub, g=g))
    elif len(h) > 1 and len(h) == len(T):
        k = np.array([])
        for hsub, Tsub in zip(h, T):
            k = np.append(k, dispersion.disper(w=((2 * np.pi) / Tsub), h=hsub, g=g))

    L = (2 * np.pi) / k

    return L
