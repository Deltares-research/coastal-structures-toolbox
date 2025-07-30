# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

import numpy as np
import numpy.typing as npt
from typing import Union
import deltares_wave_toolbox.cores.core_dispersion as dispersion


def calculate_wave_steepness_s(
    H: float | npt.NDArray[np.float64],
    T: float | npt.NDArray[np.float64],
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate wave steepness

    Determines the wave steepness based on the deep water wave lenght corresponding to the wave period supplied.

    Parameters
    ----------
    H : float | npt.NDArray[np.float64]
        Wave height (m)
    T : float | npt.NDArray[np.float64]
        Wave period Tm-1,0 (s)
    g : float, optional
        Gravitational constant (m/s^2), by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        Deep water wave steepness s (-)
    """

    s = (2 * np.pi / g) * H / np.power(T, 2)
    return s


def calculate_Irribarren_number_ksi(
    H: float | npt.NDArray[np.float64],
    T: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    s = calculate_wave_steepness_s(H, T)
    ksi = (1.0 / cot_alpha) / np.sqrt(s)
    return ksi


def calculate_critical_Irribarren_number_ksi_mc(c_pl, c_s, P, cot_alpha):

    ksi_mc = np.power(
        (c_pl / c_s) * np.power(P, 0.31) * np.sqrt(1.0 / cot_alpha),
        1.0 / (P + 0.5),
    )
    return ksi_mc


def calculate_stability_number_Ns(
    H: float | npt.NDArray[np.float64],
    D: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    Delta = calculate_buoyant_density_Delta(rho_rock, rho_water)
    Ns = H / (Delta * D)
    return Ns


def calculate_buoyant_density_Delta(
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    Delta = (rho_rock - rho_water) / rho_water
    return Delta


def calculate_M50_from_Dn50(
    Dn50: float | npt.NDArray[np.float64], rho_rock: float = 2650
) -> float | npt.NDArray[np.float64]:

    M50 = rho_rock * np.power(Dn50, 3)
    return M50


def calculate_Dn50_from_M50(
    M50: float | npt.NDArray[np.float64], rho_rock: float = 2650
) -> float | npt.NDArray[np.float64]:

    Dn50 = np.power(M50 / rho_rock, 1 / 3)
    return Dn50


def check_usage_Dn50_or_M50(
    Dn50: float | npt.NDArray[np.float64],
    M50: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    if np.all(np.isnan(Dn50)) and np.all(np.isnan(M50)):
        raise ValueError("Either Dn50 or M50 should be provided")

    if np.all(np.isnan(Dn50)) and not np.all(np.isnan(M50)):
        if np.all(np.isnan(rho_armour)):
            warnings.warn(
                "rho_armour is not provided, the default value of 2650 kg/m3 is used."
            )
            Dn50 = calculate_Dn50_from_M50(M50)
        else:
            Dn50 = calculate_Dn50_from_M50(M50, rho_rock=rho_armour)
    return Dn50


def calculate_S_from_Nod(
    Nod: float | npt.NDArray[np.float64],
    G: float | npt.NDArray[np.float64],
    nv: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Estimates S value from Nod value

    Estimation of S damage value from Nod damage value according to CEM (2006)
    and the Rock Manual (2007, 2012)

    Parameters
    ----------
    Nod : float | npt.NDArray[np.float64]
        Number of displaced units normalized to 1 unit width of structure
    G : float | npt.NDArray[np.float64]
        Gradation factor depending on armour layer, G = 1 for concrete armour
         units and 1.2 - 1.6 for stone armor
    nv : float | npt.NDArray[np.float64]
        Porosity depending on armour layer, generally between 0.4 and 0.6

    In general the S is about twice the value of Nod

    Returns
    -------
    S: float | npt.NDArray[np.float64]
        Damage value based on eroded cross sectional area
    """

    factor = G * (1 - nv)
    S = Nod / factor

    return S


def calculate_Nod_from_S(
    S: float | npt.NDArray[np.float64],
    G: float | npt.NDArray[np.float64],
    nv: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Estimates Nod value from S value

    Estimation of Nod damage value from S damage value according to CEM (2006)
    and the Rock Manual (2007, 2012)

    Parameters
    ----------
    S : float | npt.NDArray[np.float64]
        Damage value based on eroded cross sectional area
    G : float | npt.NDArray[np.float64]
        Gradation factor depending on armour layer, G = 1 for concrete armour
         units and 1.2 - 1.6 for stone armor
    nv : float | npt.NDArray[np.float64]
        Porosity depending on armour layer, generally between 0.4 and 0.6

    In general the Nod is about half the value of S

    Returns
    -------
    Nod: float | npt.NDArray[np.float64]
        Number of displaced units normalized to 1 unit width of structure
    """
    factor = G * (1 - nv)
    Nod = S * factor

    return Nod


def check_usage_stabilitynumber(
    Hs: float | npt.NDArray[np.float64] = None,
    Dn: float | npt.NDArray[np.float64] = None,
    Delta: float | npt.NDArray[np.float64] = None,
    Ns: float | npt.NDArray[np.float64] = None,
) -> Union[float | npt.NDArray[np.float64], str]:
    """Calculates missing value from stability number Ns = Hs / Delta*Dn

    Parameter that is None in the input will be calculated

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m), by default None
    Dn : float | npt.NDArray[np.float64], optional
        Nomincal diameter (m), by default None
    Delta : float | npt.NDArray[np.float64], optional
        Relative buoyant density of material, by default None
    Ns : float | npt.NDArray[np.float64], optional
        Stability number, by default None

    Returns
    -------
    Union[float | npt.NDArray[np.float64], str]
        (missing parameter value, missing parameter as calculated)

    Raises
    ------
    ValueError
        More then one missing variable
    ValueError
        No missing variable
    """
    allchecks = [Hs is None, Dn is None, Delta is None, Ns is None]
    if sum(allchecks) > 1:
        raise ValueError("More then one missing variable")
    elif sum(allchecks) == 0:
        raise ValueError("No missing variable")

    if Hs is None and not (Dn is None or Delta is None or Ns is None):
        # calculate Hs
        out1 = Dn * Delta * Ns
        out2 = "Hs"
    elif Dn is None and not (Hs is None or Delta is None or Ns is None):
        # calculate Dn
        out1 = Hs / Ns / Delta
        out2 = "Dn"
    elif Delta is None and not (Dn is None or Hs is None or Ns is None):
        # calculate Delta
        out1 = Hs / Ns / Dn
        out2 = "Delta"
    elif Ns is None and not (Dn is None or Hs is None or Delta is None):
        # calculate Ns
        out1 = Hs / (Delta * Dn)
        out2 = "Ns"

    return out1, out2


def calculate_local_wavelength(
    T: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate local wave length for wave with period T at depth h, using approximation of dispersion relation

    Parameters
    ----------
    T : float | npt.NDArray[np.float64]
        Wave period
    h : float | npt.NDArray[np.float64]
        Water depth
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    L : float | npt.NDArray[np.float64]
        Wave length at local water depth
    """

    # implementation of disper handles arrays for T, but not for h, hence this implementation
    if isinstance(h, (float, int)):
        k = dispersion.disper(w=((2 * np.pi) / T), h=h, g=g)[0]
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
