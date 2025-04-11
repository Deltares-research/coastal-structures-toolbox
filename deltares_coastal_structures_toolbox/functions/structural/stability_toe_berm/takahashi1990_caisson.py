# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt
import warnings

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity(
    Ns: float | npt.NDArray[np.float64] = np.nan,
    Bt: float | npt.NDArray[np.float64] = np.nan,
):
    if not np.any(np.isnan(Ns)):
        core_utility.check_variable_validity_range(
            "Stability number Ns",
            "Takahashi et al (1990)",
            Ns,
            0,
            np.inf,
        )

    if not np.any(np.isnan(Bt)):
        core_utility.check_variable_validity_range(
            "Toe width Bt",
            "Takahashi et al (1990)",
            Bt,
            0,
            np.inf,
        )


def calculate_nominal_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Tp: float | npt.NDArray[np.float64],
    hacc: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """calculate nominal diameter Dn50 for toe structure of caisson using Takahashi eta al 1990

    For more information, please refer to:
    Takahashi, S., K. Tanimoto and K. Shimosako, 1990. “Wave and block forces on a caisson
        covered with wave dissipating blocks.” Report of Port and Harbour Research Institute, Vol.
        30, No.4, Yokosuka, Japan, p. 3-34.

    AND

    Tanimoto, K., T. Yagyu and Y. Goda, 1983. “Irregular wave tests for composite breakwater
        foundations.” In proc. 18th int. conf. on Coastal Engineering, 14-19 Nov. 1982, Vol.III,
        ASCE, New York, p. 2144-2163

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Tp : float | npt.NDArray[np.float64]
        Wave period at the peak of the spectrum (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
     g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    Ns = calculate_stability_number_Ns(Hs=Hs, Tp=Tp, hacc=hacc, Bt=Bt, beta=beta, g=g)

    Dn50 = core_physics.check_usage_stabilitynumber(Hs=Hs, Ns=Ns, Delta=Delta)
    Dn50 = Dn50[0]
    # check_validity(Hs=Hm0, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Dn50


def calculate_significant_wave_height_Hs(
    Dn50: float | npt.NDArray[np.float64],
    Tp: float | npt.NDArray[np.float64],
    hacc: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """calculate significant wave height Hs for toe structure of caisson using Takahashi eta al 1990

    For more information, please refer to:
    Takahashi, S., K. Tanimoto and K. Shimosako, 1990. “Wave and block forces on a caisson
        covered with wave dissipating blocks.” Report of Port and Harbour Research Institute, Vol.
        30, No.4, Yokosuka, Japan, p. 3-34.

    AND

    Tanimoto, K., T. Yagyu and Y. Goda, 1983. “Irregular wave tests for composite breakwater
        foundations.” In proc. 18th int. conf. on Coastal Engineering, 14-19 Nov. 1982, Vol.III,
        ASCE, New York, p. 2144-2163

    For this method an iterative method is used

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    Tp : float | npt.NDArray[np.float64]
        Mean energy wave period or spectral wave period (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
     g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81


    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    # first guess Ns and iteration
    Ns = 1.8
    Hs_start = core_physics.check_usage_stabilitynumber(Dn=Dn50, Ns=Ns, Delta=Delta)[0]

    Ns = calculate_stability_number_Ns(
        Hs=Hs_start, Tp=Tp, hacc=hacc, Bt=Bt, beta=beta, g=g
    )
    Hs_new = core_physics.check_usage_stabilitynumber(Dn=Dn50, Ns=Ns, Delta=Delta)[0]

    # span coarse between Hs_prev and Hs_new*2
    Hs_table_in = np.arange(
        start=np.minimum(Hs_start, Hs_new),
        step=Hs_start / 20,
        stop=np.maximum(Hs_start, Hs_new),
    )
    Hs_table_out = np.array([])

    for Hsin in Hs_table_in:
        Ns = calculate_stability_number_Ns(
            Hs=Hsin, Tp=Tp, hacc=hacc, Bt=Bt, beta=beta, g=g
        )
        Hs_table_out = np.append(
            Hs_table_out,
            core_physics.check_usage_stabilitynumber(Dn=Dn50, Ns=Ns, Delta=Delta)[0],
        )

    ind = np.argmin(np.abs(Hs_table_in - Hs_table_out))

    fromto = [
        np.minimum(Hs_table_in[ind - 1], Hs_table_out[ind - 1]),
        np.maximum(Hs_table_in[ind + 1], Hs_table_out[ind + 1]),
    ]

    Hs_table_in = np.arange(start=fromto[0], stop=fromto[1], step=0.0005)
    Hs_table_out = np.array([])

    for Hsin in Hs_table_in:
        Ns = calculate_stability_number_Ns(
            Hs=Hsin, Tp=Tp, hacc=hacc, Bt=Bt, beta=beta, g=g
        )
        Hs_table_out = np.append(
            Hs_table_out,
            core_physics.check_usage_stabilitynumber(Dn=Dn50, Ns=Ns, Delta=Delta)[0],
        )
    ind = np.argmin(np.abs(Hs_table_in - Hs_table_out))
    absdiff = np.abs(Hs_table_in[ind] - Hs_table_out[ind])

    if not absdiff < 0.001:
        warnings.warn("Hs precision is not better than {}m".format(absdiff))

    Hs = Hs_table_out[ind]

    # check_validity(Hs=Hm0, tt=tt, ht=ht, cot_alpha_armour_slope=cot_alpha_armour_slope)

    return Hs


def calculate_significant_depth_above_toe_hacc(
    Dn50: float | npt.NDArray[np.float64],
    Tp: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    rho_rock: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """calculate water level above toe structure of caisson using Takahashi et al 1990

    For more information, please refer to:
    Takahashi, S., K. Tanimoto and K. Shimosako, 1990. “Wave and block forces on a caisson
        covered with wave dissipating blocks.” Report of Port and Harbour Research Institute, Vol.
        30, No.4, Yokosuka, Japan, p. 3-34.

    AND

    Tanimoto, K., T. Yagyu and Y. Goda, 1983. “Irregular wave tests for composite breakwater
        foundations.” In proc. 18th int. conf. on Coastal Engineering, 14-19 Nov. 1982, Vol.III,
        ASCE, New York, p. 2144-2163

    For this method an iterative method is used

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        Nominal diameter of toe armour (m)
    Tp : float | npt.NDArray[np.float64]
        Mean energy wave period or spectral wave period (s)
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    rho_rock : float | npt.NDArray[np.float64]
        Density of rock material (kg/m^3)
    rho_water : float | npt.NDArray[np.float64]
        Density of water (kg/m^3)
     g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_rock, rho_water=rho_water
    )

    # determine Ns to be used
    Ns = core_physics.check_usage_stabilitynumber(Dn=Dn50, Hs=Hs, Delta=Delta)[0]

    # set iteration parameters
    dht = Hs / 20
    hacc_in = dht
    Ns_out = calculate_stability_number_Ns(
        Hs=Hs, Tp=Tp, hacc=hacc_in, Bt=Bt, beta=beta, g=g
    )

    hacc_array = np.array([])
    Ns_array = np.array([])

    firsteval = Ns < Ns_out

    while firsteval == (Ns < Ns_out):
        hacc_in += dht
        Ns_out = calculate_stability_number_Ns(
            Hs=Hs, Tp=Tp, hacc=hacc_in, Bt=Bt, beta=beta, g=g
        )

        hacc_array = np.append(hacc_array, hacc_in)
        Ns_array = np.append(Ns_array, Ns_out)

    hacc = hacc_array[-1]

    # make more precise if needed, precision of 1mm is deemed sufficient:
    if dht > 0.001:
        hacc_in = hacc - dht
        dht = 0.001
        Ns_out = calculate_stability_number_Ns(
            Hs=Hs, Tp=Tp, hacc=hacc_in, Bt=Bt, beta=beta, g=g
        )
        hacc_array = np.array([])
        Ns_array = np.array([])
        firsteval = Ns < Ns_out

        while firsteval == (Ns < Ns_out):
            hacc_in += dht
            Ns_out = calculate_stability_number_Ns(
                Hs=Hs, Tp=Tp, hacc=hacc_in, Bt=Bt, beta=beta, g=g
            )

            hacc_array = np.append(hacc_array, hacc_in)
            Ns_array = np.append(Ns_array, Ns_out)
        hacc = hacc_array[-1]

    return hacc


def calculate_stability_number_Ns(
    Hs: float | npt.NDArray[np.float64],
    Tp: float | npt.NDArray[np.float64],
    hacc: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64],
    g: float | npt.NDArray[np.float64] = 9.81,
) -> float | npt.NDArray[np.float64]:
    """calculate stability number Ns for for toe structure of caisson using Takahashi eta al 1990

    For more information, please refer to:
    Takahashi, S., K. Tanimoto and K. Shimosako, 1990. “Wave and block forces on a caisson
        covered with wave dissipating blocks.” Report of Port and Harbour Research Institute, Vol.
        30, No.4, Yokosuka, Japan, p. 3-34.

    AND

    Tanimoto, K., T. Yagyu and Y. Goda, 1983. “Irregular wave tests for composite breakwater
        foundations.” In proc. 18th int. conf. on Coastal Engineering, 14-19 Nov. 1982, Vol.III,
        ASCE, New York, p. 2144-2163
    DOI: https://doi.org/10.9753/icce.v18.128

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Incident wave height near the toe (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Mean energy wave period or spectral wave period (s)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe (m)
    Bt : float | npt.NDArray[np.float64]
        Width of toe structure (m)
    g : float | npt.NDArray[np.float64], optional
        Gravitational acceleration, by default 9.81

    Returns
    -------
    Ns : float | npt.NDArray[np.float64]
        Stability number Hs/Delta*Dn (-)
    """

    beta = np.deg2rad(beta)

    Lp = core_physics.calculate_local_wavelength(T=Tp, h=hacc, g=g)
    k = (2 * np.pi) / Lp

    hacck2 = 2 * k * hacc
    kappa_1 = (hacck2) / (np.sinh(hacck2))

    kappa_2a = 0.45 * (np.sin(beta)) ** 2 * (np.cos(k * Bt * np.cos(beta))) ** 2
    kappa_2b = (np.cos(beta) ** 2) * (np.sin(k * Bt * np.cos(beta))) ** 2
    kappa_2 = np.maximum(kappa_2a, kappa_2b)

    kappa = kappa_1 * kappa_2
    a = (1 - kappa) / kappa ** (1 / 3)
    b = (1 - kappa) ** 2 / kappa ** (1 / 3)
    Ns = np.maximum(
        1.8,
        (1.3 * a * hacc / Hs + 1.8 * np.exp(-1.5 * b * (hacc / Hs))),
    )

    check_validity(Ns=Ns, Bt=Bt)

    return Ns
