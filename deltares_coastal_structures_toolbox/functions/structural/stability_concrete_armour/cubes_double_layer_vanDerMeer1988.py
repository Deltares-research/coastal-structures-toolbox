# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility

unit_properties = {
    "nv": 0.47,
    "kt": 1.10,
    "nlayers": 2,
    "Delta_x_fact_Dn": 1.7,  # although irregular placement
    "Delta_y_fact_Dn": 0.85,
}


def check_validity_range_vanDerMeer1988(
    Hs: float | npt.NDArray[np.float64],
    Tm: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
):

    if not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Significant wave height Hs",
            "Double layer cubes vander Meer 1988",
            Hs,
            0.01,
            20,
        )

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "Number of waves N_waves",
            "Double layer cubes vander Meer 1988",
            N_waves,
            0,
            7500,
        )

    if not np.any(np.isnan(Tm)):
        core_utility.check_variable_validity_range(
            "Wave period Tm",
            "Double layer cubes vander Meer 1988",
            Tm,
            0.5,
            30,
        )

    if not np.any(np.isnan(rho_armour)):
        core_utility.check_variable_validity_range(
            "Armour density rho_armour",
            "Double layer cubes vander Meer 1988",
            rho_armour,
            2000,
            3100,
        )

    if not np.any(np.isnan(cot_alpha)):
        core_utility.check_variable_validity_range(
            "Cotangent of outer structure slope cot_alpha",
            "Double layer cubes vander Meer 1988",
            cot_alpha,
            1.5,
            1.5,
        )

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tm)):
        som = core_physics.calculate_wave_steepness_s(H=Hs, T=Tm)
        core_utility.check_variable_validity_range(
            "Wave steepness using Tm and Hs",
            "Double layer cubes vander Meer 1988",
            som,
            0.005,
            0.07,
        )


def calculate_nominal_diameter_Dn_vanDerMeer1988(
    Hs: float | npt.NDArray[np.float64],
    Tm: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: (
        float | npt.NDArray[np.float64]
    ),  # should we include this? it is not an input, but it should be a warning
    Nod: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Determine required nominal diameter Dn based on Hs and Nod for double layer cubes, using van der Meer 1988

    For more details see: van der Meer 1988 (PhD thesis) and Rock Manual:
    van der Meer 1988, available here: https://repository.tudelft.nl/record/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591

    For more properties, see also unit_properties

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tm : float | npt.NDArray[np.float64]
        Mean wave period (s)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
        Formula only valid for 1.5, otherwise a warning is raised
    Nod: float | npt.NDArray[np.float64], optional
        Damage number, the number of displaced units per width Dn across armour face (-)

    Returns
    -------
    Dn: float | npt.NDArray[np.float64]
        Nominal block diameter, or equivalent cube size (m)
    """

    check_validity_range_vanDerMeer1988(
        Hs=Hs,
        Tm=Tm,
        rho_armour=rho_armour,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Nod=Nod,
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    s0m = core_physics.calculate_wave_steepness_s(H=Hs, T=Tm)
    Ns = ((6.7 * ((Nod**0.4) / (N_waves**0.3))) + 1.0) * s0m**-0.1
    Dn = core_physics.check_usage_stabilitynumber(Ns=Ns, Hs=Hs, Delta=Delta)[0]

    return Dn


def calculate_wave_height_Hs_vanDerMeer1988(
    Dn: float | npt.NDArray[np.float64],
    s0m: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Nod: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Determine required nominal diameter Hs based on Dn and Nod for double layer cubes, using van der Meer 1988

    For more details see: van der Meer 1988 (PhD thesis) and Rock Manual:
    van der Meer 1988, available here: https://repository.tudelft.nl/record/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591

    For more properties, see also unit_properties

    Parameters
    ----------
    Dn: float | npt.NDArray[np.float64]
        Nominal block diameter, or equivalent cube size (m)
    s0m : float | npt.NDArray[np.float64]
        Fictitious wave steepness for mean period wave (-)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
        Formula only valid for 1.5, otherwise a warning is raised
    Nod: float | npt.NDArray[np.float64], optional
        Damage number, the number of displaced units per width Dn across armour face (-)

    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    Ns = ((6.7 * ((Nod**0.4) / (N_waves**0.3))) + 1.0) * s0m**-0.1
    Hs = core_physics.check_usage_stabilitynumber(Ns=Ns, Dn=Dn, Delta=Delta)[0]
    Tm = ((Hs / s0m) / (9.81 / (2 * np.pi))) ** 0.5

    check_validity_range_vanDerMeer1988(
        Hs=Hs,
        Tm=Tm,
        rho_armour=rho_armour,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Nod=Nod,
    )

    return Hs


def calculate_damage_Nod_vanDerMeer1988(
    Hs: float | npt.NDArray[np.float64],
    Dn: float | npt.NDArray[np.float64],
    s0m: float | npt.NDArray[np.float64],
    rho_water: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    N_waves: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Determine required nominal diameter Hs based on Dn and Nod for double layer cubes, using van der Meer 1988

    For more details see: van der Meer 1988 (PhD thesis) and Rock Manual:
    van der Meer 1988, available here: https://repository.tudelft.nl/record/uuid:67e5692c-0905-4ddd-8487-37fdda9af6b4
    or more information in the Rock Manual (2007 / 2012):
    https://kennisbank-waterbouw.nl/DesignCodes/rockmanual/BWchapter%205.pdf page 591

    For more properties, see also unit_properties

    Parameters
    ----------
    Dn: float | npt.NDArray[np.float64]
        Nominal block diameter, or equivalent cube size (m)
    s0m : float | npt.NDArray[np.float64]
        Fictitious wave steepness for mean period wave (-)
    rho_water : float | npt.NDArray[np.float64]
        Water density (kg/m^3)
    rho_armour : float | npt.NDArray[np.float64]
        Armour density (kg/m^3)
    N_waves : float | npt.NDArray[np.float64]
        Number of waves
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
        Formula only valid for 1.5, otherwise a warning is raised
    Nod: float | npt.NDArray[np.float64], optional
        Damage number, the number of displaced units per width Dn across armour face (-)

    Returns
    -------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=rho_water
    )

    Ns = core_physics.check_usage_stabilitynumber(Dn=Dn, Hs=Hs, Delta=Delta)[0]

    Nod = ((((Ns / s0m**-0.1) - 1) / 6.7) * N_waves**0.3) ** (1 / 0.4)

    Tm = ((Hs / s0m) / (9.81 / (2 * np.pi))) ** 0.5

    check_validity_range_vanDerMeer1988(
        Hs=Hs,
        Tm=Tm,
        rho_armour=rho_armour,
        N_waves=N_waves,
        cot_alpha=cot_alpha,
        Nod=Nod,
    )

    return Nod
