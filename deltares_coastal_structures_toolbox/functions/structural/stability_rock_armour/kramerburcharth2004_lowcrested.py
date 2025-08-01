# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Kramer & Burcharth (2004) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    the conclusions of Kramer & Burcharth (2004). When parameters are nan (by default), they are not checked.

    For more details see Kramer & Burcharth (2004), available here: https://doi.org/10.1061/40733(147)12

    Parameters
    ----------
    Rc : float | npt.NDArray[np.float64], optional
        Crest freeboard of the structure (m), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    """

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Dn50)):
        core_utility.check_variable_validity_range(
            "Relative crest height Rc / Dn50",
            "Kramer & Burcharth, 2004",
            Rc / Dn50,
            -3.0,
            2.0,
        )

    return


def calculate_nominal_rock_diameter_Dn50(
    Hs: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Ns_init: float = 2.0,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
) -> float | npt.NDArray[np.float64]:
    """Calculate the nominal rock diameter Dn50 for low-crested structures with the Kramer & Burcharth (2004) formula.

    Here, eq. 4 from Kramer & Burcharth (2004) is implemented.

    For more details see Kramer & Burcharth (2004), available here: https://doi.org/10.1061/40733(147)12

    Parameters
    ----------
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    Ns_init : float, optional
        Initial stability number Ns (-) for the iterative solution, by default 2.0
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tolerance : float, optional
        Tolerance for convergence of the iterative solution, by default 1e-5

    Returns
    -------
    float | npt.NDArray[np.float64]
        The nominal rock diameter Dn50 (m)
    """

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    n_iter = 0

    Dn50_diff = np.inf
    Dn50_prev = np.inf
    Dn50 = Hs / (Delta * Ns_init)

    while np.max(Dn50_diff) > tolerance and n_iter < max_iter:
        n_iter += 1

        Dn50 = (1.0 / 1.36) * ((Hs / Delta) - Rc * (0.06 * (Rc / Dn50) - 0.23))

        Dn50_diff = np.abs(Dn50 - Dn50_prev)
        Dn50_prev = Dn50

    check_validity_range(
        Rc=Rc,
        Dn50=Dn50,
    )
    return Dn50


def calculate_significant_wave_height_Hs(
    Rc: float | npt.NDArray[np.float64],
    rho_armour: float | npt.NDArray[np.float64],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:
    """Calculate the maximum significant wave height Hs for low-crested structures with the Kramer & Burcharth (2004)
    formula.

    Here, eq. 4 from Kramer & Burcharth (2004) is implemented.

    For more details see Kramer & Burcharth (2004), available here: https://doi.org/10.1061/40733(147)12

    Parameters
    ----------
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    rho_armour : float | npt.NDArray[np.float64]
        Armour rock density (kg/m^3)
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan

    Returns
    -------
    float | npt.NDArray[np.float64]
        The significant wave height Hs (m)
    """

    Dn50 = core_physics.check_usage_Dn50_or_M50(
        Dn50=Dn50, M50=M50, rho_armour=rho_armour
    )

    Delta = core_physics.calculate_buoyant_density_Delta(
        rho_rock=rho_armour, rho_water=1025
    )

    Hs = (0.06 * np.power(Rc / Dn50, 2.0) - 0.23 * Rc / Dn50 + 1.36) * Delta * Dn50

    check_validity_range(
        Rc=Rc,
        Dn50=Dn50,
    )
    return Hs
