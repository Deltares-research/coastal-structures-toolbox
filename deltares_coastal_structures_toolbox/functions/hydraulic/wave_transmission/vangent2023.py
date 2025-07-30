# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    Rc: float | npt.NDArray[np.float64] = np.nan,
    B: float | npt.NDArray[np.float64] = np.nan,
    structure_type: str = np.nan,
) -> None:
    """Check the parameter values vs the validity range of the Van Gent et al. (2023) formula.

    For all parameters supplied, their values are checked versus the range of test conditions specified in
    the conclusions of Van Gent et al. (2023). When parameters are nan (by default), they are not checked.

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104344

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64], optional
        Spectral significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    Rc : float | npt.NDArray[np.float64], optional
        Crest freeboard of the structure (m), by default np.nan
    B : float | npt.NDArray[np.float64], optional
        Crest width of the structure (m), by default np.nan
    structure_type : str, optional
        Type of structure, by default np.nan
    """

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Rc)):
        if structure_type == "permeable":
            core_utility.check_variable_validity_range(
                "Dimensionless crest height Rc / Hm0",
                "Van Gent et al. (2023)",
                Rc / Hm0,
                -2.5,
                2.5,
            )
        else:
            core_utility.check_variable_validity_range(
                "Dimensionless crest height Rc / Hm0",
                "Van Gent et al. (2023)",
                Rc / Hm0,
                -2.5,
                0.0,
            )

    if not np.any(np.isnan(B)) and not np.any(np.isnan(Tmm10)):
        Lmm10 = core_physics.calculate_local_wavelength(T=Tmm10, h=1e100)
        core_utility.check_variable_validity_range(
            "Dimensionless crest width B / Lm-1,0",
            "Van Gent et al. (2023)",
            B / Lmm10,
            0.017,
            0.27,
        )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(B)):
        core_utility.check_variable_validity_range(
            "Dimensionless crest width B / Hm0",
            "Van Gent et al. (2023)",
            B / Hm0,
            0.9,
            10.6,
        )

    return


def calculate_wave_transmission_Kt(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    B: float | npt.NDArray[np.float64],
    structure_type: str = "permeable",
) -> float | npt.NDArray[np.float64]:
    """Calculate the wave transmission coefficient with the Van Gent et al. (2023) formula.

    Here, eq. 6 from Van Gent et al. (2023) is implemented. The coefficients in this formula
    depend on the type of structure, as listed in Table 1 of Van Gent et al. (2023). The
    possible structure types are:
        - impermeable
        - permeable
        - perforated
        - perforated_with_screen
        - perforated_with_perforated_screen

    For more details, see: https://doi.org/10.1016/j.coastaleng.2023.104344

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    B : float | npt.NDArray[np.float64]
        Crest width of the structure (m)
    structure_type : str, optional
        Type of structure, by default "permeable"

    Returns
    -------
    float | npt.NDArray[np.float64]
        Wave transmission coefficient Kt (-)

    Raises
    ------
    ValueError
        If a non-valid structure type is provided.
    """

    match structure_type:
        case "impermeable":
            c1 = 0.47
            c2 = 3.1
            c3 = 0.75
            c4 = 0.0
            c5 = 0.5
        case "permeable":
            c1 = 0.43
            c2 = 3.1
            c3 = 0.75
            c4 = -0.25
            c5 = 0.5
        case "perforated":
            c1 = 0.13
            c2 = 3.1
            c3 = 0.75
            c4 = -0.15
            c5 = 0.82
        case "perforated_with_screen":
            c1 = 0.40
            c2 = 3.1
            c3 = 0.75
            c4 = -0.15
            c5 = 0.5
        case "perforated_with_perforated_screen":
            c1 = 0.17
            c2 = 3.1
            c3 = 0.75
            c4 = -0.15
            c5 = 0.76
        case _:
            raise ValueError(
                f"Invalid structure type '{structure_type}'. "
                "Valid options are: impermeable, permeable, perforated, "
                "perforated_with_screen, perforated_with_perforated_screen."
            )

    # Calculate deep water wavelength by supplying a very large depth
    Lmm10 = core_physics.calculate_local_wavelength(T=Tmm10, h=1e100)

    Kt = c1 * np.tanh(-((Rc / Hm0) + c2 * np.power(B / Lmm10, c3) + c4)) + c5

    return Kt
