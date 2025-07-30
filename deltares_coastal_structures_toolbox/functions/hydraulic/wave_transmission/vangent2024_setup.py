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
    hc: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    structure_type: str = "impermeable",
) -> None:

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)

        match structure_type:
            case "impermeable":
                core_utility.check_variable_validity_range(
                    "Wave steepness sm-1,0", "Van Gent (2024)", smm10, 0.01, 0.04
                )
            case "permeable":
                core_utility.check_variable_validity_range(
                    "Wave steepness sm-1,0", "Van Gent (2024)", smm10, 0.008, 0.04
                )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Rc)):
        match structure_type:
            case "impermeable":
                core_utility.check_variable_validity_range(
                    "Dimensionless freeboard Rc/Hm0",
                    "Van Gent (2024)",
                    Rc / Hm0,
                    -1.0,
                    0.0,
                )
            case "permeable":
                core_utility.check_variable_validity_range(
                    "Dimensionless freeboard Rc/Hm0",
                    "Van Gent (2024)",
                    Rc / Hm0,
                    -0.8,
                    0.0,
                )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(hc)):
        match structure_type:
            case "permeable":
                core_utility.check_variable_validity_range(
                    "Dimensionless submerged structure height hc/Hm0",
                    "Van Gent (2024)",
                    hc / Hm0,
                    1.5,
                    3.0,
                )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(hc)):
        match structure_type:
            case "permeable":
                core_utility.check_variable_validity_range(
                    "Dimensionless submerged structure crest width B/Hm0",
                    "Van Gent (2024)",
                    B / Hm0,
                    0.94,
                    2.35,
                )

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Dn50)):
        match structure_type:
            case "permeable":
                core_utility.check_variable_validity_range(
                    "Dimensionless median stone diameter Dn50/Hm0",
                    "Van Gent (2024)",
                    Dn50 / Hm0,
                    0.19,
                    0.47,
                )

    return


def calculate_structure_induced_setup_delta_impermeable(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    delta_diml = calculate_dimensionless_structure_induced_setup_impermeable(
        Hm0=Hm0, Tmm10=Tmm10, Rc=Rc
    )

    delta = delta_diml * Hm0

    return delta


def calculate_dimensionless_structure_induced_setup_impermeable(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    smm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)

    delta_diml = 0.34 * np.power(smm10, 0.4) * np.power(Rc / Hm0 + 1.6, 3.5)

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        Rc=Rc,
        structure_type="impermeable",
    )

    return delta_diml


def calculate_structure_induced_setup_delta_permeable(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    hc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    delta_diml = calculate_dimensionless_structure_induced_setup_permeable(
        Hm0=Hm0, Tmm10=Tmm10, Rc=Rc, hc=hc
    )

    delta = delta_diml * Hm0

    return delta


def calculate_dimensionless_structure_induced_setup_permeable(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    hc: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:

    smm10 = core_physics.calculate_wave_steepness_s(H=Hm0, T=Tmm10)

    delta_diml = (
        0.22
        * np.power(smm10, 0.2)
        * np.power(hc / Hm0, -0.7)
        * np.power(Rc / Hm0 + 1.5, 1.5)
    )

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        Rc=Rc,
        hc=hc,
        structure_type="permeable",
    )

    return delta_diml
