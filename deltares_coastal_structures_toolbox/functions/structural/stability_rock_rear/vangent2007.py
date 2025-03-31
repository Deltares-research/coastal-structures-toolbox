# SPDX-License-Identifier: GPL-3.0-or-later
# import warnings

import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


def check_validity_range(
    Rc: float | npt.NDArray[np.float64] = np.nan,
    Rc2_front: float | npt.NDArray[np.float64] = np.nan,
    Rc2_rear: float | npt.NDArray[np.float64] = np.nan,
    Gc: float | npt.NDArray[np.float64] = np.nan,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    B_element: float | npt.NDArray[np.float64] = np.nan,
    h: float | npt.NDArray[np.float64] = np.nan,
    Hs: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    z1p: float | npt.NDArray[np.float64] = np.nan,
    S: float | npt.NDArray[np.float64] = np.nan,
    N_waves: int | npt.NDArray[np.int32] = np.nan,
) -> None:
    """Check the parameter values vs the validity range as defined in Van Gent (2007).

    For all parameters supplied, their values are checked versus the range of test conditions specified in Table 3
    (Van Gent, 2007). When parameters are nan (by default), they are not checked.

    For more details see Van Gent (2007), available here https://doi.org/10.1142/9789814282024_0002 or here
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    Rc : float | npt.NDArray[np.float64], optional
        Freeboard of the structure (m), by default np.nan
    Rc2_front : float | npt.NDArray[np.float64], optional
        Vertical distance between top of rock material at the crest and the top of the crest element (m),
        by default np.nan
    Rc2_rear : float | npt.NDArray[np.float64], optional
        Vertical distance between still-water level and the lowest point of the crest element at the rear
        side (m), by default np.nan
    Gc : float | npt.NDArray[np.float64], optional
        Width of the crest in front of crest element (m), by default np.nan
    Dn50 : float | npt.NDArray[np.float64], optional
        Nominal rock diameter (m), by default np.nan
    B_element : float | npt.NDArray[np.float64], optional
        Width of the crest element (m), by default np.nan
    h : float | npt.NDArray[np.float64], optional
        Water depth at the toe of the structure (m), by default np.nan
    Hs : float | npt.NDArray[np.float64], optional
        Significant wave height (m), by default np.nan
    Tmm10 : float | npt.NDArray[np.float64], optional
        Spectral wave period Tm-1,0 (s), by default np.nan
    z1p : float | npt.NDArray[np.float64], optional
        Wave runup height exceeded by 1% of waves, by default np.nan
    S : float | npt.NDArray[np.float64], optional
        Damage number (-), by default np.nan
    N_waves : int | npt.NDArray[np.int32], optional
        Number of waves (-), by default np.nan
    """

    if not np.any(np.isnan(Hs)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hs, Tmm10)
        core_utility.check_variable_validity_range(
            "sm-1,0", "Van Gent (2007)", smm10, 0.023, 0.054
        )

    if not np.any(np.isnan(h)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "h/Hs", "Van Gent (2007)", h / Hs, 3.3, 11.0
        )  # TODO double check! BREAKWAT um says 1.1 - 3.3, paper says 3.3 - 11.0

    if not np.any(np.isnan(N_waves)):
        core_utility.check_variable_validity_range(
            "N_waves", "Van Gent (2007)", N_waves, 1000, 10000
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc/Hs", "Van Gent (2007)", Rc / Hs, 0.5, 1.9
        )

    if not np.any(np.isnan(Rc)) and not np.any(np.isnan(Dn50)):
        core_utility.check_variable_validity_range(
            "Rc/Dn50", "Van Gent (2007)", Rc / Dn50, 7.0, 22.0
        )

    if (
        not np.any(np.isnan(Rc))
        and not np.any(np.isnan(Hs))
        and not np.any(np.isnan(z1p))
    ):
        core_utility.check_variable_validity_range(
            "(z1% - Rc)/Hs",
            "Van Gent (2007)",
            (z1p - Rc) / Hs,
            0.2,
            1.3,
        )

    if not np.any(np.isnan(Rc2_front)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc2,front/Hs", "Van Gent (2007)", Rc2_front / Hs, 0.0, 0.5
        )

    if not np.any(np.isnan(B_element)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "B_element/Hs", "Van Gent (2007)", B_element / Hs, 0.5, 1.5
        )

    if not np.any(np.isnan(Gc)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Gc/Hs", "Van Gent (2007)", Gc / Hs, 0.5, 1.1
        )

    if not np.any(np.isnan(Dn50)) and not np.any(np.isnan(Hs)):
        Ns = core_physics.calculate_stability_number_Ns(
            Hs, Dn50, rho_rock=2650, rho_water=1025
        )
        core_utility.check_variable_validity_range(
            "Stability number Ns", "Van Gent (2007)", Ns, 4.0, 12.0
        )

    if not np.any(np.isnan(Rc2_rear)) and not np.any(np.isnan(Hs)):
        core_utility.check_variable_validity_range(
            "Rc2,rear/Hs", "Van Gent (2007)", Rc2_rear / Hs, 0.0, 1.3
        )

    if not np.any(np.isnan(S)):
        core_utility.check_variable_validity_range(
            "S", "Van Gent (2007)", S, 2.0, 100.0
        )

    return


def calculate_damage_number_S(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc2_front: float | npt.NDArray[np.float64],
    Rc2_rear: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    M50: float | npt.NDArray[np.float64] = np.nan,
    rho_rock: float | npt.NDArray[np.float64] = np.nan,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number S for rock at the rear side of a rubble mound structure with a crest element
    following Van Gent (2007).

    For more details see Van Gent (2007), available here https://doi.org/10.1142/9789814282024_0002 or here
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    cot_phi : float | npt.NDArray[np.float64]
        Cotangent of the rear-side slope of the structure (-)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor for the wave runup (-)
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Rc2_front : float | npt.NDArray[np.float64]
        Vertical distance between top of rock material at the crest and the top of the crest element (m)
    Rc2_rear : float | npt.NDArray[np.float64]
        Vertical distance between still-water level and the lowest point of the crest element at the rear
        side (m)
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)
    Dn50 : float | npt.NDArray[np.float64], optional
        Median nominal rock diameter (m), by default np.nan
    M50 : float | npt.NDArray[np.float64], optional
        Median rock mass (kg), by default np.nan
    rho_rock : float | npt.NDArray[np.float64], optional
        Rock density (kg/m^3), by default np.nan

    Returns
    -------
    float | npt.NDArray[np.float64]
        The damage number S (-)

    Raises
    ------
    ValueError
        Raises an error when neither Dn50 nor M50 is provided
    """
    z1p = vangent2001.calculate_wave_runup_height_z1p(Tmm10, gamma, cot_alpha, Hs=Hs)

    # TODO ? replace Rc2_front with Ac and calculate Rc2_front = Rc - Ac

    Dn50 = core_physics.check_usage_Dn50_or_M50(Dn50, M50, rho_rock)

    S = (
        0.00025
        * np.power(cot_phi, 1.25)
        * np.power((z1p - Rc) / Dn50, 2)
        * np.power(Rc / Dn50, 0.5)
        * (1 + 5 * Rc2_rear / (Rc - Rc2_rear))
        * np.power(1 + Rc2_front / Hs, -1)
        * np.sqrt(N_waves)
    )

    check_validity_range(
        Dn50=Dn50,
        Hs=Hs,
        Rc=Rc,
        z1p=z1p,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
        S=S,
    )

    return S


def calculate_nominal_rock_diameter_Dn50(
    cot_alpha: float | npt.NDArray[np.float64],
    cot_phi: float | npt.NDArray[np.float64],
    gamma: float | npt.NDArray[np.float64],
    S: float | npt.NDArray[np.float64],
    Hs: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Rc2_front: float | npt.NDArray[np.float64],
    Rc2_rear: float | npt.NDArray[np.float64],
    N_waves: int | npt.NDArray[np.int32],
) -> float | npt.NDArray[np.float64]:
    """Calculate the damage number minimum Dn50 for armour at the rear side of a rubble mound structure with a crest
    element following Van Gent (2007).

    For more details see Van Gent (2007), available here https://doi.org/10.1142/9789814282024_0002 or here
    https://www.researchgate.net/publication/259258925_REAR-SIDE_STABILITY_OF_RUBBLE_MOUND_STRUCTURES_WITH_CREST_ELEMENTS

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        Cotangent of the front-side slope of the structure (-)
    cot_phi : float | npt.NDArray[np.float64]
        Cotangent of the rear-side slope of the structure (-)
    gamma : float | npt.NDArray[np.float64]
        Reduction factor for the wave runup (-)
    S : float | npt.NDArray[np.float64]
        Damage number (-)
    Hs : float | npt.NDArray[np.float64]
        Significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Rc : float | npt.NDArray[np.float64]
        Freeboard of the structure (m)
    Rc2_front : float | npt.NDArray[np.float64]
        Vertical distance between top of rock material at the crest and the top of the crest element (m)
    Rc2_rear : float | npt.NDArray[np.float64]
        Vertical distance between still-water level and the lowest point of the crest element at the rear
        side (m)
    N_waves : int | npt.NDArray[np.int32]
        Number of waves (-)

    Returns
    -------
    float | npt.NDArray[np.float64]
        The median nominal rock diameter Dn50 (m)
    """

    z1p = vangent2001.calculate_wave_runup_height_z1p(Tmm10, gamma, cot_alpha, Hs=Hs)

    # TODO ? replace Rc2_front with Ac and calculate Rc2_front = Rc - Ac

    Dn50 = (
        0.036
        * np.power(cot_phi, 0.5)
        * np.power(z1p - Rc, 0.8)
        * np.power(Rc, 0.2)
        * np.power((1 + 5 * Rc2_rear / (Rc - Rc2_rear)), 0.4)
        * np.power(1 + Rc2_front / Hs, -0.4)
        * np.power(S / np.sqrt(N_waves), -0.4)
    )

    check_validity_range(
        S=S,
        Hs=Hs,
        Rc=Rc,
        z1p=z1p,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
        Dn50=Dn50,
    )
    return Dn50


# def calculate_maximum_Hs(
#     cot_alpha: float | npt.NDArray[np.float64],
#     cot_phi: float | npt.NDArray[np.float64],
#     gamma: float | npt.NDArray[np.float64],
#     S: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     Dn50: float | npt.NDArray[np.float64],
#     Rc: float | npt.NDArray[np.float64],
#     Rc2_front: float | npt.NDArray[np.float64],
#     Rc2_rear: float | npt.NDArray[np.float64],
#     N_waves: int | npt.NDArray[np.int32],
#     smm10_init: float = 0.04,
#     Hs_tolerance: float = 0.01,
#     max_iterations: int = 100,
#     g: float = 9.81,
# ) -> float | npt.NDArray[np.float64]:
#     # TODO - naar kijken, want moet iteratief worden opgelost (z1% hangt af van Hs)

#     # Hs_iter_m1 = smm10_init * np.power(Tmm10, 2) / (2 * np.pi / g)

#     # z1p_iter = Rc + 0.2 * Hs_iter_m1

#     # for iteration in range(max_iterations):

#     #     Hs_iter = Rc2_front / (
#     #         (np.sqrt(N_waves) / S)
#     #         * 0.00025
#     #         * np.power(cot_phi, 1.25)
#     #         * np.power((z1p_iter - Rc) / Dn50, 2)
#     #         * np.power(Rc / Dn50, 0.5)
#     #         * (1 + 5 * Rc2_rear / (Rc - Rc2_rear))
#     #         - 1
#     #     )

#     #     z1p_iter = wave_runup_vangent2001.calculate_wave_runup_height_z1p(
#     #         Tmm10, gamma, cot_alpha, Hs=Hs_iter
#     #     )

#     #     if np.abs(Hs_iter - Hs_iter_m1) < Hs_tolerance:
#     #         check_validity_range_vangent2007(
#     #             S=S,
#     #             Hs=Hs_iter,
#     #             Rc=Rc,
#     #             z1p=z1p_iter,
#     #             Rc2_front=Rc2_front,
#     #             Rc2_rear=Rc2_rear,
#     #             N_waves=N_waves,
#     #             Dn50=Dn50,
#     #         )
#     #         break
#     #     else:
#     #         Hs_iter_m1 = Hs_iter

#     #     if iteration == max_iterations - 1:
#     #         warnings.warn(
#     #             "Maximum number of iterations reached, convergence not achieved"
#     #         )

#     Hs_iter_m1 = Rc2_front

#     for iteration in range(max_iterations):
#         z1p_iter = vangent2001.calculate_wave_runup_height_z1p(
#             Tmm10, gamma, cot_alpha, Hs=Hs_iter_m1
#         )

#         Hs_iter = vangent2001.iterate_significant_wave_height_Hs(
#             H_init=Hs_iter_m1,
#             Tmm10=Tmm10,
#             z1p=z1p_iter,
#             gamma=gamma,
#             cot_alpha=cot_alpha,
#         )

#         Hs_iter2 = Rc2_front / (
#             (np.sqrt(N_waves) / S)
#             * 0.00025
#             * np.power(cot_phi, 1.25)
#             * np.power((z1p_iter - Rc) / Dn50, 2)
#             * np.power(Rc / Dn50, 0.5)
#             * (1 + 5 * Rc2_rear / (Rc - Rc2_rear))
#             - 1
#         )
#         if np.abs(Hs_iter - Hs_iter_m1) < Hs_tolerance:
#             check_validity_range(
#                 S=S,
#                 Hs=Hs_iter,
#                 Rc=Rc,
#                 z1p=z1p_iter,
#                 Rc2_front=Rc2_front,
#                 Rc2_rear=Rc2_rear,
#                 N_waves=N_waves,
#                 Dn50=Dn50,
#             )
#             break
#         else:
#             Hs_iter_m1 = Hs_iter

#         if iteration == max_iterations - 1:
#             warnings.warn(
#                 "Maximum number of iterations reached, convergence not achieved"
#             )

#     return Hs_iter
