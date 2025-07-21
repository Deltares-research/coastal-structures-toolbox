# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.core_utility as core_utility
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def check_validity_range(
    Hm0: float | npt.NDArray[np.float64] = np.nan,
    Tmm10: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
) -> None:

    if not np.any(np.isnan(Hm0)) and not np.any(np.isnan(Tmm10)):
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        core_utility.check_variable_validity_range(
            "Wave steepness sm-1,0", "TAW (2002)", smm10, 0.0, 0.07
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
    ):
        ksi_smm10 = core_physics.calculate_Irribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "Irribarren number ksi_m-1,0",
            "TAW (2002)",
            ksi_smm10,
            0.0,
            7.0,
        )

    if (
        not np.any(np.isnan(Hm0))
        and not np.any(np.isnan(Tmm10))
        and not np.any(np.isnan(cot_alpha))
        and not np.any(np.isnan(gamma_b))
    ):
        ksi_smm10 = core_physics.calculate_Irribarren_number_ksi(Hm0, Tmm10, cot_alpha)
        core_utility.check_variable_validity_range(
            "gamma_b * ksi_m-1,0",
            "TAW (2002)",
            gamma_b * ksi_smm10,
            0.5,
            10.0,
        )

    if not np.any(np.isnan(cot_alpha_down)):
        core_utility.check_variable_validity_range(
            "Cotangent alpha lower slope", "TAW (2002)", cot_alpha_down, 1.0, 7.0
        )

    if not np.any(np.isnan(cot_alpha_up)):
        core_utility.check_variable_validity_range(
            "Cotangent alpha upper slope", "TAW (2002)", cot_alpha_up, 1.0, 7.0
        )

    if not np.any(np.isnan(beta)):
        core_utility.check_variable_validity_range(
            "Incident wave angle beta", "TAW (2002)", beta, 0.0, 90.0
        )

    if not np.any(np.isnan(gamma_b)):
        core_utility.check_variable_validity_range(
            "Influence factor berm gamma_b", "TAW (2002)", gamma_b, 0.4, 1.0
        )

    if not np.any(np.isnan(gamma_f)):
        core_utility.check_variable_validity_range(
            "Influence factor roughness gamma_f", "TAW (2002)", gamma_f, 0.4, 1.0
        )

    if not np.any(np.isnan(gamma_beta)):
        core_utility.check_variable_validity_range(
            "Influence factor oblique waves gamma_beta",
            "TAW (2002)",
            gamma_beta,
            0.4,
            1.0,
        )

    if not np.any(np.isnan(gamma_v)):
        core_utility.check_variable_validity_range(
            "Influence factor vertical wall gamma_v", "TAW (2002)", gamma_v, 0.4, 1.0
        )

    return


def calculate_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    g: float = 9.81,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    Hm0_swell : float | npt.NDArray[np.float64]
        _description_
    Tmm10 : float | npt.NDArray[np.float64]
        _description_
    Rc : float | npt.NDArray[np.float64]
        _description_
    Ac : float | npt.NDArray[np.float64]
        _description_
    cot_alpha : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    db : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    g : float, optional
        _description_, by default 9.81
    design_calculation : bool, optional
        _description_, by default True
    include_influence_wind : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        _description_
    """

    q_diml, max_reached = calculate_dimensionless_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        Rc=Rc,
        Ac=Ac,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        beta=beta,
        B_berm=B_berm,
        db=db,
        Dn50=Dn50,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
        design_calculation=design_calculation,
        include_influence_wind=include_influence_wind,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    beta: float | npt.NDArray[np.float64] = np.nan,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    Dn50: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = np.nan,
    gamma_b: float | npt.NDArray[np.float64] = np.nan,
    gamma_v: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    design_calculation: bool = True,
    include_influence_wind: bool = False,
) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:
    """_summary_

    _extended_summary_

    Eq. B1

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    Hm0_swell : float | npt.NDArray[np.float64]
        _description_
    Tmm10 : float | npt.NDArray[np.float64]
        _description_
    Rc : float | npt.NDArray[np.float64]
        _description_
    Ac : float | npt.NDArray[np.float64]
        _description_
    cot_alpha : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    B_berm : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    db : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    Dn50 : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_b : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_v : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    design_calculation : bool, optional
        _description_, by default True
    include_influence_wind : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    if np.isnan(gamma_beta):
        gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

    # TODO check with Marcel how to handle composite slopes (maybe irrelevant for RMB)
    if wave_runup_taw2002.check_composite_slope(
        cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
    ):
        z2p_for_slope = wave_runup_taw2002.iteration_procedure_z2p(
            Hm0=Hm0,
            Tmm10=Tmm10,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
            gamma_f=gamma_f,
            gamma_beta=gamma_beta,
        )

        cot_alpha = wave_runup_taw2002.determine_average_slope(
            Hm0=Hm0,
            z2p=z2p_for_slope,
            cot_alpha_down=cot_alpha_down,
            cot_alpha_up=cot_alpha_up,
            B_berm=B_berm,
            db=db,
        )

    ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
        H=Hm0, T=Tmm10, cot_alpha=cot_alpha
    )

    if np.isnan(gamma_b):
        gamma_b = calculate_influence_berm_gamma_b(
            Hm0=Hm0,
            smm10=ksi_mm10,
            Ac=Ac,
            B_berm=B_berm,
            BL=Ac - db,
        )

    if np.isnan(gamma_f):
        if np.isnan(Dn50):
            raise ValueError("Either gamma_f or Dn50 should be provided")
        smm10 = core_physics.calculate_wave_steepness_s(Hm0, Tmm10)
        gamma_f = calculate_influence_friction_gamma_f(Dn50=Dn50, Hm0=Hm0, smm10=smm10)

    q_diml_eqB1 = (
        6.8
        * np.power(cot_alpha, -1.0)
        * np.exp(
            -5.0
            * (Rc - 0.4 * Hm0_swell)
            / (gamma_f * gamma_b * gamma_v * gamma_beta * ksi_mm10 * Hm0)
        )
    )

    q_diml_max = q_diml_max_equation(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        ksi_mm10=ksi_mm10,
        cot_alpha=cot_alpha,
        Rc=Rc,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_v=gamma_v,
        gamma_beta=gamma_beta,
    )

    q_diml = np.min([q_diml_eqB1, q_diml_max], axis=0)
    max_reached = np.min([q_diml_eqB1, q_diml_max], axis=0) == q_diml_max

    if include_influence_wind:
        gamma_w = calculate_influence_wind_gamma_w(
            Rc=Rc,
            Ac=Ac,
            Hm0=Hm0,
            q_diml=q_diml,
        )
        q_diml = q_diml * gamma_w

    # TODO Check with Marcel: default should be design value following Eq. B8???
    # TODO Check with Marcel: first apply wind influence, then design values?
    if design_calculation:
        q_diml = np.power(q_diml, 0.857)

    check_validity_range(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f=gamma_f,
        gamma_b=gamma_b,
        gamma_beta=gamma_beta,
        gamma_v=gamma_v,
    )

    return q_diml, max_reached


def q_diml_max_equation(
    Hm0: float | npt.NDArray[np.float64],
    Hm0_swell: float | npt.NDArray[np.float64],
    ksi_mm10: float | npt.NDArray[np.float64],
    cot_alpha: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    gamma_f: float | npt.NDArray[np.float64],
    gamma_b: float | npt.NDArray[np.float64],
    gamma_v: float | npt.NDArray[np.float64],
    gamma_beta: float | npt.NDArray[np.float64],
    c2: float = 0.8,
    c3: float = -2.5,
):
    """_summary_

    _extended_summary_

    Eq. B2

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    Hm0_swell : float | npt.NDArray[np.float64]
        _description_
    ksi_mm10 : float | npt.NDArray[np.float64]
        _description_
    cot_alpha : float | npt.NDArray[np.float64]
        _description_
    Rc : float | npt.NDArray[np.float64]
        _description_
    gamma_f : float | npt.NDArray[np.float64]
        _description_
    gamma_b : float | npt.NDArray[np.float64]
        _description_
    gamma_v : float | npt.NDArray[np.float64]
        _description_
    gamma_beta : float | npt.NDArray[np.float64]
        _description_
    c2 : float, optional
        _description_, by default 0.8
    c3 : float, optional
        _description_, by default -2.5

    Returns
    -------
    _type_
        _description_
    """

    q_diml_max = (
        c2
        * np.power(cot_alpha, -1.0)
        * np.exp(
            c3
            * (Rc - 0.4 * Hm0_swell)
            / (
                gamma_f
                * gamma_b
                * gamma_v
                * gamma_beta
                * np.power(ksi_mm10, 0.24)
                * Hm0
            )
        )
    )

    return q_diml_max


def calculate_influence_friction_gamma_f(
    Dn50: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    smm10: float | npt.NDArray[np.float64],
    c_f1: float = 0.70,
    c_f2: float = 0.05,  # TODO Check whether this should be 0.05 or 0.1, conflicting values in the paper
    smm10_lim: float = 0.012,
):
    """_summary_

    _extended_summary_

    Eqs. B3a and B3b

    Parameters
    ----------
    Dn50 : float | npt.NDArray[np.float64]
        _description_
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    smm10 : float | npt.NDArray[np.float64]
        _description_
    c_f1 : float, optional
        _description_, by default 0.70
    c_f2 : float, optional
        _description_, by default 0.05
    conflictingvaluesinthepapersmm10_lim : float, optional
        _description_, by default 0.012

    Returns
    -------
    _type_
        _description_
    """

    gamma_f1 = 1.0 - c_f1 * np.power(Dn50 / Hm0, c_f2)

    gamma_f = np.where(
        smm10 >= smm10_lim,
        gamma_f1,
        gamma_f1 + 12.0 * (smm10_lim - smm10) * (1.0 - gamma_f1),
    )
    return gamma_f


def calculate_influence_berm_gamma_b(
    Hm0: float | npt.NDArray[np.float64],
    smm10: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    BL: float | npt.NDArray[np.float64],
    c_b1: float = 18.0,
    c_b2: float = 1.3,
    c_b3: float = 0.34,
    c_b4: float = 0.2,
):
    """_summary_

    _extended_summary_

    Eq. B4

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    smm10 : float | npt.NDArray[np.float64]
        _description_
    Ac : float | npt.NDArray[np.float64]
        _description_
    B_berm : float | npt.NDArray[np.float64]
        _description_
    BL : float | npt.NDArray[np.float64]
        _description_

    Returns
    -------
    _type_
        _description_
    """

    gamma_b = 1.0 - c_b1 * np.power(smm10 * B_berm / Hm0, c_b2) * (
        1.0 - c_b3 * np.power(BL / (smm10 * Ac), c_b4)
    )

    return gamma_b


def calculate_influence_oblique_waves_gamma_beta(
    beta: float | npt.NDArray[np.float64],
    c_beta: float = 0.35,
) -> float | npt.NDArray[np.float64]:
    """_summary_

    _extended_summary_

    Eq. B6

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        _description_
    c_beta : float, optional
        _description_, by default 0.35

    Returns
    -------
    float | npt.NDArray[np.float64]
        _description_
    """

    gamma_beta = (1 - c_beta) * np.power(np.cos(np.radians(beta)), 2) + c_beta

    return gamma_beta


def calculate_influence_wave_wall_gamma_v(
    cot_alpha: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    c_v: float = 0.45,
) -> float | npt.NDArray[np.float64]:
    """_summary_

    _extended_summary_

    Eqs. B5a and B5b

    Parameters
    ----------
    cot_alpha : float | npt.NDArray[np.float64]
        _description_
    Rc : float | npt.NDArray[np.float64]
        _description_
    Ac : float | npt.NDArray[np.float64]
        _description_
    c_v : float, optional
        _description_, by default 0.45

    Returns
    -------
    float | npt.NDArray[np.float64]
        _description_
    """

    gamma_v = np.where(
        cot_alpha <= 4.0,
        1.0 + c_v * (Rc - Ac) / Rc,
        1.0 + 0.1125 * cot_alpha * (Rc - Ac) / Rc,
    )

    return gamma_v


def calculate_influence_wind_gamma_w(
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    q_diml: float | npt.NDArray[np.float64],
    c_w1: float = 0.075,
    c_w2: float = -0.3,
):
    """_summary_

    _extended_summary_

    Eq. B7

    Parameters
    ----------
    hc : float | npt.NDArray[np.float64]
        _description_
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    q_diml : float | npt.NDArray[np.float64]
        _description_

    Returns
    -------
    _type_
        _description_
    """

    hc = Rc - Ac
    gamma_w = 1.0 + c_w1 * (hc / Hm0) * np.power(q_diml, c_w2)

    return gamma_w


# def calculate_crest_freeboard_Rc(
#     Hm0: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     q: float | npt.NDArray[np.float64],
#     cot_alpha: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
#     beta: float | npt.NDArray[np.float64] = np.nan,
#     B_berm: float | npt.NDArray[np.float64] = 0.0,
#     db: float | npt.NDArray[np.float64] = 0.0,
#     gamma_beta: float | npt.NDArray[np.float64] = np.nan,
#     gamma_b: float | npt.NDArray[np.float64] = np.nan,
#     gamma_f: float | npt.NDArray[np.float64] = 1.0,
#     gamma_v: float | npt.NDArray[np.float64] = 1.0,
#     sigma: float | npt.NDArray[np.float64] = 0,
#     g: float = 9.81,
#     use_best_fit: bool = False,
# ) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

#     Rc_diml, max_reached = calculate_dimensionless_crest_freeboard(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         gamma_beta=gamma_beta,
#         cot_alpha=cot_alpha,
#         cot_alpha_down=cot_alpha_down,
#         cot_alpha_up=cot_alpha_up,
#         q=q,
#         B_berm=B_berm,
#         db=db,
#         gamma_b=gamma_b,
#         gamma_f=gamma_f,
#         gamma_v=gamma_v,
#         sigma=sigma,
#         g=g,
#         use_best_fit=use_best_fit,
#     )

#     Rc = Rc_diml * Hm0

#     return Rc, max_reached


# def calculate_dimensionless_crest_freeboard(
#     Hm0: float | npt.NDArray[np.float64],
#     Tmm10: float | npt.NDArray[np.float64],
#     q: float | npt.NDArray[np.float64],
#     cot_alpha: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
#     cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
#     beta: float | npt.NDArray[np.float64] = np.nan,
#     B_berm: float | npt.NDArray[np.float64] = 0.0,
#     db: float | npt.NDArray[np.float64] = 0.0,
#     gamma_beta: float | npt.NDArray[np.float64] = np.nan,
#     gamma_b: float | npt.NDArray[np.float64] = np.nan,
#     gamma_f: float | npt.NDArray[np.float64] = 1.0,
#     gamma_v: float | npt.NDArray[np.float64] = 1.0,
#     sigma: float | npt.NDArray[np.float64] = 0,
#     g: float = 9.81,
#     use_best_fit: bool = False,
# ) -> tuple[float | npt.NDArray[np.float64], bool | npt.NDArray[np.bool]]:

#     if use_best_fit:
#         c1 = 4.75
#         c3 = 2.6
#     else:
#         c1 = 4.3
#         c3 = 2.3

#     if sigma == 0:
#         cor1 = 0
#         cor3 = 0
#     else:
#         if use_best_fit:
#             cor1 = 0.5 * sigma
#             cor3 = 0.35 * sigma
#         else:
#             warnings.warn(
#                 (
#                     "Sigma is only applicable to the best fit coefficients! The design values of the coefficients "
#                     "alreaddy account for uncertainty with conservative coefficient values."
#                 )
#             )

#     if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
#         gamma_beta = calculate_influence_oblique_waves_gamma_beta(beta=beta)

#     if wave_runup_taw2002.check_composite_slope(
#         cot_alpha=cot_alpha, cot_alpha_down=cot_alpha_down, cot_alpha_up=cot_alpha_up
#     ):
#         z2p_for_slope = wave_runup_taw2002.iteration_procedure_z2p(
#             Hm0=Hm0,
#             Tmm10=Tmm10,
#             cot_alpha_down=cot_alpha_down,
#             cot_alpha_up=cot_alpha_up,
#             B_berm=B_berm,
#             db=db,
#             gamma_f=gamma_f,
#             gamma_beta=gamma_beta,
#         )

#         cot_alpha = wave_runup_taw2002.determine_average_slope(
#             Hm0=Hm0,
#             z2p=z2p_for_slope,
#             cot_alpha_down=cot_alpha_down,
#             cot_alpha_up=cot_alpha_up,
#             B_berm=B_berm,
#             db=db,
#         )

#     ksi_mm10 = core_physics.calculate_Irribarren_number_ksi(
#         H=Hm0, T=Tmm10, cot_alpha=cot_alpha
#     )

#     if np.isnan(gamma_b):
#         L_berm = wave_runup_taw2002.calculate_berm_length(
#             Hm0=Hm0,
#             cot_alpha=cot_alpha,
#             cot_alpha_down=cot_alpha_down,
#             cot_alpha_up=cot_alpha_up,
#             B_berm=B_berm,
#         )

#         gamma_b = wave_runup_taw2002.iteration_procedure_gamma_b(
#             Hm0=Hm0,
#             Tmm10=Tmm10,
#             cot_alpha_average=cot_alpha,
#             B_berm=B_berm,
#             L_berm=L_berm,
#             db=db,
#             gamma_f=gamma_f,
#             gamma_beta=gamma_beta,
#         )

#     gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
#         gamma_f=gamma_f, gamma_b=gamma_b, ksi_mm10=ksi_mm10
#     )

#     Rc_diml_eq24 = (
#         np.log(
#             (1.0 / 0.067)
#             * np.sqrt(1.0 / cot_alpha)
#             * (1.0 / gamma_b)
#             * (1.0 / ksi_mm10)
#             * q
#             / np.sqrt(g * Hm0**3)
#         )
#         * (-1.0 / (c1 + cor1))
#         * ksi_mm10
#         * gamma_b
#         * gamma_f_adj
#         * gamma_beta
#         * gamma_v
#     )

#     Rc_diml_max = Rc_diml_max_equation(
#         Hm0=Hm0, q=q, c3=c3, cor3=cor3, gamma_beta=gamma_beta, gamma_f=gamma_f_adj
#     )

#     Rc_diml = np.min([Rc_diml_eq24, Rc_diml_max], axis=0)
#     max_reached = np.min([Rc_diml_eq24, Rc_diml_max], axis=0) == Rc_diml_max

#     check_validity_range(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         cot_alpha=cot_alpha,
#         cot_alpha_down=cot_alpha_down,
#         cot_alpha_up=cot_alpha_up,
#         gamma_f=gamma_f,
#         gamma_b=gamma_b,
#         gamma_beta=gamma_beta,
#         gamma_v=gamma_v,
#     )

#     return Rc_diml, max_reached


# def Rc_diml_max_equation(
#     Hm0: float | npt.NDArray[np.float64],
#     q: float | npt.NDArray[np.float64],
#     gamma_beta: float | npt.NDArray[np.float64],
#     gamma_f: float | npt.NDArray[np.float64],
#     c3: float,
#     cor3: float = 0.0,
#     c2: float = 0.2,
#     g: float = 9.81,
# ):

#     Rc_diml_max = (
#         np.log((1.0 / c2) * q / np.sqrt(g * Hm0**3))
#         * (-1.0 / (c3 + cor3))
#         * gamma_f
#         * gamma_beta
#     )

#     return Rc_diml_max
