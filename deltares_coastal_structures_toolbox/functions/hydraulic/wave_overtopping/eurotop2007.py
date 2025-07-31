# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_overtopping_discharge_q_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:
    """Calculate the mean wave overtopping discharge q for simple rubble mound slopes with the EurOtop (2007) formula.

    _extended_summary_

    Parameters
    ----------
    Hm0 : float | npt.NDArray[np.float64]
        _description_
    Tmm10 : float | npt.NDArray[np.float64]
        _description_
    Rc : float | npt.NDArray[np.float64]
        _description_
    beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_beta : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    gamma_f : float | npt.NDArray[np.float64], optional
        _description_, by default 1.0
    B_berm : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    db : float | npt.NDArray[np.float64], optional
        _description_, by default 0.0
    cot_alpha : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_down : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    cot_alpha_up : float | npt.NDArray[np.float64], optional
        _description_, by default np.nan
    c2 : float, optional
        _description_, by default 0.2
    c3 : float, optional
        _description_, by default 2.3
    use_best_fit : bool, optional
        _description_, by default False
    g : float, optional
        _description_, by default 9.81

    Returns
    -------
    float | npt.NDArray[np.float64]
        _description_
    """

    q_diml = calculate_dimensionless_overtopping_discharge_q_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        c2=c2,
        c3=c3,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q


def calculate_dimensionless_overtopping_discharge_q_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:

    (
        c2,
        c3,
    ) = check_best_fit(c2=c2, c3=c3, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta
        )

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

    # TODO Check if we need gamma_f or gamma_f_adj here
    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    q_diml = wave_overtopping_taw2002.q_diml_max_equation(
        Hm0=Hm0, Rc=Rc, c2=c2, c3=c3, gamma_beta=gamma_beta, gamma_f=gamma_f_adj
    )

    return q_diml


def check_best_fit(c2: float, c3: float, use_best_fit: bool) -> tuple[float, float]:
    """Check whether best fit coefficients need to be used

    If so, return the best fit coefficients, otherwise return the input coefficients

    Parameters
    ----------
    c1 : float
        Coefficient in wave overtopping formula (-)
    c2 : float
        Coefficient in wave overtopping formula (-)
    use_best_fit : bool
        Switch to either use best fit values for the coefficients (true) or the design values (false)

    Returns
    -------
    tuple[float, float]
        Coefficients c1 and c2 in the wave runup formula (-)
    """
    if use_best_fit:
        c2 = 0.2
        c3 = 2.6

    return c2, c3


def calculate_crest_freeboard_Rc_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    Rc_diml = calculate_dimensionless_crest_freeboard_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        gamma_beta=gamma_beta,
        cot_alpha=cot_alpha,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        q=q,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        c2=c2,
        c3=c3,
        use_best_fit=use_best_fit,
        g=g,
    )

    Rc = Rc_diml * Hm0

    return Rc


def calculate_dimensionless_crest_freeboard_rubble_mound(
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_beta: float | npt.NDArray[np.float64] = np.nan,
    gamma_f: float | npt.NDArray[np.float64] = 1.0,
    B_berm: float | npt.NDArray[np.float64] = 0.0,
    db: float | npt.NDArray[np.float64] = 0.0,
    cot_alpha: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_down: float | npt.NDArray[np.float64] = np.nan,
    cot_alpha_up: float | npt.NDArray[np.float64] = np.nan,
    c2: float = 0.2,
    c3: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    (
        c2,
        c3,
    ) = check_best_fit(c2=c2, c3=c3, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta
        )

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

    # TODO Check if we need gamma_f or gamma_f_adj here
    gamma_f_adj = wave_runup_taw2002.calculate_adjusted_influence_roughness_gamma_f(
        gamma_f=gamma_f, gamma_b=1.0, ksi_mm10=ksi_mm10
    )

    Rc_diml = wave_overtopping_taw2002.Rc_diml_max_equation(
        Hm0=Hm0, q=q, c2=c2, c3=c3, gamma_beta=gamma_beta, gamma_f=gamma_f_adj
    )

    return Rc_diml
