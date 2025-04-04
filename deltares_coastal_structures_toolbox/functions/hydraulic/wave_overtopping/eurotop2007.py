# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002


def calculate_overtopping_discharge_q_rough_slope(
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
    c1: float = 0.2,
    c2: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    q_diml, max_reached = calculate_dimensionless_overtopping_discharge_q_rough_slope(
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
        c1=c1,
        c2=c2,
        use_best_fit=use_best_fit,
    )
    q = q_diml * np.sqrt(g * Hm0**3)

    return q, max_reached


def calculate_dimensionless_overtopping_discharge_q_rough_slope(
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
    c1: float = 0.2,
    c2: float = 2.3,
    use_best_fit: bool = False,
) -> float | npt.NDArray[np.float64]:

    (
        c1,
        c2,
    ) = check_best_fit(c1=c1, c2=c2, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
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

    q_diml = c1 * np.exp(-c2 * (Rc / (Hm0 * gamma_f_adj * gamma_beta)))

    return q_diml


def check_best_fit(c1: float, c2: float, use_best_fit: bool) -> tuple[float, float]:
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
        c1 = 0.2
        c2 = 2.6

    return c1, c2


def calculate_crest_freeboard_Rc_rough_slope(
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
    c1: float = 0.2,
    c2: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    Rc_diml = calculate_dimensionless_crest_freeboard_rough_slope(
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
        c1=c1,
        c2=c2,
        use_best_fit=use_best_fit,
        g=g,
    )

    Rc = Rc_diml * Hm0

    return Rc


def calculate_dimensionless_crest_freeboard_rough_slope(
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
    c1: float = 0.2,
    c2: float = 2.3,
    use_best_fit: bool = False,
    g: float = 9.81,
) -> float | npt.NDArray[np.float64]:

    (
        c1,
        c2,
    ) = check_best_fit(c1=c1, c2=c2, use_best_fit=use_best_fit)

    if wave_runup_taw2002.check_calculate_gamma_beta(beta=beta, gamma_beta=gamma_beta):
        gamma_beta = wave_runup_taw2002.calculate_influence_oblique_waves_gamma_beta(
            beta=beta, gamma_f=gamma_f
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

    Rc_diml = (
        -np.log((q / np.sqrt(g * np.power(Hm0, 3))) * (1.0 / c1))
        * (1.0 / c2)
        * gamma_f_adj
        * gamma_beta
    )

    return Rc_diml
