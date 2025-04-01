import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.vangent2001 as vangent2001


@pytest.mark.parametrize(
    (
        "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, B_berm, db, gamma_f, z2p_expected"
    ),
    (
        ([1.0, 4.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.804]),
        ([1.5, 4.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 3.870]),
        ([1.0, 8.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 3.302]),
        ([1.0, 4.0, 30.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.619]),
        # (
        #     [1.0, 4.0, 0.0, 2.5, 2.0, 0.0, 0.0, 1.0, 2.679]
        # ),  # TODO composite slopes & berms don't work yet
        # (
        #     [1.0, 4.0, 0.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.555]
        # ),  # TODO composite slopes & berms don't work yet
        # (
        #     [1.0, 4.0, 0.0, 2.0, 2.0, 5.0, 1.0, 1.0, 2.181]
        # ),  # TODO composite slopes & berms don't work yet
        # (
        #     [1.0, 4.0, 0.0, 2.0, 2.0, 5.0, -0.5, 1.0, 2.181]
        # ),  # TODO composite slopes & berms don't work yet
        ([1.0, 4.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.45, 1.262]),
        # (
        #     [1.0, 4.0, 0.0, 2.5, 2.0, 2.0, 1.0, 1.0, 2.492]
        # ),  # TODO composite slopes & berms don't work yet
        # (
        #     [1.0, 4.0, 0.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.181]
        # ),  # TODO composite slopes & berms don't work yet
    ),
)
def test_z2p_backward(
    Hm0,
    Tmm10,
    beta,
    cot_alpha_down,
    cot_alpha_up,
    B_berm,
    db,
    gamma_f,
    z2p_expected,
):
    # z2p_calculated = vangent2001.calculate_wave_runup_height_z2p(
    #     Hm0=Hm0,
    #     Tmm10=Tmm10,
    #     beta=beta,
    #     cot_alpha_down=cot_alpha_down,
    #     cot_alpha_up=cot_alpha_up,
    #     B_berm=B_berm,
    #     db=dh,
    #     gamma_f=gamma_f,
    # )

    gamma_beta = taw2002.calculate_influence_oblique_waves_gamma_beta(beta=beta)

    z2p_for_slope = taw2002.iteration_procedure_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
        db=db,
        gamma_f=gamma_f,
        gamma_beta=gamma_beta,
    )

    cot_alpha = taw2002.determine_average_slope(
        Hm0=Hm0,
        z2p=z2p_for_slope,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
        db=db,
    )

    z2p_calculated = vangent2001.calculate_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        gamma=gamma_beta * gamma_f,
    )

    assert z2p_calculated == pytest.approx(z2p_expected, abs=1e-2)
