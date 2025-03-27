import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as taw2002


@pytest.mark.parametrize(
    (
        "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, Rc, B_berm, dh, gamma_f, q_expected"
    ),
    (
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 1.023]),
        ([2.5, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 2.763]),
        ([2.0, 7.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 5.639]),
        ([2.0, 5.00, 30.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 0.4585]),
        (
            [2.0, 5.00, 0.0, 3.5, 3.0, 5.0, 0.0, 0.0, 1.0, 0.6226]
        ),  # TODO check implementation composite slopes (temporarily disabled)
        (
            [2.0, 5.00, 0.0, 4.5, 3.0, 5.0, 0.0, 0.0, 1.0, 0.2066]
        ),  # TODO remove: extra debug
        (
            [2.0, 5.00, 0.0, 2.0, 3.0, 5.0, 0.0, 0.0, 1.0, 2.486]
        ),  # TODO remove: extra debug
        (
            [2.0, 5.00, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.2227]
        ),  # TODO remove: extra debug
        ([2.0, 5.00, 0.0, 3.0, 2.0, 5.0, 0.0, 0.0, 1.0, 5.639]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 4.5, 0.0, 0.0, 1.0, 2.122]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 1.0, 1.0, 1.0, 0.5720]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 2.0, 1.0, 1.0, 0.3258]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 1.0, 0.8, 1.0, 0.5512]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 0.45, 4.075e-5]),
        ([2.0, 5.00, 0.0, 3.5, 3.0, 5.0, 2.0, 1.0, 1.0, 0.2391]),
        ([2.0, 5.00, 0.0, 3.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.751]),
    ),
)
def test_q_backward(
    Hm0,
    Tmm10,
    beta,
    cot_alpha_down,
    cot_alpha_up,
    Rc,
    B_berm,
    dh,
    gamma_f,
    q_expected,
):
    q_calculated, _ = taw2002.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)
