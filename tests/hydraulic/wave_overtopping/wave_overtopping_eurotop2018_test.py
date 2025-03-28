import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.eurotop2018 as eurotop2018


@pytest.mark.parametrize(
    (
        "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, Rc, B_berm, dh, gamma_f, gamma_v, q_expected"
    ),
    (
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.8414]),
        ([2.5, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 1.0, 2.546]),
        ([2.0, 7.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 1.0, 7.097]),
        ([2.0, 5.00, 30.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.3253]),
        ([2.0, 5.00, 0.0, 3.5, 3.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.4718]),
        ([2.0, 5.00, 0.0, 3.0, 2.0, 5.0, 0.0, 0.0, 1.0, 1.0, 6.335]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 4.5, 0.0, 0.0, 1.0, 1.0, 1.946]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 1.0, 1.0, 1.0, 1.0, 0.4296]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 2.0, 1.0, 1.0, 1.0, 0.2212]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 1.0, 0.8, 1.0, 1.0, 0.4113]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.45, 1.0, 2.132]),
        ([2.0, 5.00, 0.0, 3.0, 3.0, 5.0, 0.0, 0.0, 1.0, 0.8, 0.0929]),
        ([2.0, 5.00, 0.0, 3.5, 3.0, 5.0, 2.0, 1.0, 1.0, 1.0, 0.1522]),
        ([2.0, 5.00, 0.0, 3.0, 2.0, 5.0, 2.0, 1.0, 1.0, 1.0, 2.606]),
        ([2.0, 5.00, 30.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.45, 1.0, 0.3672]),
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
    gamma_v,
    q_expected,
):
    q_calculated, _ = eurotop2018.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        Rc=Rc,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        use_best_fit=False,
    )

    assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)


# @pytest.mark.parametrize(
#     (
#         "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, q, B_berm, dh, gamma_f, gamma_v, Rc_expected"
#     ),
#     (
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 5.02]),
#         ([2.5, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 5.78]),
#         ([2.0, 7.00, 0.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 6.50]),
#         ([2.0, 5.00, 30.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 4.52]),
#         ([2.0, 5.00, 0.0, 3.5, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 4.69]),
#         ([2.0, 5.00, 0.0, 3.0, 2.0, 1.0e-3, 0.0, 0.0, 1.0, 1.0, 6.50]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-2, 0.0, 0.0, 1.0, 1.0, 3.44]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 1.0, 1.0, 1.0, 1.0, 4.64]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 2.0, 1.0, 1.0, 1.0, 4.33]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 1.0, 0.8, 1.0, 1.0, 4.62]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 0.45, 1.0, 2.26]),
#         ([2.0, 5.00, 0.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 1.0, 0.8, 4.01]),
#         ([2.0, 5.00, 0.0, 3.5, 3.0, 1.0e-3, 2.0, 1.0, 1.0, 1.0, 4.17]),
#         ([2.0, 5.00, 0.0, 3.0, 2.0, 1.0e-3, 2.0, 1.0, 1.0, 1.0, 5.80]),
#         ([2.0, 5.00, 30.0, 3.0, 3.0, 1.0e-3, 0.0, 0.0, 0.45, 1.0, 2.03]),
#     ),
# )
# def test_Rc_backward(
#     Hm0,
#     Tmm10,
#     beta,
#     cot_alpha_down,
#     cot_alpha_up,
#     q,
#     B_berm,
#     dh,
#     gamma_f,
#     gamma_v,
#     Rc_expected,
# ):
#     Rc_calculated, _ = eurotop2018.calculate_crest_freeboard_Rc(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         cot_alpha_down=cot_alpha_down,
#         cot_alpha_up=cot_alpha_up,
#         q=q,
#         B_berm=B_berm,
#         db=dh,
#         gamma_f=gamma_f,
#         gamma_v=gamma_v,
#         use_best_fit=False,
#     )

#     assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
