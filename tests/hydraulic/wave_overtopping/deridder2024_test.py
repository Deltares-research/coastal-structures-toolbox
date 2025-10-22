import pytest



import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.deridder2024 as deridder2024


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, Rc, q_calculated"),
    (
        ([2.0, 0.01, 0.55, 3.0, 0.0322]),
        ([2.5, 0.01, 0.55, 3.0, 0.1302]),
        ([2.0, 0.02, 0.55, 3.0, 0.0086]),
        ([2.5, 0.02, 0.55, 3.0, 0.0453]),

        ([2.0, 0.01, 0.55, 2.5, 0.0780]),
        ([2.5, 0.01, 0.55, 2.5, 0.2646]),
        ([2.0, 0.02, 0.55, 2.5, 0.0260]),
        ([2.5, 0.02, 0.55, 2.5, 0.1097]),
    ),
)
def test_internal_consistency_q_Rc(
    Hm0,
    smm10_HF,
    gamma_f,
    Rc,
    q_calculated
):
    q = deridder2024.calculate_overtopping_discharge_q_eq24(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    Rc_calculated = deridder2024.calculate_crest_freeboard_discharge_q_eq24(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


# @pytest.mark.parametrize(
#     ("Hm0, Tmm10, beta, cot_alpha, Rc, gamma_f, q_expected"),
#     (
#         ([2.0, 0.01, 0.55, 3.0, 0.0322]),
#         ([2.5, 0.01, 0.55, 3.0, 0.1302]),
#         ([2.0, 0.02, 0.55, 3.0, 0.0086]),
#         ([2.5, 0.02, 0.55, 3.0, 0.0453]),

#         ([2.0, 0.01, 0.55, 2.5, 0.0780]),
#         ([2.5, 0.01, 0.55, 2.5, 0.2646]),
#         ([2.0, 0.02, 0.55, 2.5, 0.0260]),
#         ([2.5, 0.02, 0.55, 2.5, 0.1097]),
#     ),
# )
# def test_q_backward(
#     Hm0,
#     smm10_HF,
#     gamma_f,
#     Rc,
#     q_expected,
# ):
#     q_calculated = deridder2024.calculate_overtopping_discharge_q_rubble_mound(
#         Hm0=Hm0,
#         smm10_HF=smm10_HF,
#         begamma_fta=gamma_f,
#         Rc=Rc,
#     )

#     assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)


# @pytest.mark.parametrize(
#     ("Hm0, Tmm10, beta, cot_alpha, q, gamma_f, Rc_expected"),
#     (
#         ([2.0, 5.00, 0.0, 3.0, 1.0e-3, 1.0, 6.50]),
#         ([2.5, 5.00, 0.0, 3.0, 1.0e-3, 1.0, 8.49]),
#         ([2.0, 7.00, 0.0, 3.0, 1.0e-3, 1.0, 6.50]),
#         ([2.0, 5.00, 30.0, 3.0, 1.0e-3, 1.0, 6.07]),
#         ([2.0, 5.00, 0.0, 3.5, 1.0e-3, 1.0, 6.50]),
#         ([2.0, 5.00, 0.0, 3.0, 1.0e-2, 1.0, 4.50]),
#         ([2.0, 5.00, 0.0, 3.0, 1.0e-3, 0.45, 2.93]),
#         ([2.0, 5.00, 30.0, 3.0, 1.0e-3, 0.45, 2.73]),
#     ),
# )
# def test_Rc_backward(
#     Hm0,
#     Tmm10,
#     beta,
#     cot_alpha,
#     q,
#     gamma_f,
#     Rc_expected,
# ):
#     Rc_calculated = deridder2024.calculate_crest_freeboard_Rc_rubble_mound(
#         Hm0=Hm0,
#         Tmm10=Tmm10,
#         beta=beta,
#         cot_alpha=cot_alpha,
#         q=q,
#         gamma_f=gamma_f,
#         use_best_fit=False,
#     )

#     assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
