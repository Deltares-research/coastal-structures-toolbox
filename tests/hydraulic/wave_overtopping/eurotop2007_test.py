import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.eurotop2007 as eurotop2007


@pytest.mark.parametrize(
    ("Hm0, Tmm10, beta, cot_alpha, Rc, gamma_f"),
    (
        ([2.0, 5.00, 0.0, 3.0, 5.0, 1.0]),
        ([2.5, 5.00, 0.0, 3.0, 5.0, 1.0]),
        ([2.0, 7.00, 0.0, 3.0, 5.0, 1.0]),
        ([2.0, 5.00, 30.0, 3.0, 5.0, 1.0]),
        ([2.0, 5.00, 0.0, 3.5, 5.0, 1.0]),
        ([2.0, 5.00, 0.0, 3.0, 4.5, 1.0]),
        ([2.0, 5.00, 0.0, 3.0, 5.0, 0.45]),
        ([2.0, 5.00, 30.0, 3.0, 5.0, 0.45]),
    ),
)
def test_internal_consistency_q_Rc(
    Hm0,
    Tmm10,
    beta,
    cot_alpha,
    Rc,
    gamma_f,
):
    q_calculated = eurotop2007.calculate_overtopping_discharge_q_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        Rc=Rc,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    Rc_calculated = eurotop2007.calculate_crest_freeboard_Rc_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        q=q_calculated,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, beta, cot_alpha, Rc, gamma_f, q_expected"),
    (
        ([2.0, 5.00, 0.0, 3.0, 5.0, 1.0, 5.639]),
        ([2.5, 5.00, 0.0, 3.0, 5.0, 1.0, 24.890]),
        ([2.0, 7.00, 0.0, 3.0, 5.0, 1.0, 5.639]),
        ([2.0, 5.00, 30.0, 3.0, 5.0, 1.0, 3.756]),
        ([2.0, 5.00, 0.0, 3.5, 5.0, 1.0, 5.639]),
        ([2.0, 5.00, 0.0, 3.0, 4.5, 1.0, 10.022]),
        ([2.0, 5.00, 0.0, 3.0, 5.0, 0.45, 0.0050]),
        ([2.0, 5.00, 30.0, 3.0, 5.0, 0.45, 0.0020]),
    ),
)
def test_q_backward(
    Hm0,
    Tmm10,
    beta,
    cot_alpha,
    Rc,
    gamma_f,
    q_expected,
):
    q_calculated = eurotop2007.calculate_overtopping_discharge_q_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        Rc=Rc,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, beta, cot_alpha, q, gamma_f, Rc_expected"),
    (
        ([2.0, 5.00, 0.0, 3.0, 1.0e-3, 1.0, 6.50]),
        ([2.5, 5.00, 0.0, 3.0, 1.0e-3, 1.0, 8.49]),
        ([2.0, 7.00, 0.0, 3.0, 1.0e-3, 1.0, 6.50]),
        ([2.0, 5.00, 30.0, 3.0, 1.0e-3, 1.0, 6.07]),
        ([2.0, 5.00, 0.0, 3.5, 1.0e-3, 1.0, 6.50]),
        ([2.0, 5.00, 0.0, 3.0, 1.0e-2, 1.0, 4.50]),
        ([2.0, 5.00, 0.0, 3.0, 1.0e-3, 0.45, 2.93]),
        ([2.0, 5.00, 30.0, 3.0, 1.0e-3, 0.45, 2.73]),
    ),
)
def test_Rc_backward(
    Hm0,
    Tmm10,
    beta,
    cot_alpha,
    q,
    gamma_f,
    Rc_expected,
):
    Rc_calculated = eurotop2007.calculate_crest_freeboard_Rc_rubble_mound(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        q=q,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
