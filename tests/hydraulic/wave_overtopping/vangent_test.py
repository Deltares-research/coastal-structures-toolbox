import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.vangent2025 as vangent2025


@pytest.mark.parametrize(
    ("Hm0, Hm0_swell, Tmm10, beta, cot_alpha, Rc, Ac, B_berm, dh, gamma_f, gamma_v"),
    (
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.5, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.1, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 7.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 4.5, 5.0, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 4.5, 0.0, 0.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 1.0, 0.8, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.45, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 0.8]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.45, 1.0]),
    ),
)
def test_internal_consistency_q_Rc(
    Hm0,
    Hm0_swell,
    Tmm10,
    beta,
    cot_alpha,
    Rc,
    Ac,
    B_berm,
    dh,
    gamma_f,
    gamma_v,
):
    q_calculated, _ = vangent2025.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        Rc=Rc,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        design_calculation=False,
    )

    Rc_calculated, _ = vangent2025.calculate_crest_freeboard_Rc(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        q=q_calculated,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        design_calculation=False,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    (
        "Hm0, Hm0_swell, Tmm10, beta, cot_alpha, Rc, Ac, B_berm, dh, gamma_f, gamma_v, q_expected"
    ),
    (
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 4.734]),
        ([2.5, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 15.980]),
        ([2.0, 0.1, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 4.423]),
        ([2.0, 0.2, 7.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 13.433]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.936]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0, 1.009]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 4.5, 5.0, 0.0, 0.0, 1.0, 1.0, 11.063]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 4.5, 1.0, 1.0, 1.0, 1.0, 2.684]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.697]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.018]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 1.0, 0.8, 1.0, 1.0, 2.720]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.45, 1.0, 1.744e-4]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 5.0, 5.0, 0.0, 0.0, 1.0, 0.8, 0.587]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0, 0.168]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 5.0, 5.0, 0.0, 0.0, 0.45, 1.0, 4.759e-6]),
    ),
)
def test_q_backward(
    Hm0,
    Hm0_swell,
    Tmm10,
    beta,
    cot_alpha,
    Rc,
    Ac,
    B_berm,
    dh,
    gamma_f,
    gamma_v,
    q_expected,
):
    q_calculated, _ = vangent2025.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        Rc=Rc,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        design_calculation=False,
    )

    assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    (
        "Hm0, Hm0_swell, Tmm10, beta, cot_alpha, q, Ac, B_berm, dh, gamma_f, gamma_v, Rc_expected"
    ),
    (
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.916]),
        ([2.5, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 6.825]),
        ([2.0, 0.1, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.876]),
        ([2.0, 0.2, 7.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 7.472]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 4.967]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.004]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 4.5, 1.0, 1.0, 1.0, 1.0, 5.544]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 1.0, 1.0, 1.0, 1.0, 5.547]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 2.0, 1.0, 1.0, 1.0, 5.009]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 1.0, 0.8, 1.0, 1.0, 5.553]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 0.45, 1.0, 2.706]),
        ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 0.8, 4.749]),
        ([2.0, 0.2, 5.00, 0.0, 3.5, 1.0e-3, 5.0, 2.0, 1.0, 1.0, 1.0, 4.239]),
        ([2.0, 0.2, 5.00, 30.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 0.45, 1.0, 2.279]),
    ),
)
def test_Rc_backward(
    Hm0,
    Hm0_swell,
    Tmm10,
    beta,
    cot_alpha,
    q,
    Ac,
    B_berm,
    dh,
    gamma_f,
    gamma_v,
    Rc_expected,
):
    Rc_calculated, _ = vangent2025.calculate_crest_freeboard_Rc(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Tmm10=Tmm10,
        q=q,
        Ac=Ac,
        cot_alpha=cot_alpha,
        beta=beta,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        gamma_v=gamma_v,
        design_calculation=False,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
