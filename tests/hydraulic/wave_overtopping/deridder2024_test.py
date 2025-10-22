import pytest


import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.deridder2024 as deridder2024


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, Rc, q_expected"),
    (
        ([2.0, 0.01, 0.55, 3.0, 0.0321745]),
        ([2.5, 0.01, 0.55, 3.0, 0.1302259]),
        ([2.0, 0.02, 0.55, 3.0, 0.0085919]),
        ([2.5, 0.02, 0.55, 3.0, 0.0452857]),
        ([2.0, 0.01, 0.55, 2.5, 0.0780481]),
        ([2.5, 0.01, 0.55, 2.5, 0.2645923]),
        ([2.0, 0.02, 0.55, 2.5, 0.0259724]),
        ([2.5, 0.02, 0.55, 2.5, 0.1097229]),
    ),
)
def test_internal_consistency_q_Rc_eq24(Hm0, smm10_HF, gamma_f, Rc, q_expected):
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


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, Hm0_LF, gamma_f, Rc"),
    (
        ([2.0, 0.01, 0.05, 0.55, 3.0]),
        ([2.5, 0.01, 0.05, 0.55, 3.0]),
        ([2.0, 0.02, 0.05, 0.55, 3.0]),
        ([2.5, 0.02, 0.05, 0.55, 3.0]),
        ([2.0, 0.01, 0.05, 0.55, 2.5]),
        ([2.5, 0.01, 0.05, 0.55, 2.5]),
        ([2.0, 0.02, 0.05, 0.55, 2.5]),
        ([2.5, 0.02, 0.05, 0.55, 2.5]),
    ),
)
def test_internal_consistency_q_Rc_eq26(Hm0, smm10_HF, Hm0_LF, gamma_f, Rc):
    q = deridder2024.calculate_overtopping_discharge_q_eq26(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    Rc_calculated = deridder2024.calculate_crest_freeboard_discharge_q_eq26(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, Rc, q_expected"),
    (
        ([2.0, 0.01, 0.55, 3.0, 0.0321745]),
        ([2.5, 0.01, 0.55, 3.0, 0.1302259]),
        ([2.0, 0.02, 0.55, 3.0, 0.0085919]),
        ([2.5, 0.02, 0.55, 3.0, 0.0452857]),
        ([2.0, 0.01, 0.55, 2.5, 0.0780481]),
        ([2.5, 0.01, 0.55, 2.5, 0.2645923]),
        ([2.0, 0.02, 0.55, 2.5, 0.0259724]),
        ([2.5, 0.02, 0.55, 2.5, 0.1097229]),
    ),
)
def test_q_backward(
    Hm0,
    smm10_HF,
    gamma_f,
    Rc,
    q_expected,
):
    q_calculated = deridder2024.calculate_overtopping_discharge_q_eq24(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    assert q_calculated == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, q, Rc_expected"),
    (
        ([2.0, 0.01, 0.55, 1, 1.0609]),
        ([2.5, 0.01, 0.55, 1, 1.5622594]),
        ([2.0, 0.02, 0.55, 1, 0.8498935]),
        ([2.5, 0.02, 0.55, 1, 1.2514790]),
    ),
)
def test_Rc_backward(
    Hm0,
    smm10_HF,
    gamma_f,
    q,
    Rc_expected,
):
    Rc_calculated = deridder2024.calculate_crest_freeboard_discharge_q_eq24(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
