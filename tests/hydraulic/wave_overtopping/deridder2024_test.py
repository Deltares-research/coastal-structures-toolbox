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
def test_q_backward_eq24(
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
def test_Rc_backward_eq24(
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


@pytest.mark.parametrize(
    ("Hm0, smm10_HF,Hm0_LF, gamma_f, Rc, q_expected"),
    (
        ([2.0, 0.01, 0.5, 0.55, 3.0, 0.02373296]),
        ([2.5, 0.01, 0.5, 0.55, 3.0, 0.09438792]),
        ([2.0, 0.02, 0.5, 0.55, 3.0, 0.00708636]),
        ([2.5, 0.02, 0.5, 0.55, 3.0, 0.03589002]),
        ([2.0, 0.01, 0.5, 0.55, 2.5, 0.05855709]),
        ([2.5, 0.01, 0.5, 0.55, 2.5, 0.194401]),
        ([2.0, 0.02, 0.5, 0.55, 2.5, 0.0215433]),
        ([2.5, 0.02, 0.5, 0.55, 2.5, 0.08735439]),
        ([2.0, 0.01, 0.7, 0.55, 3.0, 0.02560348]),
        ([2.5, 0.01, 0.7, 0.55, 3.0, 0.10029381]),
        ([2.0, 0.02, 0.7, 0.55, 3.0, 0.00778012]),
        ([2.5, 0.02, 0.7, 0.55, 3.0, 0.03867442]),
        ([2.0, 0.01, 0.7, 0.55, 2.5, 0.06317228]),
        ([2.5, 0.01, 0.7, 0.55, 2.5, 0.20656473]),
        ([2.0, 0.02, 0.7, 0.55, 2.5, 0.02365238]),
        ([2.5, 0.02, 0.7, 0.55, 2.5, 0.09413147]),
    ),
)
def test_q_backward_eq26(
    Hm0,
    smm10_HF,
    Hm0_LF,
    gamma_f,
    Rc,
    q_expected,
):
    q_calculated = deridder2024.calculate_overtopping_discharge_q_eq26(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    assert q_calculated == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, smm10_HF,Hm0_LF, gamma_f, q, Rc_expected"),
    (
        ([2.0, 0.01, 0.5, 0.55, 1.0, 0.92894706]),
        ([2.5, 0.01, 0.5, 0.55, 1.0, 1.36656756]),
        ([2.0, 0.02, 0.5, 0.55, 1.0, 0.77425297]),
        ([2.5, 0.02, 0.5, 0.55, 1.0, 1.12971128]),
        ([2.0, 0.01, 0.5, 0.55, 0.5, 0.92894706]),
        ([2.5, 0.01, 0.5, 0.55, 0.5, 1.36656756]),
        ([2.0, 0.02, 0.5, 0.55, 0.5, 0.77425297]),
        ([2.5, 0.02, 0.5, 0.55, 0.5, 1.12971128]),
        ([2.0, 0.01, 0.7, 0.55, 1.0, 0.97094706]),
        ([2.5, 0.01, 0.7, 0.55, 1.0, 1.40856756]),
        ([2.0, 0.02, 0.7, 0.55, 1.0, 0.81625297]),
        ([2.5, 0.02, 0.7, 0.55, 1.0, 1.17171128]),
        ([2.0, 0.01, 0.7, 0.55, 0.5, 0.97094706]),
        ([2.5, 0.01, 0.7, 0.55, 0.5, 1.40856756]),
        ([2.0, 0.02, 0.7, 0.55, 0.5, 0.81625297]),
        ([2.5, 0.02, 0.7, 0.55, 0.5, 1.17171128]),
    ),
)
def test_RC_backward_eq26(
    Hm0,
    smm10_HF,
    Hm0_LF,
    gamma_f,
    q,
    Rc_expected,
):
    Rc_calculated = deridder2024.calculate_crest_freeboard_discharge_q_eq26(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-3)
