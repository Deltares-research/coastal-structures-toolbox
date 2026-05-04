import pytest


from deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping import (
    deridder2024,
)
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.deridder2026 as deridder2026


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
def test_internal_consistency_q_Rc_eq28(Hm0, smm10_HF, gamma_f, Rc, q_expected):
    q = deridder2026.calculate_overtopping_discharge_q_eq28(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    Rc_calculated = deridder2026.calculate_crest_freeboard_discharge_q_eq28(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0_HF, smm10_HF, Hm0_LF, gamma_f, Rc"),
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
def test_internal_consistency_q_Rc_eq32(Hm0_HF, smm10_HF, Hm0_LF, gamma_f, Rc):
    q = deridder2026.calculate_overtopping_discharge_q_eq32(
        Hm0_HF=Hm0_HF,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    Rc_calculated = deridder2026.calculate_crest_freeboard_discharge_q_eq32(
        Hm0_HF=Hm0_HF,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, Rc, q_expected"),
    (
        ([2.0, 0.01, 0.55, 3.0, 0.0386781]),
        ([2.5, 0.01, 0.55, 3.0, 0.1110804]),
        ([2.0, 0.02, 0.55, 3.0, 0.0139372]),
        ([2.5, 0.02, 0.55, 3.0, 0.0490916]),
        ([2.0, 0.01, 0.55, 2.5, 0.0704917]),
        ([2.5, 0.01, 0.55, 2.5, 0.1795458]),
        ([2.0, 0.02, 0.55, 2.5, 0.0301114]),
        ([2.5, 0.02, 0.55, 2.5, 0.0909180]),
    ),
)
def test_q_backward_eq28(
    Hm0,
    smm10_HF,
    gamma_f,
    Rc,
    q_expected,
):
    q_calculated = deridder2026.calculate_overtopping_discharge_q_eq28(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    assert q_calculated == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, smm10_HF, gamma_f, q, Rc_expected"),
    (
        ([2.0, 0.01, 0.55, 1, 0.2905937]),
        ([2.5, 0.01, 0.55, 1, 0.7117761]),
        ([2.0, 0.02, 0.55, 1, 0.2264203]),
        ([2.5, 0.02, 0.55, 1, 0.5545907]),
    ),
)
def test_Rc_backward_eq28(
    Hm0,
    smm10_HF,
    gamma_f,
    q,
    Rc_expected,
):
    Rc_calculated = deridder2026.calculate_crest_freeboard_discharge_q_eq28(
        Hm0=Hm0,
        smm10_HF=smm10_HF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0_HF, smm10_HF,Hm0_LF, gamma_f, Rc, q_expected"),
    (
        ([2.0, 0.01, 0.5, 0.55, 3.0, 0.04108432]),
        ([2.5, 0.01, 0.5, 0.55, 3.0, 0.092379555]),
        ([2.0, 0.02, 0.5, 0.55, 3.0, 0.0153436574]),
        ([2.5, 0.02, 0.5, 0.55, 3.0, 0.0420124525]),
        ([2.0, 0.01, 0.5, 0.55, 2.5, 0.0659142070]),
        ([2.5, 0.01, 0.5, 0.55, 2.5, 0.1348398142]),
        ([2.0, 0.02, 0.5, 0.55, 2.5, 0.0299413213]),
        ([2.5, 0.02, 0.5, 0.55, 2.5, 0.0717218841]),
        ([2.0, 0.01, 0.7, 0.55, 3.0, 0.0493552879]),
        ([2.5, 0.01, 0.7, 0.55, 3.0, 0.1069798222]),
        ([2.0, 0.02, 0.7, 0.55, 3.0, 0.0198875651]),
        ([2.5, 0.02, 0.7, 0.55, 3.0, 0.0517011511]),
        ([2.0, 0.01, 0.7, 0.55, 2.5, 0.0791838358]),
        ([2.5, 0.01, 0.7, 0.55, 2.5, 0.1561507767]),
        ([2.0, 0.02, 0.7, 0.55, 2.5, 0.0388082165]),
        ([2.5, 0.02, 0.7, 0.55, 2.5, 0.0882620211]),
    ),
)
def test_q_backward_eq32(
    Hm0_HF,
    smm10_HF,
    Hm0_LF,
    gamma_f,
    Rc,
    q_expected,
):
    q_calculated = deridder2026.calculate_overtopping_discharge_q_eq32(
        Hm0_HF=Hm0_HF,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        Rc=Rc,
    )

    assert q_calculated == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0_HF, smm10_HF,Hm0_LF, gamma_f, q, Rc_expected"),
    (
        ([2.0, 0.01, 0.5, 0.55, 0.5, 0.35684663]),
        ([2.5, 0.01, 0.5, 0.55, 0.5, 0.76734057]),
        ([2.0, 0.02, 0.5, 0.55, 0.5, 0.39438188]),
        ([2.5, 0.02, 0.5, 0.55, 0.5, 0.68464493]),
        ([2.0, 0.01, 0.7, 0.55, 0.5, 0.55084663]),
        ([2.5, 0.01, 0.7, 0.55, 0.5, 0.96134057]),
        ([2.0, 0.02, 0.7, 0.55, 0.5, 0.58838188]),
        ([2.5, 0.02, 0.7, 0.55, 0.5, 0.87864493]),
    ),
)
def test_RC_backward_eq32(
    Hm0_HF,
    smm10_HF,
    Hm0_LF,
    gamma_f,
    q,
    Rc_expected,
):
    Rc_calculated = deridder2026.calculate_crest_freeboard_discharge_q_eq32(
        Hm0_HF=Hm0_HF,
        smm10_HF=smm10_HF,
        Hm0_LF=Hm0_LF,
        gamma_f=gamma_f,
        q=q,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-3)
