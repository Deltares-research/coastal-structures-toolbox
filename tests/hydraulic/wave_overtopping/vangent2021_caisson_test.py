import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.vangent2021_caisson as vangent2021


@pytest.mark.parametrize(
    ("Hm0, Hm0_swell, beta, Rc, c, short_crested_waves, crossing_seas, parapet"),
    (
        ([2.0, 0.0, 0.0, 5.0, 1.0, True, False, False]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, True, False, False]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, True, False, False]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, True, False, False]),
        ([2.0, 0.0, 0.0, 5.0, 1.3, True, False, False]),
        ([2.5, 0.0, 0.0, 5.0, 1.3, True, False, False]),
        ([2.0, 0.0, 60.0, 5.0, 1.3, True, False, False]),
        ([2.0, 0.0, 0.0, 4.5, 1.3, True, False, False]),
        ([2.0, 0.0, 0.0, 5.0, 1.0, True, False, True]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, True, False, True]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, True, False, True]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, True, False, True]),
        ([2.0, 0.0, 0.0, 5.0, 1.0, False, False, False]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, False, False, False]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, False, False, False]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, False, False, False]),
        ([2.0, 0.2, 0.0, 5.0, 1.0, True, True, False]),
        ([2.5, 0.2, 0.0, 5.0, 1.0, True, True, False]),
        ([2.0, 0.2, 60.0, 5.0, 1.0, True, True, False]),
        ([2.0, 0.2, 0.0, 4.5, 1.0, True, True, False]),
    ),
)
def test_internal_consistency_q_Rc(
    Hm0,
    Hm0_swell,
    beta,
    Rc,
    c,
    short_crested_waves,
    crossing_seas,
    parapet,
):
    q_calculated = vangent2021.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Rc=Rc,
        beta=beta,
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    Rc_calculated = vangent2021.calculate_crest_freeboard_Rc(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        q=q_calculated,
        beta=beta,
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    assert Rc_calculated == pytest.approx(Rc, abs=1e-2)


@pytest.mark.parametrize(
    (
        "Hm0, Hm0_swell, beta, Rc, c, short_crested_waves, crossing_seas, parapet, q_expected"
    ),
    (
        ([2.0, 0.0, 0.0, 5.0, 1.0, True, False, False, 0.1033]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, True, False, False, 1.0146]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, True, False, False, 0.0109]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, True, False, False, 0.2738]),
        ([2.0, 0.0, 0.0, 5.0, 1.3, True, False, False, 0.0884]),
        ([2.5, 0.0, 0.0, 5.0, 1.3, True, False, False, 1.0389]),
        ([2.0, 0.0, 60.0, 5.0, 1.3, True, False, False, 0.0076]),
        ([2.0, 0.0, 0.0, 4.5, 1.3, True, False, False, 0.2609]),
        ([2.0, 0.0, 0.0, 5.0, 1.0, True, False, True, 0.03095]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, True, False, True, 0.3869]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, True, False, True, 0.0072]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, True, False, True, 0.0926]),
        ([2.0, 0.0, 0.0, 5.0, 1.0, False, False, False, 0.0804]),
        ([2.5, 0.0, 0.0, 5.0, 1.0, False, False, False, 0.8307]),
        ([2.0, 0.0, 60.0, 5.0, 1.0, False, False, False, 0.0138]),
        ([2.0, 0.0, 0.0, 4.5, 1.0, False, False, False, 0.2187]),
        ([2.0, 0.2, 0.0, 5.0, 1.0, True, True, False, 0.1207]),
        ([2.5, 0.2, 0.0, 5.0, 1.0, True, True, False, 1.1494]),
        ([2.0, 0.2, 60.0, 5.0, 1.0, True, True, False, 0.0132]),
        ([2.0, 0.2, 0.0, 4.5, 1.0, True, True, False, 0.3201]),
    ),
)
def test_q_backward(
    Hm0,
    Hm0_swell,
    beta,
    Rc,
    c,
    short_crested_waves,
    crossing_seas,
    parapet,
    q_expected,
):
    q_calculated = vangent2021.calculate_overtopping_discharge_q(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        Rc=Rc,
        beta=beta,
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    assert q_calculated * 1000 == pytest.approx(q_expected, abs=1e-3)


@pytest.mark.parametrize(
    (
        "Hm0, Hm0_swell, beta, q, c, short_crested_waves, crossing_seas, parapet, Rc_expected"
    ),
    (
        ([2.0, 0.0, 0.0, 0.1033, 1.0, True, False, False, 5.00]),
        ([2.5, 0.0, 0.0, 1.0146, 1.0, True, False, False, 5.00]),
        ([2.0, 0.0, 60.0, 0.0109, 1.0, True, False, False, 5.00]),
        ([2.0, 0.0, 0.0, 0.2738, 1.0, True, False, False, 4.50]),
        ([2.0, 0.0, 0.0, 0.0884, 1.3, True, False, False, 5.00]),
        ([2.5, 0.0, 0.0, 1.0389, 1.3, True, False, False, 5.00]),
        ([2.0, 0.0, 60.0, 0.0076, 1.3, True, False, False, 5.00]),
        ([2.0, 0.0, 0.0, 0.2609, 1.3, True, False, False, 4.50]),
        ([2.0, 0.0, 0.0, 0.0309, 1.0, True, False, True, 5.00]),
        ([2.5, 0.0, 0.0, 0.3869, 1.0, True, False, True, 5.00]),
        ([2.0, 0.0, 60.0, 0.0072, 1.0, True, False, True, 5.00]),
        ([2.0, 0.0, 0.0, 0.0926, 1.0, True, False, True, 4.50]),
        ([2.0, 0.0, 0.0, 0.0804, 1.0, False, False, False, 5.00]),
        ([2.5, 0.0, 0.0, 0.8307, 1.0, False, False, False, 5.00]),
        ([2.0, 0.0, 60.0, 0.0138, 1.0, False, False, False, 5.00]),
        ([2.0, 0.0, 0.0, 0.2187, 1.0, False, False, False, 4.50]),
        ([2.0, 0.2, 0.0, 0.1207, 1.0, True, True, False, 5.00]),
        ([2.5, 0.2, 0.0, 1.1494, 1.0, True, True, False, 5.00]),
        ([2.0, 0.2, 60.0, 0.0132, 1.0, True, True, False, 5.00]),
        ([2.0, 0.2, 0.0, 0.3201, 1.0, True, True, False, 4.50]),
    ),
)
def test_Rc_backward(
    Hm0,
    Hm0_swell,
    beta,
    q,
    c,
    short_crested_waves,
    crossing_seas,
    parapet,
    Rc_expected,
):
    Rc_calculated = vangent2021.calculate_crest_freeboard_Rc(
        Hm0=Hm0,
        Hm0_swell=Hm0_swell,
        q=q / 1000,
        beta=beta,
        c=c,
        short_crested_waves=short_crested_waves,
        crossing_seas=crossing_seas,
        parapet=parapet,
    )

    assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
