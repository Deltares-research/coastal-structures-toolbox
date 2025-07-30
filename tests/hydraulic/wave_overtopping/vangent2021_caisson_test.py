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


# @pytest.mark.parametrize(
#     (
#         "Hm0, Hm0_swell, Tmm10, beta, cot_alpha, q, Ac, B_berm, dh, gamma_f, gamma_v, Rc_expected"
#     ),
#     (
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.916]),
#         ([2.5, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 6.825]),
#         ([2.0, 0.1, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.876]),
#         ([2.0, 0.2, 7.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 7.472]),
#         ([2.0, 0.2, 5.00, 30.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 4.967]),
#         ([2.0, 0.2, 5.00, 0.0, 3.5, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 1.0, 5.004]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 4.5, 1.0, 1.0, 1.0, 1.0, 5.544]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 1.0, 1.0, 1.0, 1.0, 5.547]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 2.0, 1.0, 1.0, 1.0, 5.009]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 1.0, 0.8, 1.0, 1.0, 5.553]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 0.45, 1.0, 2.706]),
#         ([2.0, 0.2, 5.00, 0.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 1.0, 0.8, 4.749]),
#         ([2.0, 0.2, 5.00, 0.0, 3.5, 1.0e-3, 5.0, 2.0, 1.0, 1.0, 1.0, 4.239]),
#         ([2.0, 0.2, 5.00, 30.0, 3.0, 1.0e-3, 5.0, 0.0, 0.0, 0.45, 1.0, 2.279]),
#     ),
# )
# def test_Rc_backward(
#     Hm0,
#     Hm0_swell,
#     Tmm10,
#     beta,
#     cot_alpha,
#     q,
#     Ac,
#     B_berm,
#     dh,
#     gamma_f,
#     gamma_v,
#     Rc_expected,
# ):
#     Rc_calculated, _ = vangent2025.calculate_crest_freeboard_Rc(
#         Hm0=Hm0,
#         Hm0_swell=Hm0_swell,
#         Tmm10=Tmm10,
#         q=q,
#         Ac=Ac,
#         cot_alpha=cot_alpha,
#         beta=beta,
#         B_berm=B_berm,
#         db=dh,
#         gamma_f=gamma_f,
#         gamma_v=gamma_v,
#         design_calculation=False,
#     )

#     assert Rc_calculated == pytest.approx(Rc_expected, abs=1e-2)
