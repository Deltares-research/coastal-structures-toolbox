import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as taw2002


@pytest.mark.parametrize(
    (
        "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, B_berm, dh, gamma_f, z2p_expected"
    ),
    (
        ([2.0, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 6.515]),
        ([2.5, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 7.995]),
        ([2.0, 12.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 6.898]),
        ([2.0, 8.0, 30.0, 3.0, 3.0, 0.0, 0.0, 1.0, 6.085]),
        ([2.0, 8.0, 0.0, 3.5, 3.0, 0.0, 0.0, 1.0, 6.461]),
        ([2.0, 8.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0, 6.772]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 5.0, 1.0, 1.0, 6.176]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 5.0, -0.5, 1.0, 5.864]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.45, 3.175]),
        ([2.0, 8.0, 0.0, 3.5, 3.0, 2.0, 1.0, 1.0, 6.479]),
        ([2.0, 8.0, 0.0, 3.0, 2.0, 2.0, 1.0, 1.0, 6.813]),
    ),
)
def test_z2p_backward(
    Hm0,
    Tmm10,
    beta,
    cot_alpha_down,
    cot_alpha_up,
    B_berm,
    dh,
    gamma_f,
    z2p_expected,
):
    z2p_calculated = taw2002.calculate_wave_runup_height_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        B_berm=B_berm,
        db=dh,
        gamma_f=gamma_f,
        use_best_fit=False,
    )

    assert z2p_calculated == pytest.approx(z2p_expected, abs=1e-2)
