import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as eurotop2018


@pytest.mark.parametrize(
    (
        "Hm0, Tmm10, beta, cot_alpha_down, cot_alpha_up, B_berm, dh, gamma_f, z2p_expected"
    ),
    (
        # TODO large (!) diffs with TAW2002, check if correct
        ([2.0, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 7.111]),
        ([2.5, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 8.738]),
        ([2.0, 12.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 7.495]),
        ([2.0, 8.0, 30.0, 3.0, 3.0, 0.0, 0.0, 1.0, 6.641]),
        ([2.0, 8.0, 0.0, 3.5, 3.0, 0.0, 0.0, 1.0, 7.059]),
        ([2.0, 8.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0, 7.375]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 5.0, 1.0, 1.0, 6.176]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 5.0, -0.5, 1.0, 5.864]),
        ([2.0, 8.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.45, 3.465]),
        ([2.0, 8.0, 0.0, 3.5, 3.0, 2.0, 1.0, 1.0, 7.073]),
        ([2.0, 8.0, 0.0, 3.0, 2.0, 2.0, 1.0, 1.0, 7.414]),
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
    z2p_calculated, _ = eurotop2018.calculate_wave_runup_height_z2p(
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
