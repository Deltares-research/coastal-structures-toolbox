import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubipod_hudson1959 as hudson1959


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([5.0, 1025, 2350, 28.0, 2.25, 1.0, 0.92 * 2350]),
        ([10.0, 1025, 2350, 28.0, 2.25, 1.0, 7.35 * 2350]),
        ([5.0, 1000, 2350, 28.0, 2.25, 1.0, 0.81 * 2350]),
        ([5.0, 1025, 2600, 28.0, 2.25, 1.0, 0.55 * 2600]),
        ([5.0, 1025, 2350, 5.0, 2.25, 1.0, 5.14 * 2350]),
        ([5.0, 1025, 2350, 28.0, 1.5, 1.0, 1.38 * 2350]),
    ),
)
def test_M_nodamage_backward(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    M_expected,
):
    M_calculated = hudson1959.calculate_unit_mass_M(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    #  0.0049*2350
    assert M_calculated == pytest.approx(M_expected, abs=14)


@pytest.mark.parametrize(
    ("Hs_expected, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M"),
    (
        ([5.0, 1025, 2350, 28.0, 2.25, 1.0, 0.92 * 2350]),
        ([10.0, 1025, 2350, 28.0, 2.25, 1.0, 7.35 * 2350]),
        ([5.0, 1000, 2350, 28.0, 2.25, 1.0, 0.81 * 2350]),
        ([5.0, 1025, 2600, 28.0, 2.25, 1.0, 0.55 * 2600]),
        ([5.0, 1025, 2350, 5.0, 2.25, 1.0, 5.14 * 2350]),
        ([5.0, 1025, 2350, 28.0, 1.5, 1.0, 1.38 * 2350]),
    ),
)
def test_Hs_nodamage_backward(
    M,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    Hs_expected,
):
    Hs_calculated = hudson1959.calculate_significant_wave_height_Hs(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
