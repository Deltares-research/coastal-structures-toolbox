import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.accropodes_Hudson1959 as hudson1959


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([3.0, 1025, 2400, 12.0, 1.33, 1.0, 1681.9]),
        ([3.0, 1000, 2400, 12.0, 1.33, 1.0, 1479.6]),
        ([3.0, 1025, 2600, 12.0, 1.33, 1.0, 1212.4]),
        ([3.0, 1025, 2400, 15.0, 1.33, 1.0, 1345.5]),
        ([3.0, 1025, 2400, 12.0, 1.5, 1.0, 1491.3]),
        ([5.0, 1025, 2400, 12.0, 1.33, 1.0, 7786.7]),
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
    M_calculated = hudson1959.calculate_unit_mass_M_Hudson1959(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert M_calculated == pytest.approx(M_expected, abs=1e-1)


@pytest.mark.parametrize(
    ("M, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, Hs_expected"),
    (
        ([3000, 1025, 2400, 12.0, 1.33, 1.0, 3.64]),
        ([6000, 1025, 2400, 12.0, 1.33, 1.0, 4.58]),
        ([3000, 1000, 2400, 12.0, 1.33, 1.0, 3.80]),
        ([3000, 1025, 2600, 12.0, 1.33, 1.0, 4.06]),
        ([3000, 1025, 2400, 15.0, 1.33, 1.0, 3.92]),
        ([3000, 1025, 2400, 12.0, 1.5, 1.0, 3.79]),
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
    Hs_calculated = hudson1959.calculate_significant_wave_height_Hs_Hudson1959(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
