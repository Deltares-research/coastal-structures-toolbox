import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as hudson1959


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M50_expected"),
    (
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, 908.19]),
        ([2.0, 1025, 2650, 4.0, 2.0, 1.27, 1362.3]),
        ([2.5, 1025, 2650, 4.0, 2.0, 1.27, 2660.7]),
        ([2.5, 1000, 2650, 4.0, 2.0, 1.27, 2360.1]),
        ([2.5, 1000, 2700, 4.0, 2.0, 1.27, 2198.7]),
        ([2.5, 1000, 2700, 2.0, 2.0, 1.27, 4397.3]),
        ([2.5, 1000, 2700, 2.0, 2.0, 1.0, 2146.7]),
    ),
)
def test_internal_consistency_Hs_M50(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    M50_expected,
):
    M50_result = hudson1959.calculate_hudson1959_no_damage_M50(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    Hs_result = hudson1959.calculate_hudson1959_no_damage_Hs(
        M50=M50_result,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_result == pytest.approx(Hs, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M50_expected"),
    (
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, 908.19]),
        ([2.0, 1025, 2650, 4.0, 2.0, 1.27, 1362.3]),
        ([2.5, 1025, 2650, 4.0, 2.0, 1.27, 2660.7]),
        ([2.5, 1000, 2650, 4.0, 2.0, 1.27, 2360.1]),
        ([2.5, 1000, 2700, 4.0, 2.0, 1.27, 2198.7]),
        ([2.5, 1000, 2700, 2.0, 2.0, 1.27, 4397.3]),
        ([2.5, 1000, 2700, 2.0, 2.0, 1.0, 2146.7]),
    ),
)
def test_M50_backward(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    M50_expected,
):
    M50_calculated = hudson1959.calculate_hudson1959_no_damage_M50(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert M50_calculated == pytest.approx(M50_expected, abs=1e-1)


@pytest.mark.parametrize(
    ("M50, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, Hs_expected"),
    (
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, 3.41]),
        ([4500, 1025, 2650, 4.0, 2.0, 1.27, 2.98]),
        ([4500, 1025, 2700, 4.0, 2.0, 1.27, 3.05]),
        ([4500, 1025, 2700, 2.0, 2.0, 1.27, 2.42]),
        ([4500, 1025, 2700, 2.0, 2.0, 1.0, 3.08]),
        ([1900, 1025, 2700, 2.0, 2.0, 1.0, 2.31]),
        ([1900, 1000, 2700, 2.0, 2.0, 1.0, 2.40]),
    ),
)
def test_Hs_backward(
    M50,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    Hs_expected,
):
    Hs_calculated = hudson1959.calculate_hudson1959_no_damage_Hs(
        M50=M50,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
