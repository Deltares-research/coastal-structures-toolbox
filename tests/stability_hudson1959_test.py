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
def test_internal_consistency_nodamage_Hs_M50(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    M50_expected,
):
    M50_result = hudson1959.calculate_median_rock_mass_M50_no_damage(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    Hs_result = hudson1959.calculate_significant_wave_height_Hs_no_damage(
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
def test_M50_nodamage_backward(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    M50_expected,
):
    M50_calculated = hudson1959.calculate_median_rock_mass_M50_no_damage(
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
def test_Hs_nodamage_backward(
    M50,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    Hs_expected,
):
    Hs_calculated = hudson1959.calculate_significant_wave_height_Hs_no_damage(
        M50=M50,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, rock_type, damage_percentage, M50_expected"
    ),
    (
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 4, 908.19]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 5, 720.95]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 9, 720.95]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 10, 613.0]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 14, 613.0]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 15, 525.57]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 19, 525.57]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 20, 423.07]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 29, 423.07]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 30, 323.98]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 39, 323.98]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 40, 248.67]),
        ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 49, 248.67]),
        # ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 50, 248.67]),
    ),
)
def test_M50_damage_backward(
    Hs,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    rock_type,
    damage_percentage,
    M50_expected,
):
    M50_calculated = hudson1959.calculate_median_rock_mass_M50(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
        rock_type=rock_type,
        damage_percentage=damage_percentage,
    )

    assert M50_calculated == pytest.approx(M50_expected, abs=1e-1)


@pytest.mark.parametrize(
    (
        "M50, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, rock_type, damage_percentage, Hs_expected"
    ),
    (
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 4, 3.41]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 5, 3.68]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 9, 3.68]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 10, 3.89]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 14, 3.89]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 15, 4.09]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 19, 4.09]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 20, 4.40]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 29, 4.40]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 30, 4.81]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 39, 4.81]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 40, 5.25]),
        ([4500, 1025, 2650, 4.0, 3.0, 1.27, "rough", 49, 5.25]),
        # ([2.0, 1025, 2650, 4.0, 3.0, 1.27, "rough", 50, 248.67]),
    ),
)
def test_Hs_damage_backward(
    M50,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    rock_type,
    damage_percentage,
    Hs_expected,
):
    Hs_calculated = hudson1959.calculate_significant_wave_height_Hs(
        M50=M50,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
        rock_type=rock_type,
        damage_percentage=damage_percentage,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-1)
