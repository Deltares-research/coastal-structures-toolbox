import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.core_loc_Hudson1959 as hudson1959


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([3.0, 1025, 2400, 16.0, 1.33, 1.0, 1261.4]),
        ([3.0, 1000, 2400, 16.0, 1.33, 1.0, 1109.7]),
        ([3.0, 1025, 2600, 16.0, 1.33, 1.0, 909.28]),
        ([3.0, 1025, 2400, 13.0, 1.33, 1.0, 1552.5]),
        ([3.0, 1025, 2400, 16.0, 1.5, 1.0, 1118.5]),
        ([5.0, 1025, 2400, 16.0, 1.33, 1.0, 5840.0]),
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

    assert M_calculated == pytest.approx(M_expected, abs=1e-1)


@pytest.mark.parametrize(
    ("M, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, Hs_expected"),
    (
        ([3000, 1025, 2400, 16.0, 1.33, 1.0, 4.00]),
        ([6000, 1025, 2400, 16.0, 1.33, 1.0, 5.05]),
        ([3000, 1000, 2400, 16.0, 1.33, 1.0, 4.18]),
        ([3000, 1025, 2600, 16.0, 1.33, 1.0, 4.47]),
        ([3000, 1025, 2400, 13.0, 1.33, 1.0, 3.74]),
        ([3000, 1025, 2400, 16.0, 1.5, 1.0, 4.17]),
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


@pytest.mark.parametrize(
    ("seabed_slope_perc, KD_expected"),
    (
        ([1.1, 15.87]),
        ([2, 14.67]),
        ([4, 12]),
        ([4.9, 10.79]),
        ([5.3, 10.56]),
        ([9, 9.33]),
    ),
)
def test_KD_seabedslope(
    seabed_slope_perc,
    KD_expected,
):
    KD_calculated = hudson1959.calculate_KD_breaking_trunk_from_seabed_slope(
        seabed_slope_perc=seabed_slope_perc,
    )

    assert KD_calculated == pytest.approx(KD_expected, abs=1e-1)
