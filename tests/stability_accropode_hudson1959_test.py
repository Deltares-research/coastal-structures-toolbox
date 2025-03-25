import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.accropode_hudson1959 as hudson1959


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([3.0, 1025, 2400, 12.0, 1.33, 1.0, 1681.9]),
        ([3.0, 1000, 2400, 12.0, 1.33, 1.0, 1479.6]),
        ([3.0, 1025, 2600, 12.0, 1.33, 1.0, 1212.4]),
        ([3.0, 1025, 2400, 15.0, 1.33, 1.0, 1345.5]),
        ([3.0, 1025, 2400, 12.0, 1.5, 1.0, 1491.3]),
        ([5.0, 1025, 2400, 12.0, 1.33, 1.0, 7786.7]),
        ([3.0, 1025, 2400, 12.0, 2.0, 1.0, 1118.5]),
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
        ([3000, 1025, 2400, 12.0, 1.33, 1.0, 3.64]),
        ([6000, 1025, 2400, 12.0, 1.33, 1.0, 4.58]),
        ([3000, 1000, 2400, 12.0, 1.33, 1.0, 3.80]),
        ([3000, 1025, 2600, 12.0, 1.33, 1.0, 4.06]),
        ([3000, 1025, 2400, 15.0, 1.33, 1.0, 3.92]),
        ([3000, 1025, 2400, 12.0, 1.5, 1.0, 3.79]),
        ([3000, 1025, 2400, 12.0, 2.0, 1.0, 4.17]),
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
        ([0.9, 15]),
        ([1.1, 14.87]),
        ([4.8, 9.97]),
        ([5.2, 9.63]),
        ([8, 8.68]),
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
