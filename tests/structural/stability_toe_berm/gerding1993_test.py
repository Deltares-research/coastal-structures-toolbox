import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.structural.stability_toe_berm.gerding1993 as gerding


@pytest.mark.parametrize(
    ("Hs, ht, h, Dn50, rho_rock, rho_water, Nod_expected"),
    (
        ([3, 4, 5, 0.5, 2650, 1025, 1.62]),
        ([2, 4, 5, 0.5, 2650, 1025, 0.1086]),
        ([3, 2, 5, 0.5, 2650, 1025, 13.55]),
        ([3, 4, 6, 0.5, 2650, 1025, 1.62]),
        ([3, 4, 5, 0.75, 2650, 1025, 0.41]),
        ([3, 4, 5, 0.5, 2800, 1025, 0.9]),
        ([3, 4, 5, 0.5, 2650, 1000, 1.24]),
    ),
)
def test_Nod_backward(Hs, ht, h, Dn50, rho_rock, rho_water, Nod_expected):

    Nod_calculated = gerding.calculate_damage_Nod(
        Hs=Hs,
        ht=ht,
        h=h,
        Dn50=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Nod_calculated == pytest.approx(Nod_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs, ht, h, Dn50_expected, rho_rock, rho_water, Nod"),
    (
        ([3.0, 4, 5, 0.5, 2650, 1025, 1.62]),
        ([2.0, 4, 5, 0.5, 2650, 1025, 0.1086]),
        ([3.0, 2, 5, 0.5, 2650, 1025, 13.55]),
        ([3.0, 4, 6, 0.5, 2650, 1025, 1.62]),
        ([3.0, 4, 5, 0.75, 2650, 1025, 0.41]),
        ([3.0, 4, 5, 0.5, 2800, 1025, 0.9]),
        ([3.0, 4, 5, 0.5, 2650, 1000, 1.24]),
        ([3.0, 4, 5, 0.25, 2650, 1000, 6.9288]),
    ),
)
def test_Dn50_backward(Hs, ht, h, Dn50_expected, rho_rock, rho_water, Nod):

    Dn50_calculated = gerding.calculate_nominal_diameter_Dn50(
        Hs=Hs,
        ht=ht,
        h=h,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs, ht_expected, h, Dn50, rho_rock, rho_water, Nod"),
    (
        ([3.0, 4.0, 5, 0.5, 2650, 1025, 1.62]),
        ([2.0, 4.0, 5, 0.5, 2650, 1025, 0.1086]),
        ([3.0, 2.0, 5, 0.5, 2650, 1025, 13.55]),
        ([3.0, 4.0, 6, 0.5, 2650, 1025, 1.62]),
        ([3.0, 4.0, 5, 0.75, 2650, 1025, 0.414]),
        ([3.0, 4.0, 5, 0.5, 2800, 1025, 0.9]),
        ([3.0, 4.0, 5, 0.5, 2650, 1000, 1.24]),
        ([3.0, 4.0, 5, 0.25, 2650, 1000, 6.9288]),
        ([3.0, 5.5, 5, 0.25, 2650, 1025, 1.890]),
    ),
)
def test_ht_backward(Hs, ht_expected, h, Dn50, rho_rock, rho_water, Nod):

    ht_calculated = gerding.calculate_depth_above_toe_ht(
        Hs=Hs,
        Dn50=Dn50,
        h=h,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert ht_calculated == pytest.approx(ht_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs_expected, ht, h, Dn50, rho_rock, rho_water, Nod"),
    (
        ([3.0, 4.0, 5, 0.5, 2650, 1025, 1.62]),
        ([2.0, 4.0, 5, 0.5, 2650, 1025, 0.1086]),
        ([3.0, 2.0, 5, 0.5, 2650, 1025, 13.55]),
        ([3.0, 4.0, 6, 0.5, 2650, 1025, 1.62]),
        ([3.0, 4.0, 5, 0.75, 2650, 1025, 0.414]),
        ([3.0, 4.0, 5, 0.5, 2800, 1025, 0.9]),
        ([3.0, 4.0, 5, 0.5, 2650, 1000, 1.24]),
        ([3.0, 4.0, 5, 0.25, 2650, 1000, 6.9288]),
        ([3.0, 5.5, 5, 0.25, 2650, 1025, 1.890]),
        ([4.0, 5.5, 5, 0.25, 2650, 1025, 12.863]),
        ([3.5, 5.5, 5, 0.25, 2650, 1025, 5.281]),
    ),
)
def test_Hs_backward(Hs_expected, ht, h, Dn50, rho_rock, rho_water, Nod):

    Hs_calculated = gerding.calculate_wave_height_Hs(
        ht=ht,
        Dn50=Dn50,
        h=h,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
