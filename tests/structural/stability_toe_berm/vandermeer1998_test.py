import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.structural.stability_toe_berm.vandermeer1998 as vandermeer


@pytest.mark.parametrize(
    ("Hs, ht, h, Dn50, rho_rock, rho_water, Nod_expected"),
    (
        ([5, 5, 10, 1.0422, 2650, 1025, 1.17]),
        ([6, 5, 10, 1.0422, 2650, 1025, 3.96]),
        ([5, 3, 10, 1.0422, 2650, 1025, 7.42]),
        ([5, 5, 8, 1.0422, 2650, 1025, 0.242]),
        ([5, 5, 10, 0.827, 2650, 1025, 5.48]),
        ([5, 5, 10, 0.812, 2800, 1025, 3.44]),
        ([5, 5, 10, 0.812, 2800, 1000, 2.66]),
    ),
)
def test_Nod_backward(Hs, ht, h, Dn50, rho_rock, rho_water, Nod_expected):

    Nod_calculated = vandermeer.calculate_damage_Nod(
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
        ([5, 5, 10, 1.0422, 2650, 1025, 1.17]),
        ([6, 5, 10, 1.0422, 2650, 1025, 3.96]),
        ([5, 3, 10, 1.0422, 2650, 1025, 7.42]),
        ([5, 5, 8, 1.0422, 2650, 1025, 0.242]),
        ([5, 5, 10, 0.827, 2650, 1025, 5.48]),
        ([5, 5, 10, 0.812, 2800, 1025, 3.44]),
        ([5, 5, 10, 0.812, 2800, 1000, 2.66]),
        ([5, 5, 10, 0.683, 2800, 1000, 8.46]),
    ),
)
def test_Dn50_backward(Hs, ht, h, Dn50_expected, rho_rock, rho_water, Nod):

    Dn50_calculated = vandermeer.calculate_nominal_diameter_Dn50(
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
        ([5, 5, 10, 1.0422, 2650, 1025, 1.17]),
        ([6, 5, 10, 1.0422, 2650, 1025, 3.96]),
        ([5, 3, 10, 1.0422, 2650, 1025, 7.42]),
        ([5, 5, 8, 1.0422, 2650, 1025, 0.242]),
        ([5, 5, 10, 0.827, 2650, 1025, 5.48]),
        ([5, 5, 10, 0.812, 2800, 1025, 3.44]),
        ([5, 5, 10, 0.812, 2800, 1000, 2.66]),
        ([5, np.nan, 10, 1.02, 2800, 1000, 8.46]),
        ([5, 9, 10, 0.3756, 2800, 1000, 2.0]),
    ),
)
def test_ht_backward(Hs, ht_expected, h, Dn50, rho_rock, rho_water, Nod):

    ht_calculated = vandermeer.calculate_depth_above_toe_ht(
        Hs=Hs,
        Dn50=Dn50,
        h=h,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert ht_calculated == pytest.approx(ht_expected, abs=1e-2, nan_ok=True)


@pytest.mark.parametrize(
    ("Hs_expected, ht, h, Dn50, rho_rock, rho_water, Nod"),
    (
        ([5, 5, 10, 1.0422, 2650, 1025, 1.17]),
        ([6, 5, 10, 1.0422, 2650, 1025, 3.96]),
        ([5, 3, 10, 1.0422, 2650, 1025, 7.42]),
        ([5, 5, 8, 1.0422, 2650, 1025, 0.242]),
        ([5, 5, 10, 0.827, 2650, 1025, 5.48]),
        ([5, 5, 10, 0.812, 2800, 1025, 3.44]),
        ([5, 5, 10, 0.812, 2800, 1000, 2.66]),
    ),
)
def test_Hs_backward(Hs_expected, ht, h, Dn50, rho_rock, rho_water, Nod):

    Hs_calculated = vandermeer.calculate_wave_height_Hs(
        ht=ht,
        Dn50=Dn50,
        h=h,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
