import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.kramerburcharth2004_lowcrested as kb2004


@pytest.mark.parametrize(
    ("rho_armour, Hs, Rc, Dn50_expected"),
    (
        ([2650, 3.615, 4.0, 2.000]),
        ([2850, 4.060, 4.0, 2.000]),
        ([2650, 4.540, 4.0, 2.500]),
        ([2650, 6.532, -4.0, 2.000]),
        ([2650, 5.707, -4.0, 1.500]),
    ),
)
def test_Dn50_backward(
    rho_armour,
    Hs,
    Rc,
    Dn50_expected,
):
    Dn50_calculated = kb2004.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Rc=Rc,
        rho_armour=rho_armour,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("rho_armour, Dn50, Rc, Hs_expected"),
    (
        ([2650, 2.0, 4.0, 3.615]),
        ([2850, 2.0, 4.0, 4.060]),
        ([2650, 2.5, 4.0, 4.540]),
        ([2650, 2.0, -4.0, 6.532]),
        ([2650, 1.5, -4.0, 5.707]),
    ),
)
def test_Hs_backward(
    rho_armour,
    Dn50,
    Rc,
    Hs_expected,
):
    Hs_calculated = kb2004.calculate_significant_wave_height_Hs(
        Dn50=Dn50,
        Rc=Rc,
        rho_armour=rho_armour,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("rho_armour, Dn50, Rc"),
    (
        ([2650, 2.0, 4.0]),
        ([2850, 2.0, 4.0]),
        ([2650, 2.5, 4.0]),
        ([2650, 2.0, -4.0]),
        ([2650, 1.5, -4.0]),
    ),
)
def test_internal_consistency_Hs_Dn50(
    rho_armour,
    Dn50,
    Rc,
):
    Hs_calculated = kb2004.calculate_significant_wave_height_Hs(
        Dn50=Dn50,
        Rc=Rc,
        rho_armour=rho_armour,
    )

    Dn50_calculated = kb2004.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs_calculated,
        Rc=Rc,
        rho_armour=rho_armour,
    )

    assert Dn50_calculated == pytest.approx(Dn50, abs=1e-3)
