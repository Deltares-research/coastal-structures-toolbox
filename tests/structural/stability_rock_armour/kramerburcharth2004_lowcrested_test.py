import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.kramerburcharth2004_lowcrested as kb2004


@pytest.mark.parametrize(
    ("rho_armour, Hs, Rc, Dn50_expected"),
    (
        ([2650, 2.0, 0.3, 0.974]),
        ([2850, 2.0, 0.3, 0.872]),
        ([2650, 2.5, 0.3, 1.207]),
        ([2650, 2.0, -0.3, 0.872]),
        ([2650, 2.5, -0.3, 1.105]),
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
        ([2650, 0.8, 0.3, 1.626]),
        ([2850, 0.8, 0.3, 1.826]),
        ([2650, 0.7, 0.3, 1.412]),
        ([2650, 0.8, -0.3, 1.845]),
        ([2650, 0.7, -0.3, 1.631]),
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
        ([2650, 0.8, 0.3]),
        ([2850, 0.8, 0.3]),
        ([2650, 0.7, 0.3]),
        ([2650, 0.8, -0.3]),
        ([2650, 0.7, -0.3]),
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
