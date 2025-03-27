import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubes_single_layer_vangent2002 as cubes


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, M_expected"),
    (
        ([5, 1025, 2400, 2898.54884531896]),
        ([5, 1025, 2800, 1571.95617950626]),
        ([6, 1025, 2400, 5008.69240471116]),
        ([5, 1000, 2400, 2549.95792569423]),
    ),
)
def test_M_nodamage_backward(
    Hs,
    rho_water,
    rho_armour,
    M_expected,
):
    M_calculated = cubes.calculate_unit_mass_M_failure(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
    )

    assert M_calculated == pytest.approx(M_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hs_expected, rho_water, rho_armour, M"),
    (
        ([11.9930541248585, 1025, 2400, 40000]),
        ([14.7065179157523, 1025, 2800, 40000]),
        ([10.8964128106154, 1025, 2400, 30000]),
        ([12.5163873957614, 1000, 2400, 40000]),
    ),
)
def test_Hs_nodamage_backward(Hs_expected, rho_water, rho_armour, M):
    Hs_calculated = cubes.calculate_significant_wave_height_Hs_failure(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
