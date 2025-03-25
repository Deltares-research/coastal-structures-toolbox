# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubes_double_layer_vandermeer1988 as cubes_vanDerMeer
import deltares_coastal_structures_toolbox.functions.core_physics as core_physics


@pytest.mark.parametrize(
    ("Hs, Tm, N_waves, rho_water, rho_armour, cot_alpha, Nod, Dn_expected"),
    (
        ([3.0, 8, 3000, 1025, 2400, 1.5, 2.0, 0.875]),
        ([3.0, 15, 3000, 1025, 2400, 1.5, 2.0, 0.772]),
        ([3.0, 8, 6000, 1025, 2400, 1.5, 2.0, 0.954]),
        ([3.0, 8, 3000, 1000, 2400, 1.5, 2.0, 0.838]),
        ([3.0, 8, 3000, 1025, 2600, 1.5, 2.0, 0.764]),
        ([3.0, 8, 3000, 1025, 2400, 2.0, 2.0, 0.875]),
        ([4.0, 8, 3000, 1025, 2400, 2.0, 2.0, 1.200]),
        ([3.0, 8, 3000, 1025, 2400, 2.0, 4.0, 0.766]),
    ),
)
def test_Dn_backward(
    Hs, Tm, N_waves, rho_water, rho_armour, cot_alpha, Nod, Dn_expected
):

    Dn_calculated = cubes_vanDerMeer.calculate_nominal_diameter_Dn_vanDerMeer1988(
        Hs=Hs,
        Tm=Tm,
        N_waves=N_waves,
        rho_water=rho_water,
        rho_armour=rho_armour,
        cot_alpha=cot_alpha,
        Nod=Nod,
    )

    assert Dn_calculated == pytest.approx(Dn_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Dn, s0m, N_waves, rho_water, rho_armour, cot_alpha, Nod, Hs_expected"),
    (
        ([1.5, 0.05, 3000, 1025, 2400, 1.5, 2.0, 4.89]),
        ([2.0, 0.05, 3000, 1025, 2400, 1.5, 2.0, 6.52]),
        ([1.5, 0.03, 3000, 1025, 2400, 1.5, 2.0, 5.14]),
        ([1.5, 0.05, 6000, 1025, 2400, 1.5, 2.0, 4.48]),
        ([1.5, 0.05, 3000, 1000, 2400, 1.5, 2.0, 5.10]),
        ([1.5, 0.05, 3000, 1025, 2600, 1.5, 2.0, 5.60]),
        ([1.5, 0.05, 3000, 1025, 2400, 2.0, 2.0, 4.89]),
        ([1.5, 0.05, 3000, 1025, 2400, 1.5, 4.0, 5.58]),
    ),
)
def test_Hs_backward(
    Dn, s0m, N_waves, rho_water, rho_armour, cot_alpha, Nod, Hs_expected
):

    Hs_calculated = cubes_vanDerMeer.calculate_wave_height_Hs_vanDerMeer1988(
        Dn=Dn,
        s0m=s0m,
        N_waves=N_waves,
        rho_water=rho_water,
        rho_armour=rho_armour,
        cot_alpha=cot_alpha,
        Nod=Nod,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs, Dn, s0m, N_waves, rho_water, rho_armour, cot_alpha, Nod_expected"),
    (
        ([3.0, 1.50, 0.05, 3000, 1025, 2400, 1.5, 0.0125]),
        ([5.0, 1.50, 0.05, 3000, 1025, 2400, 1.5, 2.27]),
        ([5.0, 2.50, 0.05, 3000, 1025, 2400, 1.5, 0.0125]),
        ([5.0, 1.50, 0.03, 3000, 1025, 2400, 1.5, 1.7]),
        ([5.0, 1.50, 0.05, 6000, 1025, 2400, 1.5, 3.81]),
        ([5.0, 1.50, 0.05, 3000, 1000, 2400, 1.5, 1.78]),
        ([5.0, 1.50, 0.05, 3000, 1025, 2600, 1.5, 1.0]),
        ([5.0, 1.50, 0.05, 3000, 1025, 2400, 2.0, 2.27]),
    ),
)
def test_Nod_backward(
    Hs, Dn, s0m, N_waves, rho_water, rho_armour, cot_alpha, Nod_expected
):

    Nod_calculated = cubes_vanDerMeer.calculate_damage_Nod_vanDerMeer1988(
        Hs=Hs,
        Dn=Dn,
        s0m=s0m,
        N_waves=N_waves,
        rho_water=rho_water,
        rho_armour=rho_armour,
        cot_alpha=cot_alpha,
    )

    assert Nod_calculated == pytest.approx(Nod_expected, abs=1e-2)
