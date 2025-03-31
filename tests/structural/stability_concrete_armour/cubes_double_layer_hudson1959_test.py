# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubes_double_layer_hudson1959 as cubes_hudson


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([3.0, 1025, 2400, 6.5, 1.5, 1.0, 2753.2]),
        ([3.0, 1025, 2400, 7.5, 1.5, 1.0, 2386.1]),
        ([3.0, 1025, 2300, 7.5, 1.5, 1.0, 2868.0]),
        ([3.0, 1000, 2400, 7.5, 1.5, 1.0, 2099.1]),
        ([3.0, 1025, 2400, 7.5, 2.0, 1.0, 1789.6]),
        ([2.0, 1025, 2400, 7.5, 2.0, 1.0, 530.24]),
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
    M_calculated = cubes_hudson.calculate_unit_mass_M(
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
        ([4800, 1025, 2400, 6.5, 1.5, 1.0, 3.61]),
        ([4800, 1025, 2400, 7.5, 1.5, 1.0, 3.79]),
        ([4800, 1025, 2300, 7.5, 1.5, 1.0, 3.56]),
        ([4800, 1000, 2400, 7.5, 1.5, 1.0, 3.95]),
        ([4800, 1025, 2400, 7.5, 2.0, 1.0, 4.17]),
        ([20000, 1025, 2400, 7.5, 2.0, 1.0, 6.71]),
    ),
)
def test_M_nodamage_backward(
    M,
    rho_water,
    rho_armour,
    KD,
    cot_alpha,
    alpha_Hs,
    Hs_expected,
):
    Hs_calculated = cubes_hudson.calculate_significant_wave_height_Hs(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
