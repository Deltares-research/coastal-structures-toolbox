# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.tetrapods_hudson1959 as tetrapods


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, M_expected"),
    (
        ([3.0, 1025, 2400, 7.0, 1.5, 1.0, 2556.5]),
        ([3.0, 1025, 2400, 8.0, 1.5, 1.0, 2237.0]),
        ([3.0, 1025, 2300, 8.0, 1.5, 1.0, 2688.8]),
        ([3.0, 1000, 2400, 8.0, 1.5, 1.0, 1967.9]),
        ([3.0, 1025, 2400, 8.0, 2.0, 1.0, 1677.7]),
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
    M_calculated = tetrapods.calculate_unit_mass_M_Hudson1959(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert M_calculated == pytest.approx(M_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("M, rho_water, rho_armour, KD, cot_alpha, alpha_Hs, Hs_expected"),
    (
        ([4800, 1025, 2400, 7.0, 1.5, 1.0, 3.70]),
        ([4800, 1025, 2400, 8.0, 1.5, 1.0, 3.87]),
        ([4800, 1025, 2300, 8.0, 1.5, 1.0, 3.63]),
        ([4800, 1000, 2400, 8.0, 1.5, 1.0, 4.04]),
        ([4800, 1025, 2400, 8.0, 2.0, 1.0, 4.26]),
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
    Hs_calculated = tetrapods.calculate_significant_wave_height_Hs_Hudson1959(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        KD=KD,
        cot_alpha=cot_alpha,
        alpha_Hs=alpha_Hs,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
