import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer2021 as vandermeer2021


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, M50, S_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 0.730]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.01]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 0.597]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 0.461]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 1.03]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 2.15]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 1000.0, 1.69]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 0.539]),
        ([4.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 0.356]),
        ([4.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 2.96]),
    ),
)
def test_S_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    Hs,
    M50,
    S_expected,
):
    S_calculated = vandermeer2021.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    assert S_calculated == pytest.approx(S_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, S, Dn50_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 0.591]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 0.724]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 2.0, 0.568]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 2.0, 0.526]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 2.0, 0.633]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 2.0, 0.733]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0, 0.698]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0, 0.545]),
        ([4.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 0.512]),
        ([4.0, 0.4, 2650, 3000, 14.0, 2.0, 2.0, 0.782]),
    ),
)
def test_Dn50_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    Hs,
    S,
    Dn50_expected,
):
    Dn50_calculated = vandermeer2021.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, S, M50, Hs_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.62]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.00]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 2.76]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 2.96]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 2.39]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 1.98]),
        ([3.0, 0.4, 2650, 3000, 6.0, 3.0, 1000.0, 2.92]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 2.84]),
        ([4.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 3.17]),
        ([4.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 1.80]),
    ),
)
def test_Hs_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    S,
    M50,
    Hs_expected,
):
    Hs_calculated = vandermeer2021.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        S=S,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, S"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0]),
        ([3.6, 0.4, 2650, 3000, 14.0, 2.0, 2.0]),
    ),
)
def test_internal_consistency_S_Dn50(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    Hs,
    S,
):
    Dn50_calculated = vandermeer2021.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
    )

    S_calculated = vandermeer2021.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
    )

    assert S_calculated == pytest.approx(S, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, S"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0]),
        ([3.6, 0.4, 2650, 3000, 14.0, 2.0, 2.0]),
    ),
)
def test_internal_consistency_Hs_Dn50(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    Hs,
    S,
):
    Dn50_calculated = vandermeer2021.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
    )

    Hs_calculated = vandermeer2021.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        S=S,
        P=P,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
    )

    assert Hs_calculated == pytest.approx(Hs, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, M50"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0]),
        ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 1000.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0]),
        ([3.6, 0.4, 2650, 3000, 14.0, 2.0, 1000.0]),
    ),
)
def test_internal_consistency_S_Hs(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tmm10,
    Hs,
    M50,
):
    S_calculated = vandermeer2021.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    Hs_calculated = vandermeer2021.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        S=S_calculated,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    assert Hs_calculated == pytest.approx(Hs, abs=1e-2)
