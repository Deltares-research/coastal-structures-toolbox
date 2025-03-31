import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as vandermeer1988_modified


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tmm10, Hs, M50, S_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 1.08]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.98]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 0.885]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 0.683]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 1.53]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 3.64]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 1000.0, 2.50]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 0.798]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 3.32]),
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
    S_calculated = vandermeer1988_modified.calculate_damage_number_S(
        Hs=Hs,
        H2p=1.4 * Hs,
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
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 0.639]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 0.783]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 2.0, 0.614]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 2.0, 0.569]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 2.0, 0.685]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 2.0, 0.815]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0, 0.755]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0, 0.589]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 2.0, 0.800]),
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
    Dn50_calculated = vandermeer1988_modified.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        H2p=1.4 * Hs,
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
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.36]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 1.80]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 2.49]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 2.66]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 2.15]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 1.81]),
        ([3.0, 0.4, 2650, 3000, 6.0, 3.0, 1000.0, 2.63]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 2.56]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 1.84]),
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
    Hs_calculated = vandermeer1988_modified.calculate_significant_wave_height_Hs(
        ratio_H2p_Hs=1.4,
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
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 2.0]),
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
    Dn50_calculated = vandermeer1988_modified.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        H2p=1.4 * Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
    )

    S_calculated = vandermeer1988_modified.calculate_damage_number_S(
        Hs=Hs,
        H2p=1.4 * Hs,
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
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 2.0]),
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
    Dn50_calculated = vandermeer1988_modified.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        H2p=1.4 * Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
    )

    Hs_calculated = vandermeer1988_modified.calculate_significant_wave_height_Hs(
        ratio_H2p_Hs=1.4,
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
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 1000.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 1000.0]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0]),
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 1000.0]),
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
    S_calculated = vandermeer1988_modified.calculate_damage_number_S(
        Hs=Hs,
        H2p=1.4 * Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    Hs_calculated = vandermeer1988_modified.calculate_significant_wave_height_Hs(
        ratio_H2p_Hs=1.4,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        S=S_calculated,
        P=P,
        N_waves=N_waves,
        M50=M50,
    )

    assert Hs_calculated == pytest.approx(Hs, abs=1e-2)
