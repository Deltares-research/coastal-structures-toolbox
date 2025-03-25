import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as vandermeer1988


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tm, Hs, M50, c_pl, S_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 0.918]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 2.53]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 0.751]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 8.68, 0.580]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 8.68, 1.30]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 8.68, 2.51]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 1000.0, 8.68, 2.12]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 8.68, 0.677]),
        (
            [3.6, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 0.582]
        ),  # TODO s3p5 check implementation
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 8.68, 3.29]),  # TODO s3p5 failing
    ),
)
def test_S_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tm,
    Hs,
    M50,
    c_pl,
    S_expected,
):
    S_calculated = vandermeer1988.calculate_damage_number_S(
        Hs=Hs,
        H2p=1.4 * Hs,
        Tm=Tm,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        M50=M50,
        c_pl=c_pl,
    )

    assert S_calculated == pytest.approx(S_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tm, Hs, S, c_pl, Dn50_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 8.68, 0.618]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 8.68, 0.757]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 2.0, 8.68, 0.594]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 2.0, 8.68, 0.551]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 2.0, 8.68, 0.663]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 2.0, 8.68, 0.756]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.5, 2.0, 8.68, 0.731]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 3.0, 8.68, 0.570]),
        (
            [3.6, 0.4, 2650, 3000, 6.0, 2.0, 2.0, 8.68, 0.565]
        ),  # TODO s3p5 check implementation
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 2.0, 8.68, 0.798]),  # TODO s3p5 failing
    ),
)
def test_Dn50_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tm,
    Hs,
    S,
    c_pl,
    Dn50_expected,
):
    Dn50_calculated = vandermeer1988.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        H2p=1.4 * Hs,
        Tm=Tm,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        P=P,
        N_waves=N_waves,
        S=S,
        c_pl=c_pl,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, P, rho_armour, N_waves, Tm, S, M50, c_pl, Hs_expected"),
    (
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 2.46]),
        ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 1.88]),
        ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 2.60]),
        ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 8.68, 2.78]),
        ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 8.68, 2.24]),
        ([3.0, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 8.68, 1.93]),
        ([3.0, 0.4, 2650, 3000, 6.0, 3.0, 1000.0, 8.68, 2.74]),
        ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 8.68, 2.67]),
        (
            [3.6, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 8.68, 2.78]
        ),  # TODO s3p5 check implementation
        ([3.6, 0.4, 2650, 3000, 12.0, 2.0, 1000.0, 8.68, 1.75]),  # TODO s3p5 failing
    ),
)
def test_Hs_backward(
    cot_alpha,
    P,
    rho_armour,
    N_waves,
    Tm,
    S,
    M50,
    c_pl,
    Hs_expected,
):
    Hs_calculated = vandermeer1988.calculate_significant_wave_height_Hs(
        ratio_H2p_Hs=1.4,
        Tm=Tm,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        S=S,
        P=P,
        N_waves=N_waves,
        M50=M50,
        c_pl=c_pl,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
