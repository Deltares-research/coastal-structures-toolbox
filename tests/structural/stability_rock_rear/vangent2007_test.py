import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_rear.vangent2007 as vangent2007


@pytest.mark.parametrize(
    ("cot_alpha, cot_phi, gamma, Dn50, Hs, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves"),
    (
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([2.5, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 3.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.55, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.60, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 11.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 4.8, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.45, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.1, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 2000]),
    ),
)
def test_internal_consistency_S_Dn50(
    cot_alpha,
    cot_phi,
    gamma,
    Dn50,
    Hs,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
):
    S_calculated = vangent2007.calculate_damage_number_S(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        Dn50=Dn50,
        Hs=Hs,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    Dn50_calculated = vangent2007.calculate_nominal_rock_diameter_Dn50(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        S=S_calculated,
        Hs=Hs,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert Dn50_calculated == pytest.approx(Dn50, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, cot_phi, gamma, M50, S, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves"),
    (
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([2.5, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 3.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.55, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 700, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 800, 9.0, 11.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 4.8, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.45, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.1, 1000]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 2000]),
    ),
)
def test_internal_consistency_S_Hs(
    cot_alpha,
    cot_phi,
    gamma,
    M50,
    S,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
):
    Hs_calculated = vangent2007.calculate_maximum_significant_wave_height_Hs(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        M50=M50,
        S=S,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    S_calculated = vangent2007.calculate_damage_number_S(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        M50=M50,
        Hs=Hs_calculated,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert S_calculated == pytest.approx(S, abs=1e-1)


@pytest.mark.parametrize(
    ("cot_alpha, cot_phi, gamma, Dn50, S, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves"),
    (
        ([3.0, 2.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([2.5, 2.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 3.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.55, 0.65, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.60, 9.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 7.0, 12.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 9.0, 11.0, 5.0, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 9.0, 12.0, 4.8, 0.5, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.45, 1.0, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.5, 1.1, 1000]),
        ([3.0, 2.0, 0.47, 0.65, 9.0, 12.0, 5.0, 0.5, 1.0, 2000]),
    ),
)
def test_internal_consistency_Dn50_Hs(
    cot_alpha,
    cot_phi,
    gamma,
    Dn50,
    S,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
):
    Hs_calculated = vangent2007.calculate_maximum_significant_wave_height_Hs(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        Dn50=Dn50,
        S=S,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    Dn50_calculated = vangent2007.calculate_nominal_rock_diameter_Dn50(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        S=S,
        Hs=Hs_calculated,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert Dn50_calculated == pytest.approx(Dn50, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, cot_phi, gamma, M50, Hs, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves, S_expected"
    ),
    (
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 3.77]),
        ([2.5, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 6.65]),
        ([3.0, 3.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 6.26]),
        ([3.0, 2.0, 0.55, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 7.23]),
        ([3.0, 2.0, 0.47, 700, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 4.22]),
        ([3.0, 2.0, 0.47, 800, 6.0, 12.0, 5.0, 0.5, 1.0, 1000, 2.4]),
        ([3.0, 2.0, 0.47, 800, 7.0, 11.0, 5.0, 0.5, 1.0, 1000, 2.55]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 4.8, 0.5, 1.0, 1000, 4.20]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.45, 1.0, 1000, 3.80]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.1, 1000, 4.04]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 2000, 5.34]),
    ),
)
def test_S_backward(
    cot_alpha,
    cot_phi,
    gamma,
    M50,
    Hs,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
    S_expected,
):
    S_calculated = vangent2007.calculate_damage_number_S(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        M50=M50,
        Hs=Hs,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert S_calculated == pytest.approx(S_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, cot_phi, gamma, S, Hs, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves, Dn50_expected"
    ),
    (
        ([3.0, 2.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.474]),
        ([2.5, 2.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.595]),
        ([3.0, 3.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.580]),
        ([3.0, 2.0, 0.55, 9.0, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.615]),
        ([3.0, 2.0, 0.47, 7.0, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.524]),
        ([3.0, 2.0, 0.47, 9.0, 6.0, 12.0, 5.0, 0.5, 1.0, 1000, 0.396]),
        ([3.0, 2.0, 0.47, 9.0, 7.0, 11.0, 5.0, 0.5, 1.0, 1000, 0.405]),
        ([3.0, 2.0, 0.47, 9.0, 7.0, 12.0, 4.8, 0.5, 1.0, 1000, 0.494]),
        ([3.0, 2.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.45, 1.0, 1000, 0.475]),
        ([3.0, 2.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.5, 1.1, 1000, 0.487]),
        ([3.0, 2.0, 0.47, 9.0, 7.0, 12.0, 5.0, 0.5, 1.0, 2000, 0.544]),
    ),
)
def test_Dn50_backward(
    cot_alpha,
    cot_phi,
    gamma,
    S,
    Hs,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
    Dn50_expected,
):
    Dn50_calculated = vangent2007.calculate_nominal_rock_diameter_Dn50(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        S=S,
        Hs=Hs,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, cot_phi, gamma, M50, S, Tmm10, Rc, Rc2_front, Rc2_rear, N_waves, Hs_expected"
    ),
    (
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000, 10.54]),
        ([2.5, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000, 7.87]),
        ([3.0, 3.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000, 8.17]),
        ([3.0, 2.0, 0.55, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 1000, 7.77]),
        ([3.0, 2.0, 0.47, 700, 9.0, 12.0, 5.0, 0.5, 1.0, 1000, 9.94]),
        ([3.0, 2.0, 0.47, 800, 7.0, 12.0, 5.0, 0.5, 1.0, 1000, 9.25]),
        ([3.0, 2.0, 0.47, 800, 9.0, 11.0, 5.0, 0.5, 1.0, 1000, 12.5]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 4.8, 0.5, 1.0, 1000, 10.12]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.45, 1.0, 1000, 10.51]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.1, 1000, 10.16]),
        ([3.0, 2.0, 0.47, 800, 9.0, 12.0, 5.0, 0.5, 1.0, 2000, 8.82]),
    ),
)
def test_Hs_backward(
    cot_alpha,
    cot_phi,
    gamma,
    M50,
    S,
    Tmm10,
    Rc,
    Rc2_front,
    Rc2_rear,
    N_waves,
    Hs_expected,
):
    Hs_calculated = vangent2007.calculate_maximum_significant_wave_height_Hs(
        cot_alpha=cot_alpha,
        cot_phi=cot_phi,
        gamma=gamma,
        M50=M50,
        S=S,
        Tmm10=Tmm10,
        Rc=Rc,
        Rc2_front=Rc2_front,
        Rc2_rear=Rc2_rear,
        N_waves=N_waves,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)
