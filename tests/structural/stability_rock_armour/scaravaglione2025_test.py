import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.scaravaglione2025 as scaravaglione2025


@pytest.mark.parametrize(
    ("cot_alpha, rho_armour, Dn50_core, N_waves, Tmm10, Hm0, M50, S_expected"),
    (
        ([2.0, 2650, 0.1, 3000, 6.0, 2.0, 500.0, 3.02]),
        ([2.0, 2850, 0.1, 3000, 6.0, 2.0, 500.0, 1.88]),
        ([2.0, 2650, 0.075, 3000, 6.0, 2.0, 500.0, 3.65]),
        ([2.0, 2650, 0.1, 6000, 6.0, 2.0, 500.0, 4.28]),
        ([2.0, 2650, 0.1, 3000, 14.0, 2.0, 500.0, 7.05]),
        ([2.0, 2650, 0.1, 3000, 6.0, 2.5, 500.0, 8.25]),
        ([2.0, 2650, 0.1, 3000, 6.0, 2.0, 600.0, 2.33]),
        ([4.0, 2650, 0.1, 3000, 6.0, 2.0, 500.0, 0.534]),
        ([4.0, 2650, 0.1, 3000, 14.0, 2.0, 500.0, 1.25]),
    ),
)
def test_S_backward(
    cot_alpha,
    rho_armour,
    Dn50_core,
    N_waves,
    Tmm10,
    Hm0,
    M50,
    S_expected,
):
    S_calculated = scaravaglione2025.calculate_damage_number_S(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        N_waves=N_waves,
        M50=M50,
        Dn50_core=Dn50_core,
    )

    assert S_calculated == pytest.approx(S_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, rho_armour, N_waves, Tmm10, Hm0, S, Dn50_core, Dn50_expected"),
    (
        ([2.0, 2650, 3000, 6.0, 2.0, 2.0, 0.2, 0.532]),
        ([2.0, 2850, 3000, 6.0, 2.0, 2.0, 0.2, 0.451]),
        ([2.0, 2650, 6000, 6.0, 2.0, 2.0, 0.2, 0.584]),
        ([2.0, 2650, 3000, 14.0, 2.0, 2.0, 0.2, 0.667]),
        ([2.0, 2650, 3000, 6.0, 2.5, 2.0, 0.2, 0.694]),
        ([2.0, 2650, 3000, 6.0, 2.0, 3.0, 0.2, 0.475]),
        ([2.0, 2650, 3000, 6.0, 2.0, 2.0, 0.15, 0.582]),
        ([4.0, 2650, 3000, 6.0, 2.0, 2.0, 0.2, 0.317]),
        ([4.0, 2650, 3000, 14.0, 2.0, 2.0, 0.2, 0.413]),
    ),
)
def test_Dn50_backward(
    cot_alpha,
    rho_armour,
    N_waves,
    Tmm10,
    Hm0,
    S,
    Dn50_core,
    Dn50_expected,
):
    Dn50_calculated = scaravaglione2025.calculate_nominal_rock_diameter_Dn50(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        N_waves=N_waves,
        S=S,
        Dn50_core=Dn50_core,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-2)


# @pytest.mark.parametrize(
#     ("cot_alpha, P, rho_armour, N_waves, Tmm10, S, M50, Hs_expected"),
#     (
#         ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.36]),
#         ([2.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 1.80]),
#         ([3.0, 0.5, 2650, 3000, 6.0, 2.0, 1000.0, 2.49]),
#         ([3.0, 0.4, 2850, 3000, 6.0, 2.0, 1000.0, 2.66]),
#         ([3.0, 0.4, 2650, 6000, 6.0, 2.0, 1000.0, 2.15]),
#         ([3.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 1.91]),
#         ([3.0, 0.4, 2650, 3000, 6.0, 3.0, 1000.0, 2.63]),
#         ([3.0, 0.4, 2650, 3000, 6.0, 2.0, 1200.0, 2.56]),
#         ([4.0, 0.4, 2650, 3000, 6.0, 2.0, 1000.0, 2.85]),
#         ([4.0, 0.4, 2650, 3000, 14.0, 2.0, 1000.0, 1.62]),
#     ),
# )
# def test_Hs_backward(
#     cot_alpha,
#     P,
#     rho_armour,
#     N_waves,
#     Tmm10,
#     S,
#     M50,
#     Hs_expected,
# ):
#     Hs_calculated = vandermeer1988_modified.calculate_significant_wave_height_Hs(
#         ratio_H2p_Hs=1.4,
#         Tmm10=Tmm10,
#         cot_alpha=cot_alpha,
#         rho_armour=rho_armour,
#         S=S,
#         P=P,
#         N_waves=N_waves,
#         M50=M50,
#     )

#     assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, rho_armour, N_waves, Tmm10, Hm0, S, Dn50_core"),
    (
        ([3.0, 2650, 3000, 6.0, 2.0, 2.0, 0.2]),
        ([2.0, 2650, 3000, 6.0, 2.0, 2.0, 0.2]),
        ([3.0, 2650, 3000, 6.0, 2.0, 2.0, 0.2]),
        ([3.0, 2850, 3000, 6.0, 2.0, 2.0, 0.2]),
        ([3.0, 2650, 6000, 6.0, 2.0, 2.0, 0.2]),
        ([3.0, 2650, 3000, 14.0, 2.0, 2.0, 0.2]),
        ([3.0, 2650, 3000, 6.0, 2.5, 2.0, 0.2]),
        ([3.0, 2650, 3000, 6.0, 2.0, 3.0, 0.2]),
        ([3.6, 2650, 3000, 14.0, 2.0, 2.0, 0.2]),
        ([3.6, 2650, 3000, 14.0, 2.0, 2.0, 0.15]),
    ),
)
def test_internal_consistency_S_Dn50(
    cot_alpha,
    rho_armour,
    N_waves,
    Tmm10,
    Hm0,
    S,
    Dn50_core,
):
    Dn50_calculated = scaravaglione2025.calculate_nominal_rock_diameter_Dn50(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        N_waves=N_waves,
        S=S,
        Dn50_core=Dn50_core,
    )

    S_calculated = scaravaglione2025.calculate_damage_number_S(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
        Dn50_core=Dn50_core,
    )

    assert S_calculated == pytest.approx(S, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, Dn50_core, rho_armour, N_waves, Tmm10, Hm0, S"),
    (
        ([2.0, 0.2, 2650, 3000, 6.0, 2.0, 2.0]),
        ([3.0, 0.2, 2650, 3000, 6.0, 2.0, 2.0]),
        ([2.0, 0.15, 2650, 3000, 6.0, 2.0, 2.0]),
        ([2.0, 0.2, 2850, 3000, 6.0, 2.0, 2.0]),
        ([2.0, 0.2, 2650, 6000, 6.0, 2.0, 2.0]),
        ([2.0, 0.2, 2650, 3000, 14.0, 2.0, 2.0]),
        ([2.0, 0.2, 2650, 3000, 6.0, 2.5, 2.0]),
        ([2.0, 0.2, 2650, 3000, 6.0, 2.0, 3.0]),
    ),
)
def test_internal_consistency_Hm0_Dn50(
    cot_alpha,
    Dn50_core,
    rho_armour,
    N_waves,
    Tmm10,
    Hm0,
    S,
):
    Dn50_calculated = scaravaglione2025.calculate_nominal_rock_diameter_Dn50(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        Dn50_core=Dn50_core,
        N_waves=N_waves,
        S=S,
    )

    Hm0_calculated = scaravaglione2025.calculate_significant_wave_height_Hm0(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        Dn50_core=Dn50_core,
        S=S,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
    )

    assert Hm0_calculated == pytest.approx(Hm0, abs=1e-2)


@pytest.mark.parametrize(
    ("cot_alpha, Dn50_core, rho_armour, N_waves, Tmm10, Hm0, M50"),
    (
        ([2.0, 0.2, 2650, 3000, 6.0, 2.0, 150.0]),
        ([3.0, 0.2, 2650, 3000, 6.0, 2.0, 150.0]),
        ([2.0, 0.15, 2650, 3000, 6.0, 2.0, 150.0]),
        ([2.0, 0.2, 2850, 3000, 6.0, 2.0, 150.0]),
        ([2.0, 0.2, 2650, 6000, 6.0, 2.0, 150.0]),
        ([2.0, 0.2, 2650, 3000, 14.0, 2.0, 150.0]),
        ([2.0, 0.2, 2650, 3000, 6.0, 2.5, 150.0]),
        ([2.0, 0.2, 2650, 3000, 6.0, 2.0, 100.0]),
    ),
)
def test_internal_consistency_S_Hm0(
    cot_alpha,
    Dn50_core,
    rho_armour,
    N_waves,
    Tmm10,
    Hm0,
    M50,
):
    S_calculated = scaravaglione2025.calculate_damage_number_S(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        Dn50_core=Dn50_core,
        N_waves=N_waves,
        M50=M50,
    )

    Hm0_calculated = scaravaglione2025.calculate_significant_wave_height_Hm0(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_armour,
        Dn50_core=Dn50_core,
        S=S_calculated,
        N_waves=N_waves,
        M50=M50,
    )

    assert Hm0_calculated == pytest.approx(Hm0, abs=1e-2)
