import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.structural.stability_toe_berm.vangent2014 as vangent2014


@pytest.mark.parametrize(
    ("Hs, Tmm10, ht, udelta_expected"),
    (
        ([0.1293584, 2.147444, 0.371, 0.573936394776771]),
        ([0.1638684, 2.329373, 0.371, 0.792463418282433]),
        ([0.1998874, 2.499612, 0.371, 1.04051074924087]),
        ([0.2382805, 2.722213, 0.371, 1.35454111440137]),
        ([0.1240029, 2.126191, 0.271, 0.751532871628882]),
        ([0.1593027, 2.331838, 0.271, 1.06202331762176]),
        ([0.1825543, 2.530658, 0.271, 1.32327561214554]),
        # ([0.12500, 2.22091, 0.20000, 1.262]),
    ),
)
def test_udelta_backward(Hs, Tmm10, ht, udelta_expected):

    udelta_calculated = vangent2014.calculate_velocity_u_delta(
        Hs=Hs,
        Tmm10=Tmm10,
        ht=ht,
    )

    assert udelta_calculated == pytest.approx(udelta_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "Hs, Tmm10, ht, Bt, tt, Dn50, rho_rock, rho_water, cot_alpha_armour_slope, Nod_expected"
    ),
    (
        (
            [
                0.12936,
                2.14744,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.37442,
            ]
        ),
        (
            [
                0.23828,
                2.72221,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.83926,
            ]
        ),
        (
            [
                0.24613,
                2.02284,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.44537,
            ]
        ),
        (
            [
                0.15930,
                2.33184,
                0.27100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.88950,
            ]
        ),
        (
            [
                0.12500,
                2.22091,
                0.17100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.79031,
            ]
        ),
        (
            [
                0.12400,
                2.12619,
                0.24160,
                0.04380,
                0.05840,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.05334,
            ]
        ),
        (
            [
                0.20097,
                1.83494,
                0.24160,
                0.04380,
                0.05840,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                2.61373,
            ]
        ),
    ),
)
def test_Nod_backward(
    Hs,
    Tmm10,
    ht,
    Bt,
    tt,
    Dn50,
    rho_rock,
    rho_water,
    cot_alpha_armour_slope,
    Nod_expected,
):

    Nod_calculated = vangent2014.calculate_damage_Nod(
        Hm0=Hs,
        Tmm10=Tmm10,
        ht=ht,
        Bt=Bt,
        tt=tt,
        Dn50=Dn50,
        rho_rock=rho_rock,
        rho_water=rho_water,
        cot_alpha_armour_slope=cot_alpha_armour_slope,
    )

    assert Nod_calculated == pytest.approx(Nod_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "Hs, Tmm10, ht, Bt, tt, Dn50_expected, rho_rock, rho_water, cot_alpha_armour_slope, Nod"
    ),
    (
        (
            [
                0.12936,
                2.14744,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.37442,
            ]
        ),
        (
            [
                0.23828,
                2.72221,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.83926,
            ]
        ),
        (
            [
                0.24613,
                2.02284,
                0.37100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.44537,
            ]
        ),
        (
            [
                0.15930,
                2.33184,
                0.27100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.88950,
            ]
        ),
        (
            [
                0.12500,
                2.22091,
                0.17100,
                0.04400,
                0.02900,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                0.79031,
            ]
        ),
        (
            [
                0.12400,
                2.12619,
                0.24160,
                0.04380,
                0.05840,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                1.05334,
            ]
        ),
        (
            [
                0.20097,
                1.83494,
                0.24160,
                0.04380,
                0.05840,
                0.01460,
                2700.00000,
                1000.00000,
                2.00000,
                2.61373,
            ]
        ),
    ),
)
def test_Dn50_backward(
    Hs,
    Tmm10,
    ht,
    Bt,
    tt,
    Dn50_expected,
    rho_rock,
    rho_water,
    cot_alpha_armour_slope,
    Nod,
):

    Dn50_calculated = vangent2014.calculate_nominal_diameter_Dn50(
        Hm0=Hs,
        Tmm10=Tmm10,
        ht=ht,
        Bt=Bt,
        tt=tt,
        Nod=Nod,
        rho_rock=rho_rock,
        rho_water=rho_water,
        cot_alpha_armour_slope=cot_alpha_armour_slope,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-3)
