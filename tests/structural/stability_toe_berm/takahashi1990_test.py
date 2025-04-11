import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.structural.stability_toe_berm.takahashi1990_caisson as takahashi


@pytest.mark.parametrize(
    ("Hm0, Tp, hacc, Bt, rho_rock, rho_water, beta, Dn50_expected"),
    (
        ([2.0, 8.0, 6.0, 2.0, 2650, 1025, 0, 0.111]),
        ([4.0, 8.0, 6.0, 2.0, 2650, 1025, 0, 0.442]),
        ([4.0, 12.0, 6.0, 2.0, 2650, 1025, 0, 0.342]),
        ([4.0, 12.0, 8.0, 2.0, 2650, 1025, 0, 0.230]),
        ([4.0, 12.0, 4.0, 2.0, 2650, 1025, 0, 0.593]),
        ([4.0, 12.0, 6.0, 4.0, 2650, 1025, 0, 0.566]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1025, 0, 0.669]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 0, 0.641]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 30, 0.731]),
    ),
)
def test_Dn50_backward(Hm0, Tp, hacc, Bt, rho_rock, rho_water, beta, Dn50_expected):

    Dn50_calculated = takahashi.calculate_nominal_diameter_Dn50(
        Hs=Hm0,
        Tp=Tp,
        hacc=hacc,
        Bt=Bt,
        beta=beta,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hs_expected, Tp, hacc, Bt, rho_rock, rho_water, beta, Dn50"),
    (
        ([2.0, 8.0, 6.0, 2.0, 2650, 1025, 0, 0.111]),
        ([4.0, 8.0, 6.0, 2.0, 2650, 1025, 0, 0.442]),
        ([4.0, 12.0, 6.0, 2.0, 2650, 1025, 0, 0.342]),
        ([4.0, 12.0, 8.0, 2.0, 2650, 1025, 0, 0.230]),
        ([4.0, 12.0, 4.0, 2.0, 2650, 1025, 0, 0.593]),
        ([4.0, 12.0, 6.0, 4.0, 2650, 1025, 0, 0.566]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1025, 0, 0.669]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 0, 0.641]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 30, 0.731]),
        ([0.2, 2.0, 0.6, 0.1, 2650, 1000, 0, 0.00742]),
    ),
)
def test_Hs_backward(Hs_expected, Tp, hacc, Bt, rho_rock, rho_water, beta, Dn50):

    Hs_calculated = takahashi.calculate_significant_wave_height_Hs(
        Dn50=Dn50,
        Tp=Tp,
        hacc=hacc,
        Bt=Bt,
        beta=beta,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hs, Tp, hacc_expected, Bt, rho_rock, rho_water, beta, Dn50"),
    (
        ([2.0, 8.0, 5.98, 2.0, 2650, 1025, 0, 0.111]),
        ([4.0, 8.0, 6.0, 2.0, 2650, 1025, 0, 0.442]),
        ([4.0, 12.0, 6.0, 2.0, 2650, 1025, 0, 0.342]),
        ([4.0, 12.0, 8.0, 2.0, 2650, 1025, 0, 0.230]),
        ([4.0, 12.0, 4.0, 2.0, 2650, 1025, 0, 0.593]),
        ([4.0, 12.0, 6.0, 4.0, 2650, 1025, 0, 0.566]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1025, 0, 0.669]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 0, 0.641]),
        ([4.0, 12.0, 6.0, 4.0, 2400, 1000, 30, 0.731]),
        ([0.2, 2.0, 0.6, 0.1, 2650, 1000, 0, 0.00742]),
    ),
)
def test_hacc_backward(Hs, Tp, hacc_expected, Bt, rho_rock, rho_water, beta, Dn50):

    hacc_calculated = takahashi.calculate_significant_depth_above_toe_hacc(
        Dn50=Dn50,
        Hs=Hs,
        Tp=Tp,
        Bt=Bt,
        beta=beta,
        rho_rock=rho_rock,
        rho_water=rho_water,
    )

    assert hacc_calculated == pytest.approx(hacc_expected, abs=1e-2)
