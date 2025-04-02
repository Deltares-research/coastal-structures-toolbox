import pytest

import deltares_coastal_structures_toolbox.functions.structural.forces_caisson.goda1985 as caisson


@pytest.mark.parametrize(
    ("T, h, g, L_expected"),
    (
        ([2.0, 0.2, 9.8, 2.71]),  # values from goda (2000) tables
        ([2.0, 1.0, 9.8, 5.21]),
        ([2.0, 2.0, 9.8, 6.05]),
        ([2.0, 11.0, 9.8, 6.24]),
        ([5.0, 6.0, 9.8, 32.17]),
    ),
)
def test_L_backward(T, h, g, L_expected):

    L_calculated = caisson.calculate_local_wavelength(T=T, h=h, g=g)

    assert L_calculated == pytest.approx(L_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "HD, Hsi, Tmax, beta, h_s, d, cota_seabed, hacc, Rc, B_up, B1, rho_water, FH_expected"
    ),
    (
        ([5.07, 2.75, 8.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 466.22]),
        ([3.60, 2.0, 8.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 288.01]),
        ([5.07, 2.75, 12.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 525.38]),
        ([7.88, 4.35, 8.0, 0.0, 10.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 1177.2]),  # C2
        ([5.8, 3.05, 8.0, 0.0, 5.0, 4.0, 50, 6.0, 5.0, 10.0, 2, 1025, 574.18]),  # C1
    ),
)
def test_FH_backward(
    HD, Hsi, Tmax, beta, h_s, d, cota_seabed, hacc, Rc, B_up, B1, rho_water, FH_expected
):

    output = caisson.calculate_forces_and_reactions(
        HD=HD,
        Hsi=Hsi,
        Tmax=Tmax,
        beta=beta,
        h_s=h_s,
        d=d,
        cota_seabed=cota_seabed,
        hacc=hacc,
        Rc=Rc,
        Bup=B_up,
        B1=B1,
        rho_water=rho_water,
        g=9.81,
        return_dict=True,
    )

    if isinstance(output, dict):
        FH_calculated = output["FH"]
    else:
        FH_calculated = output[0]

    assert FH_calculated / 1000 == pytest.approx(FH_expected, abs=1e-1)


@pytest.mark.parametrize(
    (
        "HD, Hsi, Tmax, beta, h_s, d, cota_seabed, hacc, Rc, B_up, B1, rho_water, FH_expected"
    ),
    (
        ([5.07, 2.75, 8.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 52.9407]),
        ([3.60, 2.0, 8.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 35.403]),
        ([5.07, 2.75, 12.0, 0.0, 5.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 57.6835]),
        ([7.88, 4.35, 8.0, 0.0, 10.0, 4.0, 100, 6.0, 5.0, 10.0, 2, 1025, 125.089]),
    ),
)
def test_p1_backward(
    HD, Hsi, Tmax, beta, h_s, d, cota_seabed, hacc, Rc, B_up, B1, rho_water, FH_expected
):

    FH_calculated_out = caisson.calculate_forces_and_reactions(
        HD=HD,
        Hsi=Hsi,
        Tmax=Tmax,
        beta=beta,
        h_s=h_s,
        d=d,
        cota_seabed=cota_seabed,
        hacc=hacc,
        Rc=Rc,
        Bup=B_up,
        B1=B1,
        rho_water=rho_water,
        g=9.81,
        return_dict=True,
    )

    if isinstance(FH_calculated_out, dict):
        FH_calculated = FH_calculated_out["p1"]
    else:
        FH_calculated = FH_calculated_out[0]

    assert FH_calculated / 1000 == pytest.approx(FH_expected, abs=1e-1)
