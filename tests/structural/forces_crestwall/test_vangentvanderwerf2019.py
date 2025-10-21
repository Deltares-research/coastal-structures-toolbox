import numpy as np
import pytest

import deltares_coastal_structures_toolbox.functions.structural.forces_crestwall.vangentvanderwerf2019 as vangentvanderwerf2019


@pytest.mark.parametrize(
    ("z2p, Ac, beta, gamma_beta_FH_expected"),
    (
        ([0.228712, 0.097, 0, 1.0]),
        ([0.228712, 0.097, 45, 0.5659]),
        ([0.30, 0.097, 45, 0.6305]),
        ([0.228712, 0.12, 45, 0.4740]),
        ([0.228712, 0.097, 60, 0.3488]),
    ),
)
def test_gamma_beta_horizontal_force_backward(z2p, Ac, beta, gamma_beta_FH_expected):

    gamma_beta_FH_calculated = vangentvanderwerf2019.calculate_influence_oblique_waves_horizontal_force_gamma_FH_beta(
        z2p, Ac, beta
    )

    assert gamma_beta_FH_calculated == pytest.approx(gamma_beta_FH_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("z2p, Ac, beta, gamma_beta_FV_expected"),
    (
        ([0.228712, 0.097, 0, 1.0]),
        ([0.228712, 0.097, 45, 0.6334]),
        ([0.30, 0.097, 45, 0.6700]),
        ([0.228712, 0.12, 45, 0.5878]),
        ([0.228712, 0.097, 60, 0.4501]),
    ),
)
def test_gamma_beta_vertical_force_backward(z2p, Ac, beta, gamma_beta_FV_expected):

    gamma_beta_FV_calculated = vangentvanderwerf2019.calculate_influence_oblique_waves_vertical_force_gamma_FV_beta(
        z2p, Ac, beta
    )

    assert gamma_beta_FV_calculated == pytest.approx(gamma_beta_FV_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("beta, c_beta, gamma_beta_expected"),
    (
        ([0, 0.5, 1.0]),
        ([15, 0.5, 0.9665]),
        ([30, 0.5, 0.875]),
        ([45, 0.5, 0.75]),
        ([60, 0.5, 0.625]),
        ([90, 0.5, 0.5]),
    ),
)
def test_gamma_beta_backward(beta, c_beta, gamma_beta_expected):

    gamma_beta_calculated = (
        vangentvanderwerf2019.calculate_influence_oblique_waves_gamma_beta(beta, c_beta)
    )

    assert gamma_beta_calculated == pytest.approx(gamma_beta_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, cot_alpha, Ac, Rc, Hwall, cFH, FH2p_expected"),
    (
        (0.1589, 2.5, 3, 0.097, 0.150, 0.200, 1.0, 286.6440),
        (0.1589, 3.0, 3, 0.097, 0.150, 0.200, 1.0, 326.1915),
        (0.1589, 2.5, 3, 0.080, 0.150, 0.200, 1.0, 320.8319),
        (0.14, 3.0, 3, 0.097, 0.150, 0.200, 1.0, 274.8799),
        (0.1589, 2.5, 4, 0.097, 0.150, 0.200, 1.0, 207.5491),
        (0.1589, 2.5, 3, 0.097, 0.200, 0.200, 1.0, 286.6440),
        (0.1589, 2.5, 3, 0.097, 0.150, 0.250, 1.0, 358.3050),
    ),
)
def test_FH2p_perpendicular(Hm0, Tmm10, cot_alpha, Ac, Rc, Hwall, cFH, FH2p_expected):

    FH2p_calculated = vangentvanderwerf2019.calculate_FH2p_perpendicular(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        Ac=Ac,
        Rc=Rc,
        Hwall=Hwall,
        cFH=cFH,
    )

    assert FH2p_calculated == pytest.approx(FH2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, z2p, rho_water, Ac, Rc, Hwall, g, cFH, FH2p_expected"),
    (
        (0.1589, 0.228712, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 258.419),
        (0.1589, 0.250, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 300.186),
        (0.1589, 0.228712, 1000, 0.080, 0.150, 0.200, 9.81, 1.0, 291.773),
        (0.14, 0.250, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 300.186),
        (0.1589, 0.228712, 1025, 0.097, 0.150, 0.200, 9.81, 1.0, 264.879),
        (0.1589, 0.228712, 1000, 0.097, 0.200, 0.200, 9.81, 1.0, 258.419),
        (0.1589, 0.228712, 1000, 0.097, 0.150, 0.250, 9.81, 1.0, 323.024),
    ),
)
def test_FH2p_perpendicular_from_z2p(
    Hm0, z2p, rho_water, Ac, Rc, Hwall, g, cFH, FH2p_expected
):

    FH2p_calculated = vangentvanderwerf2019.calculate_FH2p_perpendicular_from_z2p(
        Hm0=Hm0,
        z2p=z2p,
        rho_water=rho_water,
        Ac=Ac,
        Rc=Rc,
        Hwall=Hwall,
        g=g,
        cFH=cFH,
    )

    assert FH2p_calculated == pytest.approx(FH2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, beta, cot_alpha, Ac, Rc, Hwall, cFH, FH2p_expected"),
    (
        (0.1589, 2.5, 0.0, 3, 0.097, 0.150, 0.200, 1.0, 286.6440),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.150, 0.200, 1.0, 226.4295),
        (0.1589, 2.5, 45.0, 3, 0.097, 0.150, 0.200, 1.0, 166.2150),
        (0.1589, 3.0, 30.0, 3, 0.097, 0.150, 0.200, 1.0, 261.0336),
        (0.1589, 2.5, 30.0, 3, 0.080, 0.150, 0.200, 1.0, 260.6174),
        (0.14, 3.0, 30.0, 3, 0.097, 0.150, 0.200, 1.0, 216.1359),
        (0.1589, 2.5, 30.0, 4, 0.097, 0.150, 0.200, 1.0, 157.2215),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.200, 0.200, 1.0, 226.4295),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.150, 0.250, 1.0, 283.0369),
    ),
)
def test_FH2p_oblique(Hm0, Tmm10, beta, cot_alpha, Ac, Rc, Hwall, cFH, FH2p_expected):

    FH2p_calculated = vangentvanderwerf2019.calculate_FH2p_oblique(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        Ac=Ac,
        Rc=Rc,
        Hwall=Hwall,
        cFH=cFH,
    )

    assert FH2p_calculated == pytest.approx(FH2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, cot_alpha, Ac, Fb, Bwall, cFV, FV2p_expected"),
    (
        (0.1589, 2.5, 3, 0.097, 0.020, 0.030, 0.4, 10.9866),
        (0.1589, 3.0, 3, 0.097, 0.020, 0.030, 0.4, 12.2819),
        (0.1589, 2.5, 3, 0.080, 0.020, 0.030, 0.4, 10.8316),
        (0.14, 3.0, 3, 0.097, 0.020, 0.030, 0.4, 10.6012),
        (0.1589, 2.5, 4, 0.097, 0.020, 0.030, 0.4, 8.3958),
        (0.1589, 2.5, 3, 0.097, 0.030, 0.030, 0.4, 8.9328),
        (0.1589, 2.5, 3, 0.097, 0.020, 0.020, 0.4, 7.3244),
        (0.1589, 2.5, 3, 0.097, 0.020, 0.030, 0.5, 13.7332),
    ),
)
def test_FV2p_perpendicular(Hm0, Tmm10, cot_alpha, Ac, Fb, Bwall, cFV, FV2p_expected):

    FV2p_calculated = vangentvanderwerf2019.calculate_FV2p_perpendicular(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        Ac=Ac,
        Bwall=Bwall,
        Fb=Fb,
        cFV=cFV,
    )

    assert FV2p_calculated == pytest.approx(FV2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("z2p, rho_water, Ac, Fb, Bwall, cFV, FV2p_expected"),
    (
        (0.228712, 1000, 0.097, 0.020, 0.030, 0.4, 10.0231),
        (0.250, 1000, 0.097, 0.020, 0.030, 0.4, 11.3912),
        (0.228712, 1000, 0.080, 0.020, 0.030, 0.4, 9.9304),
        (0.228712, 1025, 0.097, 0.020, 0.030, 0.4, 10.2736),
        (0.228712, 1000, 0.097, 0.030, 0.030, 0.4, 8.1494),
        (0.228712, 1000, 0.097, 0.020, 0.020, 0.4, 6.6820),
        (0.228712, 1000, 0.097, 0.020, 0.030, 0.5, 12.5288),
    ),
)
def test_FV2p_perpendicular_from_z2p(z2p, rho_water, Ac, Fb, Bwall, cFV, FV2p_expected):

    FV2p_calculated = vangentvanderwerf2019.calculate_FV2p_perpendicular_from_z2p(
        z2p=z2p,
        rho_water=rho_water,
        Ac=Ac,
        Fb=Fb,
        Bwall=Bwall,
        cFV=cFV,
    )

    assert FV2p_calculated == pytest.approx(FV2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, beta, cot_alpha, Ac, Fb, Bwall, cFV, FV2p_expected"),
    (
        (0.1589, 2.5, 0.0, 3, 0.097, 0.020, 0.030, 0.4, 10.9866),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.020, 0.030, 0.4, 9.0142),
        (0.1589, 2.5, 45.0, 3, 0.097, 0.020, 0.030, 0.4, 7.0419),
        (0.1589, 3.0, 30.0, 3, 0.097, 0.020, 0.030, 0.4, 10.1477),
        (0.1589, 2.5, 30.0, 3, 0.080, 0.020, 0.030, 0.4, 9.0252),
        (0.14, 3.0, 30.0, 3, 0.097, 0.020, 0.030, 0.4, 8.6770),
        (0.1589, 2.5, 30.0, 4, 0.097, 0.020, 0.030, 0.4, 6.7473),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.030, 0.030, 0.4, 7.3291),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.020, 0.020, 0.4, 6.0095),
        (0.1589, 2.5, 30.0, 3, 0.097, 0.020, 0.030, 0.5, 11.2678),
    ),
)
def test_FV2p_oblique(Hm0, Tmm10, beta, cot_alpha, Ac, Fb, Bwall, cFV, FV2p_expected):

    FV2p_calculated = vangentvanderwerf2019.calculate_FV2p_oblique(
        Hm0=Hm0,
        Tmm10=Tmm10,
        beta=beta,
        cot_alpha=cot_alpha,
        Ac=Ac,
        Fb=Fb,
        Bwall=Bwall,
        cFV=cFV,
    )

    assert FV2p_calculated == pytest.approx(FV2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("FH2p, FH01p_expected"),
    (
        (200.0, 320.0),
        (250.0, 400.0),
    ),
)
def test_FH01p_perpendicular(FH2p, FH01p_expected):

    FH01p_calculated = vangentvanderwerf2019.calculate_FH01p_perpendicular(FH2p=FH2p)

    assert FH01p_calculated == pytest.approx(FH01p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("FV2p, s0p, FV01p_expected"),
    (
        (10.0, 0.04, 16.0000),
        (12.0, 0.04, 19.2000),
        (10.0, 0.02, 22.4000),
    ),
)
def test_FV01p_perpendicular(FV2p, s0p, FV01p_expected):

    FV01p_calculated = vangentvanderwerf2019.calculate_FV01p_perpendicular(
        FV2p=FV2p, s0p=s0p
    )

    assert FV01p_calculated == pytest.approx(FV01p_expected, abs=1e-3)
