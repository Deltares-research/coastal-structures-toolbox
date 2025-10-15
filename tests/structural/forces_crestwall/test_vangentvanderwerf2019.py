import numpy as np
import pytest

import deltares_coastal_structures_toolbox.functions.structural.forces_crestwall.vangentvanderwerf2019 as vangentvanderwerf2019


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

    gamma_beta_calculated = vangentvanderwerf2019.calculate_gamma_beta(
        np.deg2rad(beta), c_beta
    )

    assert gamma_beta_calculated == pytest.approx(gamma_beta_expected, abs=1e-2)


@pytest.mark.parametrize(
    ("Hm0", "Tmm10", "cot_alpha", "gamma_f", "gamma_beta", "c0", "c1", "z2p_expected"),
    (
        (0.0804, 1.664, 2.0, 0.45, 1, 1.45, 5.0, 0.1384),  # A
        (0.0804, 2.50, 2.0, 0.45, 1, 1.45, 5.0, 0.1526),  # A
        (0.0804, 2.50, 1.5, 0.45, 1, 1.45, 5.0, 0.1597),  # A
        (0.0804, 2.50, 1.5, 0.40, 1, 1.45, 5.0, 0.1419),  # A
        (0.0804, 2.50, 1.5, 0.40, 0.88, 1.45, 5.0, 0.1242),  # A
        (0.0804, 1.664, 2.0, 0.45, 1, 1.55, 5.4, 0.1490),  # A
        (0.120, 2.50, 2.0, 0.45, 1, 1.45, 5.0, 0.2184),  # A
    ),
)
def test_z2p_backward(Hm0, Tmm10, cot_alpha, gamma_f, gamma_beta, c0, c1, z2p_expected):

    z2p_calculated = vangentvanderwerf2019.calculate_z2p(
        Hm0=Hm0,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        gamma_f=gamma_f,
        gamma_beta=gamma_beta,
        c0=c0,
        c1=c1,
    )

    assert z2p_calculated == pytest.approx(z2p_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0", "z2p", "rho_water", "Ac", "Rc", "Hwall", "g", "cFH", "FH2p_expected"),
    (
        (0.1589, 0.228712, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 258.419),
        (0.1589, 0.250, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 300.186),
        (0.1589, 0.228712, 1000, 0.080, 0.150, 0.200, 9.81, 1.0, 291.773),
        (0.14, 0.250, 1000, 0.097, 0.150, 0.200, 9.81, 1.0, 300.186),
        (0.1589, 0.228712, 1025, 0.097, 0.150, 0.200, 9.81, 1.0, 264.879),
        (0.1589, 0.228712, 1000, 0.097, 0.200, 0.200, 9.81, 1.0, 258.419),
        (0.1589, 0.228712, 1000, 0.097, 0.150, 0.250, 9.81, 1.0, 323.024),
    ),  # B
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
