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

    gamma_beta_calculated = (
        vangentvanderwerf2019.calculate_influence_oblique_waves_gamma_beta(beta, c_beta)
    )

    assert gamma_beta_calculated == pytest.approx(gamma_beta_expected, abs=1e-2)


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
