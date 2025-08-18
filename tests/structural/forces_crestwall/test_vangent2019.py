import numpy as np
import pytest

import deltares_coastal_structures_toolbox.functions.structural.forces_crestwall.vangent2019 as vangent2019


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

    gamma_beta_calculated = vangent2019.calculate_gamma_beta(np.deg2rad(beta), c_beta)

    assert gamma_beta_calculated == pytest.approx(gamma_beta_expected, abs=1e-2)
