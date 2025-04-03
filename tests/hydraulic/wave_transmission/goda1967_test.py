import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.goda1967_caisson as goda1967_caisson


@pytest.mark.parametrize(
    ("Hsi, Rc, alpha_1, beta_1, Kt_expected"),
    (
        ([2.0, -1.0, 1.8, 0.1, 0.67]),
        ([5.0, -1.0, 1.8, 0.1, 0.54]),
        ([2.0, 1.0, 1.8, 0.1, 0.25]),
        ([2.0, 1.0, 2.2, 0.4, 0.20]),
        ([5.0, 1.0, 2.2, 0.4, 0.29]),
        ([5.0, 2.0, 2.2, 0.4, 0.23]),
    ),
)
def test_Kt_permeable_backward(Hsi, Rc, alpha_1, beta_1, Kt_expected):

    Kt_calculated = goda1967_caisson.calculate_wave_transmission_Kt(
        Hsi=Hsi,
        Rc=Rc,
        alpha_1=alpha_1,
        beta_1=beta_1,
    )

    assert Kt_calculated == pytest.approx(Kt_expected, abs=0.0049)


@pytest.mark.parametrize(
    ("Hsi, Rc, alpha_1, beta_1, Kt_expected"),
    (
        (
            [
                np.arange(start=1.0, step=2.0, stop=11.1),
                2.0,
                2.2,
                0.4,
                np.array([0.03, 0.155, 0.2297, 0.2649, 0.2851, 0.2982]),
            ]
        ),
    ),
)
def test_Kt_permeable_backward_array(Hsi, Rc, alpha_1, beta_1, Kt_expected):

    Kt_calculated = goda1967_caisson.calculate_wave_transmission_Kt(
        Hsi=Hsi,
        Rc=Rc,
        alpha_1=alpha_1,
        beta_1=beta_1,
    )

    assert Kt_calculated == pytest.approx(Kt_expected, abs=0.0049)
