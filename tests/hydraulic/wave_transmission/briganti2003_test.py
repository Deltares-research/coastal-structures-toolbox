import pytest
import numpy as np
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.briganti2003 as briganti2003


@pytest.mark.parametrize(
    ("Hsi, Tpi, Rc, B, cot_alpha, Kt_expected"),
    (
        ([2.0, 8.0, -1.0, 25, 1.333, 0.26]),
        ([2.0, 8.0, -1.0, 10, 1.333, 0.56]),
        ([1.0, 8.0, -1.0, 20, 1.333, 0.42]),
        ([2.0, 20.0, -1.0, 20, 1.333, 0.29]),
        ([2.0, 10.0, -1.0, 20, 1.333, 0.28]),
        ([2.0, 8.0, 3, 20, 1.333, 0.05]),
        ([2.0, 8.0, 1, 20, 1.333, 0.05]),
        ([2.0, 8.0, 1, 10, 1.333, 0.16]),
        ([2.0, 8.0, -1, 10, 2.5, 0.49]),
    ),
)
def test_Kt_permeable_backward(Hsi, Tpi, Rc, B, cot_alpha, Kt_expected):

    Kt_calculated = briganti2003.calculate_wave_transmission_Kt(
        Hsi=Hsi,
        Tpi=Tpi,
        Rc=Rc,
        B=B,
        cot_alpha=cot_alpha,
    )

    assert Kt_calculated == pytest.approx(Kt_expected, abs=0.0049)


@pytest.mark.parametrize(
    ("Hsi, Tpi, Rc, B, cot_alpha, Kt_expected"),
    (
        (
            [
                2.0,
                8.0,
                -1.0,
                np.arange(start=5.0, stop=21, step=5),
                1.33,
                np.array([0.65, 0.56, 0.52, 0.28]),
            ]
        ),
    ),
)
def test_Kt_permeable_backward_array(Hsi, Tpi, Rc, B, cot_alpha, Kt_expected):

    Kt_calculated = briganti2003.calculate_wave_transmission_Kt(
        Hsi=Hsi,
        Tpi=Tpi,
        Rc=Rc,
        B=B,
        cot_alpha=cot_alpha,
    )

    assert Kt_calculated == pytest.approx(Kt_expected, abs=0.0049)
