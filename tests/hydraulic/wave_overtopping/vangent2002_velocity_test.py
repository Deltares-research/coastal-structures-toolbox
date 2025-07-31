import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.vangent2002_velocity as vg2002_velocity


@pytest.mark.parametrize(
    ("Hs, zXp, Rc, Bc, gamma_f, gamma_f_Crest, uXp_expected"),
    (
        ([2.0, 5.50, 5.00, 2.0, 1.0, 1.0, 3.423]),
        ([2.5, 5.50, 5.00, 2.0, 1.0, 1.0, 3.486]),
        ([2.0, 5.75, 5.00, 2.0, 1.0, 1.0, 4.192]),
        ([2.0, 5.50, 5.25, 2.0, 1.0, 1.0, 2.420]),
        ([2.0, 5.50, 5.00, 2.5, 1.0, 1.0, 3.347]),
        ([2.0, 5.50, 5.00, 2.0, 0.8, 1.0, 3.827]),
        ([2.0, 5.50, 5.00, 2.0, 1.0, 0.8, 3.061]),
    ),
)
def test_uXp_backward(
    Hs,
    zXp,
    Rc,
    Bc,
    gamma_f,
    gamma_f_Crest,
    uXp_expected,
):
    uXp_calculated = vg2002_velocity.calculate_maximum_wave_overtopping_velocity_uXp(
        Hs=Hs,
        zXp=zXp,
        Rc=Rc,
        Bc=Bc,
        gamma_f=gamma_f,
        gamma_f_Crest=gamma_f_Crest,
    )

    assert uXp_calculated == pytest.approx(uXp_expected, abs=1e-2)
