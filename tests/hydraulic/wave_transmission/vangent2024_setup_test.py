import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.vangent2024_setup as vangent2024


@pytest.mark.parametrize(
    ("Hm0, Tmm10, Rc, delta_expected"),
    (
        ([2.0, 8.0, -1.0, 0.1986]),
        ([2.5, 8.0, -1.0, 0.3680]),
        ([2.0, 9.0, -1.0, 0.1807]),
        ([2.0, 8.0, -0.5, 0.4067]),
    ),
)
def test_delta_impermeable_backward(Hm0, Tmm10, Rc, delta_expected):

    delta_calculated = vangent2024.calculate_structure_induced_setup_delta_impermeable(
        Hm0=Hm0,
        Tmm10=Tmm10,
        Rc=Rc,
    )

    assert delta_calculated == pytest.approx(delta_expected, abs=1e-3)


@pytest.mark.parametrize(
    ("Hm0, Tmm10, Rc, hc, delta_expected"),
    (
        ([2.0, 8.0, -1.0, 4.0, 0.1239]),
        ([2.5, 8.0, -1.0, 4.0, 0.2183]),
        ([2.0, 9.0, -1.0, 4.0, 0.1182]),
        ([2.0, 8.0, -0.5, 4.0, 0.1731]),
        ([2.0, 8.0, -1.0, 3.0, 0.1515]),
    ),
)
def test_delta_permeable_backward(Hm0, Tmm10, Rc, hc, delta_expected):

    delta_calculated = vangent2024.calculate_structure_induced_setup_delta_permeable(
        Hm0=Hm0,
        Tmm10=Tmm10,
        Rc=Rc,
        hc=hc,
    )

    assert delta_calculated == pytest.approx(delta_expected, abs=1e-3)
