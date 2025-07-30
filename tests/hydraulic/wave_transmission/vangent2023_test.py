import pytest

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_transmission.vangent2023 as vangent2023


@pytest.mark.parametrize(
    ("Hm0, Tmm10, Rc, B, structure_type, Kt_expected"),
    (
        ([2.0, 8.0, -1.0, 25, "permeable", 0.357]),
        ([1.0, 8.0, -1.0, 25, "permeable", 0.565]),
        ([2.0, 20.0, -1.0, 25, "permeable", 0.689]),
        ([2.0, 8.0, -0.5, 25, "permeable", 0.270]),
        ([2.0, 8.0, -1.0, 15, "permeable", 0.501]),
        ([2.0, 8.0, -1.0, 25, "impermeable", 0.249]),
        ([1.0, 8.0, -1.0, 25, "impermeable", 0.455]),
        ([2.0, 20.0, -1.0, 25, "impermeable", 0.603]),
        ([2.0, 8.0, -0.5, 25, "impermeable", 0.176]),
        ([2.0, 8.0, -1.0, 15, "impermeable", 0.386]),
        ([2.0, 8.0, -1.0, 25, "perforated", 0.766]),
        ([1.0, 8.0, -1.0, 25, "perforated", 0.827]),
        ([2.0, 20.0, -1.0, 25, "perforated", 0.866]),
        ([2.0, 8.0, -0.5, 25, "perforated", 0.742]),
        ([2.0, 8.0, -1.0, 15, "perforated", 0.807]),
        ([2.0, 8.0, -1.0, 25, "perforated_with_screen", 0.332]),
        ([1.0, 8.0, -1.0, 25, "perforated_with_screen", 0.521]),
        ([2.0, 20.0, -1.0, 25, "perforated_with_screen", 0.642]),
        ([2.0, 8.0, -0.5, 25, "perforated_with_screen", 0.259]),
        ([2.0, 8.0, -1.0, 15, "perforated_with_screen", 0.461]),
        ([2.0, 8.0, -1.0, 25, "perforated_with_perforated_screen", 0.689]),
        ([1.0, 8.0, -1.0, 25, "perforated_with_perforated_screen", 0.769]),
        ([2.0, 20.0, -1.0, 25, "perforated_with_perforated_screen", 0.821]),
        ([2.0, 8.0, -0.5, 25, "perforated_with_perforated_screen", 0.658]),
        ([2.0, 8.0, -1.0, 15, "perforated_with_perforated_screen", 0.743]),
    ),
)
def test_Kt_backward(Hm0, Tmm10, Rc, B, structure_type, Kt_expected):

    Kt_calculated = vangent2023.calculate_wave_transmission_Kt(
        Hm0=Hm0,
        Tmm10=Tmm10,
        Rc=Rc,
        B=B,
        structure_type=structure_type,
    )

    assert Kt_calculated == pytest.approx(Kt_expected, abs=1e-3)
