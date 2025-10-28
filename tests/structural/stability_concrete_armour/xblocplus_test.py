import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.xblocplus as XblocPlus


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, V_expected"),
    (
        ([5, 1030, 2400, 4.25 / 1.25]),
        ([7, 1030, 2400, 13.99 / 1.5]),
        ([5, 1000, 2400, 3.64 / 1.25]),
        ([5, 1030, 2500, 3.44 / 1.25]),
    ),
)
def test_M_backward(
    Hs,
    rho_water,
    rho_armour,
    V_expected,
):
    M_calculated = XblocPlus.calculate_unit_mass_M(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
    )

    M_expected = rho_armour * V_expected

    assert M_calculated == pytest.approx(M_expected, abs=12)


@pytest.mark.parametrize(
    ("Hs, rho_water, rho_armour, V_expected, corr_fact"),
    (
        ([5, 1030, 2400, 4.25, 1.25]),
        ([7, 1030, 2400, 13.99, 1.5]),
        ([5, 1000, 2400, 3.64, 1.25]),
        ([5, 1030, 2500, 3.44, 1.25]),
    ),
)
def test_M_corrfactor_backward(
    Hs,
    rho_water,
    rho_armour,
    V_expected,
    corr_fact,
):
    M_calculated = XblocPlus.calculate_unit_mass_M(
        Hs=Hs,
        rho_water=rho_water,
        rho_armour=rho_armour,
        total_correction_factor=corr_fact,
    )

    M_expected = rho_armour * V_expected

    # the below tolerance of 12 is based on 0.0049*2400
    assert M_calculated == pytest.approx(M_expected, abs=12)


@pytest.mark.parametrize(
    ("Hs_expected, rho_water, rho_armour, V"),
    (
        ([5, 1030, 2400, 4.25 / 1.25]),
        ([7, 1030, 2400, 13.99 / 1.5]),
        ([5, 1000, 2400, 3.64 / 1.25]),
        ([5, 1030, 2500, 3.44 / 1.25]),
    ),
)
def test_Hs_backward(
    Hs_expected,
    rho_water,
    rho_armour,
    V,
):
    M = V * rho_armour
    Hs_calculated = XblocPlus.calculate_wave_height_Hs_from_M(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=0.01)


@pytest.mark.parametrize(
    ("Hs_expected, rho_water, rho_armour, V, corr_fact"),
    (
        ([5, 1030, 2400, 4.25, 1.25]),
        ([7, 1030, 2400, 13.99, 1.5]),
        ([5, 1000, 2400, 3.64, 1.25]),
        ([5, 1030, 2500, 3.44, 1.25]),
    ),
)
def test_Hs_corrfactor_backward(
    Hs_expected,
    rho_water,
    rho_armour,
    V,
    corr_fact,
):
    M = V * rho_armour
    Hs_calculated = XblocPlus.calculate_wave_height_Hs_from_M(
        M=M,
        rho_water=rho_water,
        rho_armour=rho_armour,
        total_correction_factor=corr_fact,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=0.01)


# testing correction factors:


@pytest.mark.parametrize(
    ("perc_slope, corrfact_expected"),
    (
        ([1, 1.0]),
        ([2, 1.0]),
        ([3, 1.0]),
        ([4, 1.1]),
        ([5, 1.25]),
        ([6, 1.25]),
        ([7, 1.5]),
        ([8, 1.5]),
        ([9, 1.5]),
        ([10, 2.0]),
    ),
)
def test_corrfactor_perc_slope_seabed_backward(
    perc_slope,
    corrfact_expected,
):

    corrfact_calculated = (
        XblocPlus.calculate_correctionfactor_unit_mass_M_by_percslope_seabed(
            perc_slope=perc_slope
        )
    )

    assert corrfact_calculated == pytest.approx(corrfact_expected, abs=0.01)


@pytest.mark.parametrize(
    ("low_core_permeability, core_impermeable, corrfact_expected"),
    (
        ([True, False, 1.25]),
        ([False, True, 1.5]),
        ([True, True, 1.5]),
        ([False, False, 1.0]),
    ),
)
def test_corrfactor_core_type_backward(
    low_core_permeability,
    core_impermeable,
    corrfact_expected,
):

    corrfact_calculated = (
        XblocPlus.switch_correctionfactor_unit_mass_M_by_core_permeability(
            core_impermeable=core_impermeable,
            low_core_permeability=low_core_permeability,
        )
    )

    assert corrfact_calculated == pytest.approx(corrfact_expected, abs=0.01)


@pytest.mark.parametrize(
    ("rel_freeboard, corrfact_expected"),
    (
        ([2 / 1, 1.0]),
        ([1.5 / 1, 1.0]),
        ([1.0 / 1, 1.0]),
        ([0.99 / 1, 1.25]),
        ([0.6 / 1, 1.25]),
        ([0.5 / 1, 1.25]),
        ([0.499 / 1, 1.5]),  # numbers also derived from calculator at website
    ),
)
def test_corrfactor_low_crested_backward(
    rel_freeboard,
    corrfact_expected,
):

    corrfact_calculated = (
        XblocPlus.calculate_correctionfactor_unit_mass_M_by_relative_freeboard(
            rel_freeboard=rel_freeboard
        )
    )

    assert corrfact_calculated == pytest.approx(corrfact_expected, abs=0.01)
