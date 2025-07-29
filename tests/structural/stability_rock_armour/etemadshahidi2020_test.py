import pytest

import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.etemadshahidi2020 as es2020


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, Hs, M50, m, use_depth_limited_version, S_expected"
    ),
    (
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 0.372]),
        ([2.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 0.840]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 0.348]),
        ([3.0, 100.0, 2850, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 0.212]),
        ([3.0, 100.0, 2650, 2500, 3000, 6.0, 2.0, 1000.0, 0.0, False, 0.368]),
        ([3.0, 100.0, 2650, 2650, 6000, 6.0, 2.0, 1000.0, 0.0, False, 0.563]),
        ([3.0, 100.0, 2650, 2650, 3000, 14.0, 2.0, 1000.0, 0.0, False, 2.033]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.5, 1000.0, 0.0, False, 0.960]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1200.0, 0.0, False, 0.266]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.04, True, 0.800]),
    ),
)
def test_S_backward(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    Hs,
    M50,
    m,
    use_depth_limited_version,
    S_expected,
):
    S_calculated = es2020.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        M50=M50,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert S_calculated == pytest.approx(S_expected, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, Hs, S, m, use_depth_limited_version, Dn50_expected"
    ),
    (
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.350, 0.0, False, 0.731]),
        ([2.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.913, 0.0, False, 0.712]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 0.328, 0.0, False, 0.731]),
        ([3.0, 100.0, 2850, 2650, 3000, 6.0, 2.0, 0.199, 0.0, False, 0.713]),
        ([3.0, 100.0, 2650, 2500, 3000, 6.0, 2.0, 0.347, 0.0, False, 0.730]),
        ([3.0, 100.0, 2650, 2650, 6000, 6.0, 2.0, 0.530, 0.0, False, 0.731]),
        ([3.0, 100.0, 2650, 2650, 3000, 14.0, 2.0, 2.210, 0.0, False, 0.712]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.5, 0.904, 0.0, False, 0.730]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.250, 0.0, False, 0.776]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.350, 0.04, True, 0.839]),
    ),
)
def test_Dn50_backward(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    Hs,
    S,
    m,
    use_depth_limited_version,
    Dn50_expected,
):
    Dn50_calculated = es2020.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        S=S,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert Dn50_calculated == pytest.approx(Dn50_expected, abs=1e-3)


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, S, M50, m, use_depth_limited_version, Hs_expected"
    ),
    (
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 2.972]),
        ([2.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 2.379]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 3.018]),
        ([3.0, 100.0, 2850, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False, 3.392]),
        ([3.0, 100.0, 2650, 2500, 3000, 6.0, 2.0, 1000.0, 0.0, False, 2.978]),
        ([3.0, 100.0, 2650, 2650, 6000, 6.0, 2.0, 1000.0, 0.0, False, 2.695]),
        ([3.0, 100.0, 2650, 2650, 3000, 14.0, 2.0, 1000.0, 0.0, False, 1.993]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.5, 1000.0, 0.0, False, 3.132]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1200.0, 0.0, False, 3.216]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.04, True, 2.481]),
    ),
)
def test_Hs_backward(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    S,
    M50,
    m,
    use_depth_limited_version,
    Hs_expected,
):
    Hs_calculated = es2020.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        S=S,
        N_waves=N_waves,
        M50=M50,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert Hs_calculated == pytest.approx(Hs_expected, abs=1e-3)


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, Hs, m, use_depth_limited_version, S"
    ),
    (
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.0, False, 0.350]),
        ([2.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.0, False, 0.913]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 0.0, False, 0.328]),
        ([3.0, 100.0, 2850, 2650, 3000, 6.0, 2.0, 0.0, False, 0.199]),
        ([3.0, 100.0, 2650, 2500, 3000, 6.0, 2.0, 0.0, False, 0.347]),
        ([3.0, 100.0, 2650, 2650, 6000, 6.0, 2.0, 0.0, False, 0.530]),
        ([3.0, 100.0, 2650, 2650, 3000, 14.0, 2.0, 0.0, False, 2.210]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.5, 0.0, False, 0.904]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.0, False, 0.250]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.04, True, 0.350]),
    ),
)
def test_internal_consistency_S_Dn50(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    Hs,
    m,
    use_depth_limited_version,
    S,
):
    Dn50_calculated = es2020.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        S=S,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    S_calculated = es2020.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert S_calculated == pytest.approx(S, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, Hs, S, m, use_depth_limited_version"
    ),
    (
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.350, 0.0, False]),
        ([2.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.913, 0.0, False]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 0.328, 0.0, False]),
        ([3.0, 100.0, 2850, 2650, 3000, 6.0, 2.0, 0.199, 0.0, False]),
        ([3.0, 100.0, 2650, 2500, 3000, 6.0, 2.0, 0.347, 0.0, False]),
        ([3.0, 100.0, 2650, 2650, 6000, 6.0, 2.0, 0.530, 0.0, False]),
        ([3.0, 100.0, 2650, 2650, 3000, 14.0, 2.0, 2.210, 0.0, False]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.5, 0.904, 0.0, False]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.250, 0.0, False]),
        ([3.0, 100.0, 2650, 2650, 3000, 6.0, 2.0, 0.350, 0.04, True]),
    ),
)
def test_internal_consistency_Hs_Dn50(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    Hs,
    S,
    m,
    use_depth_limited_version,
):
    Dn50_calculated = es2020.calculate_nominal_rock_diameter_Dn50(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        S=S,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    Hs_calculated = es2020.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        S=S,
        N_waves=N_waves,
        Dn50=Dn50_calculated,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert Hs_calculated == pytest.approx(Hs, abs=1e-2)


@pytest.mark.parametrize(
    (
        "cot_alpha, M50_core, rho_armour, rho_core, N_waves, Tmm10, Hs, M50, m, use_depth_limited_version"
    ),
    (
        ([3.0, 200.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([2.0, 200.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 150.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2850, 2650, 3000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2500, 3000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2650, 6000, 6.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2650, 3000, 14.0, 2.0, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2650, 3000, 6.0, 2.5, 1000.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2650, 3000, 6.0, 2.0, 1200.0, 0.0, False]),
        ([3.0, 200.0, 2650, 2650, 3000, 6.0, 2.0, 1000.0, 0.04, True]),
    ),
)
def test_internal_consistency_S_Hs(
    cot_alpha,
    M50_core,
    rho_armour,
    rho_core,
    N_waves,
    Tmm10,
    Hs,
    M50,
    m,
    use_depth_limited_version,
):
    S_calculated = es2020.calculate_damage_number_S(
        Hs=Hs,
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        N_waves=N_waves,
        M50=M50,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    Hs_calculated = es2020.calculate_significant_wave_height_Hs(
        Tmm10=Tmm10,
        cot_alpha=cot_alpha,
        rho_armour=rho_armour,
        rho_core=rho_core,
        S=S_calculated,
        N_waves=N_waves,
        M50=M50,
        M50_core=M50_core,
        m=m,
        use_depth_limited_version=use_depth_limited_version,
    )

    assert Hs_calculated == pytest.approx(Hs, abs=1e-2)
