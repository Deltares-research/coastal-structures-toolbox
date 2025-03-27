import pytest

import deltares_coastal_structures_toolbox.functions.structural.forces_caisson.goda1985 as caisson


@pytest.mark.parametrize(
    ("T, h, g, L_expected"),
    (
        ([2.0, 0.2, 9.8, 2.71]),  # values from goda (2000) tables
        ([2.0, 1.0, 9.8, 5.21]),
        ([2.0, 2.0, 9.8, 6.05]),
        ([2.0, 11.0, 9.8, 6.24]),
        ([5.0, 6.0, 9.8, 32.17]),
    ),
)
def test_L_backward(T, h, g, L_expected):

    L_calculated = caisson.calculate_local_wavelength(T=T, h=h, g=g)

    assert L_calculated == pytest.approx(L_expected, abs=1e-2)


# @pytest.mark.parametrize(
#     ("HD, Tmax, beta, h_s, d, hacc, Rc, B_up, rho_water, FH_expected"),
#     (([5.07, 8.0, 0.0, 5.0, 4.0, 6.0, 5.0, 10.0, 1025, 466.22]),),
# )
# def test_FH_backward(HD, Tmax, beta, h_s, d, hacc, Rc, B_up, rho_water, FH_expected):

#     FH_calculated = (
#         caisson.calculate_forces(
#             HD=HD,
#             Tmax=Tmax,
#             beta=beta,
#             h_s=h_s,
#             d=d,
#             hacc=hacc,
#             Rc=Rc,
#             B_up=B_up,
#             rho_water=rho_water,
#         )[0]
#         / 1000
#     )

#     assert FH_calculated == pytest.approx(FH_expected, abs=1e-2)
