import numpy as np

import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.eurotop2018 as wave_overtopping_eurotop2018
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as stability_rock_vandermeer1988
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as stability_rock_vandermeer1988_modified
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_rear.vangent2007 as stability_rear_with_crest_vangent2007


cot_alpha = 2.5
B_berm = 3.0
db = 0.5
Rc_var = np.linspace(2.0, 6.0, 1000)
q_max = 10e-3
Hm0 = 2.0
Tmm10 = 5.0
Tp = 1.1 * Tmm10
beta = 30.0


q_taw2002, _ = wave_overtopping_taw2002.calculate_overtopping_discharge_q(
    Hm0=Hm0,
    Tmm10=Tmm10,
    beta=beta,
    Rc=Rc_var,
    cot_alpha=cot_alpha,
    B_berm=B_berm,
    db=db,
)

q_eurotop2018, _ = wave_overtopping_eurotop2018.calculate_overtopping_discharge_q(
    Hm0=Hm0,
    Tmm10=Tmm10,
    beta=beta,
    Rc=Rc_var,
    cot_alpha=cot_alpha,
    B_berm=B_berm,
    db=db,
)


# q_calculated = wave_overtopping_taw2002.calculate_overtopping_discharge_q(
#     Hm0=2.0,
#     Tmm10=5.0,
#     beta=0.0,
#     gamma_f=1.0,
#     Rc=5.0,
#     B_berm=0.0,
#     db=0.0,
#     use_best_fit=False,
#     cot_alpha_down=3.0,
#     cot_alpha_up=3.0,
# )

# z2p_calculated = wave_runup_taw2002.calculate_wave_runup_height_z2p(
#     Hm0=np.array(
#         [
#             2.0,
#             2.5,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#         ]
#     ),
#     Tmm10=np.array(
#         [
#             8.0,
#             8.0,
#             12.0,
#             8.0,
#             8.0,
#             8.0,
#             8.0,
#             8.0,
#             8.0,
#         ]
#     ),
#     beta=np.array(
#         [
#             0.0,
#             0.0,
#             0.0,
#             30.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#         ]
#     ),
#     cot_alpha_down=np.array(
#         [
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.5,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#         ]
#     ),
#     cot_alpha_up=np.array(
#         [
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             2.0,
#             3.0,
#             3.0,
#             3.0,
#         ]
#     ),
#     B_berm=np.array(
#         [
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             2.0,
#             2.0,
#             2.0,
#             0.0,
#             0.0,
#         ]
#     ),
#     dh=np.array(
#         [
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             -0.5,
#             0.0,
#             1.0,
#             1.0,
#         ]
#     ),
#     gamma_f=np.array(
#         [
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             0.45,
#         ]
#     ),
# )

# Hs_vdM1988 = stability_rock_vandermeer1988.calculate_significant_wave_height_Hs(
#     ratio_H2p_Hs=np.array(
#         [
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#             1.4,
#         ]
#     ),
#     Tm=np.array(
#         [
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             12.0,
#             6.0,
#             6.0,
#             14.0,
#         ]
#     ),
#     cot_alpha=np.array(
#         [
#             3.0,
#             2.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             4.0,  # 3.6,
#         ]
#     ),
#     rho_armour=np.array(
#         [
#             2650,
#             2650,
#             2650,
#             2850,
#             2650,
#             2650,
#             2650,
#             2650,
#             2650,
#         ]
#     ),
#     P=np.array(
#         [
#             0.4,
#             0.4,
#             0.5,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#         ]
#     ),
#     N_waves=np.array(
#         [
#             3000,
#             3000,
#             3000,
#             3000,
#             6000,
#             3000,
#             3000,
#             3000,
#             3000,
#         ]
#     ),
#     M50=np.array(
#         [
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1200.0,
#             1000.0,
#         ]
#     ),
#     S=np.array(
#         [
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             3.0,
#             2.0,
#             2.0,
#         ]
#     ),
# )

# Hs_vdM1988_mod = (
#     stability_rock_vandermeer1988_modified.calculate_significant_wave_height_Hs(
#         ratio_H2p_Hs=np.array(
#             [
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#                 1.4,
#             ]
#         ),
#         Tmm10=np.array(
#             [
#                 6.0,
#                 6.0,
#                 6.0,
#                 6.0,
#                 6.0,
#                 12.0,
#                 6.0,
#                 6.0,
#                 14.0,
#             ]
#         ),
#         cot_alpha=np.array(
#             [
#                 3.0,
#                 2.0,
#                 3.0,
#                 3.0,
#                 3.0,
#                 3.0,
#                 3.0,
#                 3.0,
#                 4.0,  # 3.6,
#             ]
#         ),
#         rho_armour=np.array(
#             [
#                 2650,
#                 2650,
#                 2650,
#                 2850,
#                 2650,
#                 2650,
#                 2650,
#                 2650,
#                 2650,
#             ]
#         ),
#         P=np.array(
#             [
#                 0.4,
#                 0.4,
#                 0.5,
#                 0.4,
#                 0.4,
#                 0.4,
#                 0.4,
#                 0.4,
#                 0.4,
#             ]
#         ),
#         N_waves=np.array(
#             [
#                 3000,
#                 3000,
#                 3000,
#                 3000,
#                 6000,
#                 3000,
#                 3000,
#                 3000,
#                 3000,
#             ]
#         ),
#         M50=np.array(
#             [
#                 1000.0,
#                 1000.0,
#                 1000.0,
#                 1000.0,
#                 1000.0,
#                 1000.0,
#                 1000.0,
#                 1200.0,
#                 1000.0,
#             ]
#         ),
#         S=np.array(
#             [
#                 2.0,
#                 2.0,
#                 2.0,
#                 2.0,
#                 2.0,
#                 2.0,
#                 3.0,
#                 2.0,
#                 2.0,
#             ]
#         ),
#     )
# )

# S_vdM1988_mod = stability_rock_vandermeer1988_modified.calculate_damage_number_S(
#     Hs=np.array(
#         [
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.5,
#             2.0,
#             2.0,
#         ]
#     ),
#     H2p=1.4
#     * np.array(
#         [
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.5,
#             2.0,
#             2.0,
#         ]
#     ),
#     Tmm10=np.array(
#         [
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             12.0,
#             6.0,
#             6.0,
#             12.0,
#         ]
#     ),
#     cot_alpha=np.array(
#         [
#             3.0,
#             2.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.6,
#         ]
#     ),
#     rho_armour=np.array(
#         [
#             2650,
#             2650,
#             2650,
#             2850,
#             2650,
#             2650,
#             2650,
#             2650,
#             2650,
#         ]
#     ),
#     P=np.array(
#         [
#             0.4,
#             0.4,
#             0.5,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#         ]
#     ),
#     N_waves=np.array(
#         [
#             3000,
#             3000,
#             3000,
#             3000,
#             6000,
#             3000,
#             3000,
#             3000,
#             3000,
#         ]
#     ),
#     M50=np.array(
#         [
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1200.0,
#             1000.0,
#         ]
#     ),
# )

# S_vdM1988 = stability_rock_vandermeer1988.calculate_damage_number_S(
#     Hs=np.array(
#         [
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.5,
#             2.0,
#             2.0,
#         ]
#     ),
#     H2p=1.4
#     * np.array(
#         [
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.0,
#             2.5,
#             2.0,
#             2.0,
#         ]
#     ),
#     Tm=np.array(
#         [
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             6.0,
#             12.0,
#             6.0,
#             6.0,
#             12.0,
#         ]
#     ),
#     cot_alpha=np.array(
#         [
#             3.0,
#             2.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.0,
#             3.6,
#         ]
#     ),
#     rho_armour=np.array(
#         [
#             2650,
#             2650,
#             2650,
#             2850,
#             2650,
#             2650,
#             2650,
#             2650,
#             2650,
#         ]
#     ),
#     P=np.array(
#         [
#             0.4,
#             0.4,
#             0.5,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#             0.4,
#         ]
#     ),
#     N_waves=np.array(
#         [
#             3000,
#             3000,
#             3000,
#             3000,
#             6000,
#             3000,
#             3000,
#             3000,
#             3000,
#         ]
#     ),
#     M50=np.array(
#         [
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1000.0,
#             1200.0,
#             1000.0,
#         ]
#     ),
# )

# S_vdM1988_2 = stability_rock_vandermeer1988.calculate_damage_number_S(
#     Hs=np.array([2.0, 3.0, 2.0]),
#     H2p=1.4 * np.array([2.0, 3.0, 2.0]),
#     Tm=np.array([12.0, 8.0, 6.0]),
#     cot_alpha=np.array([3.6, 3.0, 2.0]),
#     rho_armour=np.array([2650.0, 2650.0, 2650.0]),
#     P=np.array([0.4, 0.4, 0.4]),
#     N_waves=np.array([900, 3000, 6000]),
#     M50=np.array([1000.0, 1000.0, 1000.0]),
# )


# S_result = stability_rear_with_crest_vangent2007.calculate_damage_number_S(
#     cot_alpha=3.0,
#     cot_phi=2.0,
#     gamma=0.47,
#     Dn50=0.6708,
#     Hs=7.0,
#     Tmm10=12.0,
#     Rc=5.0,
#     Rc2_front=0.5,
#     Rc2_rear=1.0,
#     N_waves=10000,
# )

# Dn50_result = (
#     stability_rear_with_crest_vangent2007.calculate_nominal_rock_diameter_Dn50(
#         cot_alpha=3.0,
#         cot_phi=2.0,
#         gamma=0.47,
#         S=9.0,
#         Hs=7.0,
#         Tmm10=12.0,
#         Rc=5.0,
#         Rc2_front=0.5,
#         Rc2_rear=1.0,
#         N_waves=1000,
#     )
# )

Hs_result = stability_rear_with_crest_vangent2007.calculate_maximum_Hs(
    cot_alpha=3.0,
    cot_phi=2.0,
    gamma=0.47,
    S=9.0,
    Tmm10=12.0,
    Rc=5.0,
    Rc2_front=0.5,
    Rc2_rear=1.0,
    N_waves=1000,
    Dn50=0.6708,
)

S_result_v = stability_rear_with_crest_vangent2007.calculate_damage_number_S(
    cot_alpha=np.array([3.0, 3.0, 3.0]),
    cot_phi=np.array([2.0, 2.0, 2.0]),
    gamma=np.array([0.47, 0.47, 0.47]),
    Dn50=np.array([0.6708, 0.6708, 0.6708]),
    Hs=np.array([7.0, 7.0, 7.0]),
    Tmm10=np.array([12.0, 12.0, 12.0]),
    Rc=np.array([5.0, 5.0, 5.0]),
    Rc2_front=np.array([0.5, 0.5, 0.5]),
    Rc2_rear=np.array([1.0, 1.0, 1.0]),
    N_waves=np.array([10000, 10000, 10000]),
)

Dn50_result_v = (
    stability_rear_with_crest_vangent2007.calculate_nominal_rock_diameter_Dn50(
        cot_alpha=np.array([3.0, 3.0, 3.0]),
        cot_phi=np.array([2.0, 2.0, 2.0]),
        gamma=np.array([0.47, 0.47, 0.47]),
        S=np.array([9.0, 9.0, 9.0]),
        Hs=np.array([7.0, 7.0, 7.0]),
        Tmm10=np.array([12.0, 12.0, 12.0]),
        Rc=np.array([5.0, 4.0, 6.0]),
        Rc2_front=np.array([0.5, 0.5, 0.5]),
        Rc2_rear=np.array([1.0, 1.0, 1.0]),
        N_waves=np.array([1000, 1000, 1000]),
    )
)


print("Hoera")
