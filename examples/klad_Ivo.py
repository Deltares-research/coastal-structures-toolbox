import numpy as np
import matplotlib.pyplot as plt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as stability_rock_vandermeer1988
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as stability_rock_vandermeer1988_modified
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_rear.vangent2007 as stability_rear_with_crest_vangent2007
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as stability_hudson1959
import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubes_double_layer_Hudson1959 as stability_cubes
import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.cubes_double_layer_vanDerMeer1988 as stability_cubes_vdM


Hs = 3.0
rho_armour = 2400

out = stability_cubes.calculate_unit_mass_M_Hudson1959(
    Hs=Hs,
    rho_water=1025,
    rho_armour=rho_armour,
    KD=6.5,
    cot_alpha=1.5,
    alpha_Hs=1.0,
)

print(out)

out = stability_cubes.calculate_significant_wave_height_Hs_Hudson1959(
    M=20000,
    rho_water=1025,
    rho_armour=rho_armour,
    KD=7.5,
    cot_alpha=2.0,
    alpha_Hs=1.0,
)

print(out)

Ns = core_physics.check_usage_stabilitynumber(Ns=None, Hs=3.0, Delta=1.7, Dn=2.0)
print(Ns)
out = core_physics.check_usage_stabilitynumber(Ns=Ns[0], Hs=None, Delta=1.7, Dn=2.0)
out = core_physics.check_usage_stabilitynumber(Ns=Ns[0], Hs=3.0, Delta=None, Dn=2.0)
out = core_physics.check_usage_stabilitynumber(Ns=Ns[0], Hs=3.0, Delta=1.7, Dn=None)
# out = core_physics.check_usage_stabilitynumber(Ns=Ns[0], Hs=3.0, Delta=None, Dn=None)


Dn = stability_cubes_vdM.calculate_nominal_diameter_Dn_vanDerMeer1988(
    3.0, 8.0, 1025, 2400, 3000, 1.5, 2.0
)
print(Dn)

s0m = core_physics.calculate_wave_steepness_s(H=3.0, T=8.0)

out = stability_cubes_vdM.calculate_wave_height_Hs_vanDerMeer1988(
    Dn=Dn,
    s0m=s0m,
    rho_water=1025,
    rho_armour=2400,
    N_waves=3000,
    cot_alpha=1.5,
    Nod=2.0,
)

print(out)


out = stability_cubes_vdM.calculate_damage_Nod_vanDerMeer1988(
    Hs=3.0,
    Dn=1.5,
    s0m=0.05,
    rho_water=1025,
    rho_armour=2400,
    N_waves=3000,
    cot_alpha=1.5,
)

print(out)
