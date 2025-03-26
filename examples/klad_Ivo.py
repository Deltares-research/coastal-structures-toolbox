import numpy as np
import matplotlib.pyplot as plt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as stability_rock_vandermeer1988
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as stability_rock_vandermeer1988_modified
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_rear.vangent2007 as stability_rear_with_crest_vangent2007
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as stability_hudson1959
import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.accropode_hudson1959 as stability_accropodes
import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.Xbloc as xbloc


out = stability_accropodes.calculate_KD_breaking_trunk_from_seabed_slope(100)
print(out)

out = stability_accropodes.calculate_KD_breaking_trunk_from_seabed_slope(4)
print(out)

out = stability_accropodes.calculate_KD_breaking_trunk_from_seabed_slope(5.5)
print(out)


allKD = np.array([16, 15, 14, 13, 12, 11, 10, 9, 8])

M = stability_accropodes.calculate_unit_mass_M(3.0, 1025, 2400, allKD)

Hs_in = 5.0

M = xbloc.xbloc_calculate_unit_mass_M(Hs_in, 1025, 2400)
V = xbloc.xbloc_calculate_unit_volume_V(Hs_in, 1025, 2400)

Hs_out = xbloc.xbloc_calculate_wave_height_Hs_from_M(
    M=M, rho_water=1025, rho_armour=2400
)

print(Hs_out)
