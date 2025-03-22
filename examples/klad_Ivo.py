import numpy as np

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_overtopping.taw2002 as wave_overtopping_taw2002
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as wave_runup_taw2002
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988 as stability_rock_vandermeer1988
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.vandermeer1988_modified as stability_rock_vandermeer1988_modified
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_rear.vangent2007 as stability_rear_with_crest_vangent2007
import deltares_coastal_structures_toolbox.functions.structural.stability_rock_armour.hudson1959 as stability_hudson1959
import deltares_coastal_structures_toolbox.functions.structural.stability_concrete_armour.tetrapods as stability_tetrapods


print(
    stability_hudson1959.calculate_significant_wave_height_Hs(
        M50=3000,
        rho_water=1025,
        rho_armour=2400,
        KD=7.0,
        cot_alpha=1.5,
        alpha_Hs=1.0,
        rock_type="rough",
        damage_percentage=0,
    )
)

print(stability_tetrapods.unit_properties)

print(core_physics.calculate_Dn50_from_M50(3000, 2400))

# Sin = 2.0
# Nod = core_physics.calculate_Nod_from_S(Sin, 1.0, 0.45)

# Sout = core_physics.calculate_S_from_Nod(Nod, 1.0, 0.45)

# print(Sin == Sout)
# print("Nod = {}, S = {}".format(Nod, Sout))
