import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.structural.forces_caisson.goda1985 as goda
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as eurotop
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as taw


B = 3
Hsi = np.arange(1, 2, 0.1)

check_is_array = [
    isinstance(B, np.ndarray),
    isinstance(Hsi, np.ndarray),
]

if not any(check_is_array):
    print("no arrays")
else:
    print("arrays")


B = 3
Hsi = 10

check_is_array = [
    isinstance(B, np.ndarray),
    isinstance(Hsi, np.ndarray),
]

if not any(check_is_array):
    print("no arrays")
else:
    print("arrays")

print("Done")
