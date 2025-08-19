import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.structural.forces_caisson.goda1985 as goda
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.eurotop2018 as eurotop
import deltares_coastal_structures_toolbox.functions.hydraulic.wave_runup.taw2002 as taw


tmp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

np.where(tmp > 5, tmp, 1)

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


class BoundaryCondition:

    # boundary condition for breakwater calculations
    condition_name = str
    spectrum_gamma = np.nan
    spectrum_Hm0 = np.nan
    spectrum_Tp = np.nan
    spectrum_values = np.array([])
    water_level = np.nan
    return_period = np.nan

    def __init__(self):
        pass

    def set_spectrum(f, S):
        pass

    @property
    def Tmm10(self):
        pass

    @Tmm10.setter
    def Tmm10(self, value):
        pass
