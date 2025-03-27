import numpy as np
import matplotlib.pyplot as plt

import deltares_coastal_structures_toolbox.functions.core_physics as core_physics
import deltares_coastal_structures_toolbox.functions.structural.forces_caisson.goda1985 as goda


print(goda.calculate_goda_local_wavelength(h=np.arange(0.1, 2.0, 0.1), Tmax=2.0, g=9.8))
