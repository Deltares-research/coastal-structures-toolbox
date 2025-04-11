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


plt.plot(Hs_table_in, label="in")
plt.plot(Hs_table_out, label="out")
plt.show

intersection = np.intersect1d(Hs_table_in, Hs_table_out)

ind = np.argmin(np.abs(Hs_table_in - Hs_table_out))

fromto = [
    np.minimum(Hs_table_in[ind], Hs_table_out[ind]) - 0.1,
    np.maximum(Hs_table_in[ind], Hs_table_out[ind]) + 0.1,
]
