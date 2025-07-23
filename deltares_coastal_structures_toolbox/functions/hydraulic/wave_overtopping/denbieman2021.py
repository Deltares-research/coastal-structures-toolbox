# SPDX-License-Identifier: GPL-3.0-or-later
import os

import numpy as np
import numpy.typing as npt
import pandas as pd


def write_input_file_XGB_Overtopping(
    beta: float | npt.NDArray[np.float64],
    h: float | npt.NDArray[np.float64],
    Hm0: float | npt.NDArray[np.float64],
    Tmm10: float | npt.NDArray[np.float64],
    Bt: float | npt.NDArray[np.float64],
    ht: float | npt.NDArray[np.float64],
    B_berm: float | npt.NDArray[np.float64],
    db: float | npt.NDArray[np.float64],
    Rc: float | npt.NDArray[np.float64],
    Ac: float | npt.NDArray[np.float64],
    Gc: float | npt.NDArray[np.float64],
    cot_alpha_down: float | npt.NDArray[np.float64],
    cot_alpha_up: float | npt.NDArray[np.float64],
    gamma_f_down: float | npt.NDArray[np.float64],
    gamma_f_up: float | npt.NDArray[np.float64],
    tan_alpha_f: float | npt.NDArray[np.float64],
    output_dir: str,
    file_name: str = "input_XGB_Overtopping",
):
    # TODO check equal lengths (or length = 1, then copy everything to the same length)
    if isinstance(beta, float):
        index = [0]
    else:
        index = None

    input_dataframe = pd.DataFrame(
        data={
            "b": beta,
            "h": h,
            "Hm0 toe": Hm0,
            "Tm-1,0 toe": Tmm10,
            "ht": ht,
            "Bt": Bt,
            "hb": db,
            "B": B_berm,
            "Rc": Rc,
            "Ac": Ac,
            "Gc": Gc,
            "cotad": cot_alpha_down,
            "cotau": cot_alpha_up,
            "gf_d": gamma_f_down,
            "gf_u": gamma_f_up,
            "tanaf": tan_alpha_f,
        },
        index=index,
    )
    input_dataframe.to_csv(os.path.join(output_dir, f"{file_name}.csv"), sep=";")

    return
