# SPDX-License-Identifier: GPL-3.0-or-later
import os

import numpy as np
import numpy.typing as npt
import pandas as pd


def write_input_file_NN_Overtopping(
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
    gamma_f: float | npt.NDArray[np.float64],
    tan_alpha_B: float | npt.NDArray[np.float64],
    output_dir: str,
    file_name: str = "input_NN_Overtopping",
):
    """Generate input file for NN Overtopping model described by Van Gent et al. (2007).

    This function creates a CSV input file with the necessary parameters for the NN Overtopping model, which
    predicts the mean wave overtopping discharge q (m^3/s/m) including uncertainties. The model is available
    as a free web tool at https://www.deltares.nl/en/software-and-data/products/overtopping-neural-network.

    All input parameters should be (1D) arrays of the same length.

    Note that the NN Overtopping model is also available as a downloadable installer here:
    https://dserie.deltares.nl/NNOvertopping/helppage.aspx

    For more details, see Van Gent et al. (2007), which is available here:
    https://doi.org/10.1016/j.coastaleng.2006.12.001

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h : float | npt.NDArray[np.float64]
        Water depth at toe of the structure (m)
    Hm0 : float | npt.NDArray[np.float64]
        Spectral significant wave height (m)
    Tmm10 : float | npt.NDArray[np.float64]
        Spectral wave period Tm-1,0 (s)
    Bt : float | npt.NDArray[np.float64]
        Toe width of the structure (m)
    ht : float | npt.NDArray[np.float64]
        Water depth above the toe of the structure (m)
    B_berm : float | npt.NDArray[np.float64]
        Berm width of the structure (m)
    db : float | npt.NDArray[np.float64]
        Berm height of the structure (m)
    Rc : float | npt.NDArray[np.float64]
        Crest freeboard of the structure (m)
    Ac : float | npt.NDArray[np.float64]
        Armour crest freeboard of the structure (m)
    Gc : float | npt.NDArray[np.float64]
        Width of the crest of the structure (m)
    cot_alpha_down : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    cot_alpha_up : float | npt.NDArray[np.float64]
        Cotangent of the lower part of the front-side slope of the structure (-)
    gamma_f : float | npt.NDArray[np.float64]
        Influence factor for surface roughness (-)
    tan_alpha_B : float | npt.NDArray[np.float64]
        Tangent of the berm slope (-).
    output_dir : str
        Directory where the input file will be saved.
    file_name : str, optional
        Name of the input file, by default "input_NN_Overtopping"
    """

    with open(os.path.join(output_dir, f"{file_name}.inp"), "w") as output_file:
        output_file.write(
            "!-------------------------------------------------------------------------------------------\n"
        )
        output_file.write("! Begin input file\n")
        output_file.write(f"! {file_name}.inp\n")
        output_file.write("! This is a comment record\n")
        output_file.write("! ===============================\n")
        output_file.write("! COLUMN#01  Angle of Wave attack\n")
        output_file.write("! COLUMN#02  Water depth in front of structure\n")
        output_file.write(
            "! COLUMN#03  Significant Wave Height at the toe of structure\n"
        )
        output_file.write("! COLUMN#04  Wave period\n")
        output_file.write("! COLUMN#05  Water depth at the toe of structure\n")
        output_file.write("! COLUMN#06  Width of toe\n")
        output_file.write("! COLUMN#07  Roughness coefficient\n")
        output_file.write("! COLUMN#08  Angle of down slope\n")
        output_file.write("! COLUMN#09  Angle of upper slope\n")
        output_file.write("! COLUMN#10  Crest Freeboard in relation to SWL\n")
        output_file.write("! COLUMN#11  Berm Width\n")
        output_file.write("! COLUMN#12  Water depth at the berm of the structure\n")
        output_file.write("! COLUMN#13  Berm slope\n")
        output_file.write("! COLUMN#14  Armour Freeboard in relation to SWL\n")
        output_file.write("! COLUMN#15  Armour Width\n")
        output_file.write("! ===============================\n")
        output_file.write("! and this is another comment record\n")
        output_file.write("* as well as this one, and the next five ones\n")
        output_file.write(
            "!-----------------------------------------------------------------------"
            + "---------------------------------------------------\n"
        )
        output_file.write(
            "!0 1   02     03        04      05       06       07      08      09      "
            + "10       11       12        13     14       15   \n"
        )
        output_file.write(
            "!degr   m     toe(m)   toe(s)   m        m                                m"
            + "        m        m                m        m    \n"
        )
        output_file.write(
            "!--------------------------------------------------------------------------"
            + "------------------------------------------------\n"
        )

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
            "gf": gamma_f,
            "cotad": cot_alpha_down,
            "cotau": cot_alpha_up,
            "Rc": Rc,
            "B": B_berm,
            "hb": db,
            "tanaB": tan_alpha_B,
            "Ac": Ac,
            "Gc": Gc,
        },
        index=index,
    )
    input_dataframe.to_csv(
        os.path.join(output_dir, f"{file_name}.inp"),
        sep=" ",
        index=False,
        header=False,
        mode="a",
    )

    return
