# SPDX-License-Identifier: GPL-3.0-or-later
import os
import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd


def create_input(
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
    trained_weights_dir: str = "",
):

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
    # input_dataframe.to_csv(os.path.join(output_dir, f"{file_name}.csv"), sep=";")

    return input_dataframe


def run_XGB_conformal(
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
    trained_weights_dir: str = "",
    trained_weights_name: str = "mapie_QR_model_XGB_v74_CI_90.pickle",
):
    """Generate input file for XGBc Overtopping model described by Den Bieman & Van Gent (2027).


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
    gamma_f_down : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the lower part of the structure (-)
    gamma_f_up : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the upper part of the structure (-)
    tan_alpha_f : float | npt.NDArray[np.float64]
        Tangent of the foreshore slope (-)
    output_dir : str
        Directory where the input file will be saved.
    trained_weights_dir : str, optional
        Directory containing the trained weights for the XGBc Overtopping model, by default ""
    trained_weights_name : str, optional
        Name of the trained weights file for the XGBc Overtopping model, by default "mapie_QR_model_XGB_v74_CI_90.pickle"
    """

    # Create input dataframe for the model
    input_dataframe = create_input(
        beta=beta,
        h=h,
        Hm0=Hm0,
        Tmm10=Tmm10,
        Bt=Bt,
        ht=ht,
        B_berm=B_berm,
        db=db,
        Rc=Rc,
        Ac=Ac,
        Gc=Gc,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f_down=gamma_f_down,
        gamma_f_up=gamma_f_up,
        tan_alpha_f=tan_alpha_f,
    )

    # load trained XGBc Overtopping model
    mapie_cqr = pickle.load(
        open(
            os.path.join(trained_weights_dir, trained_weights_name),
            "rb",
        )
    )

    X = input_dataframe.values  # TODO make this work

    y_pred, y_90p_interval = mapie_cqr.predict_interval(X)

    return y_pred, y_90p_interval


def run_XGB_bootstrap(
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
    trained_weights_dir: str = "",
):

    # Create input dataframe for the model
    input_dataframe = create_input(
        beta=beta,
        h=h,
        Hm0=Hm0,
        Tmm10=Tmm10,
        Bt=Bt,
        ht=ht,
        B_berm=B_berm,
        db=db,
        Rc=Rc,
        Ac=Ac,
        Gc=Gc,
        cot_alpha_down=cot_alpha_down,
        cot_alpha_up=cot_alpha_up,
        gamma_f_down=gamma_f_down,
        gamma_f_up=gamma_f_up,
        tan_alpha_f=tan_alpha_f,
    )

    return
