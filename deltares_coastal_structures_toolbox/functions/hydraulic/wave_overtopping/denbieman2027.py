# SPDX-License-Identifier: GPL-3.0-or-later
import os
import pickle

import mapie
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost


def create_and_scale_input(
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
):

    # Models have been trained with data Froude scaled to Hm0 = 1.0 m
    scale_to_Hm0 = 1.0

    Fr_factor = scale_to_Hm0 / Hm0

    if isinstance(beta, float):
        index = [0]
    else:
        index = None

    input_dataframe = pd.DataFrame(
        data={
            "b": beta,
            "cotad": cot_alpha_down,
            "cotau": cot_alpha_up,
            "sc_Tm-1,0 toe": Tmm10 * np.sqrt(Fr_factor),
            "sc_h": h * Fr_factor,
            "sc_ht": (h - ht) * Fr_factor,  # note: tt instead of ht
            "sc_Bt": Bt * Fr_factor,
            "sc_Rc": Rc * Fr_factor,
            "sc_B": B_berm * Fr_factor,
            "sc_hb": db * Fr_factor,
            "sc_Ac": (Ac - Rc) * Fr_factor,  # note: dAc instead of Ac
            "sc_Gc": Gc * Fr_factor,
            "tanaF": tan_alpha_f,
            "gf_u": gamma_f_up,
            "gf_diff": gamma_f_down / gamma_f_up,
        },
        index=index,
    )

    return input_dataframe, Fr_factor


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
    input_dataframe, scale_factor = create_and_scale_input(
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

    y_prediction, y_confidence_interval = mapie_cqr.predict_interval(input_dataframe)

    # convert to actual wave overtopping discharges (invert earlier Froude scaling)
    q_prediction = np.power(10.0, y_prediction) / np.power(scale_factor, 1.5)
    q_confidence_interval = np.power(10.0, y_confidence_interval) / np.power(
        scale_factor, 1.5
    )

    return q_prediction, q_confidence_interval


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
    trained_weights_dir: str = "",
    confidence_interval_width: float = 0.9,
):

    # Create input dataframe for the model
    input_dataframe, scale_factor = create_and_scale_input(
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

    # fixed number of bootstraps in the trained XGBrt model
    number_of_bootstraps = 500

    y_prediction_all = np.zeros((number_of_bootstraps, input_dataframe.shape[0]))

    for model_number in range(number_of_bootstraps):
        if model_number < 10:
            model_save_name = "xgb_model_00{}.pickle".format(model_number)
        elif 10 <= model_number < 100:
            model_save_name = "xgb_model_0{}.pickle".format(model_number)
        else:
            model_save_name = "xgb_model_{}.pickle".format(model_number)

        print("Predicting with trained model: {}".format(model_save_name))

        xgb_model = pickle.load(
            open(os.path.join(trained_weights_dir, model_save_name), "rb")
        )
        tmp_pred = []
        tmp_pred = xgb_model.predict(input_dataframe)
        y_prediction_all[model_number, :] = tmp_pred
    y_prediction = pd.DataFrame(y_prediction_all)

    q_prediction = np.power(10, y_prediction.quantile(q=0.50)) / np.power(
        scale_factor, 1.5
    )

    ci_lower = (1 - confidence_interval_width) / 2
    ci_upper = 1 - ci_lower

    q_confidence_interval_upper = np.power(
        10, y_prediction.quantile(q=ci_lower)
    ) / np.power(scale_factor, 1.5)
    q_confidence_interval_lower = np.power(
        10, y_prediction.quantile(q=ci_upper)
    ) / np.power(scale_factor, 1.5)

    q_confidence_interval = np.vstack(
        (q_confidence_interval_lower, q_confidence_interval_upper)
    )

    return q_prediction, q_confidence_interval
