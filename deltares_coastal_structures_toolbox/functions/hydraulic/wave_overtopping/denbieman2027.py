# SPDX-License-Identifier: GPL-3.0-or-later
import os
import pickle

import mapie  # noqa
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost  # noqa


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
) -> tuple[pd.DataFrame, float]:
    """Create input dataframe and Froude scale the contents

    This function creates a pandas DataFrame containing the input parameters for both
    the XGBcr and the XGBrt wave overtopping model described in Den Bieman & Van Gent (2027).
    These input parameters are Froude scaled to a Hm0 of 1.0 m, as that is the way these models
    have been trained.

    For more details, see Den Bieman & Van Gent (2027), which is available here:
        TODO DOI

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h : float | npt.NDArray[np.float64]
        Water depth at the toe of the structure (m)
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
        Cotangent of the upper part of the front-side slope of the structure (-)
    gamma_f_down : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the lower part of the structure (-)
    gamma_f_up : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the upper part of the structure (-)
    tan_alpha_f : float | npt.NDArray[np.float64]
        Tangent of the foreshore slope (-)

    Returns
    -------
    tuple[pd.DataFrame, float]
        Input dataframe, including Froude scaling and converted input parameters,
        and the scaling factor needed to scale to a Hm0 of 1.0 m
    """

    # Models have been trained with data Froude scaled to Hm0 = 1.0 m
    scale_to_Hm0 = 1.0

    Froude_factor = scale_to_Hm0 / Hm0

    if isinstance(beta, float):
        index = [0]
    else:
        index = None

    input_dataframe = pd.DataFrame(
        data={
            "b": beta,
            "cotad": cot_alpha_down,
            "cotau": cot_alpha_up,
            "sc_Tm-1,0 toe": Tmm10 * np.sqrt(Froude_factor),
            "sc_h": h * Froude_factor,
            "sc_ht": (h - ht) * Froude_factor,  # note: tt instead of ht
            "sc_Bt": Bt * Froude_factor,
            "sc_Rc": Rc * Froude_factor,
            "sc_B": B_berm * Froude_factor,
            "sc_hb": db * Froude_factor,
            "sc_Ac": (Ac - Rc) * Froude_factor,  # note: dAc instead of Ac
            "sc_Gc": Gc * Froude_factor,
            "tanaF": tan_alpha_f,
            "gf_u": gamma_f_up,
            "gf_diff": gamma_f_down / gamma_f_up,
        },
        index=index,
    )

    return input_dataframe, Froude_factor


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
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with the XGBcr conformal overtopping model described by Den Bieman & Van Gent (2027)

    Important: to use this function, the trained weights for the XGBcr Overtopping model must first
    be downloaded and saved locally in a directory to be supplied via the "trained_weights_dir"
    keyword. These trained weights are freely available here:
        TODO DOI

    The XGBcr conformal overtopping model both predicts the expected mean wave overtopping
    discharge and provides the boundaries of the associated 90% confidence intervals. Note that
    Den Bieman & Van Gent (2027) recommend using of the XGBcr conformal model (this function) over
    the XGBrt bootstrapped model (run_XGB_bootstrap function).

    For more details, see Den Bieman & Van Gent (2027), which is available here:
        TODO DOI

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h : float | npt.NDArray[np.float64]
        Water depth at the toe of the structure (m)
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
        Cotangent of the upper part of the front-side slope of the structure (-)
    gamma_f_down : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the lower part of the structure (-)
    gamma_f_up : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the upper part of the structure (-)
    tan_alpha_f : float | npt.NDArray[np.float64]
        Tangent of the foreshore slope (-)
    trained_weights_dir : str, optional
        Directory containing the trained weights for the XGBcr Overtopping model, by default ""
    trained_weights_name : str, optional
        Name of the trained model weights file, by default "mapie_QR_model_XGB_v74_CI_90.pickle"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The first array contains the predicted (50th percentile) mean wave overtopping discharge
        q (m^3/s/m).The second array contains the corresponding boundaries of the 90% confidence
        intervals for the predicted overtopping discharges.
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
    q_confidence_interval_lower = np.power(
        10.0, y_confidence_interval[:, 0, :].flatten()
    ) / np.power(scale_factor, 1.5)
    q_confidence_interval_upper = np.power(
        10.0, y_confidence_interval[:, 1, :].flatten()
    ) / np.power(scale_factor, 1.5)

    q_confidence_interval = np.vstack(
        (q_confidence_interval_lower, q_confidence_interval_upper)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with the XGBrt bootstrapped overtopping model described by Den Bieman & Van Gent (2027).

    Important: to use this function, the trained weights for the XGBrt Overtopping model must first
    be downloaded and saved locally in a directory to be supplied via the "trained_weights_dir"
    keyword. These trained weights are freely available here:
        TODO DOI

    The XGBrt bootstrapped overtopping model both predicts the expected mean wave overtopping
    discharge and provides the boundaries of an associated confidence intervals, by default the
    90% confidence interval. Note that Den Bieman & Van Gent (2027) recommend using of the XGBcr
    conformal model (run_XGB_conformal function) over the XGBrt bootstrapped model (this function).
    Additionally, the bootstrapped model tends is slower to perform predictions compared to the
    conformal model, as it needs to load and predict with 500 individual models.

    For more details, see Den Bieman & Van Gent (2027), which is available here:
        TODO DOI

    Parameters
    ----------
    beta : float | npt.NDArray[np.float64]
        Angle of wave incidence (degrees)
    h : float | npt.NDArray[np.float64]
        Water depth at the toe of the structure (m)
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
        Cotangent of the upper part of the front-side slope of the structure (-)
    gamma_f_down : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the lower part of the structure (-)
    gamma_f_up : float | npt.NDArray[np.float64]
        Influence factor for surface roughness of the upper part of the structure (-)
    tan_alpha_f : float | npt.NDArray[np.float64]
        Tangent of the foreshore slope (-)
    trained_weights_dir : str, optional
        Directory containing the trained model weights, by default ""
    confidence_interval_width : float, optional
        Width of the confidence interval for the predictions, by default 0.9

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The first array contains the predicted (50th percentile) mean wave overtopping discharge
        q (m^3/s/m).The second array contains the corresponding boundaries of the 90% confidence
        intervals for the predicted overtopping discharges.
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

    # fixed number of bootstraps in the trained XGBrt model
    # number_of_bootstraps = 500
    number_of_bootstraps = 5

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

    ci_lower = (1.0 - confidence_interval_width) / 2.0
    ci_upper = 1.0 - ci_lower

    q_confidence_interval_lower = np.power(
        10, y_prediction.quantile(q=ci_lower)
    ) / np.power(scale_factor, 1.5)
    q_confidence_interval_upper = np.power(
        10, y_prediction.quantile(q=ci_upper)
    ) / np.power(scale_factor, 1.5)

    q_confidence_interval = np.vstack(
        (q_confidence_interval_lower, q_confidence_interval_upper)
    )

    return q_prediction.values, q_confidence_interval
