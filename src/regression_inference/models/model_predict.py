import numpy as np
from ..models.predict import predict_linear, predict_logit, predict_logit_multinomial, predict_logit_ordinal
from typing import Union


def predict(model, X: np.ndarray, alpha: float, return_table: bool) -> Union[float, np.ndarray, dict, list[dict]]:

    X = np.asarray(X, dtype=float)
    X = np.atleast_2d(X)

    if model.model_type == "linear":
        # Returns: Union[float, dict]
        return predict_linear.predict(model, X, alpha, return_table)

    elif model.model_type == "logit":
        # Returns: Union[np.ndarray, dict]
        return predict_logit.predict(model, X, alpha, return_table)

    elif model.model_type == "logit_multinomial":
        # Returns: Union[np.ndarray, dict]
        return predict_logit_multinomial.predict(model, X, alpha, return_table)

    elif model.model_type == "logit_ordinal":
        # Returns: Union[np.ndarray, List[dict]]
        return predict_logit_ordinal.predict(model, X, alpha, return_table)

    else:
        raise ValueError(f"Model type: {model.model_type} is unexpected.")
