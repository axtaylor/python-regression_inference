from typing import Optional
from ..utils import validate
from ..models.fit import fit_linear, fit_logit, fit_logit_ordinal, fit_logit_multinomial
import numpy as np

# CUDA Not required. Do not throw an error here
try:
    import cupy
    from ..models.fit.accelerated_models import fit_logit_ordinal_cuda, fit_logit_multinomial_cuda
    CUDA = True
except ImportError:
    CUDA = False


def get_featureLabel(X, feature_names):
    '''
    Feature labels are assigned in a hierarchy:

    User defined names -> Pandas column names -> Fallback names.

    The assignment is always in order of the columns from the X array or dataframe
    '''

    return (
        ['const', *feature_names] if feature_names is not None
        else X.columns if hasattr(X, 'columns')
        else ['const', *[f"Feature {i}" for i in range(1, X.shape[1])]]
    )


def get_targetLabel(y, target_name):
    '''
    Target label is assigned in a hierarchy:

    User defined name -> Pandas series name -> Fallback name.
    '''

    return (
        target_name if target_name is not None
        else y.name if hasattr(y, 'name')
        else "Target"
    )


def fit(
    model,
    X:              np.ndarray,
    y:              np.ndarray,
    feature_names:  Optional[list[str]],
    target_name:    Optional[str],
    alpha:          float = 0.05,
    max_iter:       int = 100,
    tol:            float = 1e-8,
    adj_cutpoints:  bool = False,
    cuda:           bool = False,
):

    X_array, y_array = (
        validate.validate(X, y, alpha, model.model_type)
    )

    model.feature_names = get_featureLabel(X, feature_names)
    model.target = get_targetLabel(y, target_name)
    model.alpha = alpha

    model.X, model.y = (
        X_array, y_array
    )

    model.degrees_freedom = (
        model.X.shape[0] - model.X.shape[1]
    )

    if not cuda:

        if model.model_type == "linear":
            fit_linear.internal_linear(model)

        elif model.model_type == "logit":
            fit_logit.internal_logit(model, max_iter, tol)

        elif model.model_type == "logit_multinomial":
            fit_logit_multinomial.internal_multinomial_logit(
                model, max_iter, tol)

        elif model.model_type == "logit_ordinal":
            fit_logit_ordinal.internal_ordinal_logit(
                model, adj_cutpoints, max_iter, tol)

        else:
            raise ValueError(f"Model type: {model.model_type} is unexpected.")

    elif cuda:

        if not CUDA:
            raise ImportError(
                f"Could not find module 'cupy' in this Python environment.\n"
            )

        model.cuda = True

        if model.model_type == "logit_multinomial":
            fit_logit_multinomial_cuda.accelerated_multinomial_logit(
                model, max_iter, tol)

        elif model.model_type == "logit_ordinal":
            fit_logit_ordinal_cuda.accelerated_ordinal_logit(
                model, adj_cutpoints, max_iter, tol)

        else:
            raise ValueError(f"CUDA Not supported for model type: {model.model_type}")

    else:
        raise ValueError(f"Unexpected cuda specification: {cuda}")

    return model
