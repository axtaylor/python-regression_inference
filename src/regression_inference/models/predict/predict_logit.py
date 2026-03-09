import numpy as np
from scipy.stats import norm


def sigmoid(z):
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def predict(model, X, alpha, return_table):

    prediction = (
        sigmoid(X @ model.coefficients + model.intercept)
    )

    if not return_table:
        return prediction
    
    X = X.reshape(1,-1)

    prediction_features = {
        name: f'{value_at.item():.2f}'
        for name, value_at in zip(model.feature_names[1:], X[0])
    }

    X = (
        np.hstack([np.ones((X.shape[0], 1)), X])
    )

    prediction = X @ model.theta

    se_prediction = (
        (np.sqrt((X @ model.variance_coefficient @ X.T)).item())
    )

    prediction_prob = sigmoid(prediction)

    z_critical = norm.ppf(1 - alpha/2)

    ci_low_z, ci_high_z = (
        (prediction - z_critical * se_prediction),
        (prediction + z_critical * se_prediction)
    )

    ci_low_prob, ci_high_prob = (
        sigmoid(ci_low_z),
        sigmoid(ci_high_z)
    )

    z_stat = (
        prediction / se_prediction
        if se_prediction > 0
        else np.array([np.inf])
    )

    p = (
        2 * (1 - norm.cdf(abs(z_stat)))
    )

    return ({
        "features": [prediction_features],
        "prediction_prob": [np.round(prediction_prob.item(), 4)],
        "prediction_class": [int(prediction_prob.item() >= 0.5)],
        "std_error": [np.round(se_prediction, 4)],
        "z_statistic": [np.round(z_stat.item(), 4)],
        "P>|z|": [f"{p.item():.3f}"],
        f"ci_low_{alpha}": [np.round(ci_low_prob.item(), 4)],
        f"ci_high_{alpha}": [np.round(ci_high_prob.item(), 4)],
    })

