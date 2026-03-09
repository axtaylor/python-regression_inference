import numpy as np
from scipy.stats import norm


def softmax(Z: np.ndarray) -> np.ndarray:

    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(np.clip(Z_stable, -700, 700))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def predict(model, X, alpha, return_table):

    X = X.reshape(1,-1)

    prediction = softmax(
        np.column_stack(
            [
                np.zeros(X.shape[0]),
                np.asarray(
                    X, dtype=float) @ model.coefficients + model.intercept
            ]
        )
    )

    if not return_table:
        return prediction

    prediction_features = {
        name: f'{value_at.item():.2f}'
        for name, value_at in zip(model.feature_names[1:], X[0])
    }

    X = (
        np.hstack([np.ones((X.shape[0], 1)), X])
    )

    multinomial_classes, prediction_probs, prediction_linear, std_errors, z_statistics, p_values, ci_lows, ci_highs = (
        [], [], [], [], [], [], [], []
    )

    x = X[0]

    p, J = model.theta.shape

    z_crit = norm.ppf(1 - alpha / 2)

    ''' 'J' Linear Predictors -> Add Reference Class -> Convert To Probabilities '''

    prediction = x @ model.theta
    eta_full = np.r_[0.0, prediction]
    prediction_prob = softmax(eta_full[None, :])[0]
    pred_class = int(np.argmax(prediction_prob))

    '''Initial inference for class Y = 0'''

    p0, pj = (
        prediction_prob[0],  prediction_prob[1:]
    )

    g0 = np.zeros(p * J)

    for j in range(J):

        g0[j*p: (j+1)*p] = -p0 * pj[j] * x

    var_p0 = (
        g0 @ model.xtWx_inv @ g0
    )

    se_p0 = (
        np.sqrt(var_p0)
    )

    ci_low_p0, ci_high_p0 = (
        max(0.0, p0 - z_crit * se_p0),
        min(1.0, p0 + z_crit * se_p0)
    )

    multinomial_classes.append(int(model.y_classes[0]))
    prediction_probs.append(np.round(p0, 4))
    prediction_linear.append(0.0)
    std_errors.append(np.round(se_p0, 4))
    z_statistics.append(None)
    p_values.append(None)
    ci_lows.append(np.round(ci_low_p0, 4))
    ci_highs.append(np.round(ci_high_p0, 4))

    for j in range(J):

        '''Compute inference for J reference classes'''

        idx = slice(j*p, (j+1)*p)
        Vj = model.xtWx_inv[idx, idx]  # Cov block

        se_eta = np.sqrt(x @ Vj @ x)

        ''' Log odds relative to reference class'''

        ci_low_eta, ci_high_eta = (
            prediction[j] - z_crit * se_eta,
            prediction[j] + z_crit * se_eta
        )

        eta_low, eta_high = (
            eta_full.copy(),
            eta_full.copy()
        )

        eta_low[j+1], eta_high[j+1] = (
            ci_low_eta,
            ci_high_eta
        )

        p_low, p_high = (
            softmax(eta_low[None, :])[0][j+1],
            softmax(eta_high[None, :])[0][j+1]
        )

        z_stat = (
            prediction[j] / se_eta
            if se_eta > 0 else
            np.inf
        )

        p_val = (
            2 * (1 - norm.cdf(abs(z_stat)))
        )

        multinomial_classes.append(int(model.y_classes[j+1]))
        prediction_probs.append(np.round(prediction_prob[j+1], 4))
        prediction_linear.append(np.round(prediction[j], 4))
        std_errors.append(np.round(se_eta, 4))
        z_statistics.append(np.round(z_stat, 4))
        p_values.append(f"{p_val:.3f}")
        ci_lows.append(np.round(p_low, 4))
        ci_highs.append(np.round(p_high, 4))

    return {
        "features": str(prediction_features),
        "prediction_linear": [prediction_linear],
        "prediction_class": pred_class,
        "prediction_prob": [prediction_probs],
        "std_error": [std_errors],
        "z_statistic": [z_statistics],
        "P>|z|": [p_values],
        f"ci_low_{alpha}": [ci_lows],
        f"ci_high_{alpha}": [ci_highs],
    }
