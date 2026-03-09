import numpy as np
from scipy.stats import norm

PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1 - 1e-15


def predict_prob(X, beta, alpha, n_classes):

    n = X.shape[0]
    J = len(alpha)

    cumulative = np.zeros((n, J))

    for j in range(J):

        eta = alpha[j] - X @ beta

        cumulative[:, j] = (
            1 / (1 + np.exp(-np.clip(eta, -700, 700)))
        )

    categorical_pr = (
        np.zeros((n, n_classes))
    )

    categorical_pr[:, 0] = (
        cumulative[:, 0]
    )

    for j in range(1, J):
        categorical_pr[:, j] = cumulative[:, j] - cumulative[:, j-1]

    categorical_pr[:, J] = 1 - cumulative[:, J-1]

    categorical_pr = (
        np.clip(categorical_pr, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    categorical_pr /= (
        categorical_pr.sum(axis=1, keepdims=True)
    )

    return categorical_pr


def predict(model, X, alpha, return_table):

    prediction = predict_prob(
        X,
        model.coefficients,
        model.alpha_cutpoints,
        model.n_classes
    )

    if not return_table:
        return prediction.reshape(-1)

    prediction_features = {
        name: f'{value_at.item():.2f}'
        for name, value_at in zip(model.feature_names, X[0])
    }

    results = []

    for i in range(X.shape[0]):

        p_i = prediction[i]

        pred_class = (
            int(np.argmax(p_i))
        )

        # expected = (
        #    float(np.dot(p_i, np.arange(model.n_classes)))
        # )

        cumulative = np.cumsum(p_i)
        x_i = X[i]

        J = len(model.alpha_cutpoints)
        p = len(model.coefficients)

        def probability_gradient(x, beta, alpha, n_classes):

            grad = np.zeros((n_classes, p + J))
            F = np.zeros(J)
            dF_dbeta = np.zeros(J)
            dF_dalpha = np.zeros((J, J))

            for j in range(J):

                eta = alpha[j] - x @ beta

                F[j] = (
                    1 / (1 + np.exp(-np.clip(eta, -700, 700)))
                )

                f = F[j] * (1 - F[j])  # derivative of logistic

                dF_dbeta[j] = -f
                dF_dalpha[j, j] = f

            grad[0, :p] = dF_dbeta[0] * x
            grad[0, p] = dF_dalpha[0, 0]

            for j in range(1, J):

                grad[j, :p] = (dF_dbeta[j] - dF_dbeta[j-1]) * x
                grad[j, p+j-1] = -dF_dalpha[j-1, j-1]
                grad[j, p+j] = dF_dalpha[j, j]

            grad[J, :p] = -dF_dbeta[J-1] * x
            grad[J, p+J-1] = -dF_dalpha[J-1, J-1]

            return grad

        grad_p = (
            probability_gradient(x_i, model.coefficients,
                                 model.alpha_cutpoints, model.n_classes)
        )

        var_p = (
            grad_p @ model.xtWx_inv @ grad_p.T
        )

        se_p = (
            np.sqrt(np.maximum(np.diag(var_p), 1e-20))
        )

        z_stats = (
            np.where(se_p > 0, p_i / se_p, np.inf)
        )

        p_values = (
            2 * (1 - norm.cdf(np.abs(z_stats)))
        )

        z_critical = (
            norm.ppf(1 - alpha/2)
        )

        ci_low, ci_high = (
            np.clip(p_i - z_critical * se_p, 0, 1),
            np.clip(p_i + z_critical * se_p, 0, 1)
        )

        results.append({
            "features": [prediction_features],
            "prediction_class": pred_class,
            # "prediction_ex": round(expected, 4),
            "cumulative_probabilities": np.round(cumulative, 4).tolist(),
            "prediction_prob": np.round(p_i, 4).tolist(),
            "std_error": np.round(se_p, 4).tolist(),
            "z_statistic": np.round(z_stats, 4).tolist(),
            "P>|z|": [f"{p:.3f}" for p in p_values],
            f"ci_low_{alpha}": np.round(ci_low, 4).tolist(),
            f"ci_high_{alpha}": np.round(ci_high, 4).tolist(),
        })

    return results
