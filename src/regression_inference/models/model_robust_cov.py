import numpy as np
from scipy.stats import t as t_dist, norm


def robust_se(model, type, apply=False):

    n, k = model.X.shape

    if model.model_type == "linear":

        '''
        LinearRegression()

            - Leverage and squared residuals

        '''

        stat_dist, stat_name = (t_dist, 't')

        XTX_INV = model.xtx_inv

        h = (
            np.sum(model.X @ XTX_INV * model.X, axis=1)
        )

        sr = (
            model.residuals.reshape(-1, 1).flatten()**2
        )

    elif model.model_type == "logit":

        '''
        LogisticRegression()

            - Leverage and squared residuals

        '''
        def sigmoid(z):
            return np.where(
                z >= 0,
                1 / (1 + np.exp(-z)),
                np.exp(z) / (1 + np.exp(z))
            )

        stat_dist, stat_name = (norm, 'z')

        mu = (
            sigmoid(model.X @ model.theta)
        )

        W = (
            mu * (1 - mu)
        )

        XTX_INV = model.xtWx_inv

        h = (
            np.sum(model.X @ XTX_INV * model.X, axis=1) * W
        )

        response_residuals = model.y - mu

        sr = response_residuals**2

    elif model.model_type == "logit_multinomial":

        '''
        MultinomialLogisticRegression()

            - Leverage, non-scalar

        '''

        stat_dist, stat_name = (norm, 'z')

        J = model.theta.shape[1]

        Y_onehot = np.zeros((n, J+1))

        Y_onehot[np.arange(n), model.y_encoded] = 1

        Y = Y_onehot[:, 1:]

        P = model.probabilities[:, 1:]

        residuals = Y - P

        XTX_INV = model.xtWx_inv

        h = (
            np.zeros(n)
            if type in ["HC0", "HC1"] else
            None
        )

        sr = None

    else:
        residuals = []
        raise ValueError(f"Model type: {model.model_type} is unexpected.")

    # Scalar residual models only

    if model.model_type in ["linear", "logit"]:

        '''
        Robust standard errors for LinearRegression() and LogisticRegression()
        '''

        feature_names = model.feature_names

        HC_ = {
            "HC0": lambda sr, n_obs, k_regressors, leverage: sr,
            "HC1": lambda sr, n_obs, k_regressors, leverage: (n_obs / (n_obs - k_regressors)) * sr,
            "HC2": lambda sr, n_obs, k_regressors, leverage: sr / (1 - leverage),
            "HC3": lambda sr, n_obs, k_regressors, leverage: sr / ((1 - leverage) ** 2),
        }

        '''Compute the diagonal only'''
        try:

            omega_diagonal = (
                HC_[type](sr, n, k, h)
            )

            X_omega = (
                model.X * np.sqrt(omega_diagonal)[:, None]
            )
            # Multiply each model.X row by X*(diagonal weights)^(0.5)
            robust_cov = (
                XTX_INV @ (X_omega.T @ X_omega) @ XTX_INV
            )                                                   # Sandwich

            robust_se = (
                np.sqrt(np.diag(robust_cov))
            )                                                   # Diagonal extract the var-cov

            robust_stat = (
                model.theta / robust_se
            )

            if model.model_type == "linear":

                robust_p = (
                    2 * (1 - stat_dist.cdf(abs(robust_stat), model.degrees_freedom))
                )

                crit = (
                    stat_dist.ppf(1 - model.alpha/2, model.degrees_freedom)
                )

            elif model.model_type == "logit":

                robust_p = (
                    2 * (1 - stat_dist.cdf(abs(robust_stat)))
                )

                crit = (
                    stat_dist.ppf(1 - model.alpha/2)
                )
            else:
                crit = np.array([])
                robust_p = np.array([])

            robust_ci_low, robust_ci_high = (
                model.theta - crit * robust_se,
                model.theta + crit * robust_se
            )

        except ValueError:
            raise ValueError("Select 'HC0', 'HC1', 'HC2', 'HC3'")

    elif model.model_type in ["logit_multinomial"]:

        '''
        Robust standard errors for MultinomialLogisticRegression()
        '''

        feature_names = (
            list(model.feature_names)*((np.unique(model.y).shape[0])-1)
        )

        p = model.X.shape[1]
        J = model.theta.shape[1]

        omega = np.zeros((p*J, p*J))

        HC__ = {
            "HC0": 1.0,
            "HC1": n / (n - p*J),
            "HC2": None,
            "HC3": None,
        }

        try:
            scale = HC__[type]

            if scale is None:
                raise ValueError(f"{type} not supported for {model.model_type}.")

            for i in range(n):

                Xi = model.X[i][:, None]

                e_i = residuals[i]

                score = np.kron(e_i, Xi.flatten())

                omega += np.outer(score, score)

            omega *= scale

            robust_cov = (
                XTX_INV @ omega @ XTX_INV
            )

            robust_se_flat = (
                np.sqrt(np.maximum(np.diag(robust_cov), 1e-20))
            )

            theta_flat = (
                model.theta.flatten(order='F')
            )

            robust_stat_flat = (
                theta_flat / robust_se_flat
            )

            robust_p_flat = (
                2 * (1 - stat_dist.cdf(np.abs(robust_stat_flat)))
            )

            crit = (
                stat_dist.ppf(1 - model.alpha/2)
            )

            robust_ci_low_flat, robust_ci_high_flat = (
                theta_flat - crit * robust_se_flat,
                theta_flat + crit * robust_se_flat
            )

            shape = model.theta.shape

            robust_se = (
                robust_se_flat.reshape(shape, order='F')
            )

            robust_stat = (
                robust_stat_flat.reshape(shape, order='F')
            )

            robust_p = (
                robust_p_flat.reshape(shape, order='F')
            )

            robust_ci_low = (
                robust_ci_low_flat.reshape(shape, order='F')
            )

            robust_ci_high = (
                robust_ci_high_flat.reshape(shape, order='F')
            )

        except KeyError:
            raise ValueError("Select 'HC0', 'HC1', 'HC2', 'HC3'")

    else:
        raise ValueError(f"Model type: {model.model_type} is unexpected.")

    '''If 'cov_type' is set on model fitting update the covariance before attributes are frozen. '''

    if apply:

        if model.model_type == "linear":
            model.t_stat_coefficient = robust_stat

        elif model.model_type in ["logit", 'logit_multinomial']:
            model.z_stat_coefficient = robust_stat

        model.variance_coefficient = robust_cov
        model.std_error_coefficient = robust_se
        model.p_value_coefficient = robust_p
        model.ci_low = robust_ci_low
        model.ci_high = robust_ci_high

    return {
        "feature":                    feature_names,
        "robust_se":                  robust_se.flatten(order="F") if model.model_type == "logit_multinomial" else robust_se,
        f"robust_{stat_name}":        robust_stat.flatten(order="F") if model.model_type == "logit_multinomial" else robust_stat,
        "robust_p":                   robust_p.flatten(order="F") if model.model_type == "logit_multinomial" else robust_p,
        f"ci_low_{model.alpha}":      robust_ci_low.flatten(order="F") if model.model_type == "logit_multinomial" else robust_ci_low,
        f"ci_high_{model.alpha}":     robust_ci_high.flatten(order="F") if model.model_type == "logit_multinomial" else robust_ci_high,
    }
