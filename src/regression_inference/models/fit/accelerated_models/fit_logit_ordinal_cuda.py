import cupy as cp
import numpy as np
from scipy.stats import norm
import warnings

PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1 - 1e-15

# TODO: Work in progress. Results may be unreliable.


def accelerated_ordinal_logit(model, adj_cutpoints: bool, max_iter: int, tol: float) -> None:

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No CUDA devices detected")

    cp.cuda.Stream.null.synchronize()
    device = cp.cuda.Device(0)

    warnings.warn(
        f"\nCUDA Acceleration is Experimental\n"
        f"Device: {str(cp.cuda.runtime.getDeviceProperties(device.id)['name'])[2:][:-1]}\n",
        UserWarning,
        stacklevel=4,
    )

    _, model.n_features = model.X.shape

    model.n_classes = len(cp.unique(model.y))

    if model.n_classes <= 2:
        raise ValueError(
            "Ordinal logit requires 3+ classes. "
            "Use LogisticRegression() for 2 classes."
        )

    model.y_classes = cp.unique(model.y)

    model.y_encoded = (
        cp.searchsorted(model.y_classes, cp.asarray(model.y))
    )

    converged = (
        fit(model, adj_cutpoints, model.y_encoded, max_iter, tol)
    )

    if not converged['success']:
        conv_warn(max_iter, converged['message'])

    y_enc = (
        postfit(model)
    )

    model_params(model, adj_cutpoints, y_enc)


def fit(model, adj_cutpoints: bool, y: cp.ndarray, max_iter: int, tol: float):
    """Newton-Raphson with line search osn cupy array."""

    model.X = (
        cp.atleast_2d(model.X)
    )

    _, n_features = model.X.shape

    J = model.n_classes - 1

    # TODO - There is significant precision loss on scaling during fit, but it is a strong optimizer

    # Standardize features for numerical stability
    # X_mean = cp.mean(model.X, axis=0)
    # X_std = cp.std(model.X, axis=0)
    # X_std = cp.where(X_std < 1e-10, 1.0, X_std)
    # X_scaled = (model.X - X_mean) / X_std

    # Initialization

    start = (
        cp.zeros(n_features + J, dtype=cp.float64)
    )

    class_counts = (
        cp.array([cp.sum(y == k) for k in range(model.n_classes)])
    )

    cumulative_props = (
        cp.cumsum(class_counts[:-1]) / len(y)
    )

    cumulative_props = (
        cp.clip(cumulative_props, 0.01, 0.99)
    )

    start[n_features:] = (
        cp.log(cumulative_props / (1 - cumulative_props))
    )

    res = newton_raphson_optimize(
        negativeLL_gpu,
        gradient_gpu,
        hessian_gpu,
        start,
        args=(model.X, y, model.n_classes),
        max_iter=max_iter,
        tol=tol
    )

    # TODO: Scaling params

    # beta_scaled = res['x'][:n_features]
    # alpha_scaled = res['x'][n_features:]
    # alpha_scaled_sorted = cp.sort(alpha_scaled)

    # Unscale

    # model.coefficients = beta_scaled / X_std
    # intercept_adjustment = cp.sum(X_mean * beta_scaled / X_std)
    # model.alpha_cutpoints = alpha_scaled_sorted + intercept_adjustment

    beta = (
        res['x'][:n_features]
    )

    alpha = (
        res['x'][n_features:]
    )

    alpha_sorted = (
        cp.sort(alpha)
    )

    model.coefficients = beta

    model.alpha_cutpoints = alpha_sorted

    # Ordering?
    if not cp.all(cp.diff(model.alpha_cutpoints) > 0):
        warnings.warn(
            "Cutpoints are not strictly increasing. This may indicate model misspecification.",
            UserWarning
        )

    # TODO: Scaling

    # Compute Hessian at final solution with SCALED data

    # params_sorted = cp.concatenate([beta_scaled, alpha_scaled_sorted])
    # H_scaled = hessian_gpu(params_sorted, X_scaled, y, model.n_classes)

    params_sorted = (
        cp.concatenate([beta, alpha_sorted])
    )

    # Hessian optimized for cupy
    H = (
        hessian_gpu(params_sorted, model.X, y, model.n_classes)
    )

    # Cov matrix scaled

    # try:
    #    cov_scaled = cp.linalg.inv(H_scaled)
    # except:
    #    cov_scaled = cp.linalg.pinv(H_scaled)

    # Unscale

    # scale_matrix = cp.diag(cp.concatenate([1.0 / X_std, cp.ones(J, dtype=cp.float64)]))
    # model.xtWx_inv = scale_matrix @ cov_scaled @ scale_matrix

    try:
        model.xtWx_inv = cp.linalg.inv(H)
    except:
        model.xtWx_inv = cp.linalg.pinv(H)

    # Statsmodels like method

    model.theta_cutpoints = (
        cp.empty_like(model.alpha_cutpoints)
    )

    model.theta_cutpoints[0] = model.alpha_cutpoints[0]

    diffs = cp.diff(model.alpha_cutpoints)

    diffs = cp.maximum(diffs, 1e-10)

    model.theta_cutpoints[1:] = cp.log(diffs)

    model.theta = (
        cp.concatenate([model.coefficients, model.theta_cutpoints])
        if adj_cutpoints else
        # MASS:polr like method
        cp.concatenate([model.coefficients, model.alpha_cutpoints])
    )

    return res


def newton_raphson_optimize(
    fun,
    grad_fun,
    hess_fun,
    x0,
    args=(),
    max_iter=100,
    tol=1e-5
):
    """Newton-Raphson optimization with damped updates and line search."""

    x = (
        cp.asarray(x0, dtype=cp.float64)
    )

    success = False
    message = "Maximum iterations reached"

    for iteration in range(max_iter):

        # Compute gradient and Hessian
        g = (
            grad_fun(x, *args)
        )

        H = (
            hess_fun(x, *args)
        )

        # Check convergence
        grad_norm = cp.linalg.norm(g)

        if grad_norm < tol:
            success = True
            message = "Optimization converged"
            break

        # Add damping for stability
        damping = (
            1e-6 * cp.eye(len(x), dtype=cp.float64)
        )

        H_damped = (
            H + damping
        )

        # Solve for Newton direction
        try:
            direction = -cp.linalg.solve(H_damped, g)

        except:
            # Fallback to gradient descent if Hessian is singular
            direction = -g / (cp.linalg.norm(g) + 1e-10)

        # Line search with backtracking
        alpha = (
            backtracking_line_search(fun, x, direction, g, args)
        )

        x_new = x + alpha * direction

        if not cp.all(cp.isfinite(x_new)):
            message = "Numerical issues encountered"
            break

        x = x_new

    f = fun(x, *args)

    return {
        'x': x,
        'fun': f,
        'success': success,
        'message': message,
        'nit': iteration + 1
    }


def backtracking_line_search(fun, x, direction, gradient, args, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=30):
    """Backtracking line search with Armijo condition."""

    alpha = (
        alpha_init
    )

    f_current = (
        fun(x, *args)
    )

    grad_dot_dir = (
        cp.dot(gradient, direction)
    )

    for _ in range(max_iter):

        x_new = (
            x + alpha * direction
        )

        f_new = (
            fun(x_new, *args)
        )

        # Armijo condition
        if f_new <= f_current + c * alpha * grad_dot_dir:
            return alpha

        alpha *= rho

    return alpha


def negativeLL_gpu(params, X, y, n_classes):

    params = (
        cp.asarray(params, dtype=cp.float64)
    )

    X = (
        cp.asarray(X, dtype=cp.float64)
    )

    y = (
        cp.asarray(y, dtype=cp.int32)
    )

    n, p = X.shape
    J = n_classes - 1

    beta = params[:p]
    alpha = params[p:]

    # Linear predictor
    eta = X @ beta

    # Cumulative probabilities using stable logistic
    z = alpha[:, cp.newaxis] - eta[cp.newaxis, :]       # Shape: (J, n)

    cumulative = (
        1.0 / (1.0 + cp.exp(-cp.clip(z, -500, 500)))
    )

    cumulative = cumulative.T

    # Category probabilities

    probs = (
        cp.zeros((n, n_classes), dtype=cp.float64)
    )

    probs[:, 0] = cumulative[:, 0]

    for j in range(1, J):
        probs[:, j] = cumulative[:, j] - cumulative[:, j-1]

    probs[:, -1] = 1.0 - cumulative[:, -1]

    # Normalization
    probs = (
        cp.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    probs = (
        probs / probs.sum(axis=1, keepdims=True)
    )

    ll = (
        cp.sum(cp.log(probs[cp.arange(n), y]))
    )

    return -ll  # As cupy scalar


def gradient_gpu(params, X, y, n_classes):

    params = (
        cp.asarray(params, dtype=cp.float64)
    )

    X = (
        cp.asarray(X, dtype=cp.float64)
    )

    y = (
        cp.asarray(y, dtype=cp.int32)
    )

    n, p = X.shape
    J = n_classes - 1

    beta = params[:p]
    alpha = params[p:]

    eta = X @ beta

    # cumulative probabilities and densities

    z = (
        alpha[:, cp.newaxis] - eta[cp.newaxis, :]
    )

    z_clipped = (
        cp.clip(z, -500, 500)
    )

    exp_neg_z = (
        cp.exp(-z_clipped)
    )

    cumulative = (
        1.0 / (1.0 + exp_neg_z)
    )

    densities = (
        exp_neg_z / ((1.0 + exp_neg_z) ** 2)
    )

    cumulative = cumulative.T  # Shape: (n, J)

    densities = densities.T  # Shape: (n, J)

    # Category probabilities
    probs = (
        cp.zeros((n, n_classes), dtype=cp.float64)
    )

    probs[:, 0] = cumulative[:, 0]

    for j in range(1, J):
        probs[:, j] = cumulative[:, j] - cumulative[:, j-1]

    probs[:, -1] = (
        1.0 - cumulative[:, -1]
    )

    probs = (
        cp.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    # probabilities for observed classes
    pi = probs[cp.arange(n), y]  # Shape: (n,)

    # gradient contributions
    grad_beta = (
        cp.zeros(p, dtype=cp.float64)
    )

    grad_alpha = (
        cp.zeros(J, dtype=cp.float64)
    )

    # First category (yi == 0)
    mask_0 = (y == 0)

    if cp.any(mask_0):

        weights_0 = densities[mask_0, 0] / pi[mask_0]  # Shape: (n_0,)

        grad_beta += cp.sum((weights_0[:, cp.newaxis] * X[mask_0]), axis=0)

        grad_alpha[0] -= cp.sum(weights_0)

    # Last category (yi == n_classes - 1)
    mask_last = (y == n_classes - 1)

    if cp.any(mask_last):

        weights_last = densities[mask_last, J-1] / \
            pi[mask_last]  # Shape: (n_last,)

        grad_beta -= cp.sum((weights_last[:,
                            cp.newaxis] * X[mask_last]), axis=0)

        grad_alpha[J-1] += cp.sum(weights_last)

    # Middle categories (0 < yi < n_classes - 1)
    for k in range(1, n_classes - 1):

        mask_k = (y == k)

        if cp.any(mask_k):

            # Shape: (n_k,)
            weights_k = (densities[mask_k, k] -
                         densities[mask_k, k-1]) / pi[mask_k]

            grad_beta += cp.sum((weights_k[:, cp.newaxis] * X[mask_k]), axis=0)

            grad_alpha[k-1] += cp.sum(densities[mask_k, k-1] / pi[mask_k])

            grad_alpha[k] -= cp.sum(densities[mask_k, k] / pi[mask_k])

    grad = (
        cp.concatenate([grad_beta, grad_alpha])
    )

    return grad


def hessian_gpu(params, X, y, n_classes):

    params = (
        cp.asarray(params, dtype=cp.float64)
    )

    X = (
        cp.asarray(X, dtype=cp.float64)
    )

    y = (
        cp.asarray(y, dtype=cp.int32)
    )

    n, p = X.shape
    J = n_classes - 1
    dim = p + J

    beta = params[:p]
    alpha = params[p:]

    eta = X @ beta

    # cumulative probabilities and densities
    z = (
        alpha[:, cp.newaxis] - eta[cp.newaxis, :]
    )

    z_clipped = (
        cp.clip(z, -500, 500)
    )

    exp_neg_z = (
        cp.exp(-z_clipped)
    )

    cumprobs = (
        1.0 / (1.0 + exp_neg_z)
    )

    densities = (
        exp_neg_z / ((1.0 + exp_neg_z) ** 2)
    )

    cumprobs = cumprobs.T

    densities = densities.T

    probs = (
        cp.zeros((n, n_classes), dtype=cp.float64)
    )

    probs[:, 0] = (
        cumprobs[:, 0]
    )

    for j in range(1, J):
        probs[:, j] = cumprobs[:, j] - cumprobs[:, j-1]

    probs[:, -1] = 1.0 - cumprobs[:, -1]

    probs = (
        cp.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    # observed classes
    pi = probs[cp.arange(n), y]  # Shape: (n,)

    # gradient for each observation
    g_batch = (
        cp.zeros((n, dim), dtype=cp.float64)
    )

    # First category (yi == 0)
    mask_0 = (y == 0)

    if cp.any(mask_0):

        weights_0 = densities[mask_0, 0] / pi[mask_0]  # Shape: (n_0,)

        g_batch[mask_0, :p] = weights_0[:, cp.newaxis] * X[mask_0]

        g_batch[mask_0, p] = -weights_0

    # Last category (yi == n_classes - 1)
    mask_last = (y == n_classes - 1)

    if cp.any(mask_last):

        weights_last = densities[mask_last, J-1] / \
            pi[mask_last]  # Shape: (n_last,)

        g_batch[mask_last, :p] = -weights_last[:, cp.newaxis] * X[mask_last]

        g_batch[mask_last, p + J - 1] = weights_last

    # Middle categories
    for k in range(1, n_classes - 1):

        mask_k = (y == k)

        if cp.any(mask_k):

            # Shape: (n_k,)
            weights_k = (densities[mask_k, k] -
                         densities[mask_k, k-1]) / pi[mask_k]

            g_batch[mask_k, :p] = weights_k[:, cp.newaxis] * X[mask_k]

            g_batch[mask_k, p + k - 1] = densities[mask_k, k-1] / pi[mask_k]

            g_batch[mask_k, p + k] = -densities[mask_k, k] / pi[mask_k]

    # Compute Hessian as sum of outer products
    # H = sum_i (g_i @ g_i.T) = G.T @ G where G is (n, dim)
    H = (
        g_batch.T @ g_batch
    )

    return H


def predict(X: cp.ndarray, beta: cp.ndarray, alpha: cp.ndarray, n_classes: int) -> cp.ndarray:

    n = X.shape[0]
    J = len(alpha)

    eta = X @ beta

    z = (
        alpha[:, cp.newaxis] - eta[cp.newaxis, :]
    )

    cumulative = (
        1 / (1 + cp.exp(-cp.clip(z, -500, 500)))
    )

    cumulative = cumulative.T

    categorical_pr = (
        cp.zeros((n, n_classes), dtype=cp.float64)
    )

    categorical_pr[:, 0] = cumulative[:, 0]

    for j in range(1, J):
        categorical_pr[:, j] = cumulative[:, j] - cumulative[:, j-1]

    categorical_pr[:, -1] = 1 - cumulative[:, -1]

    categorical_pr = (
        cp.clip(categorical_pr, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    categorical_pr = (
        categorical_pr / categorical_pr.sum(axis=1, keepdims=True)
    )

    return categorical_pr


def postfit(model) -> cp.ndarray:

    model.probabilities = predict(
        model.X,
        model.coefficients,
        model.alpha_cutpoints,
        model.n_classes
    )

    model.predictions = (
        cp.argmax(model.probabilities, axis=1)
    )

    model.classification_accuracy = (
        cp.mean(model.predictions == model.y_encoded)
    )

    n = len(model.y_encoded)

    y_onehot = (
        cp.zeros((n, model.n_classes), dtype=cp.float64)
    )

    y_onehot[cp.arange(n), model.y_encoded] = 1

    return y_onehot


def model_params(model, adj_cutpoints: bool, y_enc: cp.ndarray) -> None:

    y_hat_prob = (
        cp.clip(model.probabilities, PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    model.log_likelihood = (
        cp.sum(y_enc * cp.log(y_hat_prob))
    )

    model.deviance = (
        -2 * model.log_likelihood
    )

    n_samples, _ = y_enc.shape

    class_probabilities = (
        cp.clip(cp.mean(y_enc, axis=0), PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    model.null_log_likelihood = (
        cp.sum(y_enc * cp.log(class_probabilities))
    )

    model.null_deviance = (
        -2 * model.null_log_likelihood
    )

    n_params = (
        model.n_features + (model.n_classes - 1)
    )

    model.aic = (
        -2 * model.log_likelihood + 2 * n_params
    )

    model.bic = (
        -2 * model.log_likelihood + n_params * cp.log(n_samples)
    )

    model.pseudo_r_squared = (
        1 - (model.log_likelihood / model.null_log_likelihood)
    )

    model.lr_statistic = (
        -2 * (model.null_log_likelihood - model.log_likelihood)
    )

    if adj_cutpoints:

        # Transform covariance for adjusted cutpoints
        model.variance_coefficient = (
            transform_covariance(model)
        )

        model.std_error_coefficient = (
            cp.sqrt(cp.maximum(cp.diag(model.variance_coefficient), 1e-20))
        )

    else:

        # Use covariance directly for alpha parameterization
        model.variance_coefficient = model.xtWx_inv

        model.std_error_coefficient = (
            cp.sqrt(cp.maximum(cp.diag(model.xtWx_inv), 1e-20))
        )

    model.z_stat_coefficient = (
        model.theta / cp.maximum(model.std_error_coefficient, 1e-10)
    )

    z_stats_np = (
        cp.asnumpy(cp.abs(model.z_stat_coefficient))
    )

    model.p_value_coefficient = (
        cp.asarray(2 * (1 - norm.cdf(z_stats_np)))
    )

    z_crit = norm.ppf(1 - model.alpha / 2)

    model.ci_low, model.ci_high = (
        model.theta - z_crit * model.std_error_coefficient,
        model.theta + z_crit * model.std_error_coefficient
    )

    predicted_class = (
        cp.argmax(y_hat_prob, axis=1)
    )

    actual_class = (
        model.y_encoded
    )

    model.residuals = (
        (actual_class != predicted_class).astype(cp.float64)
    )

    for attr in vars(model):
        value = getattr(model, attr)

        if isinstance(value, cp.ndarray):
            setattr(model, attr, value.get())


def transform_covariance(model) -> cp.ndarray:
    """
    Transform covariance matrix from alpha to theta parameterization.
    Uses delta method: Var(g(theta)) = g'(theta) * Var(theta) * g'(theta)^T

    Where theta[j] = log(alpha[j] - alpha[j-1]) for j > 0
          theta[0] = alpha[0]
    """

    p = len(model.coefficients)
    J = len(model.alpha_cutpoints)
    V_alpha = model.xtWx_inv  # Covariance in alpha parameterization

    dim = p + J

    # Jacobian matrix: d(beta, theta) / d(beta, alpha)
    G = cp.zeros((dim, dim), dtype=cp.float64)

    # Beta part (no transformation)
    G[:p, :p] = cp.eye(p, dtype=cp.float64)

    # Theta cutpoint transformations
    # theta[0] = alpha[0]  =>  d(theta[0])/d(alpha[0]) = 1
    G[p, p] = 1.0

    # For j >= 1: theta[j] = log(alpha[j] - alpha[j-1])
    # d(theta[j])/d(alpha[j]) = 1 / (alpha[j] - alpha[j-1])
    # d(theta[j])/d(alpha[j-1]) = -1 / (alpha[j] - alpha[j-1])

    for j in range(1, J):

        denom = (
            model.alpha_cutpoints[j] - model.alpha_cutpoints[j-1]
        )

        denom = cp.maximum(denom, 1e-10)  # Protect against small differences

        G[p + j, p + j] = 1.0 / denom          # d/d(alpha[j])

        G[p + j, p + j - 1] = -1.0 / denom     # d/d(alpha[j-1])

    # Apply delta method: Var(theta) = G * Var(alpha) * G^T
    V_theta = G @ V_alpha @ G.T

    return V_theta


def conv_warn(max_iter: int, message: str = ""):
    warnings.warn(
        f"\nOptimization did not converge after {max_iter} iterations.\n"
        f"Optimizer message: {message}\n"
        f"Consider:\n"
        f"- Increasing max_iter\n"
        f"- Adjusting tolerance\n"
        f"- Scaling features\n"
        f"- Checking for separation issues\n"
        f"- Ensuring sufficient samples per class\n",
        UserWarning,
        stacklevel=5
    )
