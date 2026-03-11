import numpy as np
import cupy as cp
import warnings
from scipy.stats import norm

COND_THRESHOLD = 1e10
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1 - 1e-15


def softmax(Z: cp.ndarray) -> cp.ndarray:

    Z_stable = Z - cp.max(Z, axis=1, keepdims=True)

    exp_Z = cp.exp(cp.clip(Z_stable, -700, 700))

    return exp_Z / cp.sum(exp_Z, axis=1, keepdims=True)


def accelerated_multinomial_logit(model, max_iter: int, tol: float) -> None:

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

    n_samples, model.n_features = model.X.shape

    model.n_classes = len(cp.unique(model.y))

    if model.n_classes <= 2:
        raise ValueError(
            "Multinomial logit requires 3+ classes. "
            "Use LogisticRegression() for 2 classes."
        )

    model.y_classes = cp.unique(model.y)

    model.y_encoded = (
        cp.searchsorted(model.y_classes, cp.asarray(model.y))
    )

    y_onehot = (
        cp.zeros((n_samples, model.n_classes))
    )

    y_onehot[cp.arange(n_samples), model.y_encoded] = 1

    standard_fit(model, y_onehot, n_samples, max_iter, tol)


def standard_fit(model, y_enc: cp.ndarray, n_samples: int, max_iter: int, tol: float) -> None:

    model.X = cp.asarray(model.X)

    p = model.n_features

    J = model.n_classes - 1

    Y = y_enc[:, 1:]

    theta = cp.zeros((p, J))

    for _ in range(max_iter):

        Z = model.X @ theta

        Z_full = cp.column_stack([cp.zeros(n_samples), Z])

        P = softmax(Z_full)

        Pj = P[:, 1:]

        grad = model.X.T @ (Y - Pj)

        grad_flat = grad.flatten(order="F")

        H = standard_hessian(model.X, P)

        try:
            step = cp.linalg.solve(H, grad_flat)
        except cp.linalg.LinAlgError:
            step = cp.linalg.pinv(H) @ grad_flat

        theta_new = theta + step.reshape(p, J, order="F")

        if cp.max(cp.abs(theta_new - theta)) < tol:
            break

        theta = theta_new

    model.theta = theta

    model.probabilities = (
        predict_prob(model.X, model.theta)
    )

    model.predictions = (
        cp.argmax(model.probabilities, axis=1)
    )

    model.classification_accuracy = (
        cp.mean(model.predictions == model.y_encoded)
    )

    H = standard_hessian(model.X, model.probabilities)

    cond = np.linalg.cond(H.get())
    if cond > COND_THRESHOLD:
        cond_warn(cond)

    try:

        model.xtWx_inv = cp.linalg.inv(H)

    except cp.linalg.LinAlgError:
        warnings.warn(
            "Hessian is singular, using pseudo-inverse. Standard errors may be unreliable.",
            UserWarning
        )

        model.xtWx_inv = cp.linalg.pinv(H)

    model.intercept = model.theta[0, :]

    model.coefficients = model.theta[1:, :]

    model_params(model, y_enc)


def standard_hessian(X: cp.ndarray, probs: cp.ndarray) -> cp.ndarray:

    _, n_features = X.shape

    n_classes = probs.shape[1]

    n_alt = n_classes - 1

    P = probs[:, 1:]  # shape: (n_samples, n_alt)

    H = (
        cp.zeros((n_features * n_alt, n_features * n_alt))
    )

    for j in range(n_alt):

        for k in range(n_alt):

            # (j,k) contribution
            if j == k:

                # Diagonal block: X^T diag(p_j * (1 - p_j)) X
                w = P[:, j] * (1 - P[:, j])

            else:

                # Off-diagonal block: -X^T diag(p_j * p_k) X
                w = -P[:, j] * P[:, k]

            X_weighted = X.T * w    # (n_features, n_samples)

            block = X_weighted @ X  # (n_features, n_features)

            # Place in H
            H[j*n_features:(j+1)*n_features,
              k*n_features:(k+1)*n_features] = block

    return H


def predict_prob(X: cp.ndarray, theta: cp.ndarray) -> cp.ndarray:

    n_samples = X.shape[0]

    Z = X @ theta

    Z_full = cp.column_stack([cp.zeros(n_samples), Z])

    return softmax(Z_full)


def model_params(model, y_enc: cp.ndarray):

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

    class_probs = (
        cp.clip(cp.mean(y_enc, axis=0), PROB_CLIP_MIN, PROB_CLIP_MAX)
    )

    model.null_log_likelihood = (
        cp.sum(y_enc * cp.log(class_probs))
    )

    model.null_deviance = (
        -2 * model.null_log_likelihood
    )

    n_params = model.theta.size

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

    model.variance_coefficient = model.xtWx_inv

    model.std_error_coefficient = (
        cp.sqrt(cp.maximum(cp.diag(model.variance_coefficient), 1e-20))
    )

    theta_flat = model.theta.flatten(order="F")

    model.z_stat_coefficient = (
        theta_flat / model.std_error_coefficient
    )

    model.p_value_coefficient = (
        2 * (1 - norm.cdf(np.abs(model.z_stat_coefficient.get())))
    )

    z_crit = (
        norm.ppf(1 - model.alpha / 2)
    )

    model.ci_low, model.ci_high = (
        theta_flat - z_crit * model.std_error_coefficient,
        theta_flat + z_crit * model.std_error_coefficient
    )

    # Reshape step for correct ordering by theta_flat

    shape = model.theta.shape

    model.std_error_coefficient = (
        model.std_error_coefficient.reshape(shape, order="F")
    )

    model.z_stat_coefficient = (
        model.z_stat_coefficient.reshape(shape, order="F")
    )

    model.p_value_coefficient = (
        model.p_value_coefficient.reshape(shape, order="F")
    )

    model.ci_low = (
        model.ci_low.reshape(shape, order="F")
    )

    model.ci_high = (
        model.ci_high.reshape(shape, order="F")
    )

    predicted_class = cp.argmax(y_hat_prob, axis=1)

    model.residuals = (model.y_encoded != predicted_class).astype(float)

    for attr in vars(model):
        value = getattr(model, attr)

        if isinstance(value, cp.ndarray):
            setattr(model, attr, value.get())


def cond_warn(cond: float):
    warnings.warn(
        f"\nHessian matrix is ill-conditioned (cond={cond:.2e}).\n"
        f"Results may be unreliable. Consider:\n"
        f"- Removing collinear features\n"
        f"- Scaling features\n"
        f"- Increasing sample size per class\n",
        UserWarning,
        stacklevel=5
    )


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


def separation_warn(max_coef: float):
    warnings.warn(
        f"\nLarge coefficients detected (max |θ| = {max_coef:.2f}).\n"
        f"This may indicate separation in the data:\n"
        f"- Perfect or quasi-complete separation\n"
        f"- Classes are linearly separable\n"
        f"- Standard errors may be unreliable\n",
        UserWarning,
        stacklevel=5
    )
