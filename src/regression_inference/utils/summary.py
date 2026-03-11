import numpy as np


def summary(*args):

    col_width, col_span, models = (
        15,
        30,
        list(args)
    )
    for i, model in enumerate(models):
        if model.theta is None:
            raise ValueError(f"Error: Model {i+1} is not fitted.")

    if len(set(m.model_type for m in models)) > 1:
        raise ValueError("Error: Cannot stack different model types.")

    format_length = col_span + (len(models)*col_width)

    TITLE_MAP = {
        "linear": "OLS Regression Results",
        "logit": "Logistic Regression Results",
        "logit_multinomial": "Multinomial Regression Results",
        "logit_ordinal": "Ordinal Regression Results",
    }

    model_type = models[0].model_type

    try:
        title = TITLE_MAP[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")

    header = (
        f"{'=' * format_length}\n"
        f"{title}\n"
        f"{'-' * format_length}\n"
        f"{'Dependent:':<{col_span}}"
        + "".join(f"{m.target:>{col_width}}" for m in models)
        + "\n"
        f"{'-' * format_length}\n"
    )

    old_feature_names = []
    all_features = []
    for model in models:

        if model.model_type == "logit_ordinal":

            old_feature_names.append(model.feature_names)

            p = len(model.feature_names)
            remainder = model.theta.shape[0] - p

            model.feature_names = np.concatenate([
                np.array(model.feature_names),
                np.array([f"{i}:{i+1}" for i in range(remainder)])
            ])

        for feature in model.feature_names:
            if feature not in all_features:
                all_features.append(feature)

    rows = []

    if not models[0].model_type == "logit_multinomial":
        for feature in all_features:
            coef_row = f"{feature:<{col_span}}"
            se_row = " " * col_span

            for model in models:
                if feature in model.feature_names:
                    feature_index = list(model.feature_names).index(feature)
                    coef = model.theta[feature_index]
                    se = model.std_error_coefficient[feature_index]
                    p = model.p_value_coefficient[feature_index]

                    stars = (
                        "***" if p < 0.01 else
                        "**" if p < 0.05 else
                        "*" if p < 0.1 else
                        ""
                    )
                    coef_fmt = (
                        f"{coef:.4f}{stars}"
                        if abs(coef) > 0.0001
                        else f"{coef:.2e}{stars}"
                    )
                    se_fmt = (
                        f"({se:.4f})"
                        if abs(se) > 0.0001
                        else f"({se:.2e})"
                    )
                    coef_row += f"{coef_fmt:>{col_width}}"
                    se_row += f"{se_fmt:>{col_width}}"
                else:
                    coef_row += " " * col_width
                    se_row += " " * col_width

            rows.append(" ")
            rows.append(coef_row)
            rows.append(se_row)
    else:
        J = models[0].theta.shape[1]  # number of non-reference classes

        for j in range(J):

            col_num = int(models[0].y_classes[j+1])
            rows.append(f"{'Class:':<{col_span}}{col_num:>{col_width}}\n")

            for feature in all_features:
                coef_row = f"{feature:<{col_span}}"
                se_row = " " * col_span

                for model in models:
                    i = list(model.feature_names).index(feature)

                    coef = model.theta[i, j]
                    se = model.std_error_coefficient[i, j]
                    p = model.p_value_coefficient[i, j]

                    stars = (
                        "***" if p < 0.01 else
                        "**" if p < 0.05 else
                        "*" if p < 0.1 else
                        ""
                    )
                    coef_fmt = (
                        f"{coef:.4f}{stars}"
                        if abs(coef) > 0.0001
                        else f"{coef:.2e}{stars}"
                    )
                    se_fmt = (
                        f"({se:.4f})"
                        if abs(se) > 0.0001
                        else f"({se:.2e})"
                    )
                    coef_row += f"{coef_fmt:>{col_width}}"
                    se_row += f"{se_fmt:>{col_width}}"

                rows.append(coef_row)
                rows.append(se_row)
                rows.append(" ")

            rows.append(f"{'-'*format_length}")

    if models[0].model_type == "linear":
        stats_lines = [
            ("R-squared", "r_squared"),
            ("Adjusted R-squared", "r_squared_adjusted"),
            ("F Statistic", "f_statistic"),
            ("Observations", lambda m: m.X.shape[0]),
            ("Log Likelihood", "log_likelihood"),
            ("AIC", "aic"),
            ("BIC", "bic"),
            ("TSS", "tss"),
            ("RSS", "rss"),
            ("ESS", "ess"),
            ("MSE", "mse"),
        ]

    elif models[0].model_type in ["logit", "logit_multinomial", "logit_ordinal"]:
        stats_lines = [
            ("Accuracy", "classification_accuracy"),
            ("Pseudo R-squared", "pseudo_r_squared"),
            ("LR Statistic", "lr_statistic"),
            ("Observations", lambda m: m.X.shape[0]),
            ("Log Likelihood", "log_likelihood"),
            ("Null Log Likelihood", "null_log_likelihood"),
            ("Deviance", "deviance"),
            ("Null Deviance", "null_deviance"),
            ("AIC", "aic"),
            ("BIC", "bic")
        ]
    else:
        stats_lines = ""

    stats = f"\n{'-'*format_length}\n"

    for label, attr in stats_lines:
        stat_row = f"{label:<{col_span}}"
        for model in models:
            stat_row += f"{(attr(model) if callable(attr) else getattr(model, attr)):>{col_width}.3f}"
        stats += stat_row + "\n"

    # Reset feature names to remove the ordinal
    if models[0].model_type == "logit_ordinal":
        for i, model in enumerate(models):
            model.feature_names = old_feature_names[i]

    return (
        header +
        "\n".join(rows) + "\n" +
        stats +
        f"{'='*format_length}\n"
        "*p<0.1; **p<0.05; ***p<0.01\n"
    )

