import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100


def multi_model_chain_predict(
        df,
        path,
        frac,
        c,
        epsilon,
        gamma,
        max_depth,
        min_samples_leaf,
        min_samples_split):
    predictions = {}
    error_metrics = {}

    x_pred_loess = df[path[0]].values
    x_pred_svr = df[path[0]].values
    x_pred_cart = df[path[0]].values

    for i in range(1, len(path)):
        y = df[path[i]].values

        # --- LOESS ---
        x = x_pred_loess.reshape(-1, 1)
        sort_idx = np.argsort(x.ravel())
        x_sorted = x.ravel()[sort_idx]
        y_sorted = y[sort_idx]

        unique_x = np.unique(x_sorted)

        if len(unique_x) <= 1:
            print(f"\n[Step {i}] Warning: LOESS predictor for '{path[i]}' is constant.")
            print("Predictions will equal target values. LOESS smoothing is not possible.")
        elif len(unique_x) < 10:
            print(f"\n[Step {i}] Warning: LOESS predictor has very few unique values ({len(unique_x)}).")
            print("Suggestion: Increase 'frac' to 0.3–0.5 to improve LOESS stability.")
        with np.errstate(divide='ignore', invalid='ignore'):
            loess_result = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)
        loess_preds = np.empty_like(loess_result)
        loess_preds[sort_idx] = loess_result

        # --- SVR ---
        x = x_pred_svr.reshape(-1, 1)
        # svr = SVR(kernel='rbf')
        # C balances fitting the training data vs. smoothness.
        # Low C → smoother model, risk of underfitting.
        # High C → fits data closely, risk of overfitting.
        # Recommended default range: 1 → 10
        #
        # epsilon defines a “tube” around predictions where no penalty is applied
        # Small epsilon → model tries to predict closely, sensitive to noise.
        # Large epsilon → ignores small errors, smoother predictions.
        # Recommended default range: 0.05 → 0.2
        #
        # gamma controls how far the influence of a single support vector reaches.
        # Small gamma → very smooth, risk of underfitting.
        # Large gamma → very flexible, risk of overfitting.
        # Recommended starting value: 'scale' (automatic, usually robust)
        # Alternative manual values: 0.01, 0.1, 0.5, 1 — useful for very small datasets or high-dimensional features

        svr = SVR(kernel='rbf', C=c, epsilon=epsilon, gamma=gamma)
        svr.fit(x, y)
        svr_preds = svr.predict(x)

        # --- CART ---
        x = x_pred_cart.reshape(-1, 1)
        cart = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42)
        cart.fit(x, y)
        cart_preds = cart.predict(x)

        col_name = path[i]
        predictions[f"{col_name}_loess"] = loess_preds
        predictions[f"{col_name}_svr"] = svr_preds
        predictions[f"{col_name}_cart"] = cart_preds

        # Chyby
        edge = (path[i - 1], path[i])
        error_metrics[edge] = {
            "rmse": [
                np.sqrt(np.mean((y - loess_preds) ** 2)),
                np.sqrt(np.mean((y - svr_preds) ** 2)),
                np.sqrt(np.mean((y - cart_preds) ** 2)),
            ],
            "mae": [
                np.mean(np.abs(y - loess_preds)),
                np.mean(np.abs(y - svr_preds)),
                np.mean(np.abs(y - cart_preds)),
            ],
            "smape": [
                smape(y, loess_preds),
                smape(y, svr_preds),
                smape(y, cart_preds),
            ],
        }

        x_pred_loess = loess_preds
        x_pred_svr = svr_preds
        x_pred_cart = cart_preds

    df_result = df.copy()
    for key, val in predictions.items():
        df_result[key] = val

    return df_result, error_metrics


def print_error_metrics(error_metrics):
    models = ["LOESS", "SVR", "CART"]
    for edge, metrics in error_metrics.items():
        print(f"\nError metrics for {edge[0]} → {edge[1]}")
        for metric_name, values in metrics.items():
            for model_name, val in zip(models, values):
                if metric_name == 'smape':
                    print(f"{'sMAPE'} ({model_name}): {val:.4f} %")
                else:
                    print(f"{metric_name.upper()} ({model_name}): {val:.4f}")
