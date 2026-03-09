from analysis.regregresneModely import predict_loess, predict_svr, predict_cart
from analysis.metriky import smape, rmse, mae

def multi_model_chained_predict(
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
        x_loess = x_pred_loess.reshape(-1, 1)
        loess_preds = predict_loess(x_loess, y, frac)

        # --- SVR ---
        x_svr = x_pred_svr.reshape(-1, 1)
        svr_preds = predict_svr(x_svr, y, c, epsilon, gamma)

        # --- CART ---
        x_cart = x_pred_cart.reshape(-1, 1)
        cart_preds = predict_cart(
            x_cart, y,
            max_depth,
            min_samples_split,
            min_samples_leaf
        )

        ###
        col_name = path[i]
        predictions[f"{col_name}_loess"] = loess_preds
        predictions[f"{col_name}_svr"] = svr_preds
        predictions[f"{col_name}_cart"] = cart_preds

        # Chyby
        edge = (path[i - 1], path[i])
        error_metrics[edge] = {
            "rmse": [
                rmse(y, loess_preds),
                rmse(y, svr_preds),
                rmse(y, cart_preds),
            ],
            "mae": [
                mae(y, loess_preds),
                mae(y, svr_preds),
                mae(y, cart_preds),
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
        print(f"\nError metrics for {edge[0]} – {edge[1]}")
        for metric_name, values in metrics.items():
            for model_name, val in zip(models, values):
                if metric_name == 'smape':
                    print(f"{'sMAPE'} ({model_name}): {val:.4f} %")
                else:
                    print(f"{metric_name.upper()} ({model_name}): {val:.4f}")
