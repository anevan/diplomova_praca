import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess


def predict_loess(x, y, frac):
    sort_idx = np.argsort(x.ravel())
    x_sorted = x.ravel()[sort_idx]
    y_sorted = y[sort_idx]

    loess_result = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)

    loess_preds = np.empty_like(loess_result)
    loess_preds[sort_idx] = loess_result

    return loess_preds


def predict_svr(x, y, c, epsilon, gamma):
    svr = SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
    svr.fit(x, y)
    return svr.predict(x)


def predict_cart(x, y, max_depth, min_samples_split, min_samples_leaf):
    cart = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    cart.fit(x, y)
    return cart.predict(x)