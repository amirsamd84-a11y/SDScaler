import numpy as np
import pandas as pd

from .helpers import pick_method

METHODS = ("auto", "standard", "minmax", "robust")


class SDScaler:
    """
    basically StandardScaler/MinMaxScaler/RobustScaler but it picks one
    for you per column instead of you having to guess which one to use.

    set method to "standard"/"minmax"/"robust" if you want to force one
    instead of letting it decide (default is "auto")
    """

    def __init__(self, method="auto", outlier_cutoff=0.05, skew_cutoff=1.0):
        if method not in METHODS:
            raise ValueError(f"method has to be one of {METHODS}, got '{method}'")

        self.method = method
        self.outlier_cutoff = outlier_cutoff
        self.skew_cutoff = skew_cutoff

        self.stats_ = {}
        self.columns_ = None
        self.is_df = False
        self.fitted = False

    def fit(self, X):
        self.is_df = isinstance(X, pd.DataFrame)

        if self.is_df:
            self.columns_ = list(X.columns)
            data = X.values
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.columns_ = list(range(X.shape[1]))
            data = X

        self.stats_ = {}
        for i, col in enumerate(self.columns_):
            col_data = data[:, i]
            method = pick_method(col_data, self.skew_cutoff, self.outlier_cutoff) \
                if self.method == "auto" else self.method
            self.stats_[col] = self._fit_one(col_data, method)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("call fit() before transform()")

        if self.is_df:
            out = X.copy()
            for col in self.columns_:
                out[col] = self._apply_one(X[col].values, self.stats_[col])
            return out

        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 1
        if squeeze:
            X = X.reshape(-1, 1)

        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(self.columns_):
            out[:, i] = self._apply_one(X[:, i], self.stats_[col])

        return out.ravel() if squeeze else out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.fitted:
            raise RuntimeError("call fit() before inverse_transform()")

        if self.is_df:
            out = X.copy()
            for col in self.columns_:
                out[col] = self._undo_one(X[col].values, self.stats_[col])
            return out

        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 1
        if squeeze:
            X = X.reshape(-1, 1)

        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(self.columns_):
            out[:, i] = self._undo_one(X[:, i], self.stats_[col])

        return out.ravel() if squeeze else out

    def summary(self):
        if not self.fitted:
            raise RuntimeError("call fit() before summary()")
        return pd.DataFrame(
            [{"column": c, "method": s["method"]} for c, s in self.stats_.items()]
        )

    # everything below here is just internal helpers, don't call these directly

    def _fit_one(self, x, method):
        if method == "standard":
            mean, std = x.mean(), x.std()
            return {"method": "standard", "mean": mean, "std": std or 1.0}

        if method == "minmax":
            lo, hi = x.min(), x.max()
            span = hi - lo
            return {"method": "minmax", "min": lo, "max": hi, "span": span or 1.0}

        if method == "robust":
            med = np.median(x)
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            return {"method": "robust", "median": med, "iqr": iqr or 1.0}

        raise ValueError(f"unknown method {method}")

    def _apply_one(self, x, s):
        x = np.asarray(x, dtype=float)
        if s["method"] == "standard":
            return (x - s["mean"]) / s["std"]
        if s["method"] == "minmax":
            return (x - s["min"]) / s["span"]
        if s["method"] == "robust":
            return (x - s["median"]) / s["iqr"]

    def _undo_one(self, x, s):
        x = np.asarray(x, dtype=float)
        if s["method"] == "standard":
            return x * s["std"] + s["mean"]
        if s["method"] == "minmax":
            return x * s["span"] + s["min"]
        if s["method"] == "robust":
            return x * s["iqr"] + s["median"]

    def __repr__(self):
        return f"SDScaler(method='{self.method}')"
