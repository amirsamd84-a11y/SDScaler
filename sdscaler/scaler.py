import warnings

import numpy as np
import pandas as pd

from .helpers import pick_method

METHODS = ("auto", "standard", "minmax", "robust", "log")


class SDScaler:
    """
    basically StandardScaler/MinMaxScaler/RobustScaler but it picks one
    for you per column instead of you having to guess which one to use.

    scaler = SDScaler()
    scaled = scaler.fit_transform(df)
    scaler.summary()   # shows what got picked for each column

    set method to "standard"/"minmax"/"robust"/"log" if you want to force
    one instead of letting it decide (default is "auto")

    pass log_cols if you want specific columns log-scaled regardless of
    what auto-pick would've chosen - useful for stuff like income that's
    always heavily skewed. auto-pick will also choose log on its own for
    skewed, strictly-positive columns, log_cols just forces it.

    if X is a dataframe, non-numeric columns get left alone (passthrough)
    instead of crashing - only numeric columns actually get scaled.
    """

    def __init__(self, method="auto", outlier_cutoff=0.05, skew_cutoff=1.0, log_cols=None):
        if method not in METHODS:
            raise ValueError(f"method has to be one of {METHODS}, got '{method}'")

        self.method = method
        self.outlier_cutoff = outlier_cutoff
        self.skew_cutoff = skew_cutoff
        self.log_cols = log_cols or []

        self.stats_ = {}
        self.columns_ = None
        self.numeric_columns_ = None
        self.is_df = False
        self.fitted = False

    def fit(self, X):
        self.is_df = isinstance(X, pd.DataFrame)

        if self.is_df:
            self.columns_ = list(X.columns)
            self.numeric_columns_ = [
                c for c in self.columns_ if pd.api.types.is_numeric_dtype(X[c])
            ]
            nan_cols = [c for c in self.numeric_columns_ if X[c].isna().any()]
            if nan_cols:
                warnings.warn(
                    f"these columns have missing values, stats were computed "
                    f"ignoring them but the scaled output will still have NaN "
                    f"in those spots: {nan_cols}"
                )
            data_lookup = {c: X[c].to_numpy(dtype=float) for c in self.numeric_columns_}
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.columns_ = list(range(X.shape[1]))
            self.numeric_columns_ = list(self.columns_)
            data_lookup = {c: X[:, i] for i, c in enumerate(self.columns_)}

        self.stats_ = {}
        for col in self.numeric_columns_:
            col_data = data_lookup[col]
            if col in self.log_cols:
                method = "log"
            elif self.method == "auto":
                method = pick_method(col_data, self.skew_cutoff, self.outlier_cutoff)
            else:
                method = self.method
            self.stats_[col] = self._fit_one(col_data, method)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("call fit() before transform()")

        if self.is_df:
            out = X.copy()
            for col in self.numeric_columns_:
                out[col] = self._apply_one(X[col].to_numpy(dtype=float), self.stats_[col])
            return out

        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 1
        if squeeze:
            X = X.reshape(-1, 1)

        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(self.numeric_columns_):
            out[:, i] = self._apply_one(X[:, i], self.stats_[col])

        return out.ravel() if squeeze else out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.fitted:
            raise RuntimeError("call fit() before inverse_transform()")

        if self.is_df:
            out = X.copy()
            for col in self.numeric_columns_:
                out[col] = self._undo_one(X[col].to_numpy(dtype=float), self.stats_[col])
            return out

        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 1
        if squeeze:
            X = X.reshape(-1, 1)

        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(self.numeric_columns_):
            out[:, i] = self._undo_one(X[:, i], self.stats_[col])

        return out.ravel() if squeeze else out

    def summary(self):
        if not self.fitted:
            raise RuntimeError("call fit() before summary()")

        rows = []
        for col in self.columns_:
            if col in self.stats_:
                rows.append({"column": col, "method": self.stats_[col]["method"]})
            else:
                rows.append({"column": col, "method": "passthrough (not numeric)"})
        return pd.DataFrame(rows)

    def to_column_transformer(self):
        """
        builds an sklearn ColumnTransformer that mirrors whatever this
        scaler decided, in case you'd rather plug it into an sklearn
        Pipeline instead of calling .transform() directly. needs sklearn.
        non-numeric columns get passed through untouched, same as here.
        """
        if not self.fitted:
            raise RuntimeError("call fit() before to_column_transformer()")

        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        except ImportError:
            raise ImportError("to_column_transformer() needs scikit-learn installed")

        transformers = []
        for col, stat in self.stats_.items():
            if stat["method"] == "standard":
                transformers.append((f"{col}_standard", StandardScaler(), [col]))
            elif stat["method"] == "minmax":
                transformers.append((f"{col}_minmax", MinMaxScaler(), [col]))
            elif stat["method"] == "robust":
                transformers.append((f"{col}_robust", RobustScaler(), [col]))
            elif stat["method"] == "log":
                transformers.append((f"{col}_log", _LogScaler(), [col]))

        return ColumnTransformer(transformers, remainder="passthrough")

    # internal, don't call directly

    def _fit_one(self, x, method):
        if method == "standard":
            mean, std = np.nanmean(x), np.nanstd(x)
            return {"method": "standard", "mean": mean, "std": std or 1.0}

        if method == "minmax":
            lo, hi = np.nanmin(x), np.nanmax(x)
            span = hi - lo
            return {"method": "minmax", "min": lo, "max": hi, "span": span or 1.0}

        if method == "robust":
            med = np.nanmedian(x)
            q1, q3 = np.nanpercentile(x, [25, 75])
            iqr = q3 - q1
            return {"method": "robust", "median": med, "iqr": iqr or 1.0}

        if method == "log":
            m = np.nanmin(x)
            # shift so everything's positive before taking the log - if the
            # column's already all positive this is just offset=0 and does
            # nothing
            offset = (1 - m) if m <= 0 else 0.0
            shifted = np.clip(x + offset, 1e-6, None)
            logged = np.log(shifted)
            mean, std = np.nanmean(logged), np.nanstd(logged)
            return {"method": "log", "offset": offset, "mean": mean, "std": std or 1.0}

        raise ValueError(f"unknown method {method}")

    def _apply_one(self, x, s):
        x = np.asarray(x, dtype=float)
        if s["method"] == "standard":
            return (x - s["mean"]) / s["std"]
        if s["method"] == "minmax":
            return (x - s["min"]) / s["span"]
        if s["method"] == "robust":
            return (x - s["median"]) / s["iqr"]
        if s["method"] == "log":
            shifted = np.clip(x + s["offset"], 1e-6, None)
            logged = np.log(shifted)
            return (logged - s["mean"]) / s["std"]

    def _undo_one(self, x, s):
        x = np.asarray(x, dtype=float)
        if s["method"] == "standard":
            return x * s["std"] + s["mean"]
        if s["method"] == "minmax":
            return x * s["span"] + s["min"]
        if s["method"] == "robust":
            return x * s["iqr"] + s["median"]
        if s["method"] == "log":
            logged = x * s["std"] + s["mean"]
            shifted = np.exp(logged)
            return shifted - s["offset"]

    def __repr__(self):
        return f"SDScaler(method='{self.method}')"


class _LogScaler:
    """
    small helper class used by to_column_transformer() for log-method
    columns - sklearn's ColumnTransformer needs something with its own
    fit/transform, this is just our log logic wrapped up that way.
    """

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float).ravel()
        m = np.nanmin(x)
        self.offset_ = (1 - m) if m <= 0 else 0.0
        shifted = np.clip(x + self.offset_, 1e-6, None)
        logged = np.log(shifted)
        self.mean_ = np.nanmean(logged)
        self.std_ = np.nanstd(logged) or 1.0
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float).ravel()
        shifted = np.clip(x + self.offset_, 1e-6, None)
        logged = np.log(shifted)
        return ((logged - self.mean_) / self.std_).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self