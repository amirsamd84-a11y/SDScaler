# figures out which scaling method fits a column best
# just checks for outliers and skew
import numpy as np


def skewness(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0

    mean = x.mean()
    std = x.std()
    if std == 0:
        return 0.0

    return np.mean(((x - mean) / std) ** 3)


def outlier_frac(x):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = (x < low) | (x > high)
    return outliers.sum() / len(x)


def pick_method(x, skew_cutoff=1.0, outlier_cutoff=0.05):
    if outlier_frac(x) > outlier_cutoff:
        return "robust"
    if abs(skewness(x)) > skew_cutoff:
        return "minmax"
    return "standard"
