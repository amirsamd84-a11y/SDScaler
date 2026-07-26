# figures out which scaling method fits a column best
# just checks for outliers and skew, nothing fancy

import numpy as np


def skewness(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0

    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0:
        return 0.0

    return np.nanmean(((x - mean) / std) ** 3)


def outlier_frac(x):
    # classic 1.5*IQR rule from stats class
    x = np.asarray(x, dtype=float)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = (x < low) | (x > high)
    return np.nansum(outliers) / len(x)


def pick_method(x, skew_cutoff=1.0, outlier_cutoff=0.05):
    if outlier_frac(x) > outlier_cutoff:
        return "robust"

    if abs(skewness(x)) > skew_cutoff:
        # log only makes sense if the column is strictly positive, otherwise
        # fall back to minmax like before
        if np.nanmin(x) > 0:
            return "log"
        return "minmax"

    return "standard"