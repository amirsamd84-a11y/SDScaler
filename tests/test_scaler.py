import numpy as np
import pandas as pd
import pytest

from sdscaler import SDScaler


def sample_df():
    return pd.DataFrame({
        "normal": [10, 12, 11, 13, 9, 10, 12, 11],
        "skewed": [1, 2, 1, 2, 1, 2, 1, 100],
        "outliers": [5, 6, 5, 7, 6, 5, 6, 500],
    })


def test_shape_stays_same():
    df = sample_df()
    out = SDScaler().fit_transform(df)
    assert out.shape == df.shape


def test_picks_valid_methods():
    df = sample_df()
    s = SDScaler().fit(df)
    for stat in s.stats_.values():
        assert stat["method"] in ("standard", "minmax", "robust")


def test_outlier_col_goes_robust():
    df = sample_df()
    s = SDScaler().fit(df)
    assert s.stats_["outliers"]["method"] == "robust"


def test_roundtrip():
    df = sample_df()
    s = SDScaler()
    scaled = s.fit_transform(df)
    back = s.inverse_transform(scaled)
    np.testing.assert_allclose(back.values, df.values, rtol=1e-6)


def test_forced_method():
    df = sample_df()
    s = SDScaler(method="standard").fit(df)
    for stat in s.stats_.values():
        assert stat["method"] == "standard"


def test_transform_needs_fit_first():
    with pytest.raises(RuntimeError):
        SDScaler().transform(sample_df())


def test_bad_method_name():
    with pytest.raises(ValueError):
        SDScaler(method="whatever")


def test_works_with_numpy():
    arr = np.array([[1, 2], [3, 4], [5, 100]])
    out = SDScaler().fit_transform(arr)
    assert out.shape == (3, 2)
